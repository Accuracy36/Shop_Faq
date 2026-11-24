"""句向量编码器与损失函数。

三类编码器共用同一接口：tokenize(texts) 得到张量字典，forward(**batch) 得到句向量（未归一化）。
    bilstm       字嵌入 + 双向 LSTM，从零训练
    transformer  字嵌入 + 自己实现的 Transformer 编码器，从零训练
    rbt3/roberta 预训练中文 RoBERTa（3 层 / 12 层），微调
"""
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import MAX_LEN, PRETRAINED, PROC, normalize, read_json, write_json


def masked_mean(hidden, mask):
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)


class CharTokenizer:
    """按字切分。0 为 [PAD]，1 为 [UNK]。"""

    def __init__(self, vocab):
        self.vocab = vocab
        self.index = {c: i for i, c in enumerate(vocab)}

    def __call__(self, texts):
        ids = [[self.index.get(c, 1) for c in t[:MAX_LEN]] or [1] for t in texts]
        width = max(len(x) for x in ids)
        out = torch.zeros(len(ids), width, dtype=torch.long)
        for i, x in enumerate(ids):
            out[i, :len(x)] = torch.tensor(x)
        return dict(input_ids=out, attention_mask=(out != 0).long())


class BiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, dim=256, layers=2, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.LSTM(dim, dim, num_layers=layers, batch_first=True, bidirectional=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.proj = nn.Linear(4 * dim, dim)

    def forward(self, input_ids, attention_mask):
        lengths = attention_mask.sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(self.drop(self.embed(input_ids)), lengths, batch_first=True,
                                                   enforce_sorted=False)
        hidden, _ = self.rnn(packed)
        hidden, _ = nn.utils.rnn.pad_packed_sequence(hidden, batch_first=True, total_length=input_ids.size(1))
        mean = masked_mean(hidden, attention_mask)
        peak = hidden.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e4).max(1).values
        return self.proj(torch.cat([mean, peak], dim=-1))


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力：Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V。"""

    def __init__(self, dim, heads, dropout):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.head_dim = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        batch, length, dim = x.shape
        q, k, v = self.qkv(x).view(batch, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)          # (batch, heads, length, length)
        scores = scores.masked_fill(mask[:, None, None, :] == 0, torch.finfo(scores.dtype).min)
        weights = self.drop(scores.softmax(-1))
        return self.out((weights @ v).transpose(1, 2).reshape(batch, length, dim))


class TransformerLayer(nn.Module):
    """Pre-LN 结构：x + Attn(LN(x))，再 x + FFN(LN(x))。"""

    def __init__(self, dim, heads, ffn, dropout):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads, dropout)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = x + self.drop(self.attn(self.norm1(x), mask))
        return x + self.drop(self.ffn(self.norm2(x)))


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, dim=256, heads=4, layers=4, ffn=1024, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.pos = nn.Embedding(MAX_LEN, dim)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList(TransformerLayer(dim, heads, ffn, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)
        self.scale = math.sqrt(dim)
        nn.init.normal_(self.embed.weight, std=dim ** -0.5)
        nn.init.normal_(self.pos.weight, std=dim ** -0.5)
        with torch.no_grad():
            self.embed.weight[0].zero_()

    def forward(self, input_ids, attention_mask):
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        x = self.drop(self.embed(input_ids) * self.scale + self.pos(positions))
        for layer in self.layers:
            x = layer(x, attention_mask)
        return masked_mean(self.norm(x), attention_mask)


class BertEncoder(nn.Module):
    """预训练 BERT 类模型 + 池化（cls：取 [CLS] 位置的输出；mean：对有效位置取平均）。"""

    def __init__(self, path, pooling='mean'):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(str(path), add_pooling_layer=False)
        self.pooling = pooling

    def forward(self, input_ids, attention_mask):
        hidden = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return hidden[:, 0] if self.pooling == 'cls' else masked_mean(hidden, attention_mask)


class SentenceEncoder(nn.Module):
    """编码器 + 分词器 + 配置，负责保存、载入和批量编码。"""

    def __init__(self, arch, pooling='mean'):
        super().__init__()
        self.config = dict(arch=arch, pooling=pooling)
        if arch in PRETRAINED:
            from transformers import AutoTokenizer
            self.net = BertEncoder(PRETRAINED[arch], pooling)
            self.hf_tokenizer = AutoTokenizer.from_pretrained(str(PRETRAINED[arch]))
        else:
            vocab = read_json(PROC / 'char_vocab.json')
            self.char_tokenizer = CharTokenizer(vocab)
            self.net = BiLSTMEncoder(len(vocab)) if arch == 'bilstm' else TransformerEncoder(len(vocab))

    def enable_gradient_checkpointing(self):
        """梯度检查点（只对预训练模型）：前向时不保存各层的中间结果，反向传播时重算。

        重算时恢复了随机数状态（Dropout 的掩码相同），所以计算结果与不用时一样，只是用约三成的时间换显存。
        """
        if self.config['arch'] in PRETRAINED:
            self.net.bert.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    def tokenize(self, texts):
        if self.config['arch'] in PRETRAINED:
            enc = self.hf_tokenizer(list(texts), padding=True, truncation=True, max_length=MAX_LEN + 2,
                                    return_tensors='pt')
            return dict(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'])
        return self.char_tokenizer(texts)

    def forward(self, input_ids, attention_mask):
        return self.net(input_ids, attention_mask)

    @torch.no_grad()
    def encode(self, texts, batch_size=256, normalized=True):
        """把规范化后的句子编码为 L2 归一化的向量（numpy，float32）。"""
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        texts = [normalize(t) for t in texts]
        order = np.argsort([len(t) for t in texts])
        out = np.zeros((len(texts), 0), dtype=np.float32)
        for start in range(0, len(texts), batch_size):
            idx = order[start:start + batch_size]
            batch = {k: v.to(device) for k, v in self.tokenize([texts[i] for i in idx]).items()}
            with torch.autocast(device.type, enabled=device.type == 'cuda'):
                emb = self(**batch).float()
            if normalized:
                emb = F.normalize(emb, dim=-1)
            if out.shape[1] == 0:
                out = np.zeros((len(texts), emb.size(1)), dtype=np.float32)
            out[idx] = emb.cpu().numpy()
        self.train(was_training)
        return out

    def save(self, directory, extra=None):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), directory / 'encoder.pt')
        write_json(directory / 'encoder_config.json', {**self.config, **(extra or {})})

    @classmethod
    def load(cls, directory, device='cpu'):
        directory = Path(directory)
        config = read_json(directory / 'encoder_config.json')
        model = cls(config['arch'], config['pooling'])
        model.net.load_state_dict(torch.load(directory / 'encoder.pt', map_location='cpu', weights_only=True))
        return model.to(device).eval()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# ----------------------------- 损失函数 -----------------------------

def cosent_loss(sim, labels, scale=20.0):
    """CoSENT 排序损失：要求批内每个正例对的余弦相似度都高于每个负例对。

    loss = log(1 + sum_{i 为负例, j 为正例} exp(scale * (sim_i - sim_j)))
    """
    sim = sim * scale
    diff = sim[:, None] - sim[None, :]                       # diff[i, j] = sim_i - sim_j
    wanted = (labels[:, None] == 0) & (labels[None, :] == 1)   # i 负例、j 正例
    diff = diff.masked_fill(~wanted, -1e12).flatten()
    return torch.logsumexp(torch.cat([diff.new_zeros(1), diff]), dim=0)


def info_nce_loss(emb_a, emb_b, labels=None, temperature=0.05, text_ids=None):
    """批内负例的对比损失（InfoNCE，双向）。

    每个正例对 (a_i, b_i) 以批内其他句子的 b_j（及反方向的 a_j）作为负例。
    labels 给出时只有标签为 1 的句对作为锚点，标签为 0 的句对只提供负例；
    text_ids 给出时，与正例文本相同的候选不计作负例。
    """
    emb_a, emb_b = F.normalize(emb_a, dim=-1), F.normalize(emb_b, dim=-1)
    logits = emb_a @ emb_b.T / temperature
    if text_ids is not None:
        ids_a, ids_b = text_ids
        same_b = ids_b[:, None] == ids_b[None, :]              # 候选 b_j 与正例 b_i 文本相同
        same_a = ids_a[:, None] == ids_a[None, :]
        eye = torch.eye(len(logits), dtype=torch.bool, device=logits.device)
        logits_ab = logits.masked_fill(same_b & ~eye, -1e4)
        logits_ba = logits.T.masked_fill(same_a & ~eye, -1e4)
    else:
        logits_ab, logits_ba = logits, logits.T
    target = torch.arange(len(logits), device=logits.device)
    loss = F.cross_entropy(logits_ab, target, reduction='none') + F.cross_entropy(logits_ba, target, reduction='none')
    if labels is not None:
        keep = labels == 1
        if not keep.any():
            return logits.sum() * 0.0
        loss = loss[keep]
    return loss.mean() / 2
