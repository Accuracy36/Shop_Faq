"""第一阶段：在公开句对数据（LCQMC + BQ + AFQMC）上训练通用的句子匹配双塔模型。

示例：
    python src/train_matcher.py --arch transformer --loss both --epochs 10 --lr 1e-3 --name s1_transformer
    python src/train_matcher.py --arch roberta --loss both --epochs 3 --lr 3e-5 --name s1_roberta_seed1 --seed 1
"""
import argparse
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from common import MODELS, PROC, load_faq, read_jsonl, set_seed, write_json
from models import SentenceEncoder, cosent_loss, count_parameters, info_nce_loss
from retrieval import DenseScorer, FaqIndex, best_threshold, load_eval_sets, retrieval_metrics


def make_optimizer(model, lr, weight_decay=0.01):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 or 'norm' in name.lower() else decay).append(p)
    return torch.optim.AdamW([dict(params=decay, weight_decay=weight_decay),
                              dict(params=no_decay, weight_decay=0.0)], lr=lr)


def make_scheduler(optimizer, total_steps, warmup_ratio=0.05):
    warmup = max(1, int(total_steps * warmup_ratio))

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup))))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def make_batches(train, batch_size, rng, chunk=50):
    """打乱后每 chunk 个批次的数据按句长排序再切批，批次顺序再打乱。

    同一批内句长接近，补齐的位置少，训练明显更快；批内负例仍来自随机的一小块数据。
    """
    order = rng.permutation(len(train))
    batches = []
    for start in range(0, len(order), batch_size * chunk):
        part = sorted(order[start:start + batch_size * chunk], key=lambda i: train[i]['len'])
        batches += [part[k:k + batch_size] for k in range(0, len(part), batch_size)]
    batches = [b for b in batches if len(b) == batch_size]
    return [batches[i] for i in rng.permutation(len(batches))]


def encode_pairs(model, texts_a, texts_b, device):
    """两侧句子拼成一个批次前向一次，再拆开。"""
    batch = {k: v.to(device) for k, v in model.tokenize(list(texts_a) + list(texts_b)).items()}
    emb = model(**batch).float()
    return emb[:len(texts_a)], emb[len(texts_a):]


@torch.no_grad()
def evaluate_dev(model, dev_pairs, faq, eval_sets):
    """验证集指标：句对 AUC，以及真实问句验证集上的 FAQ 检索指标（索引含相似问）。"""
    emb_a = model.encode([r['a'] for r in dev_pairs])
    emb_b = model.encode([r['b'] for r in dev_pairs])
    sim = (emb_a * emb_b).sum(1)
    labels = np.array([r['label'] for r in dev_pairs])
    out = dict(pair_auc=float(roc_auc_score(labels, sim)))
    scorer = DenseScorer(model)
    for tag, use_similar in (('std', False), ('exp', True)):
        index = FaqIndex(faq, scorer, use_similar=use_similar)
        scores, _ = index.score([r['query'] for r in eval_sets['real_dev']])
        m = retrieval_metrics(scores, eval_sets['real_dev'])
        out[f'faq_top1_{tag}'], out[f'faq_mrr_{tag}'] = m['top1'], m['mrr']
        if use_similar:
            out['faq_e2e_acc'] = best_threshold(scores, eval_sets['real_dev'])[1]
    # 选模型用的验证分数：检索 MRR 与端到端准确率（均在真实问句验证集上、索引含相似问）的平均
    out['dev_score'] = (out['faq_mrr_exp'] + out['faq_e2e_acc']) / 2
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', required=True, choices=['bilstm', 'transformer', 'rbt3', 'roberta'])
    ap.add_argument('--loss', default='both', choices=['cosent', 'infonce', 'both'])
    ap.add_argument('--pooling', default='mean', choices=['mean', 'cls'])
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--temperature', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--evals-per-epoch', type=int, default=1)
    ap.add_argument('--max-train', type=int, default=0, help='只用前若干条训练数据（调试用）')
    ap.add_argument('--init', default='', help='从已有模型目录继续训练')
    ap.add_argument('--name', required=True)
    ap.add_argument('--grad-ckpt', action='store_true', help='梯度检查点：省显存，计算结果不变')
    ap.add_argument('--mem-frac', type=float, default=0.55, help='本进程最多使用的显存比例')
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        # 显存上限设为 4.5 GB 左右：超出时直接报错，而不是溢出到共享内存后变得极慢
        torch.cuda.set_per_process_memory_fraction(args.mem_frac)
    out_dir = MODELS / args.name
    train = read_jsonl(PROC / 'pairs_train.jsonl')
    if args.max_train:
        train = train[:args.max_train]
    dev_pairs = read_jsonl(PROC / 'pairs_dev.jsonl')
    faq, eval_sets = load_faq(), load_eval_sets()
    text_id = {}
    for r in train:
        r['ia'] = text_id.setdefault(r['a'], len(text_id))
        r['ib'] = text_id.setdefault(r['b'], len(text_id))
        r['len'] = max(len(r['a']), len(r['b']))

    model = SentenceEncoder.load(MODELS / args.init, device) if args.init else SentenceEncoder(args.arch, args.pooling)
    model.to(device).train()
    if args.grad_ckpt:
        model.enable_gradient_checkpointing()
    steps_per_epoch = len(train) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    optimizer = make_optimizer(model, args.lr)
    scheduler = make_scheduler(optimizer, total_steps)
    scaler = torch.amp.GradScaler(enabled=device.type == 'cuda')
    eval_every = max(1, steps_per_epoch // args.evals_per_epoch)
    print(f'{args.name}: {count_parameters(model):,} 参数, {len(train)} 句对, {total_steps} 步', flush=True)

    rng = np.random.RandomState(args.seed)
    history, curve, best, step, window = [], [], None, 0, []
    start = time.time()
    for epoch in range(args.epochs):
        batches = make_batches(train, args.batch_size, rng)
        for b in range(steps_per_epoch):
            rows = [train[i] for i in batches[b]]
            labels = torch.tensor([r['label'] for r in rows], device=device)
            ids = (torch.tensor([r['ia'] for r in rows], device=device),
                   torch.tensor([r['ib'] for r in rows], device=device))
            with torch.autocast(device.type, enabled=device.type == 'cuda'):
                emb_a, emb_b = encode_pairs(model, [r['a'] for r in rows], [r['b'] for r in rows], device)
            loss = emb_a.new_zeros(())
            if args.loss in ('cosent', 'both'):
                loss = loss + cosent_loss(F.cosine_similarity(emb_a, emb_b), labels)
            if args.loss in ('infonce', 'both'):
                loss = loss + info_nce_loss(emb_a, emb_b, labels, args.temperature, ids)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            window.append(loss.item())
            if step % 50 == 0:
                curve.append(dict(step=step, loss=float(np.mean(window[-50:]))))
            if step % 500 == 0:
                print(f'  step {step}/{total_steps}  loss {np.mean(window[-500:]):.4f}  '
                      f'{time.time() - start:.0f}s', flush=True)
            if step % eval_every == 0 or step == total_steps:
                metrics = evaluate_dev(model, dev_pairs, faq, eval_sets)
                record = dict(step=step, epoch=step / steps_per_epoch, train_loss=float(np.mean(window)),
                              seconds=time.time() - start, **metrics)
                window = []
                history.append(record)
                # 以验证分数选择模型，并列时取靠后的
                if best is None or record['dev_score'] >= best['dev_score']:
                    best = record
                    model.save(out_dir, extra=dict(stage='pairs', args=vars(args), best=record,
                                                   parameters=count_parameters(model)))
                print(' '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}' for k, v in record.items()),
                      flush=True)
                model.train()
    write_json(out_dir / 'history.json', dict(args=vars(args), history=history, loss_curve=curve, best=best,
                                              train_seconds=time.time() - start,
                                              peak_memory_gb=torch.cuda.max_memory_allocated() / 2 ** 30
                                              if device.type == 'cuda' else None,
                                              parameters=count_parameters(model)))
    print(f'最佳：step {best["step"]}  {best}', flush=True)


if __name__ == '__main__':
    import sys
    import warnings
    sys.stdout.reconfigure(encoding='utf-8')
    warnings.filterwarnings('ignore')
    try:
        import transformers
        transformers.logging.set_verbosity_error()
    except ImportError:
        pass
    main()
