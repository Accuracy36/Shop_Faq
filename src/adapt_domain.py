"""第二阶段：领域适配。从第一阶段的模型出发，使用店铺 FAQ、标注过的真实问句和淘宝客服语料（ECD）。

    faq     有监督对比学习：每条 FAQ 是一个类别，标准问、相似问互为正例；
            --real 加入标注过的真实问句（范围内的作正例）；--oos 每步再取若干范围外真实问句作负例；
            --pseudo 加入伪标注句子
    simcse  无监督 SimCSE：ECD 的无标签用户发言，同一句话两次 Dropout 前向互为正例
    mine    伪标注：用模型为 ECD 用户发言检索 FAQ，把高置信度的句子记为该 FAQ 的真实说法（不训练）

示例：
    python src/adapt_domain.py --mode faq --init s1_roberta_seed1 --name s2_faq_seed1
    python src/adapt_domain.py --mode faq --init s1_roberta_seed1 --real --oos 60 --name s2_real_oos_seed1
    python src/adapt_domain.py --mode simcse --init s1_roberta_seed1 --name s2_simcse_seed1
    python src/adapt_domain.py --mode mine --init s2_real_oos_seed1 --name pseudo_seed1
"""
import argparse
import collections
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import FAQ_DIR, MODELS, PROC, RESULTS, load_faq, normalize, read_json, read_jsonl, set_seed, write_json
from models import SentenceEncoder, count_parameters, info_nce_loss
from retrieval import DenseScorer, FaqIndex, best_threshold, load_eval_sets, load_real_train, retrieval_metrics
from train_matcher import make_optimizer


def load_ecd_utterances(min_len=4):
    rows = []
    for line in open(PROC / 'ecd_train_utts.tsv', encoding='utf-8'):
        text, count = line.rstrip('\n').split('\t')
        if len(text) >= min_len:
            rows.append((text, int(count)))
    return rows


@torch.no_grad()
def dev_metrics(model, faq, eval_sets, extra=None, reject=None):
    """真实问句验证集上的指标；索引配置（是否含真实问句、拒识样例）与最终使用时一致。"""
    index = FaqIndex(faq, DenseScorer(model), use_similar=True, extra=extra, reject=reject)
    scores, _, decision = index.score_full([r['query'] for r in eval_sets['real_dev']])
    m = retrieval_metrics(scores, eval_sets['real_dev'])
    threshold, acc = best_threshold(decision, eval_sets['real_dev'])
    return dict(faq_top1_exp=m['top1'], faq_mrr_exp=m['mrr'], faq_e2e_acc=acc, threshold=threshold,
                dev_score=(m['mrr'] + acc) / 2)


def mine_pseudo_labels(model, faq, eval_sets, extra, precision=0.95, per_faq=30):
    """给 ECD 用户发言打伪标签。

    置信度阈值在真实问句验证集上确定：取使“最高相似度不低于阈值的问句中，排名第一的 FAQ 正确”的比例
    达到 precision 的最小阈值。每条 FAQ 最多保留 per_faq 句，优先保留在语料中出现次数多的说法。
    """
    index = FaqIndex(faq, DenseScorer(model), use_similar=True, extra=extra)
    dev = eval_sets['real_dev']
    scores, _ = index.score([r['query'] for r in dev])
    conf, top = scores.max(1), scores.argmax(1)
    correct = np.array([bool(r['accept']) and top[i] in r['accept'] for i, r in enumerate(dev)])
    threshold, stats = None, None
    for t in np.sort(np.unique(conf)):
        keep = conf >= t
        if keep.sum() >= 20 and correct[keep].mean() >= precision:
            threshold, stats = float(t), dict(dev_selected=int(keep.sum()), dev_precision=float(correct[keep].mean()))
            break
    if threshold is None:
        raise SystemExit('验证集上达不到要求的精度')
    known = {normalize(t) for t in index.texts} | {r['query'] for r in read_jsonl(FAQ_DIR / 'real_train.jsonl')}
    utterances = [(t, c) for t, c in load_ecd_utterances() if t not in known]
    scores, _ = index.score([t for t, _ in utterances])
    conf, top = scores.max(1), scores.argmax(1)
    picked = collections.defaultdict(list)
    for i in np.argsort([-c for _, c in utterances], kind='stable'):
        if conf[i] >= threshold and len(picked[int(top[i])]) < per_faq:
            picked[int(top[i])].append(dict(text=utterances[i][0], faq_id=int(top[i]), key=faq[int(top[i])]['key'],
                                            score=round(float(conf[i]), 4), count=utterances[i][1]))
    rows = [r for faq_id in sorted(picked) for r in picked[faq_id]]
    stats.update(threshold=threshold, candidates=len(utterances), above_threshold=int((conf >= threshold).sum()),
                 selected=len(rows), faqs_covered=len(picked))
    return rows, stats


def contrastive_loss(emb_a, emb_b, emb_o, blocked, temperature, oos_margin, oos_weight):
    """每个类别一对 (a_i, b_i)。候选包括其他类别的句子和范围外句子 o_k；blocked[i, j] 为真表示
    类别 j 的句子也是 a_i 或 b_i 可以接受的答案，不当作负例。范围外句子另加一项间隔损失，
    要求它与任何 FAQ 句子的余弦相似度不超过 oos_margin。"""
    a, b = F.normalize(emb_a, dim=-1), F.normalize(emb_b, dim=-1)
    logits = (a @ b.T / temperature).masked_fill(blocked, -1e4)
    logits_ab, logits_ba = logits, logits.T
    extra = a.new_zeros(())
    if emb_o is not None:
        o = F.normalize(emb_o, dim=-1)
        sim_ao, sim_bo = a @ o.T, b @ o.T
        logits_ab = torch.cat([logits_ab, sim_ao / temperature], dim=1)
        logits_ba = torch.cat([logits_ba, sim_bo / temperature], dim=1)
        if oos_weight > 0:
            extra = oos_weight * (F.relu(sim_ao - oos_margin).mean() + F.relu(sim_bo - oos_margin).mean()) / 2
    target = torch.arange(len(a), device=a.device)
    return (F.cross_entropy(logits_ab, target) + F.cross_entropy(logits_ba, target)) / 2 + extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['faq', 'simcse', 'mine'])
    ap.add_argument('--init', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--real', action='store_true', help='训练时加入标注过的真实问句（范围内的作正例）')
    ap.add_argument('--oos', type=int, default=0, help='每步加入多少个范围外真实问句作负例')
    ap.add_argument('--oos-margin', type=float, default=0.3)
    ap.add_argument('--oos-weight', type=float, default=1.0)
    ap.add_argument('--pseudo', default='', help='伪标注文件名（results/pseudo/ 下，不含扩展名），并入 FAQ 训练')
    ap.add_argument('--real-index', action='store_true', help='验证时索引中加入真实训练问句')
    ap.add_argument('--reject', action='store_true', help='验证时使用拒识样例')
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--eval-every', type=int, default=25)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--batch-size', type=int, default=128, help='simcse 的批大小')
    ap.add_argument('--temperature', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--grad-ckpt', action='store_true', help='梯度检查点：省显存，计算结果不变')
    ap.add_argument('--mem-frac', type=float, default=0.55, help='本进程最多使用的显存比例')
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    faq, eval_sets = load_faq(), load_eval_sets()
    if args.init.startswith('raw_'):                    # 不经过第一阶段，直接从预训练模型开始
        model = SentenceEncoder(args.init[4:], 'mean').to(device)
    else:
        model = SentenceEncoder.load(MODELS / args.init, device)
    real_in, real_out = load_real_train()
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(args.mem_frac)
    if args.grad_ckpt:
        model.enable_gradient_checkpointing()
    index_extra = real_in if args.real_index else None
    if args.pseudo and args.real_index:           # 伪标注句子参与训练时，验证用的索引也包含它们，与最终评测一致
        index_extra = real_in + [(r['text'], r['faq_id'])
                                 for r in read_json(RESULTS / 'pseudo' / f'{args.pseudo}.json')['rows']]
    index_reject = real_out if args.reject else None

    if args.mode == 'mine':
        rows, stats = mine_pseudo_labels(model, faq, eval_sets, index_extra)
        write_json(RESULTS / 'pseudo' / f'{args.name}.json', dict(init=args.init, stats=stats, rows=rows))
        print(stats, flush=True)
        return

    # 每个类别的句子，以及每个句子可以接受的全部类别（真实问句可能对应多条 FAQ）
    groups = {e['id']: [normalize(s) for s in [e['question']] + e['similar']] for e in faq}
    accept = {t: {faq_id} for faq_id, texts in groups.items() for t in texts}
    if args.real:
        for r in read_jsonl(FAQ_DIR / 'real_train.jsonl'):
            if r['accept'] and r['query'] not in accept:
                groups[r['faq_id']].append(r['query'])
                accept[r['query']] = set(r['accept'])
    if args.pseudo:
        for r in read_json(RESULTS / 'pseudo' / f'{args.pseudo}.json')['rows']:
            if r['text'] not in accept:
                groups[r['faq_id']].append(r['text'])
                accept[r['text']] = {r['faq_id']}
    class_ids = list(groups)
    utterances = [t for t, _ in load_ecd_utterances()] if args.mode == 'simcse' else None
    rng = np.random.RandomState(args.seed)
    optimizer = make_optimizer(model, args.lr)
    scaler = torch.amp.GradScaler(enabled=device.type == 'cuda')
    out_dir = MODELS / args.name

    def save(record):
        model.save(out_dir, extra=dict(stage=args.mode, args=vars(args), best=record,
                                       parameters=count_parameters(model)))

    def embed(texts):
        return model(**{k: v.to(device) for k, v in model.tokenize(texts).items()}).float()

    history = [dict(step=0, train_loss=None, seconds=0.0, **dev_metrics(model, faq, eval_sets, index_extra,
                                                                         index_reject))]
    best = history[0]
    save(best)
    print(history[0], flush=True)
    start, window = time.time(), []
    model.train()
    for step in range(1, args.steps + 1):
        with torch.autocast(device.type, enabled=device.type == 'cuda'):
            if args.mode == 'faq':
                pairs = [rng.choice(len(groups[c]), 2, replace=False) for c in class_ids]
                a = [groups[c][i] for c, (i, _) in zip(class_ids, pairs)]
                b = [groups[c][j] for c, (_, j) in zip(class_ids, pairs)]
                blocked = torch.tensor([[i != j and (cj in accept[a[i]] or cj in accept[b[i]]
                                                     or ci in accept[a[j]] or ci in accept[b[j]])
                                         for j, cj in enumerate(class_ids)] for i, ci in enumerate(class_ids)],
                                       device=device)
                oos = [real_out[i] for i in rng.choice(len(real_out), args.oos, replace=False)] if args.oos else None
                loss = contrastive_loss(embed(a), embed(b), embed(oos) if oos else None, blocked, args.temperature,
                                        args.oos_margin, args.oos_weight)
            else:
                batch = [utterances[i] for i in rng.choice(len(utterances), args.batch_size, replace=False)]
                # 同一批句子前向两次，两次的 Dropout 不同，得到正例对
                loss = info_nce_loss(embed(batch), embed(batch), temperature=args.temperature)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        window.append(loss.item())
        if step % args.eval_every == 0 or step == args.steps:
            record = dict(step=step, train_loss=float(np.mean(window)), seconds=time.time() - start,
                          **dev_metrics(model, faq, eval_sets, index_extra, index_reject))
            window = []
            history.append(record)
            if record['dev_score'] > best['dev_score']:          # 严格更好才替换；选中 step 0 表示不做适配
                best = record
                save(best)
            print(' '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}' for k, v in record.items()),
                  flush=True)
            model.train()
    write_json(out_dir / 'history.json', dict(args=vars(args), history=history, best=best,
                                              group_sizes=[len(groups[c]) for c in class_ids]))
    print(f'最佳：step {best["step"]}  dev_score {best["dev_score"]:.4f}', flush=True)


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
