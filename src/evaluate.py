"""评测一个模型或基线，结果写入 results/eval/<名称>.json，逐条预测写入 results/eval/<名称>_pred.jsonl。

    python src/evaluate.py --model s1_roberta_seed1 --latency
    python src/evaluate.py --baseline tfidf        # tfidf / bm25 / raw_rbt3 / raw_roberta（未微调的预训练模型）

指标：
    句对判定（LCQMC、BQ、AFQMC 测试集）：AUC、准确率、F1（阈值在验证集上选取）
    FAQ 检索（只看知识库范围内的问句）：Top-1、Top-3、MRR@10；索引分“只含标准问”和“标准问+相似问”两种
    端到端（含范围外问句）：在真实问句验证集上选阈值，在真实问句测试集上统计
        准确率 =（答对的范围内问句 + 被拒答的范围外问句）/ 全部问句
    开放域误答率：500 个 LCQMC 测试集问句（与购物无关）中被回答的比例
"""
import argparse
import random
import time

import numpy as np
import torch

from common import MODELS, PROC, RESULTS, load_faq, read_jsonl, write_json, write_jsonl
from models import SentenceEncoder, count_parameters
from retrieval import (BM25Scorer, DenseScorer, FaqIndex, TfidfScorer, best_threshold, end_to_end, load_eval_sets,
                       load_real_train, oos_auroc, outcomes, pair_metrics, retrieval_metrics)


def open_domain_queries(n=500, seed=2026):
    rows = [r for r in read_jsonl(PROC / 'pairs_test.jsonl') if r['source'] == 'LCQMC']
    texts = sorted({r['a'] for r in rows})
    random.Random(seed).shuffle(texts)
    return texts[:n]


def evaluate_pairs(similarity):
    """similarity(texts_a, texts_b) 返回逐对相似度。"""
    dev, test = read_jsonl(PROC / 'pairs_dev.jsonl'), read_jsonl(PROC / 'pairs_test.jsonl')
    sim_dev = similarity([r['a'] for r in dev], [r['b'] for r in dev])
    sim_test = similarity([r['a'] for r in test], [r['b'] for r in test])
    y_dev, y_test = np.array([r['label'] for r in dev]), np.array([r['label'] for r in test])
    src_dev, src_test = np.array([r['source'] for r in dev]), np.array([r['source'] for r in test])
    out = {}
    for name in ('LCQMC', 'BQ', 'AFQMC'):
        d = src_dev == name if (src_dev == name).any() else np.ones(len(dev), dtype=bool)   # AFQMC 没有验证集
        t = src_test == name
        out[name] = pair_metrics(sim_dev[d], y_dev[d], sim_test[t], y_test[t])
    out['mean_auc'] = float(np.mean([out[k]['auc'] for k in ('LCQMC', 'BQ', 'AFQMC')]))
    out['mean_accuracy'] = float(np.mean([out[k]['accuracy'] for k in ('LCQMC', 'BQ', 'AFQMC')]))
    return out


def evaluate_faq(scorer, faq, sets, extra=None, reject=None):
    out, predictions = {}, []
    open_queries = open_domain_queries()
    for tag, use_similar in (('std', False), ('exp', True)):
        index = FaqIndex(faq, scorer, use_similar=use_similar, extra=extra if use_similar else None,
                         reject=reject if use_similar else None)
        scores = {name: index.score_full([r['query'] for r in rows]) for name, rows in sets.items()}
        res = {name: retrieval_metrics(scores[name][0], rows) for name, rows in sets.items()}
        threshold, dev_acc = best_threshold(scores['real_dev'][2], sets['real_dev'])
        res['threshold'] = threshold
        res['e2e_dev'] = end_to_end(scores['real_dev'][2], sets['real_dev'], threshold)
        res['e2e_test'] = end_to_end(scores['real_test'][2], sets['real_test'], threshold)
        res['oos_auroc_test'] = oos_auroc(scores['real_test'][2], sets['real_test'])
        open_scores = index.score_full(open_queries)[2]
        res['open_domain_answer_rate'] = float((open_scores.max(1) >= threshold).mean())
        res['index_size'] = len(index.texts)
        out[tag] = res
        if use_similar:
            for name in ('constructed', 'real_test'):
                s, best, decision = scores[name]
                result = outcomes(decision, sets[name], threshold)
                for i, row in enumerate(sets[name]):
                    top = np.argsort(-s[i])[:3]
                    predictions.append(dict(
                        set=name, query=row['query'], gold=row['key'], outcome=result[i],
                        in_scope=bool(row['accept']), hit=bool(row['accept']) and int(top[0]) in row['accept'],
                        confidence=round(float(decision[i].max()), 4),
                        top=[dict(key=faq[j]['key'], score=round(float(s[i, j]), 4), matched=index.texts[best[i, j]])
                             for j in top]))
    return out, predictions


def measure_latency(encoder, faq, queries, device, extra=None, reject=None, threads=4):
    """单条查询的平均耗时（规范化、编码、与索引中全部句子比较、按 FAQ 取最高分），索引与评测时相同。

    CPU 上限定 threads 个线程，模拟普通服务器上的单个服务进程。
    """
    encoder.to(device)
    previous = torch.get_num_threads()
    if device == 'cpu':
        torch.set_num_threads(threads)
    index = FaqIndex(faq, DenseScorer(encoder), use_similar=True, extra=extra, reject=reject)
    for q in queries[:10]:
        index.score_full([q])
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    for q in queries:
        index.score_full([q])
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = 1000 * (time.perf_counter() - start) / len(queries)
    torch.set_num_threads(previous)
    return elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='')
    ap.add_argument('--baseline', default='', choices=['', 'tfidf', 'bm25', 'raw_rbt3', 'raw_roberta'])
    ap.add_argument('--name', default='')
    ap.add_argument('--latency', action='store_true')
    ap.add_argument('--extra-index', default='', help='伪标注文件名（results/pseudo/ 下），其中的句子加入索引')
    ap.add_argument('--real-index', action='store_true', help='把真实训练问句（范围内）加入索引')
    ap.add_argument('--reject', action='store_true', help='把真实训练问句（范围外）作为拒识样例')
    args = ap.parse_args()
    name = args.name or args.model or args.baseline
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    faq, sets = load_faq(), load_eval_sets()
    result = dict(name=name)
    extra, encoder, reject = [], None, None
    real_in, real_out = load_real_train()
    if args.real_index:
        extra += real_in
    if args.reject:
        reject = real_out

    if args.baseline in ('tfidf', 'bm25'):
        scorer = TfidfScorer() if args.baseline == 'tfidf' else BM25Scorer()
        if args.baseline == 'tfidf':
            # 句对相似度：IDF 在训练集句子上统计，相似度为两个 TF-IDF 向量的余弦
            pair_scorer = TfidfScorer()
            train = read_jsonl(PROC / 'pairs_train.jsonl')
            pair_scorer.vectorizer.fit([r['a'] for r in train] + [r['b'] for r in train])

            def similarity(a, b):
                va, vb = pair_scorer.vectorizer.transform(a), pair_scorer.vectorizer.transform(b)
                return np.asarray(va.multiply(vb).sum(1)).ravel()
            result['pairs'] = evaluate_pairs(similarity)
    else:
        if args.baseline:
            encoder = SentenceEncoder(args.baseline[4:], 'mean').to(device).eval()
        else:
            encoder = SentenceEncoder.load(MODELS / args.model, device)
            if args.extra_index:
                from common import read_json
                extra += [(r['text'], r['faq_id'])
                         for r in read_json(RESULTS / 'pseudo' / f'{args.extra_index}.json')['rows']]
        scorer = DenseScorer(encoder)
        result['parameters'] = count_parameters(encoder)
        result['pairs'] = evaluate_pairs(lambda a, b: (encoder.encode(a) * encoder.encode(b)).sum(1))

    result['index_config'] = dict(real_index=args.real_index, reject=args.reject, pseudo=args.extra_index)
    result['faq'], predictions = evaluate_faq(scorer, faq, sets, extra, reject)
    if args.latency and encoder is not None:
        queries = [r['query'] for r in sets['real_test'][:100]]
        result['latency_ms'] = dict(cpu=measure_latency(encoder, faq, queries, 'cpu', extra, reject))
        if device == 'cuda':
            result['latency_ms']['cuda'] = measure_latency(encoder, faq, queries, 'cuda', extra, reject)
    write_json(RESULTS / 'eval' / f'{name}.json', result)
    write_jsonl(RESULTS / 'eval' / f'{name}_pred.jsonl', predictions)
    exp = result['faq']['exp']
    line = (f"{name}: 构造集 Top1 {exp['constructed']['top1']:.3f}  真实集 Top1 {exp['real_test']['top1']:.3f} "
            f"MRR {exp['real_test']['mrr']:.3f}  端到端 {exp['e2e_test']['accuracy']:.3f}  "
            f"AUROC {exp['oos_auroc_test']:.3f}  开放域误答 {exp['open_domain_answer_rate']:.3f}")
    if 'pairs' in result:
        line += f"  句对AUC {result['pairs']['mean_auc']:.3f}"
    print(line, flush=True)


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
