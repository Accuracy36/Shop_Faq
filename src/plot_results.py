"""汇总 results/eval 下的评测结果并画图：results/tables.json、results/error_cases.json、results/figures/*.png。

缺少的结果会被跳过，所以可以在实验做到一半时运行。t-SNE 图需要载入模型（加 --tsne）。
"""
import argparse
import collections
import json

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from common import MODELS, RESULTS, load_faq, read_json, read_jsonl, write_json

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
EVAL, FIG = RESULTS / 'eval', RESULTS / 'figures'
SEEDS = (1, 2, 3)


def load(name):
    path = EVAL / f'{name}.json'
    return read_json(path) if path.exists() else None


def load_pred(name, subset='real_test'):
    path = EVAL / f'{name}_pred.jsonl'
    return [r for r in read_jsonl(path) if r['set'] == subset] if path.exists() else None


def metrics(result):
    """把一次评测的结果摊平成一行（索引：标准问+相似问，以及该次评测指定的真实样例、拒识样例）。"""
    exp, std = result['faq']['exp'], result['faq']['std']
    e2e = exp['e2e_test']
    row = dict(constructed_top1=exp['constructed']['top1'], constructed_top1_std_index=std['constructed']['top1'],
               real_top1=exp['real_test']['top1'], real_top3=exp['real_test']['top3'], real_mrr=exp['real_test']['mrr'],
               real_top1_std_index=std['real_test']['top1'], e2e=e2e['accuracy'], in_scope_acc=e2e['in_scope_accuracy'],
               oos_reject=e2e['oos_reject_rate'], answer_precision=e2e['answer_precision'],
               answer_rate=e2e['answer_rate'], auroc=exp['oos_auroc_test'], open_domain=exp['open_domain_answer_rate'],
               threshold=exp['threshold'], e2e_dev=exp['e2e_dev']['accuracy'])
    if 'pairs' in result:
        row.update(pair_auc=result['pairs']['mean_auc'], pair_acc=result['pairs']['mean_accuracy'],
                   **{f'{k.lower()}_{m}': result['pairs'][k][m] for k in ('LCQMC', 'BQ', 'AFQMC')
                      for m in ('accuracy', 'auc', 'f1')})
    if 'parameters' in result:
        row['parameters'] = result['parameters']
    for device, value in result.get('latency_ms', {}).items():
        row[f'latency_{device}'] = value
    return row


def aggregate(names):
    """多个随机种子的均值和标准差；只有一个结果时标准差为 0。"""
    rows = [metrics(r) for r in (load(n) for n in names) if r is not None]
    if not rows:
        return None
    out = dict(n=len(rows))
    for key in rows[0]:
        values = np.array([r[key] for r in rows if key in r], dtype=float)
        out[key] = dict(mean=float(values.mean()), std=float(values.std(ddof=1)) if len(values) > 1 else 0.0)
    return out


def seeds(prefix):
    return [f'{prefix}_seed{s}' for s in SEEDS]


def build_tables():
    spec = collections.OrderedDict()
    spec['encoders'] = [                       # 编码器对比：只做第一阶段 / 完整方案
        ('TF-IDF（不训练）', ['tfidf'], None), ('BM25（不训练）', ['bm25'], ['bm25_real']),
        ('RoBERTa，未微调', ['raw_roberta'], None),
        ('BiLSTM，从零训练', ['s1_bilstm_seed1'], ['s2_final_bilstm_seed1']),
        ('Transformer，从零训练', ['s1_transformer_seed1'], ['s2_final_transformer_seed1']),
        ('RBT3（3 层）', seeds('s1_rbt3'), seeds('s2_final_rbt3')),
        ('RoBERTa-wwm-ext（12 层）', seeds('s1_roberta'), seeds('s2_final_roberta'))]
    spec['loss'] = [('CoSENT', ['s1_rbt3_cosent'], None), ('InfoNCE', ['s1_rbt3_infonce'], None),
                    ('CoSENT + InfoNCE，平均池化', ['s1_rbt3_seed1'], None),
                    ('CoSENT + InfoNCE，[CLS] 池化', ['s1_rbt3_cls'], None)]
    spec['adaptation'] = [
        ('① 只做阶段一；索引：标准问+相似问', ['s1_roberta_seed1'], None),
        ('② ① + 索引加入真实问句', ['s1_roberta_seed1_realidx'], None),
        ('③ ② + 拒识样例', ['s1_roberta_seed1_realidx_reject'], None),
        ('④ 阶段二只用标准问、相似问微调', ['s2_faqonly_roberta_seed1'], None),
        ('⑤ 阶段二改为无监督 SimCSE（ECD）', ['s2_simcse_roberta_seed1'], None),
        ('⑥ 阶段二加入真实问句作正例；索引同②', ['s2_real_roberta_seed1'], None),
        ('⑦ ⑥ + 范围外问句作负例；只用阈值拒答', ['s2_final_roberta_seed1_threshold_only'], None),
        ('⑧ ⑦ + 拒识样例（完整方案）', ['s2_final_roberta_seed1'], None),
        ('⑨ 不做阶段一，直接做阶段二', ['s2_nostage1_roberta_seed1'], None),
        ('⑩ ⑧ + 索引加入伪标注句子', ['s2_final_roberta_seed1_pseudoidx'], None),
        ('⑪ ⑧ + 伪标注句子参与训练和索引', ['s2_pseudo_roberta_seed1'], None)]
    tables = {}
    for table, rows in spec.items():
        tables[table] = [dict(label=label, stage1=aggregate(a), final=aggregate(b) if b else None)
                         for label, a, b in rows]
    write_json(RESULTS / 'tables.json', tables)
    return tables


def cell(agg, key, scale=100, digits=1):
    if agg is None or key not in agg:
        return '—'
    m, s = agg[key]['mean'] * scale, agg[key]['std'] * scale
    return f'{m:.{digits}f}±{s:.{digits}f}' if agg['n'] > 1 else f'{m:.{digits}f}'


# ---------------------------------------------------------------- 图
def plot_training():
    names = [('s1_bilstm_seed1', 'BiLSTM', '0.6', '--'), ('s1_transformer_seed1', 'Transformer（从零）', '0.35', '-.'),
             ('s1_rbt3_seed1', 'RBT3', '0.5', '-'), ('s1_roberta_seed1', 'RoBERTa', 'black', '-')]
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 8.2), constrained_layout=True)
    found = False
    for name, label, color, style in names:
        path = MODELS / name / 'history.json'
        if not path.exists():
            continue
        found = True
        h = read_json(path)
        total = h['history'][-1]['step']
        curve = h['loss_curve']
        smooth = np.convolve([c['loss'] for c in curve], np.ones(10) / 10, mode='valid')
        axes[0].plot(np.array([c['step'] for c in curve][9:]) / total, smooth, style, color=color, label=label, lw=1.3)
        epochs = [r['epoch'] / h['history'][-1]['epoch'] for r in h['history']]
        axes[1].plot(epochs, [r['pair_auc'] for r in h['history']], style, color=color, marker='o', ms=3, label=label)
        axes[2].plot(epochs, [r['faq_top1_exp'] for r in h['history']], style, color=color, marker='o', ms=3, label=label)
    if not found:
        plt.close(fig)
        return
    for ax, title in zip(axes, ('训练损失（CoSENT + InfoNCE，滑动平均）', '公开句对验证集 AUC', '真实问句验证集 Top-1（未做领域适配）')):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('训练进度（占总步数的比例）')
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=9)
    fig.savefig(FIG / 'training_stage1.png', dpi=200)
    plt.close(fig)


def plot_stage2():
    names = [('s2_faqonly_roberta_seed1', '只用标准问、相似问', '0.6', '--'),
             ('s2_real_roberta_seed1', '加入真实问句', '0.4', '-.'),
             ('s2_final_roberta_seed1', '真实问句 + 范围外负例 + 拒识样例', 'black', '-')]
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 8.2), constrained_layout=True)
    fig.get_layout_engine().set(h_pad=0.13, hspace=0.07)
    found = False
    for name, label, color, style in names:
        path = MODELS / name / 'history.json'
        if not path.exists():
            continue
        found = True
        h = read_json(path)['history']
        steps = [r['step'] for r in h]
        axes[0].plot(steps[1:], [r['train_loss'] for r in h[1:]], style, color=color, label=label)
        axes[1].plot(steps, [r['faq_top1_exp'] for r in h], style, color=color, marker='o', ms=3, label=label)
        axes[2].plot(steps, [r['faq_e2e_acc'] for r in h], style, color=color, marker='o', ms=3, label=label)
    if not found:
        plt.close(fig)
        return
    axes[0].set_yscale('log')
    for ax, title in zip(axes, ('阶段二训练损失', '真实问句验证集 Top-1', '真实问句验证集端到端准确率')):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('阶段二迭代步数')
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=9)
    fig.savefig(FIG / 'training_stage2.png', dpi=200)
    plt.close(fig)


def plot_compare(tables):
    rows = [r for r in tables['encoders'] if r['stage1']]
    labels = [r['label'].replace('TF-IDF（不训练）', 'TF-IDF').replace('BM25（不训练）', 'BM25')
              .replace('RoBERTa，未微调', 'RoBERTa\n未微调').replace('BiLSTM，从零训练', 'BiLSTM')
              .replace('Transformer，从零训练', 'Transformer').replace('RBT3（3 层）', 'RBT3')
              .replace('RoBERTa-wwm-ext（12 层）', 'RoBERTa\n微调') for r in rows]
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 7.0), constrained_layout=True)
    x = np.arange(len(rows))
    for ax, key, title in zip(axes, ('real_top1', 'e2e'), ('真实问句测试集 Top-1（范围内 203 句）', '真实问句测试集端到端准确率（398 句）')):
        a = [100 * r['stage1'][key]['mean'] for r in rows]
        ae = [100 * r['stage1'][key]['std'] for r in rows]
        b = [100 * r['final'][key]['mean'] if r['final'] else np.nan for r in rows]
        be = [100 * r['final'][key]['std'] if r['final'] else 0 for r in rows]
        ax.bar(x - 0.2, a, 0.4, yerr=ae, color='0.75', capsize=2, label='阶段一或未训练')
        ax.bar(x + 0.2, b, 0.4, yerr=be, color='0.25', capsize=2, label='完整方案')
        ax.set_xticks(x, labels, fontsize=9)
        ax.set_ylabel('%')
        ax.set_ylim(0, 105)
        ax.set_title(title, fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    axes[0].legend(fontsize=9, loc='upper left')
    fig.savefig(FIG / 'encoder_comparison.png', dpi=200)
    plt.close(fig)


def plot_similarity():
    names = [('raw_roberta', 'RoBERTa，未微调'), ('s1_roberta_seed1', '阶段一（通用句对）'),
             ('s2_final_roberta_seed1_threshold_only', '阶段二（领域适配）')]
    available = [(n, l) for n, l in names if load_pred(n)]
    if not available:
        return
    fig, axes = plt.subplots(len(available), 1, figsize=(6.4, 2.8 * len(available)), constrained_layout=True,
                             squeeze=False)
    bins = np.linspace(-0.1, 1.0, 34)
    for ax, (name, label) in zip(axes[:, 0], available):
        pred = load_pred(name)
        inside = [r['top'][0]['score'] for r in pred if r['in_scope']]
        outside = [r['top'][0]['score'] for r in pred if not r['in_scope']]
        ax.hist(inside, bins=bins, color='0.2', alpha=0.85, label=f'范围内（{len(inside)} 句）')
        ax.hist(outside, bins=bins, color='0.7', alpha=0.75, label=f'范围外（{len(outside)} 句）')
        threshold = load(name)['faq']['exp']['threshold']
        ax.axvline(threshold, color='black', ls='--', lw=1)
        ax.text(threshold, ax.get_ylim()[1] * 0.93, f' 阈值 {threshold:.2f}', fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel('与最相近 FAQ 的余弦相似度')
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel('句数')
    axes[0][0].legend(fontsize=9, loc='upper left')
    fig.savefig(FIG / 'similarity_distribution.png', dpi=200)
    plt.close(fig)


def plot_roc_and_threshold():
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 7.0), constrained_layout=True)
    names = [('bm25', 'BM25', '0.7', ':'), ('raw_roberta', 'RoBERTa，未微调', '0.55', '-.'),
             ('s1_roberta_seed1', '阶段一', '0.4', '--'), ('s2_final_roberta_seed1_threshold_only', '阶段二，只用阈值', '0.15', '-'),
             ('s2_final_roberta_seed1', '阶段二 + 拒识样例', 'black', '-')]
    found = False
    for name, label, color, style in names:
        pred = load_pred(name)
        if not pred:
            continue
        found = True
        y = np.array([r['in_scope'] for r in pred])
        conf = np.array([r['confidence'] for r in pred])
        fpr, tpr, _ = roc_curve(y, conf)
        auroc = load(name)['faq']['exp']['oos_auroc_test']
        axes[0].plot(fpr, tpr, style, color=color, lw=1.6 if color == 'black' else 1.2, label=f'{label}（AUROC {auroc:.3f}）')
    if not found:
        plt.close(fig)
        return
    axes[0].plot([0, 1], [0, 1], color='0.85', lw=0.8)
    axes[0].set_xlabel('范围外问句被回答的比例')
    axes[0].set_ylabel('范围内问句被回答的比例')
    axes[0].set_title('区分“范围内 / 范围外”问句的 ROC 曲线（真实问句测试集）', fontsize=10)
    axes[0].legend(fontsize=9, loc='lower right')
    axes[0].grid(alpha=0.3)

    name = 's2_final_roberta_seed1'
    pred = load_pred(name)
    if pred:
        conf = np.array([r['confidence'] for r in pred])
        inside = np.array([r['in_scope'] for r in pred])
        hit = np.array([r['hit'] for r in pred])
        ts = np.linspace(0.2, 0.98, 79)
        acc = [((conf >= t) & hit).sum() / len(pred) + ((conf < t) & ~inside).sum() / len(pred) for t in ts]
        precision = [((conf >= t) & hit).sum() / max((conf >= t).sum(), 1) for t in ts]
        rate = [(conf >= t).mean() for t in ts]
        axes[1].plot(ts, acc, '-', color='black', label='端到端准确率')
        axes[1].plot(ts, precision, '--', color='0.3', label='回答的正确率')
        axes[1].plot(ts, rate, ':', color='0.3', label='回答的比例')
        threshold = load(name)['faq']['exp']['threshold']
        axes[1].axvline(threshold, color='0.5', lw=0.8)
        axes[1].text(threshold, 0.05, f' 验证集选出的阈值 {threshold:.2f}', fontsize=9)
        axes[1].set_xlabel('回答阈值')
        axes[1].set_title('阈值对完整方案的影响（真实问句测试集）', fontsize=10)
        axes[1].legend(fontsize=9)
        axes[1].grid(alpha=0.3)
    fig.savefig(FIG / 'roc_threshold.png', dpi=200)
    plt.close(fig)


def plot_category_and_errors(name='s2_final_roberta_seed1'):
    pred = load_pred(name)
    if not pred:
        return
    faq = {e['key']: e for e in load_faq()}
    cats = list(dict.fromkeys(e['category'] for e in faq.values())) + ['范围外']
    counts = {c: collections.Counter() for c in cats}
    for r in pred:
        counts[faq[r['gold']]['category'] if r['in_scope'] else '范围外'][r['outcome']] += 1
    order = [('correct', '答对', '0.2'), ('rejected', '正确拒答', '0.45'), ('wrong_answer', '答错（答了别的 FAQ）', '0.65'),
             ('missed', '漏答（该答却拒答）', '0.8'), ('false_answer', '误答（该拒却回答）', '0.92')]
    fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    bottom = np.zeros(len(cats))
    for key, label, color in order:
        values = np.array([counts[c][key] for c in cats])
        ax.bar(cats, values, bottom=bottom, color=color, edgecolor='black', lw=0.4, label=label)
        bottom += values
    for i, c in enumerate(cats):
        total = sum(counts[c].values())
        good = counts[c]['correct'] + counts[c]['rejected']
        ax.text(i, bottom[i] + 2, f'{100 * good / max(total, 1):.0f}%', ha='center', fontsize=9)
    ax.set_ylabel('句数')
    ax.set_title('真实测试集分类结果\n柱上数字为处理正确的比例', fontsize=10)
    ax.tick_params(axis='x', labelrotation=25, labelsize=9)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(axis='y', alpha=0.3)
    fig.savefig(FIG / 'category_outcomes.png', dpi=200)
    plt.close(fig)

    errors = collections.defaultdict(list)
    confusion = collections.Counter()
    for r in pred:
        if r['outcome'] in ('wrong_answer', 'missed', 'false_answer'):
            errors[r['outcome']].append(dict(query=r['query'], gold=r['gold'], predicted=r['top'][0]['key'],
                                             confidence=r['confidence'], matched=r['top'][0]['matched']))
        if r['outcome'] == 'wrong_answer':
            confusion[(r['gold'], r['top'][0]['key'])] += 1
    write_json(RESULTS / 'error_cases.json', dict(
        model=name, counts=dict(collections.Counter(r['outcome'] for r in pred)),
        confusions=[dict(gold=g, predicted=p, n=n) for (g, p), n in confusion.most_common(15)], **errors))


def plot_latency(tables):
    rows = [r for r in tables['encoders'] if r['final'] and 'latency_cpu' in r['final']]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.4), constrained_layout=True)
    y = np.arange(len(rows))
    for device, offset, color, label in [('cpu', -.18, '0.75', 'CPU'), ('cuda', .18, '0.25', 'GPU')]:
        values = [r['final'][f'latency_{device}']['mean'] for r in rows]
        errors = [r['final'][f'latency_{device}']['std'] for r in rows]
        ax.barh(y + offset, values, height=.33, color=color, xerr=errors, capsize=2, label=label)
        for pos, value, error in zip(y + offset, values, errors):
            ax.text(value + error + .8, pos, f'{value:.2f}', va='center', fontsize=9)
    ax.set_yticks(y, ['BiLSTM', 'Transformer', 'RBT3', 'RoBERTa'])
    ax.invert_yaxis()
    ax.set_xlim(0, 88)
    ax.set_xlabel('单请求时延（毫秒，编码 + 检索）')
    ax.set_title('完整方案 CPU 与 GPU 时延', fontsize=10)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='x', alpha=0.3)
    fig.savefig(FIG / 'latency.png', dpi=200)
    plt.close(fig)


def plot_tsne(names=(('raw_roberta', 'RoBERTa，未微调'), ('s2_final_roberta_seed1', '完整方案'))):
    """真实问句测试集中范围内问句的句向量（t-SNE 降到二维），按 FAQ 类别标记。"""
    import torch
    from sklearn.manifold import TSNE
    from models import SentenceEncoder
    from retrieval import load_eval_sets
    faq = load_faq()
    rows = [r for r in load_eval_sets()['real_test'] if r['accept']]
    cats = list(dict.fromkeys(e['category'] for e in faq))
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X']
    fig, axes = plt.subplots(len(names), 1, figsize=(6.4, 4.2 * len(names)), constrained_layout=True, squeeze=False)
    for ax, (name, label) in zip(axes[:, 0], names):
        if name.startswith('raw_'):
            encoder = SentenceEncoder(name[4:], 'mean').eval()
        else:
            encoder = SentenceEncoder.load(MODELS / name, 'cpu')
        with torch.no_grad():
            emb = encoder.encode([r['query'] for r in rows])
        xy = TSNE(n_components=2, init='pca', perplexity=25, random_state=0, metric='cosine').fit_transform(emb)
        for k, cat in enumerate(cats):
            idx = [i for i, r in enumerate(rows) if faq[r['faq_id']]['category'] == cat]
            ax.scatter(xy[idx, 0], xy[idx, 1], marker=markers[k], s=22, color=str(0.1 + 0.12 * k), edgecolor='black',
                       lw=0.3, label=cat)
        ax.set_title(label, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0][0].legend(fontsize=9)
    fig.savefig(FIG / 'tsne.png', dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tsne', action='store_true')
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    tables = build_tables()
    plot_training()
    plot_stage2()
    plot_compare(tables)
    plot_similarity()
    plot_roc_and_threshold()
    plot_category_and_errors()
    plot_latency(tables)
    if args.tsne:
        plot_tsne()
    keys = ('constructed_top1', 'real_top1', 'real_mrr', 'e2e', 'in_scope_acc', 'oos_reject', 'auroc', 'open_domain')
    for table, rows in tables.items():
        print(f'== {table}: ' + ' | '.join(keys))
        for r in rows:
            for tag in ('stage1', 'final'):
                if r[tag]:
                    print(f'  {r["label"]} [{tag}] ' + ' | '.join(cell(r[tag], k) for k in keys))


if __name__ == '__main__':
    import sys
    import warnings
    sys.stdout.reconfigure(encoding='utf-8')
    warnings.filterwarnings('ignore')
    main()
