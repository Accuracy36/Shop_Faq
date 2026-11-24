"""数据集统计与核验：results/data_stats.json、results/figures/data_overview.png。只读数据，不需要模型。

核验内容：各数据集之间没有不该有的重复（评测问句不出现在任何训练、索引文本中），标注合法。
"""
import collections

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common import FAQ_DIR, PROC, RESULTS, load_faq, normalize, read_json, read_jsonl, write_json

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
FIG = RESULTS / 'figures'


def length_stats(texts):
    lengths = np.array([len(t) for t in texts])
    return dict(n=int(len(lengths)), mean=float(lengths.mean()), median=float(np.median(lengths)),
                p95=float(np.quantile(lengths, 0.95)), max=int(lengths.max()))


def main():
    faq = load_faq()
    pairs = {s: read_jsonl(PROC / f'pairs_{s}.jsonl') for s in ('train', 'dev', 'test')}
    real = {s: read_jsonl(FAQ_DIR / f'real_{s}.jsonl') for s in ('train', 'dev', 'test')}
    constructed = read_jsonl(FAQ_DIR / 'test_constructed.jsonl')
    ecd = [line.rstrip('\n').split('\t') for line in open(PROC / 'ecd_train_utts.tsv', encoding='utf-8')]
    ecd_texts = [t for t, _ in ecd]
    index_texts = [normalize(s) for e in faq for s in [e['question']] + e['similar']]
    constructed_texts = [normalize(r['query']) for r in constructed]

    stats = dict(
        pairs={split: {src: dict(total=sum(r['source'] == src for r in rows),
                                 positive=sum(r['source'] == src and r['label'] == 1 for r in rows))
                       for src in ('LCQMC', 'BQ', 'AFQMC')} for split, rows in pairs.items()},
        pair_sentence_length={src: length_stats([r['a'] for r in pairs['train'] if r['source'] == src]
                                                + [r['b'] for r in pairs['train'] if r['source'] == src])
                              for src in ('LCQMC', 'BQ', 'AFQMC')},
        ecd=dict(distinct_utterances=len(ecd), occurrences=sum(int(c) for _, c in ecd), length=length_stats(ecd_texts),
                 **read_json(PROC / 'ecd_stats.json')),
        faq=dict(entries=len(faq), categories=dict(collections.Counter(e['category'] for e in faq)),
                 index_texts=len(index_texts), index_length=length_stats(index_texts),
                 answer_length=length_stats([e['answer'] for e in faq])),
        constructed=dict(n=len(constructed), length=length_stats(constructed_texts)),
        real={split: dict(n=len(rows), in_scope=sum(bool(r['accept']) for r in rows),
                          oos=sum(not r['accept'] for r in rows),
                          multi_label=sum(len(r['accept']) > 1 for r in rows),
                          faqs_covered=len({r['faq_id'] for r in rows if r['accept']}),
                          length=length_stats([r['query'] for r in rows]),
                          category=dict(collections.Counter(
                              faq[r['faq_id']]['category'] if r['accept'] else '范围外' for r in rows)))
              for split, rows in real.items()},
        char_vocab=len(read_json(PROC / 'char_vocab.json')))

    # ---- 核验
    sets = {s: {r['query'] for r in rows} for s, rows in real.items()}
    pair_keys = {s: {(r['a'], r['b']) for r in rows} | {(r['b'], r['a']) for r in rows} for s, rows in pairs.items()}
    keys = {e['key'] for e in faq}
    checks = dict(
        real_train_dev_overlap=len(sets['train'] & sets['dev']),
        real_train_test_overlap=len(sets['train'] & sets['test']),
        real_dev_test_overlap=len(sets['dev'] & sets['test']),
        eval_queries_in_ecd_unlabeled=len((sets['dev'] | sets['test']) & set(ecd_texts)),
        eval_queries_in_index=len((sets['dev'] | sets['test'] | set(constructed_texts)) & set(index_texts)),
        constructed_in_real_train=len(set(constructed_texts) & sets['train']),
        pairs_train_test_overlap=len(pair_keys['train'] & pair_keys['test']),
        pairs_train_dev_overlap=len(pair_keys['train'] & pair_keys['dev']),
        labels_valid=all(r['key'] == 'oos' or r['key'] in keys for rows in real.values() for r in rows),
        every_faq_has_6_similar=all(len(e['similar']) == 6 for e in faq),
        every_faq_has_3_constructed=set(collections.Counter(r['key'] for r in constructed).values()) == {3})
    stats['checks'] = checks
    write_json(RESULTS / 'data_stats.json', stats)

    # ---- 图：句长分布、真实问句的类别构成、真实问句中最常见的 FAQ
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 8.4), constrained_layout=True)
    bins = np.arange(0, 42, 2)
    series = [('公开句对（训练集）', [len(r['a']) for r in pairs['train'][::20]], 'black', '-'),
              ('ECD 用户发言', [len(t) for t in ecd_texts[::10]], '0.45', '--'),
              ('FAQ 标准问与相似问', [len(t) for t in index_texts], '0.2', ':'),
              ('真实问句（验证+测试）', [len(r['query']) for r in real['dev'] + real['test']], '0.6', '-.')]
    for label, lengths, color, style in series:
        hist, _ = np.histogram(lengths, bins=bins, density=True)
        axes[0].plot(bins[:-1] + 1, hist, style, color=color, label=label, lw=1.5)
    axes[0].set_xlabel('句长（规范化后的字符数）')
    axes[0].set_ylabel('比例密度')
    axes[0].set_title('各数据来源的句长分布', fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    cats = list(stats['faq']['categories']) + ['范围外']
    width = 0.27
    for k, (split, label, color) in enumerate((('train', '真实训练集', '0.25'), ('dev', '真实验证集', '0.55'),
                                                ('test', '真实测试集', '0.8'))):
        total = stats['real'][split]['n']
        values = [100 * stats['real'][split]['category'].get(c, 0) / total for c in cats]
        axes[1].bar(np.arange(len(cats)) + (k - 1) * width, values, width, color=color, label=label)
    axes[1].set_xticks(np.arange(len(cats)), cats, rotation=30, ha='right', fontsize=9)
    axes[1].set_ylabel('占比（%）')
    axes[1].set_title('真实问句的类别构成', fontsize=10)
    axes[1].legend(fontsize=9)
    axes[1].grid(axis='y', alpha=0.3)

    counter = collections.Counter(r['key'] for r in real['dev'] + real['test'] if r['accept'])
    by_key = {e['key']: e['question'].rstrip('？。') for e in faq}
    top = counter.most_common(12)[::-1]
    axes[2].barh(range(len(top)), [c for _, c in top], color='0.4')
    axes[2].set_yticks(range(len(top)), [by_key[k] if len(by_key[k]) <= 11 else by_key[k][:10] + '…' for k, _ in top], fontsize=9)
    axes[2].set_xlabel('句数（验证集 + 测试集）')
    axes[2].set_title(f'真实问句最常问到的 12 条 FAQ（共涉及 {len(counter)} 条）', fontsize=10)
    axes[2].grid(axis='x', alpha=0.3)
    fig.savefig(FIG / 'data_overview.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    for k, v in checks.items():
        print(k, v)
    print({s: {k: v for k, v in d.items() if k not in ('category', 'length')} for s, d in stats['real'].items()})
    print(stats['pairs'])
    print(stats['pair_sentence_length'], stats['ecd']['length'], stats['faq']['index_length'])


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    main()
