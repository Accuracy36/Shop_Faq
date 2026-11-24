"""数据准备：解析 FAQ 源文件，整理公开句对数据集，抽取 ECD 中的用户问句，建立字表并统计。

用法（在项目根目录执行）：
    python src/prepare_data.py faq        # 解析 data/faq/faq_source.txt
    python src/prepare_data.py pairs      # LCQMC / BQ / AFQMC 句对
    python src/prepare_data.py ecd        # ECD 用户问句与待标注的真实问句候选
    python src/prepare_data.py finalize   # 标注完成后：划分真实问句验证集/测试集、建立字表、统计
    python src/prepare_data.py trainsample     # 从 ECD 训练部分抽取待标注的训练样本
    python src/prepare_data.py finalize_train  # 训练样本标注完成后生成 real_train.jsonl
"""
import collections
import random
import re
import sys

import pandas as pd

from common import FAQ_DIR, PROC, RAW, join_ecd, normalize, read_json, read_jsonl, write_json, write_jsonl

SEED = 2026
N_CANDIDATES = 700          # 抽取多少条真实问句用于逐条标注（标注时剔除 2 条）
N_REAL_DEV = 300            # 其中多少条作为验证集，其余为测试集


def build_faq():
    entries, tests, cur = [], [], None
    for line in (FAQ_DIR / 'faq_source.txt').read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line.startswith('## '):
            key, category, question = [s.strip() for s in line[3:].split('|')]
            cur = dict(id=len(entries), key=key, category=category, question=question, answer='', similar=[])
            entries.append(cur)
        elif line.startswith('A:'):
            cur['answer'] = line[2:].strip()
        elif line.startswith('S:'):
            cur['similar'] = [s.strip() for s in line[2:].split('|')]
        elif line.startswith('T:'):
            tests += [dict(query=s.strip(), key=cur['key'], faq_id=cur['id']) for s in line[2:].split('|')]
    keys = [e['key'] for e in entries]
    assert len(set(keys)) == len(keys), '标识重复'
    for e in entries:
        assert e['answer'] and len(e['similar']) == 6, e['key']
    per = collections.Counter(t['key'] for t in tests)
    assert all(per[k] == 3 for k in keys), per
    # 标准问、相似问、测试问规范化后不得重复，测试问不得出现在索引或训练文本中
    seen = collections.Counter(normalize(s) for e in entries for s in [e['question']] + e['similar'])
    seen.update(normalize(t['query']) for t in tests)
    dup = [s for s, c in seen.items() if c > 1]
    assert not dup, f'重复文本: {dup}'
    write_json(FAQ_DIR / 'faq.json', entries)
    write_jsonl(FAQ_DIR / 'test_constructed.jsonl', tests)
    cats = collections.Counter(e['category'] for e in entries)
    print(f'FAQ {len(entries)} 条, 相似问 {sum(len(e["similar"]) for e in entries)} 条, 构造测试问 {len(tests)} 条')
    print(dict(cats))


def build_pairs():
    stats = {}
    out = collections.defaultdict(list)
    held_out = set()        # 验证集、测试集中的句对（不分先后顺序），训练集中相同的句对要去掉
    for name in ('LCQMC', 'BQ', 'AFQMC'):
        for split in ('validation', 'test', 'train'):
            df = pd.read_parquet(RAW / name / f'{split}.parquet')
            rows, seen, removed = [], set(), 0
            for a, b, y in zip(df['sentence1'], df['sentence2'], df['score']):
                a, b = normalize(a), normalize(b)
                if int(y) not in (0, 1) or not a or not b or (a, b) in seen:
                    continue
                key = (a, b) if a <= b else (b, a)
                if split == 'train':
                    if key in held_out:           # 规范化后与验证集、测试集撞上的句对
                        removed += 1
                        continue
                else:
                    held_out.add(key)
                seen.add((a, b))
                rows.append(dict(a=a, b=b, label=int(y), source=name))
            stats[f'{name}/{split}'] = dict(raw=len(df), kept=len(rows), positive=sum(r['label'] for r in rows),
                                            removed_as_held_out=removed)
            # AFQMC 官方测试集不含标签，因此用它的验证集作测试集
            target = {'train': 'train', 'validation': 'dev', 'test': 'test'}[split]
            if name == 'AFQMC':
                target = {'train': 'train', 'validation': 'test', 'test': None}[split]
            if target and rows:
                out[target] += rows
    for split, rows in out.items():
        write_jsonl(PROC / f'pairs_{split}.jsonl', rows)
        print(split, len(rows), dict(collections.Counter(r['source'] for r in rows)))
    write_json(PROC / 'pairs_stats.json', stats)
    print(stats)


def ecd_sessions(split):
    with open(RAW / 'ECD' / f'{split}.txt', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if parts[0] == '1':
                yield parts[1:-1], parts[-1]


def usable(u):
    return 2 <= len(u) <= 30 and not re.search(r'\d{6,}', u) and 'http' not in u


def build_ecd():
    # 1) 训练集全部用户发言（无标签），用于领域自适应
    counter, sessions = collections.Counter(), 0
    for ctx, _ in ecd_sessions('train'):
        sessions += 1
        for i, u in enumerate(ctx):
            if (len(ctx) - 1 - i) % 2 == 0:          # 回复由客服发出，往前隔一句是用户
                u = join_ecd(u)
                if usable(u):
                    counter[u] += 1
    # 2) 验证集、测试集中每段对话的最后一句用户发言，作为真实问句候选
    faq = read_json(FAQ_DIR / 'faq.json')
    known = {normalize(s) for e in faq for s in [e['question']] + e['similar']}
    known |= {normalize(t['query']) for t in read_jsonl(FAQ_DIR / 'test_constructed.jsonl')}
    cands = []
    for split in ('dev', 'test'):
        for ctx, _ in ecd_sessions(split):
            u = join_ecd(ctx[-1])
            if usable(u) and u not in known:
                cands.append(u)
    distinct = list(dict.fromkeys(cands))
    random.Random(SEED).shuffle(distinct)
    picked = distinct[:N_CANDIDATES]
    with open(FAQ_DIR / 'ecd_eval_candidates.tsv', 'w', encoding='utf-8') as f:
        for i, u in enumerate(picked):
            f.write(f'{i}\t{u}\n')
    # 训练用的无标签语料中去掉被抽为评测候选的句子
    picked_set = set(picked)
    with open(PROC / 'ecd_train_utts.tsv', 'w', encoding='utf-8') as f:
        for u, c in counter.most_common():
            if u not in picked_set:
                f.write(f'{u}\t{c}\n')
    stats = dict(train_sessions=sessions, train_customer_utts=sum(counter.values()), train_distinct=len(counter),
                 train_distinct_after_removing_eval=len(counter) - len(picked_set & set(counter)),
                 eval_last_turn_utts=len(cands), eval_distinct=len(distinct), eval_sampled=len(picked))
    write_json(PROC / 'ecd_stats.json', stats)
    print(stats)


def finalize():
    faq = read_json(FAQ_DIR / 'faq.json')
    key2id = {e['key']: e['id'] for e in faq}
    cands = dict(line.rstrip('\n').split('\t') for line in open(FAQ_DIR / 'ecd_eval_candidates.tsv', encoding='utf-8'))
    # 标注文件只列出知识库范围内的句子；每行可给出多个可接受的 FAQ，第一个为主标签
    labels = {}
    for line in open(FAQ_DIR / 'ecd_eval_annotation.txt', encoding='utf-8'):
        if line.strip() and not line.startswith('#'):
            idx, keys = line.split()
            labels[idx] = keys.split(',')
    rows = []
    with open(FAQ_DIR / 'ecd_eval_labels.tsv', 'w', encoding='utf-8') as f:
        for idx, query in cands.items():
            keys = labels.get(idx, ['oos'])
            if keys == ['drop']:
                continue
            assert keys == ['oos'] or all(k in key2id for k in keys), (idx, keys)
            accept = [key2id[k] for k in keys if k != 'oos']
            rows.append(dict(query=query, key=keys[0], faq_id=accept[0] if accept else -1, accept=accept))
            f.write(f'{query}\t{",".join(keys)}\n')
    assert set(labels) <= set(cands)
    random.Random(SEED).shuffle(rows)
    dev, test = rows[:N_REAL_DEV], rows[N_REAL_DEV:]
    write_jsonl(FAQ_DIR / 'real_dev.jsonl', dev)
    write_jsonl(FAQ_DIR / 'real_test.jsonl', test)
    # 字表：句对训练集、ECD 无标签语料、FAQ 文本中出现不少于 3 次的字符
    chars = collections.Counter()
    for r in read_jsonl(PROC / 'pairs_train.jsonl'):
        chars.update(r['a'])
        chars.update(r['b'])
    for line in open(PROC / 'ecd_train_utts.tsv', encoding='utf-8'):
        u, c = line.rstrip('\n').split('\t')
        for ch in u:
            chars[ch] += int(c)
    for e in faq:
        for s in [e['question']] + e['similar']:
            chars.update(normalize(s))
    vocab = ['[PAD]', '[UNK]'] + [c for c, n in chars.most_common() if n >= 3]
    write_json(PROC / 'char_vocab.json', vocab)
    stats = dict(faq=len(faq), categories=dict(collections.Counter(e['category'] for e in faq)),
                 similar=sum(len(e['similar']) for e in faq),
                 constructed_test=len(read_jsonl(FAQ_DIR / 'test_constructed.jsonl')),
                 real_dev=dict(total=len(dev), in_scope=sum(r['faq_id'] >= 0 for r in dev)),
                 real_test=dict(total=len(test), in_scope=sum(r['faq_id'] >= 0 for r in test)),
                 real_label_counts=dict(collections.Counter(r['key'] for r in rows).most_common()),
                 char_vocab=len(vocab))
    write_json(PROC / 'faq_stats.json', stats)
    print(stats)


def build_train_sample(top=200, rand=500):
    """从 ECD 训练部分的用户发言中抽取待标注的训练样本：出现次数最多的 top 句 + 其余句子中随机 rand 句。

    ecd_train_utts.tsv 已去掉被抽作评测的句子，所以训练样本与真实问句验证集、测试集没有重复。
    """
    rows = [line.rstrip('\n').split('\t') for line in open(PROC / 'ecd_train_utts.tsv', encoding='utf-8')]
    head, rest = rows[:top], rows[top:]
    random.Random(SEED + 1).shuffle(rest)
    picked = head + rest[:rand]
    with open(FAQ_DIR / 'ecd_train_candidates.tsv', 'w', encoding='utf-8') as f:
        for i, (u, c) in enumerate(picked):
            f.write(f'{i}\t{u}\t{c}\n')
    print(len(picked), 'candidates; top counts', picked[0], picked[top - 1], 'random part max count',
          max(int(c) for _, c in picked[top:]))


def finalize_train():
    """读入训练样本的标注，生成 real_train.jsonl。"""
    key2id = {e['key']: e['id'] for e in read_json(FAQ_DIR / 'faq.json')}
    cands = {}
    for line in open(FAQ_DIR / 'ecd_train_candidates.tsv', encoding='utf-8'):
        idx, u, c = line.rstrip('\n').split('\t')
        cands[idx] = (u, int(c))
    labels = {}
    for line in open(FAQ_DIR / 'ecd_train_annotation.txt', encoding='utf-8'):
        if line.strip() and not line.startswith('#'):
            idx, keys = line.split()
            labels[idx] = keys.split(',')
    assert set(labels) <= set(cands)
    eval_queries = {r['query'] for name in ('real_dev', 'real_test') for r in read_jsonl(FAQ_DIR / f'{name}.jsonl')}
    rows = []
    for idx, (query, count) in cands.items():
        keys = labels.get(idx, ['oos'])
        if keys == ['drop']:
            continue
        assert query not in eval_queries, query
        assert keys == ['oos'] or all(k in key2id for k in keys), (idx, keys)
        accept = [key2id[k] for k in keys if k != 'oos']
        rows.append(dict(query=query, key=keys[0], faq_id=accept[0] if accept else -1, accept=accept, count=count))
    write_jsonl(FAQ_DIR / 'real_train.jsonl', rows)
    in_scope = [r for r in rows if r['accept']]
    stats = dict(total=len(rows), in_scope=len(in_scope), oos=len(rows) - len(in_scope),
                 faqs_covered=len({r['faq_id'] for r in in_scope}),
                 label_counts=dict(collections.Counter(r['key'] for r in rows).most_common()))
    write_json(PROC / 'real_train_stats.json', stats)
    print(stats)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    {'faq': build_faq, 'pairs': build_pairs, 'ecd': build_ecd, 'finalize': finalize,
     'trainsample': build_train_sample, 'finalize_train': finalize_train}[sys.argv[1]]()
