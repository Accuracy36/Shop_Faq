"""Validate completeness and recompute saved prediction aggregates without retraining."""
import collections
import json
import math
import statistics
from pathlib import Path
from common import FAQ_DIR, MODELS, RESULTS, read_json, read_jsonl, write_json


def main():
    checks, details = {}, {}
    stats = read_json(RESULTS / 'data_stats.json')['checks']
    checks['数据划分无交叉'] = all(v is True if isinstance(v, bool) else v == 0 for v in stats.values())
    first = ['roberta_seed1', 'roberta_seed2', 'roberta_seed3', 'rbt3_seed1', 'rbt3_seed2', 'rbt3_seed3',
             'transformer_seed1', 'bilstm_seed1', 'rbt3_cosent', 'rbt3_infonce', 'rbt3_cls']
    second = ['faqonly_roberta_seed1', 'simcse_roberta_seed1', 'real_roberta_seed1', 'nostage1_roberta_seed1']
    second += ['final_' + name for name in first[:8]]
    pseudo = (RESULTS / 'pseudo/pseudo_seed1.json').exists()
    if pseudo:
        second += ['pseudo_roberta_seed1']
    for prefix, names in [('s1_', first), ('s2_', second)]:
        rows = []
        for name in names:
            d = MODELS / (prefix + name)
            files = all((d / p).exists() for p in ['encoder.pt', 'encoder_config.json', 'history.json'])
            if files:
                h, c = read_json(d / 'history.json'), read_json(d / 'encoder_config.json')
                last = h['history'][-1]
                finished = (last['epoch'] == h['args']['epochs'] if prefix == 's1_' else last['step'] == 200)
                selected = c['best'] == h['best']
            else:
                finished = selected = False
            rows.append(dict(name=prefix + name, files=files, finished=finished, selected=selected))
        checks[prefix + '模型与轮次完整'] = all(r['files'] and r['finished'] and r['selected'] for r in rows)
        details[prefix] = rows
    sizes = {'real_test': len(read_jsonl(FAQ_DIR / 'real_test.jsonl')),
             'constructed': len(read_jsonl(FAQ_DIR / 'test_constructed.jsonl'))}
    predictions = []
    for path in sorted((RESULTS / 'eval').glob('*.json')):
        pred_path = path.with_name(path.stem + '_pred.jsonl')
        if not pred_path.exists():
            predictions.append(dict(name=path.stem, matched=False, reason='missing predictions'))
            continue
        data = read_json(path)
        rows = read_jsonl(pred_path)
        test = [r for r in rows if r['set'] == 'real_test']
        scope = [r for r in test if r['in_scope']]
        counts = collections.Counter(r['set'] for r in rows)
        e2e = sum(r['outcome'] in ['correct', 'rejected'] for r in test) / len(test)
        top1 = sum(r['hit'] for r in scope) / len(scope)
        exp = data['faq']['exp']
        matched = counts == sizes and abs(e2e - exp['e2e_test']['accuracy']) < 1e-12
        matched = matched and abs(top1 - exp['real_test']['top1']) < 1e-12
        predictions.append(dict(name=path.stem, matched=matched, e2e=e2e, real_top1=top1))
    checks['逐条预测与汇总一致'] = bool(predictions) and all(r['matched'] for r in predictions)
    expected = 33 if pseudo else 31
    checks['评测配置齐全'] = len(predictions) == expected
    tables = read_json(RESULTS / 'tables.json')
    checks['三种子统计完整'] = all(r['stage1']['n'] == 3 and r['final']['n'] == 3
                                  for r in tables['encoders'] if r['label'].startswith(('RBT3', 'RoBERTa-wwm')))
    aggregates = []
    for arch, label in [('rbt3', 'RBT3'), ('roberta', 'RoBERTa-wwm')]:
        row = next(r for r in tables['encoders'] if r['label'].startswith(label))
        for stage, prefix in [('stage1', 's1_'), ('final', 's2_final_')]:
            evaluations = [read_json(RESULTS / f'eval/{prefix}{arch}_seed{i}.json')['faq']['exp'] for i in [1, 2, 3]]
            for metric, section, key in [('real_top1', 'real_test', 'top1'), ('e2e', 'e2e_test', 'accuracy')]:
                values = [e[section][key] for e in evaluations]
                actual = row[stage][metric]
                aggregates.append(abs(actual['mean'] - statistics.mean(values)) < 1e-12 and
                                  abs(actual['std'] - statistics.stdev(values)) < 1e-12)
    checks['均值与样本标准差复算一致'] = all(aggregates)
    if pseudo:
        mined = read_json(RESULTS / 'pseudo/pseudo_seed1.json')
        counts = collections.Counter(r['faq_id'] for r in mined['rows'])
        checks['伪标注遵守预设筛选条件'] = (mined['stats']['dev_selected'] >= 20 and
            mined['stats']['dev_precision'] >= .95 and len(mined['rows']) == mined['stats']['selected'] and
            max(counts.values()) <= 30 and len(counts) == mined['stats']['faqs_covered'])
    details['predictions'] = predictions
    details['evaluation_count'] = len(predictions)
    details['pseudo_available'] = pseudo
    write_json(RESULTS / 'verification.json', dict(checks=checks, details=details))
    for key, value in checks.items():
        print('PASS' if value else 'FAIL', key)
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    main()
