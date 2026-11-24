"""网页 Demo：电商客服 FAQ 自动问答。

    python demo/app.py --model s2_final_roberta_seed1 --real-index --reject [--port 7860]

浏览器打开 http://127.0.0.1:7860 。每次提问：规范化 → 编码为句向量 → 与索引中所有标准问、相似问计算余弦相似度
→ 每条 FAQ 取最高分 → 最高分不低于阈值、且最相近的索引句不是拒识样例则回答，否则不作答并给出“猜你想问”和转人工提示。
阈值取自 results/eval/<模型名>.json（在真实问句验证集上选出），也可以用 --threshold 指定。
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from common import MODELS, RESULTS, load_faq, normalize, read_json  # noqa: E402
from models import SentenceEncoder  # noqa: E402
from retrieval import DenseScorer, FaqIndex, load_real_train  # noqa: E402

app = Flask(__name__, static_folder=str(Path(__file__).resolve().parent / 'static'))
STATE = {}


def build(args):
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    encoder = SentenceEncoder.load(MODELS / args.model, device)
    faq = load_faq()
    real_in, real_out = load_real_train()
    extra = list(real_in) if args.real_index else []
    if args.pseudo:
        extra += [(r['text'], r['faq_id']) for r in read_json(RESULTS / 'pseudo' / f'{args.pseudo}.json')['rows']]
    index = FaqIndex(faq, DenseScorer(encoder), use_similar=True, extra=extra,
                     reject=real_out if args.reject else None)
    threshold = args.threshold
    if threshold is None:
        threshold = read_json(RESULTS / 'eval' / f'{args.eval_name or args.model}.json')['faq']['exp']['threshold']
    index.score(['你好'])                                   # 预热
    STATE.update(index=index, faq=faq, threshold=float(threshold), device=device, model=args.model,
                 parameters=sum(p.numel() for p in encoder.parameters()))


@app.route('/')
def home():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/info')
def info():
    faq = STATE['faq']
    hot = ['courier', 'ship_time', 'return_policy', 'coupon', 'invoice', 'refund_time']
    by_key = {e['key']: e for e in faq}
    return jsonify(model=STATE['model'], device=STATE['device'], threshold=round(STATE['threshold'], 3),
                   faq_count=len(faq), index_size=len(STATE['index'].texts), reject_size=len(STATE['index'].reject),
                   parameters=STATE['parameters'],
                   hot=[by_key[k]['question'] for k in hot])


@app.route('/api/ask', methods=['POST'])
def ask():
    query = str((request.get_json(silent=True) or {}).get('query', '')).strip()[:200]
    if not normalize(query):
        return jsonify(status='empty', answer='请输入您想咨询的问题。', candidates=[], normalized='', latency_ms=0)
    index, faq = STATE['index'], STATE['faq']
    start = time.perf_counter()
    scores, best, decision = index.score_full([query])
    latency = 1000 * (time.perf_counter() - start)
    order = np.argsort(-scores[0])[:3]
    candidates = [dict(question=faq[j]['question'], category=faq[j]['category'], score=round(float(scores[0, j]), 3),
                       matched=index.texts[best[0, j]]) for j in order]
    top = int(order[0])
    near_reject = bool(decision[0].max() < 0 <= scores[0, top])      # 最相近的索引句是拒识样例
    answered = float(decision[0].max()) >= STATE['threshold']
    if answered:
        answer = faq[top]['answer']
    else:
        answer = '抱歉，这个问题我暂时回答不了，建议联系人工客服（服务时间 9:00–23:00）。您也可以看看下面的问题是否有帮助。'
    return jsonify(status='answered' if answered else 'rejected', answer=answer, normalized=normalize(query),
                   question=faq[top]['question'] if answered else None,
                   category=faq[top]['category'] if answered else None,
                   confidence=round(float(scores[0, top]), 3), threshold=round(STATE['threshold'], 3),
                   near_reject=near_reject,
                   candidates=candidates, latency_ms=round(latency, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--pseudo', default='')
    ap.add_argument('--real-index', action='store_true', help='索引中加入标注过的真实问句')
    ap.add_argument('--reject', action='store_true', help='使用拒识样例')
    ap.add_argument('--eval-name', default='', help='读取阈值的评测结果名，默认与模型名相同')
    ap.add_argument('--threshold', type=float, default=None)
    ap.add_argument('--port', type=int, default=7860)
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()
    import warnings
    warnings.filterwarnings('ignore')
    import transformers
    transformers.logging.set_verbosity_error()
    build(args)
    print(f'模型 {args.model}，设备 {STATE["device"]}，阈值 {STATE["threshold"]:.3f}，'
          f'索引 {len(STATE["index"].texts)} 句：http://127.0.0.1:{args.port}', flush=True)
    app.run(host='127.0.0.1', port=args.port, debug=False, threaded=False)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
