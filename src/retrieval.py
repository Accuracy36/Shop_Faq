"""FAQ 检索与评测指标。

打分器（Scorer）统一接口：fit_index(texts) 建立索引，query(texts) 返回 (查询数, 索引文本数) 的相似度矩阵。
FaqIndex 把索引文本映射回 FAQ 条目：一条 FAQ 的得分取它名下所有索引文本（标准问、相似问）得分的最大值。
"""
import collections
import math

import numpy as np
from sklearn.metrics import roc_auc_score

from common import FAQ_DIR, normalize, read_jsonl


class DenseScorer:
    """句向量余弦相似度。"""

    def __init__(self, encoder):
        self.encoder = encoder

    def fit_index(self, texts):
        self.index = self.encoder.encode(texts)

    def query(self, texts):
        return self.encoder.encode(texts) @ self.index.T


def char_ngrams(text):
    text = normalize(text)
    return list(text) + [text[i:i + 2] for i in range(len(text) - 1)]


class TfidfScorer:
    """字的一元、二元组 TF-IDF 向量的余弦相似度（不含任何训练）。"""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(analyzer=char_ngrams, sublinear_tf=True)

    def fit_index(self, texts):
        self.index = self.vectorizer.fit_transform(texts)

    def query(self, texts):
        return np.asarray((self.vectorizer.transform(texts) @ self.index.T).todense())


class BM25Scorer:
    """Okapi BM25，词项为字的一元、二元组。分数未归一化。"""

    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b

    def fit_index(self, texts):
        self.docs = [collections.Counter(char_ngrams(t)) for t in texts]
        self.lengths = np.array([sum(d.values()) for d in self.docs], dtype=np.float64)
        self.avg_len = self.lengths.mean()
        df = collections.Counter(term for d in self.docs for term in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.postings = collections.defaultdict(list)
        for i, d in enumerate(self.docs):
            for term, tf in d.items():
                self.postings[term].append((i, tf))

    def query(self, texts):
        out = np.zeros((len(texts), len(self.docs)))
        norm = self.k1 * (1 - self.b + self.b * self.lengths / self.avg_len)
        for row, text in enumerate(texts):
            for term in set(char_ngrams(text)):
                for i, tf in self.postings.get(term, ()):
                    out[row, i] += self.idf[term] * tf * (self.k1 + 1) / (tf + norm[i])
        return out


class FaqIndex:
    def __init__(self, faq, scorer, use_similar=True, extra=None, reject=None):
        """extra：额外加入索引的 (文本, FAQ 编号) 列表，例如标注过的真实问句。
        reject：拒识样例，即已知不在知识库范围内的真实问句；与某个拒识样例最相近的查询直接拒答。"""
        self.faq, self.scorer = faq, scorer
        self.texts, self.owner = [], []
        for e in faq:
            for s in [e['question']] + (e['similar'] if use_similar else []):
                self.texts.append(s)
                self.owner.append(e['id'])
        for text, faq_id in extra or []:
            self.texts.append(text)
            self.owner.append(faq_id)
        self.owner = np.array(self.owner)
        self.reject = list(reject or [])
        scorer.fit_index(self.texts + self.reject)

    def score(self, queries):
        """返回 (查询数, FAQ 数) 的得分矩阵，以及每条 FAQ 得分最高的索引文本下标。"""
        scores, best, _ = self.score_full(queries)
        return scores, best

    def score_full(self, queries):
        """另外返回用于“回答还是拒答”判定的得分：最相近的索引句是拒识样例时，该行全部置为 -1。"""
        full = self.scorer.query(queries)
        sim = full[:, :len(self.texts)]
        scores = np.full((len(queries), len(self.faq)), -np.inf)
        best = np.zeros((len(queries), len(self.faq)), dtype=np.int64)
        for faq_id in range(len(self.faq)):
            cols = np.where(self.owner == faq_id)[0]
            sub = sim[:, cols]
            scores[:, faq_id] = sub.max(1)
            best[:, faq_id] = cols[sub.argmax(1)]
        decision = scores
        if self.reject:
            decision = scores.copy()
            decision[full[:, len(self.texts):].max(1) > scores.max(1)] = -1.0
        return scores, best, decision


def load_real_train():
    """标注过的真实训练问句：返回 (范围内问句的 (文本, FAQ 编号) 列表, 范围外问句列表)。"""
    rows = read_jsonl(FAQ_DIR / 'real_train.jsonl')
    return [(r['query'], r['faq_id']) for r in rows if r['accept']], [r['query'] for r in rows if not r['accept']]


def load_eval_sets():
    sets = dict(constructed=read_jsonl(FAQ_DIR / 'test_constructed.jsonl'),
                real_dev=read_jsonl(FAQ_DIR / 'real_dev.jsonl'), real_test=read_jsonl(FAQ_DIR / 'real_test.jsonl'))
    for row in sets['constructed']:
        row['accept'] = [row['faq_id']]
    return sets


def retrieval_metrics(scores, rows):
    """只统计知识库范围内的问句。命中：排在前面的 FAQ 属于该问句可接受的 FAQ。"""
    ranks = []
    order = np.argsort(-scores, axis=1)
    for i, row in enumerate(rows):
        if row['accept']:
            ranks.append(min(int(np.where(order[i] == a)[0][0]) for a in row['accept']) + 1)
    ranks = np.array(ranks)
    return dict(n=len(ranks), top1=float((ranks <= 1).mean()), top3=float((ranks <= 3).mean()),
                mrr=float(np.where(ranks <= 10, 1.0 / ranks, 0.0).mean()))


def outcomes(scores, rows, threshold):
    """端到端结果：最高分不低于阈值则回答排名第一的 FAQ，否则拒答（转人工）。"""
    top, conf = scores.argmax(1), scores.max(1)
    result = []
    for i, row in enumerate(rows):
        answered = conf[i] >= threshold
        if row['accept']:
            result.append('correct' if answered and top[i] in row['accept'] else
                          ('wrong_answer' if answered else 'missed'))
        else:
            result.append('false_answer' if answered else 'rejected')
    return result


def end_to_end(scores, rows, threshold):
    c = collections.Counter(outcomes(scores, rows, threshold))
    n_in, n_out = c['correct'] + c['wrong_answer'] + c['missed'], c['false_answer'] + c['rejected']
    answered = c['correct'] + c['wrong_answer'] + c['false_answer']
    return dict(accuracy=(c['correct'] + c['rejected']) / max(len(rows), 1),
                in_scope_accuracy=c['correct'] / max(n_in, 1),
                oos_reject_rate=c['rejected'] / max(n_out, 1),
                answer_precision=c['correct'] / max(answered, 1),
                answer_rate=answered / max(len(rows), 1), counts=dict(c))


def best_threshold(scores, rows):
    """在验证集上选端到端准确率最高的阈值（并列时取中间值）。"""
    conf = np.unique(scores.max(1))
    cands = np.concatenate([[conf[0] - 1e-6], (conf[:-1] + conf[1:]) / 2, [conf[-1] + 1e-6]])
    accs = np.array([end_to_end(scores, rows, t)['accuracy'] for t in cands])
    best = np.where(accs >= accs.max() - 1e-12)[0]
    return float(cands[best[len(best) // 2]]), float(accs.max())


def oos_auroc(scores, rows):
    """用最高相似度区分“范围内”和“范围外”问句的 AUROC。"""
    y = np.array([1 if r['accept'] else 0 for r in rows])
    return float(roc_auc_score(y, scores.max(1)))


def pair_metrics(sim_dev, y_dev, sim_test, y_test):
    """句对判定：AUC，以及用验证集上准确率最高的阈值得到的测试集准确率和 F1。"""
    from sklearn.metrics import f1_score
    order = np.sort(np.unique(np.round(sim_dev, 4)))
    cands = order[::max(1, len(order) // 400)]
    accs = [((sim_dev >= t) == y_dev).mean() for t in cands]
    t = float(cands[int(np.argmax(accs))])
    pred = sim_test >= t
    return dict(n=int(len(y_test)), auc=float(roc_auc_score(y_test, sim_test)), accuracy=float((pred == y_test).mean()),
                f1=float(f1_score(y_test, pred)), threshold=t)
