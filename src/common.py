"""公共工具：路径、文本规范化、随机种子、读写 jsonl、FAQ 载入。"""
import json
import random
import re
import unicodedata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / 'data' / 'raw'
PROC = ROOT / 'data' / 'processed'
FAQ_DIR = ROOT / 'data' / 'faq'
MODELS = ROOT / 'models'
RESULTS = ROOT / 'results'
PRETRAINED = {'roberta': MODELS / 'pretrained' / 'chinese-roberta-wwm-ext',
              'rbt3': MODELS / 'pretrained' / 'rbt3'}
MAX_LEN = 40          # 规范化后的最大字符数；三个句对数据集的 99 分位长度约为 40

_SPACE = re.compile(r'\s+')
_ASCII_GAP = re.compile(r'(?<=[a-z0-9]) (?=[a-z0-9])')


def normalize(text):
    """统一的文本规范化：NFKC（全角转半角）、小写、去掉标点和符号、去掉多余空白。

    淘宝客服语料（ECD）发布时已去掉标点，这里对所有来源做同样处理，保证训练、检索和线上输入一致。
    英文、数字之间保留一个空格，中文之间不留空格。
    """
    text = unicodedata.normalize('NFKC', str(text)).lower()
    chars = [' ' if unicodedata.category(c)[0] in 'PSZC' else c for c in text]
    text = _SPACE.sub(' ', ''.join(chars)).strip()
    text = _ASCII_GAP.sub('\x00', text).replace(' ', '').replace('\x00', ' ')
    return text[:MAX_LEN]


def join_ecd(tokens):
    """ECD 的句子已分词并以空格分隔；normalize 会去掉中文之间的空格，拼回原句。"""
    return normalize(tokens)


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def load_faq():
    """返回 FAQ 列表，每项含 id、key、category、question、answer、similar。"""
    return read_json(FAQ_DIR / 'faq.json')
