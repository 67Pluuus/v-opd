from ...smp import *
from .multiple_choice import extract_answer_from_item
import numpy as np
import re

FAIL_MSG = 'Failed to obtain answer via API.'

TASK_CATEGORIES = [
    'SR','IMC','TCI','TA','MHR','PAR','CTI',
]


def get_dimension_rating(data_path, score_col='score', type_col='question_type'):
    data = load(data_path)
    acc_by_type = {}
    for qtype, group in data.groupby(type_col):
        correct = (group[score_col] == 1).sum()
        total = len(group)
        acc = correct / total if total > 0 else 0
        acc_by_type[qtype] = {
            'correct': int(correct),
            'total': int(total),
            'acc': acc
        }

    total_correct = (data[score_col] == 1).sum()
    total_count = len(data)
    total_acc = total_correct / total_count if total_count > 0 else 0

    result = {
        'acc_by_type': acc_by_type,
        'total': {
            'correct': int(total_correct),
            'total': int(total_count),
            'acc': total_acc
        }
    }

    return result


def extract_option(pred):
    """
    Extract option letter A-F from model output.

    Common model outputs:
    - With tags: <answer>A</answer> or <answer>A. ...</answer>
    - Without tags: "A", "A.", "A: ...", "The answer is A", etc.
    """
    if pred is None:
        return 'WRONG'

    text = str(pred)

    # Prefer explicit <answer>...</answer> if present
    try:
        matches = re.findall(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL | re.IGNORECASE)
    except Exception:
        matches = []

    candidates = []
    if matches:
        candidates.append(matches[-1])
    # Also consider the full text as fallback
    candidates.append(text)

    for cand in candidates:
        s = cand.strip()
        # 1) letter at beginning: A / A. / A: / A) / (A) / [A]
        m = re.search(r'^\s*[\(\[\{]?\s*([A-F])\s*[\)\]\}]?\s*[\.\:\)]?\b', s, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        # 2) explicit phrasing: "answer is A"
        m = re.search(r'\b(answer|final)\b[^A-F]*([A-F])\b', s, re.IGNORECASE)
        if m:
            return m.group(2).upper()
        # 3) last resort: pick the last standalone option letter in the string
        ms = re.findall(r'\b([A-F])\b', s, re.IGNORECASE)
        if ms:
            return ms[-1].upper()

    return 'WRONG'
