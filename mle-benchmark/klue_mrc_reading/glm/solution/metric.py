"""Char-level F1 metric (KorQuAD-style) for evaluation."""
import re
import collections


def normalize_answer(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def char_f1(pred: str, gold: str) -> float:
    pred = normalize_answer(pred)
    gold = normalize_answer(gold)
    if pred == gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    pc = list(pred); gc = list(gold)
    common = collections.Counter(pc) & collections.Counter(gc)
    nc = sum(common.values())
    if nc == 0:
        return 0.0
    precision = nc / len(pc)
    recall = nc / len(gc)
    return 2 * precision * recall / (precision + recall)


def evaluate(preds, golds):
    assert len(preds) == len(golds)
    f1s = [char_f1(p, g) for p, g in zip(preds, golds)]
    return sum(f1s) / len(f1s) if f1s else 0.0
