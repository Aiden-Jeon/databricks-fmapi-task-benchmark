"""Fast baseline: longest-match dictionary tagging built from train entities."""
import sys
from collections import Counter, defaultdict

import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from data import load, format_entities, micro_f1  # noqa


def build_dict(rows, min_count=1, min_prec=0.0):
    cnt = defaultdict(Counter)      # surface -> type counts
    occ = Counter()                 # surface -> occurrences in sentences
    for r in rows:
        for s, t in r["ents"]:
            cnt[s][t] += 1
    for r in rows:
        sent = r["sentence"]
        for s in cnt:
            if s in sent:
                occ[s] += sent.count(s)
    d = {}
    for s, c in cnt.items():
        typ, n = c.most_common(1)[0]
        tot = sum(c.values())
        if tot < min_count:
            continue
        prec = tot / max(occ[s], 1)
        if prec < min_prec:
            continue
        d[s] = typ
    return d


def tag_sentence(sent, d, maxlen):
    ents = []
    i, n = 0, len(sent)
    while i < n:
        hit = None
        for L in range(min(maxlen, n - i), 0, -1):
            sub = sent[i:i + L]
            if sub in d:
                hit = (sub, d[sub])
                break
        if hit:
            ents.append(hit)
            i += len(hit[0])
        else:
            i += 1
    return ents


def main():
    rows, _, _ = load("train.csv")
    n = len(rows)
    tr, va = rows[: int(n * 0.9)], rows[int(n * 0.9):]
    for mp in (0.0, 0.3, 0.5, 0.7):
        d = build_dict(tr, min_prec=mp)
        ml = max(len(s) for s in d)
        preds = [tag_sentence(r["sentence"], d, ml) for r in va]
        golds = [r["ents"] for r in va]
        print("min_prec", mp, "dict", len(d), "P/R/F1", micro_f1(golds, preds))

    d = build_dict(rows, min_prec=0.5)
    ml = max(len(s) for s in d)
    test = load("test.csv", with_labels=False)
    out = pd.DataFrame({
        "id": [r["id"] for r in test],
        "entities": [format_entities(tag_sentence(r["sentence"], d, ml)) for r in test],
    })
    out.to_csv("outputs/submission.csv", index=False)
    print("wrote outputs/submission.csv", out.shape)


if __name__ == "__main__":
    main()
