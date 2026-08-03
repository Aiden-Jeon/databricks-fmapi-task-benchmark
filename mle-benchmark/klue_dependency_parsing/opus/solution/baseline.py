"""Quick heuristic baseline for KLUE-DP: head = next eojeol, root = last eojeol.
deprel = most frequent label given the dependent eojeol's 2-char suffix (backoff)."""
import json
import os
from collections import Counter, defaultdict

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    tr = pd.read_csv(os.path.join(ROOT, "train.csv"))
    te = pd.read_csv(os.path.join(ROOT, "test.csv"))

    c2 = defaultdict(Counter)
    c1 = defaultdict(Counter)
    allc = Counter()
    for toks, parse in zip(tr.tokens.map(json.loads), tr.parse):
        items = parse.split("|")
        for w, it in zip(toks, items):
            rel = it.split(":")[1]
            c2[w[-2:]][rel] += 1
            c1[w[-1:]][rel] += 1
            allc[rel] += 1
    rootc = Counter()
    for toks, parse in zip(tr.tokens.map(json.loads), tr.parse):
        rootc[parse.split("|")[-1].split(":")[1]] += 1
    root_rel = rootc.most_common(1)[0][0]

    def rel_of(w):
        if len(c2[w[-2:]]) and sum(c2[w[-2:]].values()) >= 3:
            return c2[w[-2:]].most_common(1)[0][0]
        if len(c1[w[-1:]]):
            return c1[w[-1:]].most_common(1)[0][0]
        return allc.most_common(1)[0][0]

    rows = []
    for i, toks in zip(te.id, te.tokens.map(json.loads)):
        n = len(toks)
        parts = []
        for k, w in enumerate(toks):
            if k == n - 1:
                parts.append("0:%s" % root_rel)
            else:
                parts.append("%d:%s" % (k + 2, rel_of(w)))
        rows.append((i, "|".join(parts)))
    out = pd.DataFrame(rows, columns=["id", "parse"])
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    out.to_csv(os.path.join(ROOT, "outputs", "submission.csv"), index=False)
    print(out.shape, out.head())


if __name__ == "__main__":
    main()
