"""Sanity checks: (1) DP recovers gold trees from oracle scores; (2) baseline submission."""
import os
import sys
import collections
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dp_lib import decode, load  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

tr = load(pd.read_csv(os.path.join(ROOT, "train.csv")), True)
te = load(pd.read_csv(os.path.join(ROOT, "test.csv")), False)

# ---- (1) oracle decode test
bad = 0
for it in tr:
    n = it["n"]
    sc = np.zeros((n, n))
    for d in range(n - 1):
        sc[d, it["gold_heads"][d]] = 1.0
    p = decode(sc, n)
    if p != it["gold_heads"]:
        bad += 1
print("oracle decode mismatches: %d / %d" % (bad, len(tr)))

# ---- (2) trivial baseline: head = next word, root = last; label = most common by suffix2
cnt = collections.defaultdict(collections.Counter)
glob = collections.Counter()
for it in tr:
    for d in range(it["n"]):
        w = it["words"][d]
        key = (w[-2:], d == it["n"] - 1)
        cnt[key][it["gold_rels"][d]] += 1
        glob[it["gold_rels"][d]] += 1
default = glob.most_common(1)[0][0]
rows = []
for it in te:
    n = it["n"]
    parts = []
    for d in range(n):
        key = (it["words"][d][-2:], d == n - 1)
        c = cnt.get(key)
        lab = c.most_common(1)[0][0] if c else default
        parts.append("%d:%s" % (0 if d == n - 1 else d + 2, lab))
    rows.append((it["id"], "|".join(parts)))
out = os.path.join(ROOT, "outputs", "submission.csv")
os.makedirs(os.path.dirname(out), exist_ok=True)
pd.DataFrame(rows, columns=["id", "parse"]).to_csv(out, index=False)
print("baseline written:", out)
