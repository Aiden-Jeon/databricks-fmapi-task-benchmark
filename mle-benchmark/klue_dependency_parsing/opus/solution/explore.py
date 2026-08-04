import json, collections, pandas as pd, numpy as np, os
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tr = pd.read_csv(os.path.join(D, "train.csv"))
te = pd.read_csv(os.path.join(D, "test.csv"))
print(tr.shape, te.shape)

def parse_row(r):
    toks = json.loads(r["tokens"])
    items = r["parse"].split("|")
    heads = [int(x.split(":")[0]) for x in items]
    rels = [x.split(":")[1] for x in items]
    return toks, heads, rels

n_tok = 0
n_right = 0     # head > dep
n_left = 0
n_root = 0
root_is_last = 0
n_multi_root = 0
lens = []
head_dist = collections.Counter()
rel_cnt = collections.Counter()
nonproj = 0
badlen = 0
root_positions = collections.Counter()
for _, r in tr.iterrows():
    toks, heads, rels = parse_row(r)
    if len(toks) != len(heads):
        badlen += 1
        continue
    n = len(toks)
    lens.append(n)
    roots = [i for i, h in enumerate(heads) if h == 0]
    n_root += len(roots)
    if len(roots) != 1:
        n_multi_root += 1
    for rt in roots:
        root_positions[n - 1 - rt] += 1
        if rt == n - 1:
            root_is_last += 1
    for i, (h, rl) in enumerate(zip(heads, rels), start=1):
        n_tok += 1
        rel_cnt[rl] += 1
        if h == 0:
            head_dist["ROOT"] += 1
        elif h > i:
            n_right += 1
            head_dist[h - i] += 1
        else:
            n_left += 1
            head_dist[-(i - h)] += 1
    # projectivity check
    for i, h in enumerate(heads, start=1):
        if h == 0:
            continue
        lo, hi = min(i, h), max(i, h)
        for j in range(lo + 1, hi):
            hj = heads[j - 1]
            if hj != 0 and (hj < lo or hj > hi):
                nonproj += 1
                break
        else:
            continue
        break

print("tokens", n_tok, "sents", len(lens))
print("badlen", badlen)
print("head right(h>d):", n_right, n_right / n_tok)
print("head left(h<d):", n_left, n_left / n_tok)
print("roots total", n_root, "multi-root sents", n_multi_root)
print("root is last token:", root_is_last, root_is_last / len(lens))
print("root pos from end:", root_positions.most_common(6))
print("nonprojective sents:", nonproj, nonproj / len(lens))
print("len stats", np.mean(lens), np.percentile(lens, [50, 90, 99]), max(lens))
print("head dist top:", head_dist.most_common(12))
print("n rels", len(rel_cnt), rel_cnt.most_common(40))
# test lens
tl = [len(json.loads(t)) for t in te["tokens"]]
print("test len", np.mean(tl), max(tl), sum(tl))
