import os, sys, collections, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dp_lib import load, word_counts, set_vocab
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import train as T

tr_df = pd.read_csv(os.path.join(ROOT, "train.csv"))
wc = word_counts([tr_df]); set_vocab({w for w, c in wc.items() if c >= 2})
data = load(tr_df, True)
rng = np.random.RandomState(0)
perm = rng.permutation(len(data)); ndev = 480
dev = [data[i] for i in perm[:ndev]]; train = [data[i] for i in perm[ndev:]]

fm = T.FeatMap(); T.build_arc_cache(train, fm); fm.frozen = True; T.build_arc_cache(dev, fm)
m = T.ArcModel(fm.size()); order = np.arange(len(train))
for ep in range(14):
    rng.shuffle(order); T.arc_epoch(m, train, order)
w = m.averaged()
T.arc_predict(dev, w, m)

by_gd = collections.Counter(); by_gd_ok = collections.Counter()
by_len = collections.Counter(); by_len_ok = collections.Counter()
pred_dist = collections.Counter()
for it in dev:
    n = it["n"]
    for d in range(n - 1):
        g = it["gold_heads"][d]; p = it["pred_heads"][d]
        k = min(g - d, 6)
        by_gd[k] += 1; by_gd_ok[k] += (g == p)
        lb = min(n // 5, 6)
        by_len[lb] += 1; by_len_ok[lb] += (g == p)
        if g != p:
            pred_dist[(k, min(p - d, 6))] += 1
print("acc by gold head-distance (1..6+):")
for k in sorted(by_gd):
    print("  dist %s  n=%5d acc=%.3f" % (k, by_gd[k], by_gd_ok[k] / by_gd[k]))
print("acc by sentence-length bucket (n//5):")
for k in sorted(by_len):
    print("  len~%d  n=%5d acc=%.3f" % (k * 5, by_len[k], by_len_ok[k] / by_len[k]))
print("top confusions (gold_dist -> pred_dist):", pred_dist.most_common(10))
tot = sum(by_gd.values()); ok = sum(by_gd_ok.values())
print("non-root arc acc %.4f" % (ok / tot))
