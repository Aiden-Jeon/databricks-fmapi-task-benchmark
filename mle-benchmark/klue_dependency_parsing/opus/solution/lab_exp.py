"""Fast standalone experiments for the deprel-labelling model (gold heads)."""
import os, sys, time, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dp_lib
from dp_lib import load, word_counts, set_vocab, lab_feats
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from train2 import FeatMap, _pack

tr_df = pd.read_csv(os.path.join(ROOT, "train.csv"))
wc = word_counts([tr_df]); set_vocab({w for w, c in wc.items() if c >= 2})
data = load(tr_df, True)
rng = np.random.RandomState(0); perm = rng.permutation(len(data))
dev = [data[i] for i in perm[:480]]; train = [data[i] for i in perm[480:]]
labels = sorted({r for it in data for r in it["gold_rels"]})
lab2i = {l: i for i, l in enumerate(labels)}
print("labels", len(labels))


class LM:
    def __init__(self, size, nlab, mira):
        self.W = np.zeros((size, nlab)); self.WA = np.zeros((size, nlab))
        self.last = np.zeros(size, dtype=np.int64); self.t = 0; self.mira = mira

    def upd(self, feats, gl, pl, loss, C=1.0):
        uq, cnt = np.unique(feats, return_counts=True)
        k = uq != 0; uq = uq[k]; cnt = cnt[k].astype(float)
        if not len(uq): return
        if self.mira:
            sd = float(self.W[uq, gl] @ cnt - self.W[uq, pl] @ cnt)
            nrm = 2.0 * float(cnt @ cnt)
            eta = (loss - sd) / nrm
            if eta <= 0: return
            eta = min(eta, C)
        else:
            eta = 1.0
        self.t += 1
        self.WA[uq] += self.W[uq] * (self.t - self.last[uq])[:, None]
        self.W[uq, gl] += eta * cnt; self.W[uq, pl] -= eta * cnt
        self.last[uq] = self.t

    def avg(self):
        o = (self.WA + self.W * (self.t - self.last)[:, None]) / max(self.t, 1)
        o[0] = 0.0; return o


def run(mira, epochs, tag):
    lm = FeatMap()
    for it in train:
        it["LF"] = _pack([[lm.get(x) for x in lab_feats(it["tl"], d, it["gold_heads"][d], it["n"])]
                          for d in range(it["n"])])
    lm.frozen = True
    for it in dev:
        it["LF"] = _pack([[lm.get(x) for x in lab_feats(it["tl"], d, it["gold_heads"][d], it["n"])]
                          for d in range(it["n"])])
    m = LM(lm.size(), len(labels), mira)
    order = np.arange(len(train)); r = np.random.RandomState(1)
    best = 0.0
    for ep in range(epochs):
        r.shuffle(order)
        for si in order:
            it = train[si]; F = it["LF"]; S = m.W[F].sum(axis=1)
            for d in range(it["n"]):
                g = lab2i[it["gold_rels"][d]]
                s = S[d] + 1.0; s[g] -= 1.0
                p = int(s.argmax())
                if p != g: m.upd(F[d], g, p, 1.0)
        W = m.avg()
        nc = nt = 0
        for it in dev:
            pr = W[it["LF"]].sum(axis=1).argmax(axis=1)
            for d in range(it["n"]):
                nt += 1; nc += labels[int(pr[d])] == it["gold_rels"][d]
        acc = nc / nt
        best = max(best, acc)
        print("  %s ep %2d dev-lab-acc %.4f" % (tag, ep + 1, acc), flush=True)
    print("%s BEST %.4f  |feat|=%d" % (tag, best, lm.size()), flush=True)
    return W, lm


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    if mode in ("plain", "both"):
        run(False, 8, "plain")
    if mode in ("mira", "both"):
        W, lm = run(True, 12, "mira")
        # confusion analysis
        conf = collections.Counter()
        for it in dev:
            pr = W[it["LF"]].sum(axis=1).argmax(axis=1)
            for d in range(it["n"]):
                p = labels[int(pr[d])]; g = it["gold_rels"][d]
                if p != g: conf[(g, p)] += 1
        print("top label confusions (gold->pred):", conf.most_common(15))
