"""Train the KLUE-DP parser and write outputs/submission.csv.

usage:  python solution/train.py [--dev 0.1] [--epochs 12] [--lepochs 15] [--seed 0]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dp_lib import arc_feats, lab_feats, decode, load, word_counts, set_vocab  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEG = -1e18


# --------------------------------------------------------------------------- #
# feature caches
# --------------------------------------------------------------------------- #

class FeatMap:
    def __init__(self):
        self.d = {}          # string -> index (index 0 reserved for "unknown/pad")
        self.frozen = False

    def get(self, s):
        i = self.d.get(s, -1)
        if i >= 0:
            return i
        if self.frozen:
            return 0
        i = len(self.d) + 1
        self.d[s] = i
        return i

    def size(self):
        return len(self.d) + 1


def build_arc_cache(data, fm, verbose_tag=""):
    """For every sentence, cache padded int32 feature-index matrix of all candidate arcs."""
    t0 = time.time()
    for si, it in enumerate(data):
        n = it["n"]
        tl = it["tl"]
        rows = []
        di = []
        hi = []
        for d in range(n - 1):
            for h in range(d + 1, n):
                fs = arc_feats(tl, d, h, n)
                rows.append([fm.get(x) for x in fs])
                di.append(d)
                hi.append(h)
        if rows:
            K = max(len(r) for r in rows)
            F = np.zeros((len(rows), K), dtype=np.int32)
            for i, r in enumerate(rows):
                F[i, :len(r)] = r
        else:
            F = np.zeros((0, 1), dtype=np.int32)
        it["F"] = F
        it["di"] = np.asarray(di, dtype=np.int32)
        it["hi"] = np.asarray(hi, dtype=np.int32)
        if verbose_tag and (si + 1) % 1000 == 0:
            print("  [%s] arc-feats %d/%d  %.0fs  |feat|=%d"
                  % (verbose_tag, si + 1, len(data), time.time() - t0, fm.size()), flush=True)
    return fm


def build_lab_cache(data, lm, gold=True):
    """Cache label features for gold arcs (training) -- padded int32 matrix per sentence."""
    for it in data:
        n = it["n"]
        tl = it["tl"]
        heads = it["gold_heads"] if gold else it["pred_heads"]
        rows = [[lm.get(x) for x in lab_feats(tl, d, heads[d], n)] for d in range(n)]
        K = max(len(r) for r in rows)
        F = np.zeros((n, K), dtype=np.int32)
        for i, r in enumerate(rows):
            F[i, :len(r)] = r
        it["LF"] = F


def lab_matrix(it, lm, heads):
    n = it["n"]
    tl = it["tl"]
    rows = [[lm.get(x) for x in lab_feats(tl, d, heads[d], n)] for d in range(n)]
    K = max(len(r) for r in rows)
    F = np.zeros((n, K), dtype=np.int32)
    for i, r in enumerate(rows):
        F[i, :len(r)] = r
    return F


# --------------------------------------------------------------------------- #
# arc model: averaged structured perceptron
# --------------------------------------------------------------------------- #

class ArcModel:
    def __init__(self, size):
        self.w = np.zeros(size, dtype=np.float64)
        self.wa = np.zeros(size, dtype=np.float64)
        self.last = np.zeros(size, dtype=np.int64)
        self.t = 0

    def score(self, it, w):
        n = it["n"]
        sc = np.full((n, n), NEG, dtype=np.float64)
        F = it["F"]
        if F.shape[0]:
            sc[it["di"], it["hi"]] = w[F].sum(axis=1)
        return sc

    def update(self, gfeat, pfeat):
        self.t += 1
        idx = np.concatenate((gfeat, pfeat))
        val = np.concatenate((np.ones(len(gfeat)), -np.ones(len(pfeat))))
        uq, inv = np.unique(idx, return_inverse=True)
        delta = np.bincount(inv, weights=val, minlength=len(uq))
        keep = (uq != 0) & (delta != 0.0)
        uq = uq[keep]
        delta = delta[keep]
        if len(uq) == 0:
            return
        self.wa[uq] += self.w[uq] * (self.t - self.last[uq])
        self.w[uq] += delta
        self.last[uq] = self.t

    def averaged(self):
        wa = self.wa + self.w * (self.t - self.last)
        out = wa / max(self.t, 1)
        out[0] = 0.0
        return out


def arc_epoch(model, data, order):
    ncorr = 0
    ntot = 0
    for si in order:
        it = data[si]
        n = it["n"]
        if n <= 1:
            continue
        sc = model.score(it, model.w)
        pred = decode(sc, n)
        gold = it["gold_heads"]
        F = it["F"]
        di = it["di"]
        hi = it["hi"]
        # row lookup: arcs enumerated as d ascending, h in (d, n)
        # offset for d = sum_{j<d}(n-1-j)
        off = np.zeros(n, dtype=np.int64)
        acc = 0
        for d in range(n - 1):
            off[d] = acc
            acc += n - 1 - d
        gi = []
        pi = []
        for d in range(n - 1):
            g = gold[d]
            p = pred[d]
            ntot += 1
            if g == p:
                ncorr += 1
                continue
            gi.append(off[d] + (g - d - 1))
            pi.append(off[d] + (p - d - 1))
        ntot += 1
        ncorr += 1  # root (last token) is structurally always right
        if gi:
            gf = F[np.asarray(gi, dtype=np.int64)].ravel()
            pf = F[np.asarray(pi, dtype=np.int64)].ravel()
            model.update(gf, pf)
    return ncorr / max(ntot, 1)


def arc_predict(data, w, model):
    for it in data:
        n = it["n"]
        if n <= 1:
            it["pred_heads"] = [-1]
            continue
        sc = model.score(it, w)
        it["pred_heads"] = decode(sc, n)


# --------------------------------------------------------------------------- #
# label model: multiclass averaged perceptron
# --------------------------------------------------------------------------- #

class LabModel:
    def __init__(self, size, nlab):
        self.W = np.zeros((size, nlab), dtype=np.float64)
        self.WA = np.zeros((size, nlab), dtype=np.float64)
        self.last = np.zeros(size, dtype=np.int64)
        self.t = 0
        self.nlab = nlab

    def scores(self, F, W):
        return W[F].sum(axis=1)

    def update(self, feats, gl, pl):
        self.t += 1
        uq, cnt = np.unique(feats, return_counts=True)
        keep = uq != 0
        uq = uq[keep]
        cnt = cnt[keep].astype(np.float64)
        if len(uq) == 0:
            return
        self.WA[uq] += self.W[uq] * (self.t - self.last[uq])[:, None]
        self.W[uq, gl] += cnt
        self.W[uq, pl] -= cnt
        self.last[uq] = self.t

    def averaged(self):
        WA = self.WA + self.W * (self.t - self.last)[:, None]
        out = WA / max(self.t, 1)
        out[0] = 0.0
        return out


def lab_epoch(model, data, order, lab2i):
    nc = 0
    nt = 0
    for si in order:
        it = data[si]
        F = it["LF"]
        s = model.scores(F, model.W)
        pred = s.argmax(axis=1)
        for d in range(it["n"]):
            g = lab2i[it["gold_rels"][d]]
            p = int(pred[d])
            nt += 1
            if g == p:
                nc += 1
            else:
                model.update(F[d], g, p)
    return nc / max(nt, 1)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def evaluate(data, labs_pred):
    """LAS / UAS given it['pred_heads'] and labs_pred[i] = list of predicted labels."""
    nt = las = uas = 0
    for it, pl in zip(data, labs_pred):
        for d in range(it["n"]):
            nt += 1
            hok = it["pred_heads"][d] == it["gold_heads"][d]
            uas += hok
            las += hok and (pl[d] == it["gold_rels"][d])
    return las / nt, uas / nt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lepochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--minwf", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "submission.csv"))
    args = ap.parse_args()

    t0 = time.time()
    tr_df = pd.read_csv(os.path.join(ROOT, "train.csv"))
    te_df = pd.read_csv(os.path.join(ROOT, "test.csv"))
    if args.minwf > 1:
        wc = word_counts([tr_df])
        set_vocab({w for w, c in wc.items() if c >= args.minwf})
        print("vocab kept %d / %d (minwf=%d)"
              % (sum(1 for c in wc.values() if c >= args.minwf), len(wc), args.minwf), flush=True)
    train = load(tr_df, True)
    test = load(te_df, False)

    rng = np.random.RandomState(args.seed)
    if args.dev > 0:
        perm = rng.permutation(len(train))
        ndev = int(len(train) * args.dev)
        dev = [train[i] for i in perm[:ndev]]
        train = [train[i] for i in perm[ndev:]]
    else:
        dev = []
    print("train=%d dev=%d test=%d" % (len(train), len(dev), len(test)), flush=True)

    labels = sorted({r for it in (train + dev) for r in it["gold_rels"]})
    lab2i = {l: i for i, l in enumerate(labels)}
    print("labels:", len(labels), flush=True)

    # ---------------- arc model ----------------
    fm = FeatMap()
    build_arc_cache(train, fm, "train")
    print("arc feats built: %d  (%.0fs)" % (fm.size(), time.time() - t0), flush=True)
    fm.frozen = True
    if dev:
        build_arc_cache(dev, fm)
    build_arc_cache(test, fm)
    print("caches done (%.0fs)" % (time.time() - t0), flush=True)

    model = ArcModel(fm.size())
    order = np.arange(len(train))
    best = None
    for ep in range(args.epochs):
        rng.shuffle(order)
        acc = arc_epoch(model, train, order)
        w = model.averaged()
        msg = "ep %2d train-arc-acc %.4f" % (ep + 1, acc)
        if dev:
            arc_predict(dev, w, model)
            nt = uas = 0
            for it in dev:
                for d in range(it["n"]):
                    nt += 1
                    uas += it["pred_heads"][d] == it["gold_heads"][d]
            msg += "  dev-UAS %.4f" % (uas / nt)
        print("%s  (%.0fs)" % (msg, time.time() - t0), flush=True)
        best = w
    w = best
    np.save(os.path.join(ROOT, "solution", "_arc_w.npy"), w)

    # ---------------- label model ----------------
    lm = FeatMap()
    build_lab_cache(train, lm, gold=True)
    print("label feats: %d (%.0fs)" % (lm.size(), time.time() - t0), flush=True)
    lm.frozen = True

    lmod = LabModel(lm.size(), len(labels))
    lorder = np.arange(len(train))
    for ep in range(args.lepochs):
        rng.shuffle(lorder)
        acc = lab_epoch(lmod, train, lorder, lab2i)
        print("  lab ep %2d train-acc %.4f (%.0fs)" % (ep + 1, acc, time.time() - t0), flush=True)
    LW = lmod.averaged()

    # ---------------- dev score ----------------
    if dev:
        arc_predict(dev, w, model)
        preds = []
        for it in dev:
            F = lab_matrix(it, lm, it["pred_heads"])
            pl = LW[F].sum(axis=1).argmax(axis=1)
            preds.append([labels[i] for i in pl])
        las, uas = evaluate(dev, preds)
        print("DEV  UAS %.4f  LAS %.4f" % (uas, las), flush=True)
        # oracle: gold heads -> label accuracy ceiling
        gp = []
        for it in dev:
            F = lab_matrix(it, lm, it["gold_heads"])
            pl = LW[F].sum(axis=1).argmax(axis=1)
            gp.append([labels[i] for i in pl])
        nc = nt = 0
        for it, pl in zip(dev, gp):
            for d in range(it["n"]):
                nt += 1
                nc += pl[d] == it["gold_rels"][d]
        print("DEV  label-acc given gold head %.4f" % (nc / nt), flush=True)

    # ---------------- test predictions ----------------
    arc_predict(test, w, model)
    rows = []
    for it in test:
        n = it["n"]
        F = lab_matrix(it, lm, it["pred_heads"])
        pl = LW[F].sum(axis=1).argmax(axis=1)
        parts = []
        for d in range(n):
            h = it["pred_heads"][d]
            hh = 0 if h < 0 else h + 1
            parts.append("%d:%s" % (hh, labels[int(pl[d])]))
        rows.append((it["id"], "|".join(parts)))
    sub = pd.DataFrame(rows, columns=["id", "parse"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sub.to_csv(args.out, index=False)
    print("wrote", args.out, sub.shape, "(%.0fs)" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
