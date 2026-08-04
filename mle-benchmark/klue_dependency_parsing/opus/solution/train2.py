"""Second-order (adjacent-sibling) structured-perceptron parser for KLUE-DP.

Search space: projective, head-final trees rooted at the last word (verified to cover
100% of train.csv).  Exact O(n^3) DP with arc + adjacent-sibling + valency factors.
Deprel labels are predicted by a multiclass averaged perceptron on the decoded arcs.

usage: python solution/train2.py [--dev 0.1] [--epochs 20] [--out outputs/submission.csv]
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dp_lib import (arc_feats, sib_feats, fc_feats, lab_feats, decode2, load,  # noqa: E402
                    word_counts, set_vocab)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEG = -1e17


class FeatMap:
    def __init__(self):
        self.d = {}
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


def _pack(rows):
    if not rows:
        return np.zeros((0, 1), dtype=np.int32)
    K = max(len(r) for r in rows)
    F = np.zeros((len(rows), K), dtype=np.int32)
    for i, r in enumerate(rows):
        F[i, :len(r)] = r
    return F


C2 = np.array([math.comb(i, 2) for i in range(64)], dtype=np.int64)
C3 = np.array([math.comb(i, 3) for i in range(64)], dtype=np.int64)


def build_cache(data, fm, tag=""):
    """Cache arc / sibling / valency feature-index matrices for every sentence.

    Row layouts (allow O(1) arithmetic lookup, no dicts):
      arc  (d,h), h>d          : off[d] + (h-d-1)
      sib  (h,c1,c2), c2<c1<h  : C3[h] + C2[c1] + c2
      fc   (h)                 : h
    """
    t0 = time.time()
    for si, it in enumerate(data):
        n = it["n"]
        tl = it["tl"]
        # arcs
        rows, di, hi = [], [], []
        off = np.zeros(n + 1, dtype=np.int64)
        acc = 0
        for d in range(n - 1):
            off[d] = acc
            acc += n - 1 - d
            for h in range(d + 1, n):
                rows.append([fm.get(x) for x in arc_feats(tl, d, h, n)])
                di.append(d)
                hi.append(h)
        it["F"] = _pack(rows)
        it["di"] = np.asarray(di, dtype=np.int32)
        it["hi"] = np.asarray(hi, dtype=np.int32)
        it["off"] = off
        # siblings
        srows, sh, sc1, sc2 = [], [], [], []
        for h in range(n):
            for c1 in range(h):
                for c2 in range(c1):
                    srows.append([fm.get(x) for x in sib_feats(tl, h, c1, c2, n)])
                    sh.append(h)
                    sc1.append(c1)
                    sc2.append(c2)
        it["SF"] = _pack(srows)
        it["sh"] = np.asarray(sh, dtype=np.int32)
        it["sc1"] = np.asarray(sc1, dtype=np.int32)
        it["sc2"] = np.asarray(sc2, dtype=np.int32)
        # valency (head has >= 1 dependent)
        it["FF"] = _pack([[fm.get(x) for x in fc_feats(tl, h, n)] for h in range(n)])
        if tag and (si + 1) % 800 == 0:
            print("  [%s] cache %d/%d %.0fs |feat|=%d"
                  % (tag, si + 1, len(data), time.time() - t0, fm.size()), flush=True)
    return fm


def score_all(it, w):
    n = it["n"]
    sc = np.full((n, n), NEG, dtype=np.float64)
    if it["F"].shape[0]:
        sc[it["di"], it["hi"]] = w[it["F"]].sum(axis=1)
    sibt = np.zeros((n, n, n), dtype=np.float64)
    if it["SF"].shape[0]:
        sibt[it["sh"], it["sc1"], it["sc2"]] = w[it["SF"]].sum(axis=1)
    fcs = w[it["FF"]].sum(axis=1)
    return sc, sibt, fcs


def struct_feats(it, heads):
    """All feature indices of the structure defined by `heads` (0-based, -1 = root)."""
    n = it["n"]
    parts = []
    F, SF, FF = it["F"], it["SF"], it["FF"]
    off = it["off"]
    kids = [[] for _ in range(n)]
    arow = []
    for d in range(n - 1):
        h = heads[d]
        arow.append(off[d] + (h - d - 1))
        kids[h].append(d)
    if arow:
        parts.append(F[np.asarray(arow, dtype=np.int64)].ravel())
    srow = []
    frow = []
    for h in range(n):
        ks = kids[h]
        if not ks:
            continue
        frow.append(h)
        for j in range(len(ks) - 1):
            c2, c1 = ks[j], ks[j + 1]      # ks ascending; c1 is nearer to h
            srow.append(C3[h] + C2[c1] + c2)
    if srow:
        parts.append(SF[np.asarray(srow, dtype=np.int64)].ravel())
    if frow:
        parts.append(FF[np.asarray(frow, dtype=np.int64)].ravel())
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int32)


class Model:
    def __init__(self, size):
        self.w = np.zeros(size, dtype=np.float64)
        self.wa = np.zeros(size, dtype=np.float64)
        self.last = np.zeros(size, dtype=np.int64)
        self.t = 0

    def update(self, gf, pf, loss=None, C=1.0):
        """MIRA / structured-hinge subgradient step.  If `loss` is None this is a plain
        perceptron update; otherwise the step size is the MIRA closed form
        eta = clip((loss - w.(phi(y)-phi(yhat))) / ||phi(y)-phi(yhat)||^2, 0, C)."""
        idx = np.concatenate((gf, pf)).astype(np.int64)
        val = np.concatenate((np.ones(len(gf)), -np.ones(len(pf))))
        uq, inv = np.unique(idx, return_inverse=True)
        delta = np.bincount(inv, weights=val, minlength=len(uq))
        keep = (uq != 0) & (delta != 0.0)
        uq = uq[keep]
        delta = delta[keep]
        if not len(uq):
            return
        if loss is None:
            eta = 1.0
        else:
            sdiff = float(self.w[uq] @ delta)
            nrm = float(delta @ delta)
            eta = (loss - sdiff) / nrm
            if eta <= 0.0:
                return
            eta = min(eta, C)
        self.t += 1
        self.wa[uq] += self.w[uq] * (self.t - self.last[uq])
        self.w[uq] += eta * delta
        self.last[uq] = self.t

    def averaged(self):
        out = (self.wa + self.w * (self.t - self.last)) / max(self.t, 1)
        out[0] = 0.0
        return out


def epoch(model, data, order, mira=True, C=1.0):
    nc = nt = 0
    for si in order:
        it = data[si]
        n = it["n"]
        nt += n
        nc += 1                                    # root always correct
        if n <= 1:
            continue
        sc, sibt, fcs = score_all(it, model.w)
        gold = it["gold_heads"]
        if mira:
            # cost-augmented inference: +1 for every non-gold arc (Hamming loss)
            scc = sc.copy()
            scc[it["di"], it["hi"]] += 1.0
            ar = np.arange(n - 1)
            scc[ar, np.asarray(gold[:n - 1])] -= 1.0
            pred = decode2(scc, sibt, fcs, n)
        else:
            pred = decode2(sc, sibt, fcs, n)
        same = sum(1 for d in range(n - 1) if pred[d] == gold[d])
        nc += same
        if same != n - 1:
            loss = float(n - 1 - same) if mira else None
            model.update(struct_feats(it, gold), struct_feats(it, pred), loss, C)
    return nc / max(nt, 1)


def predict(data, w):
    for it in data:
        n = it["n"]
        if n <= 1:
            it["pred_heads"] = [-1]
            continue
        sc, sibt, fcs = score_all(it, w)
        it["pred_heads"] = decode2(sc, sibt, fcs, n)


# ------------------------------- label model ------------------------------- #

def lab_matrix(it, lm, heads):
    n = it["n"]
    tl = it["tl"]
    rows = [[lm.get(x) for x in lab_feats(tl, d, heads[d], n)] for d in range(n)]
    return _pack(rows)


class LabModel:
    def __init__(self, size, nlab):
        self.W = np.zeros((size, nlab), dtype=np.float64)
        self.WA = np.zeros((size, nlab), dtype=np.float64)
        self.last = np.zeros(size, dtype=np.int64)
        self.t = 0

    def update(self, feats, gl, pl):
        self.t += 1
        uq, cnt = np.unique(feats, return_counts=True)
        keep = uq != 0
        uq = uq[keep]
        cnt = cnt[keep].astype(np.float64)
        if not len(uq):
            return
        self.WA[uq] += self.W[uq] * (self.t - self.last[uq])[:, None]
        self.W[uq, gl] += cnt
        self.W[uq, pl] -= cnt
        self.last[uq] = self.t

    def averaged(self):
        out = (self.WA + self.W * (self.t - self.last)[:, None]) / max(self.t, 1)
        out[0] = 0.0
        return out


def lab_epoch(model, data, order, lab2i):
    nc = nt = 0
    for si in order:
        it = data[si]
        F = it["LF"]
        S = model.W[F].sum(axis=1)
        for d in range(it["n"]):
            g = lab2i[it["gold_rels"][d]]
            nt += 1
            if int(S[d].argmax()) == g:
                nc += 1
            # margin-augmented argmax: +1 for every wrong label
            s = S[d] + 1.0
            s[g] -= 1.0
            p = int(s.argmax())
            if p != g:
                model.update(F[d], g, p)
    return nc / max(nt, 1)


# ----------------------------------- main ---------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lepochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--minwf", type=int, default=2)
    ap.add_argument("--mira", type=int, default=1)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "submission.csv"))
    args = ap.parse_args()

    t0 = time.time()
    tr_df = pd.read_csv(os.path.join(ROOT, "train.csv"))
    te_df = pd.read_csv(os.path.join(ROOT, "test.csv"))
    if args.minwf > 1:
        wc = word_counts([tr_df])
        set_vocab({w for w, c in wc.items() if c >= args.minwf})
    train = load(tr_df, True)
    test = load(te_df, False)

    rng = np.random.RandomState(args.seed)
    if args.dev > 0:
        perm = rng.permutation(len(train))
        nd = int(len(train) * args.dev)
        dev = [train[i] for i in perm[:nd]]
        train = [train[i] for i in perm[nd:]]
    else:
        dev = []
    print("train=%d dev=%d test=%d" % (len(train), len(dev), len(test)), flush=True)

    labels = sorted({r for it in (train + dev) for r in it["gold_rels"]})
    lab2i = {l: i for i, l in enumerate(labels)}

    fm = FeatMap()
    build_cache(train, fm, "train")
    print("features: %d (%.0fs)" % (fm.size(), time.time() - t0), flush=True)
    fm.frozen = True
    if dev:
        build_cache(dev, fm)
    build_cache(test, fm)
    print("caches done (%.0fs)" % (time.time() - t0), flush=True)

    def dev_uas(ww):
        predict(dev, ww)
        nt = uas = 0
        for it in dev:
            for d in range(it["n"]):
                nt += 1
                uas += it["pred_heads"][d] == it["gold_heads"][d]
        return uas / nt

    # Train `--seeds` independent runs over the same (shared) feature cache and average
    # their weight vectors.  For a linear model this equals averaging the arc/sibling
    # scores of an ensemble, and reduces the variance of perceptron/MIRA training.
    order = np.arange(len(train))
    wsum = np.zeros(fm.size(), dtype=np.float64)
    for s in range(args.seeds):
        model = Model(fm.size())
        srng = np.random.RandomState(args.seed + 1000 * s)
        for ep in range(args.epochs):
            srng.shuffle(order)
            acc = epoch(model, train, order, mira=bool(args.mira), C=args.C)
            if (ep + 1) % 4 == 0 or ep == args.epochs - 1:
                print("  seed %d ep %2d train-arc-acc %.4f (%.0fs)"
                      % (s, ep + 1, acc, time.time() - t0), flush=True)
        w_s = model.averaged()
        del model
        if dev:
            print("  seed %d dev-UAS %.4f" % (s, dev_uas(w_s)), flush=True)
        wsum += w_s
        del w_s
    w = wsum / args.seeds
    if dev and args.seeds > 1:
        print("ENSEMBLE(%d) dev-UAS %.4f" % (args.seeds, dev_uas(w)), flush=True)

    # ---- labels
    lm = FeatMap()
    for it in train:
        it["LF"] = lab_matrix(it, lm, it["gold_heads"])
    lm.frozen = True
    lmod = LabModel(lm.size(), len(labels))
    lorder = np.arange(len(train))
    for ep in range(args.lepochs):
        rng.shuffle(lorder)
        a = lab_epoch(lmod, train, lorder, lab2i)
        print("  lab ep %2d acc %.4f (%.0fs)" % (ep + 1, a, time.time() - t0), flush=True)
    LW = lmod.averaged()

    def label(it, heads):
        F = lab_matrix(it, lm, heads)
        return [labels[int(i)] for i in LW[F].sum(axis=1).argmax(axis=1)]

    if dev:
        predict(dev, w)
        nt = las = uas = lac = 0
        for it in dev:
            pl = label(it, it["pred_heads"])
            gl = label(it, it["gold_heads"])
            for d in range(it["n"]):
                nt += 1
                hok = it["pred_heads"][d] == it["gold_heads"][d]
                uas += hok
                las += hok and pl[d] == it["gold_rels"][d]
                lac += gl[d] == it["gold_rels"][d]
        print("DEV UAS %.4f LAS %.4f  label-acc(gold-head) %.4f"
              % (uas / nt, las / nt, lac / nt), flush=True)

    predict(test, w)
    rows = []
    for it in test:
        pl = label(it, it["pred_heads"])
        parts = []
        for d in range(it["n"]):
            h = it["pred_heads"][d]
            parts.append("%d:%s" % (0 if h < 0 else h + 1, pl[d]))
        rows.append((it["id"], "|".join(parts)))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(rows, columns=["id", "parse"]).to_csv(args.out, index=False)
    print("wrote %s (%.0fs)" % (args.out, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
