"""Averaged structured perceptron char tagger with Viterbi decoding (numpy)."""
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from data import LABELS  # noqa
from features import NFEAT, MAX_F, sent_features, build_gaz, gaz_matches  # noqa

TAGS = ["O"] + ["B-" + l for l in LABELS] + ["I-" + l for l in LABELS]
TAG2I = {t: i for i, t in enumerate(TAGS)}
NT = len(TAGS)
NEG = -1e9


def transition_mask():
    """mask[prev, cur] = 0 allowed, NEG forbidden."""
    m = np.zeros((NT, NT), dtype=np.float32)
    for pi, pt in enumerate(TAGS):
        for ci, ct in enumerate(TAGS):
            if ct.startswith("I-"):
                typ = ct[2:]
                if not (pt == "B-" + typ or pt == "I-" + typ):
                    m[pi, ci] = NEG
    return m


MASK = transition_mask()
START_MASK = np.array([NEG if t.startswith("I-") else 0.0 for t in TAGS], dtype=np.float32)


class Tagger:
    def __init__(self, seed=0):
        self.W = np.zeros((NFEAT, NT), dtype=np.float32)
        self.T = np.zeros((NT, NT), dtype=np.float32)   # transitions
        self.rng = np.random.RandomState(seed)

    # -------------------------------------------------- scoring / decoding
    def emissions(self, ids, W=None):
        W = self.W if W is None else W
        return W[ids].sum(axis=1)

    def viterbi(self, E, T):
        n = E.shape[0]
        bp = np.empty((n, NT), dtype=np.int8)
        score = E[0] + START_MASK
        TM = T + MASK
        for i in range(1, n):
            cand = score[:, None] + TM
            bp[i] = np.argmax(cand, axis=0)
            score = cand[bp[i], np.arange(NT)] + E[i]
        path = np.empty(n, dtype=np.int8)
        path[n - 1] = int(np.argmax(score))
        for i in range(n - 1, 0, -1):
            path[i - 1] = bp[i, path[i]]
        return path

    def predict_ids(self, ids, W=None, T=None):
        E = self.emissions(ids, W)
        return self.viterbi(E, self.T if T is None else T)

    # -------------------------------------------------- training
    def train(self, data, epochs=8, shuffle_seed=13, verbose=True, dev=None,
              dev_eval=None):
        """data: list of (ids, gold_tag_idx_array)."""
        N = len(data)
        acc = np.zeros((NFEAT, NT), dtype=np.float64)
        accT = np.zeros((NT, NT), dtype=np.float64)
        last = np.zeros(NFEAT, dtype=np.int64)
        lastT = [0]
        t = 0
        rng = np.random.RandomState(shuffle_seed)
        order = np.arange(N)
        best = None
        for ep in range(epochs):
            rng.shuffle(order)
            t0 = time.time()
            wrong = tot = 0
            for k in order:
                ids, gold = data[k]
                t += 1
                pred = self.predict_ids(ids)
                tot += len(gold)
                diff = pred != gold
                nd = int(diff.sum())
                wrong += nd
                if nd:
                    rows = ids[diff].ravel()
                    g = np.repeat(gold[diff], MAX_F)
                    p = np.repeat(pred[diff], MAX_F)
                    urows = np.unique(rows)
                    dt = (t - last[urows]).astype(np.float64)
                    acc[urows] += self.W[urows] * dt[:, None]
                    last[urows] = t
                    np.add.at(self.W, (rows, g), 1.0)
                    np.add.at(self.W, (rows, p), -1.0)
                    self.W[0] = 0.0
                    # transitions
                    gt = np.stack([gold[:-1], gold[1:]])
                    pt = np.stack([pred[:-1], pred[1:]])
                    accT += self.T * float(t - lastT[0])
                    lastT[0] = t
                    np.add.at(self.T, (gt[0], gt[1]), 1.0)
                    np.add.at(self.T, (pt[0], pt[1]), -1.0)
            if verbose:
                msg = "ep %d tokerr %.4f  %.1fs" % (ep + 1, wrong / tot, time.time() - t0)
                if dev_eval is not None:
                    Wa, Ta = self.averaged(acc, last, t, accT, lastT[0])
                    f1 = dev_eval(Wa, Ta)
                    msg += "  devF1 %.4f" % f1
                    if best is None or f1 > best[0]:
                        best = (f1, Wa, Ta, ep + 1)
                print(msg, flush=True)
        Wa, Ta = self.averaged(acc, last, t, accT, lastT[0])
        self.Wavg, self.Tavg = Wa, Ta
        self.best = best
        return self

    def averaged(self, acc, last, t, accT, lastT):
        a = acc + self.W * (t - last)[:, None]
        Wa = (a / max(t, 1)).astype(np.float32)
        Wa[0] = 0.0
        Ta = ((accT + self.T * float(t - lastT)) / max(t, 1)).astype(np.float32)
        return Wa, Ta
