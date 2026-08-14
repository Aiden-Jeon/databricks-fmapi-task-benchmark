"""Structured averaged perceptron for char-level NER tagging.

Features: memm2.extract_feats(sent) gives context features per char. At both
training and decode time we append a prev-tag feature 'PT=<tag>'. During
training the PT feature comes from the *gold* prefix (like a MEMM); at decode
time we use beam search where each state carries its own prev-tag, so the PT
feature is the state's previous tag — this makes train/test feature
distributions match the structured setting.

Update: standard structured perceptron — if decoded path != gold path, for
each position where they differ (features differ, since PT may differ too we
recompute full feature sets), add gold features' weights and subtract
predicted features' weights, using the gold-prefix PT for gold and
pred-prefix PT for pred.
"""
import sys, os, time, random, pickle
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TASK_DIR, LABELS
from memm2 import (TAGS, TAG2ID, NUM_TAGS, extract_feats, build_gold_tags,
                   make_gazetteer, TRANS, STATE0, tags_to_spans)

B_OF = {li: TAG2ID[f'B-{LABELS[li]}'] for li in range(len(LABELS))}
I_OF = {li: TAG2ID[f'I-{LABELS[li]}'] for li in range(len(LABELS))}
NEG = -1e9


class AveragedPerceptron:
    def __init__(self):
        self.W = defaultdict(lambda: np.zeros(NUM_TAGS, dtype=np.float64))
        self._tot = defaultdict(lambda: np.zeros(NUM_TAGS, dtype=np.float64))
        self._ts = defaultdict(int)
        self.i = 0

    def _touch(self, f):
        w = self.W[f]
        tot = self._tot[f]
        tot += (self.i - self._ts[f]) * w
        self._ts[f] = self.i
        return w

    def update(self, f_add, tag_add, f_sub, tag_sub):
        """+1 to weights of (feature,tag_add), -1 to (feature,tag_sub)."""
        self.i += 1
        for fl, ta in zip(f_add, tag_add):
            for f in fl:
                self._touch(f)[ta] += 1.0
        for fl, tb in zip(f_sub, tag_sub):
            for f in fl:
                self._touch(f)[tb] -= 1.0

    def average(self):
        for f, w in self.W.items():
            tot = self._tot[f]
            tot += (self.i - self._ts[f]) * w
            w[:] = tot / max(1, self.i)
        self._tot = defaultdict(lambda: np.zeros(NUM_TAGS, dtype=np.float64))
        self._ts = defaultdict(int)
        self.i = 0


def scores_with_pt(per, feats, pt_name='O'):
    """Score matrix where every char gets PT=pt_name feature (used for gold-independent decode start)."""
    n = len(feats)
    S = np.zeros((n, NUM_TAGS))
    ptf = 'PT=' + pt_name
    W = per.W
    for i, fl in enumerate(feats):
        row = S[i]
        for f in fl:
            w = W.get(f)
            if w is not None:
                row += w
        w = W.get(ptf)
        if w is not None:
            row += w
    return S


def decode(per, feats, beam=20, min_ent_len=2, start_scores=None):
    """Beam search where each state is (tag, etype, entlen); the PT feature
    for position i is the state's tag at i-1. Scores are computed
    incrementally: base context score + PT-dependent score."""
    n = len(feats)
    # base scores (context features only)
    B = np.zeros((n, NUM_TAGS))
    W = per.W
    for i, fl in enumerate(feats):
        row = B[i]
        for f in fl:
            w = W.get(f)
            if w is not None:
                row += w
    # PT score lookup: pt_score[prev_tag][cur_tag]
    pt_names = TAGS
    PT = np.zeros((NUM_TAGS, NUM_TAGS))
    for pt in range(NUM_TAGS):
        w = W.get('PT=' + pt_names[pt])
        if w is not None:
            PT[pt] = w
    # beam
    cur = {}
    # at position 0, PT='O'
    for (t, e) in STATE0:
        cur[(t, e, 1 if e >= 0 else 0)] = B[0, t] + PT[TAG2ID['O'], t]
    bps = []
    for i in range(1, n):
        nxt = {}
        bp = {}
        bi = B[i]
        for (t, e, el), sc in cur.items():
            for (nt, ne) in TRANS[(t, e)]:
                if ne >= 0:
                    nel = el + 1 if nt == I_OF[ne] else 1
                else:
                    if e >= 0 and el < min_ent_len:
                        continue
                    nel = 0
                key = (nt, ne, nel)
                v = sc + bi[nt] + PT[t, nt]
                if key not in nxt or v > nxt[key]:
                    nxt[key] = v
                    bp[key] = (t, e, el)
        if len(nxt) > beam:
            top = sorted(nxt.items(), key=lambda kv: -kv[1])[:beam]
            keep = set(k for k, _ in top)
            nxt = dict(top)
            bp = {k: bp[k] for k in keep}
        bps.append(bp)
        cur = nxt
    best = None
    bestv = -1e18
    for st, v in cur.items():
        t, e, el = st
        if e >= 0 and el < min_ent_len:
            continue
        if v > bestv:
            bestv = v
            best = st
    if best is None:
        best = max(cur.items(), key=lambda kv: kv[1])[0]
    path = [0] * n
    path[n - 1] = best[0]
    st = best
    for i in range(n - 1, 0, -1):
        st = bps[i - 1][st]
        path[i - 1] = st[0]
    return path


def feats_with_pt(feats, path):
    out = []
    prev = 'O'
    for fl, t in zip(feats, path):
        out.append(fl + ['PT=' + prev])
        prev = TAGS[t]
    return out


def train(train_feats, train_tags, epochs=12, beam=12, seed=42, verbose=True,
          min_ent_len=2):
    rng = random.Random(seed)
    per = AveragedPerceptron()
    t0 = time.time()
    for ep in range(epochs):
        order = list(range(len(train_feats)))
        rng.shuffle(order)
        nerr = 0
        ntok = 0
        for si in order:
            feats = train_feats[si]
            gold = train_tags[si]
            pred = decode(per, feats, beam=beam, min_ent_len=min_ent_len)
            if pred != gold:
                nerr += sum(1 for a, b in zip(gold, pred) if a != b)
                gf = feats_with_pt(feats, gold)
                pf = feats_with_pt(feats, pred)
                per.update(gf, gold, pf, pred)
            ntok += len(gold)
        if verbose:
            print(f"  ep{ep+1}/{epochs} char-err {nerr/max(ntok,1):.4f} feats {len(per.W)} {time.time()-t0:.0f}s", flush=True)
    per.average()
    return per


def predict_sent(per, sent, gaz_map, gaz_maxlen, beam=20, min_ent_len=2):
    feats = extract_feats(sent, gaz_map, gaz_maxlen)
    path = decode(per, feats, beam=beam, min_ent_len=min_ent_len)
    return tags_to_spans(sent, path)


def save(per, path):
    W = {f: w.tolist() for f, w in per.W.items()}
    with open(path, 'wb') as fp:
        pickle.dump(W, fp)


def load(path):
    with open(path, 'rb') as fp:
        W = pickle.load(fp)
    per = AveragedPerceptron()
    for f, w in W.items():
        per.W[f] = np.array(w, dtype=np.float64)
    return per
