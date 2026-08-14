"""Logistic-regression dependency parser for KLUE-DP (eojeol-level).

Structure exploits two properties of the data:
  - every arc points right (head index > dependent index)
  - the last token is always the root
Hence picking argmax head among rightward candidates per token always
produces a valid tree.

Two models:
  1. arc model   : binary LR over (dep, candidate head) pairs
  2. label model : multiclass LR over (dep, head) pairs -> deprel
"""
import argparse
import pickle
from collections import Counter

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from common import load_train, load_test, jamo_flat, las, write_submission

PUNCT = set('.!?,;:~…·')

_suf_cache = {}


def suf(tok):
    """Suffix/shape features of one token (cached)."""
    f = _suf_cache.get(tok)
    if f is not None:
        return f
    jam = jamo_flat(tok)
    f = {
        'js1': ''.join(jam[-1:]),
        'js2': ''.join(jam[-2:]),
        'js3': ''.join(jam[-3:]),
        'js4': ''.join(jam[-4:]),
        'js5': ''.join(jam[-5:]),
        'js6': ''.join(jam[-6:]),
        'cs1': tok[-1:],
        'cs2': tok[-2:],
        'cs3': tok[-3:],
        'first': tok[0] if tok else '',
        'len': min(len(tok), 8),
        'digit': any(c.isdigit() for c in tok),
        'punct': tok[-1] in PUNCT,
    }
    _suf_cache[tok] = f
    return f


ROOT_SUF = {k: 'ROOT' for k in
            ('js1', 'js2', 'js3', 'js4', 'js5', 'js6', 'cs1', 'cs2', 'cs3',
             'first')}
ROOT_SUF.update({'len': 0, 'digit': False, 'punct': False})


def pair_feats(toks, i, j):
    """Categorical features for dependent index i and head index j (0-based).

    j == -1 denotes the artificial ROOT (used by the label model for the
    root token only).
    """
    n = len(toks)
    fi = suf(toks[i])
    fj = ROOT_SUF if j == -1 else suf(toks[j])
    f = {}
    for k in ('js2', 'js3', 'js4', 'js5', 'cs2', 'cs3'):
        f['i_' + k] = fi[k]
        f['j_' + k] = fj[k]
    f['cross33'] = (fi['js3'], fj['js3'])
    f['cross43'] = (fi['js4'], fj['js3'])
    f['cross32'] = (fi['js3'], fj['js2'])
    f['cross22'] = (fi['js2'], fj['js2'])
    f['cross44'] = (fi['js4'], fj['js4'])
    f['cross55'] = (fi['js5'], fj['js5'])
    f['cross23'] = (fi['js2'], fj['js3'])
    f['cross34'] = (fi['js3'], fj['js4'])
    f['dist'] = 'R' if j == -1 else str(min(j - i, 10))
    f['j_last'] = 'R' if j == -1 else str(j == n - 1)
    f['i_from_end'] = str(min(n - 1 - i, 5))
    f['j_from_end'] = 'R' if j == -1 else str(min(n - 1 - j, 5))
    f['n'] = str(min(n, 12))
    f['dist_jend'] = ('R' if j == -1 else
                      (str(min(j - i, 5)), str(min(n - 1 - j, 3))))
    f['i_digit'] = str(fi['digit'])
    f['j_digit'] = str(fj['digit'])
    f['i_punct'] = str(fi['punct'])
    f['j_punct'] = str(fj['punct'])
    f['i_len'] = str(fi['len'])
    f['j_len'] = str(fj['len'])
    if i > 0:
        f['im1_js3'] = suf(toks[i - 1])['js3']
        f['im1_cs2'] = suf(toks[i - 1])['cs2']
        f['im1_j'] = (suf(toks[i - 1])['js3'], fj['js3'])
    if i + 1 < n:
        f['ip1_js3'] = suf(toks[i + 1])['js3']
        f['i_ip1'] = (fi['js3'], suf(toks[i + 1])['js3'])
    if j != -1 and j + 1 < n:
        f['jp1_js3'] = suf(toks[j + 1])['js3']
    if j != -1 and j > 0:
        f['jm1_js3'] = suf(toks[j - 1])['js3']
        f['i_jm1'] = (fi['js3'], suf(toks[j - 1])['js3'])
    # between-token context for longer arcs
    if j != -1 and j - i >= 2:
        f['i_mid'] = (fi['js3'], suf(toks[i + 1])['js3'])
        if j - i >= 3:
            f['i_mid2'] = (fi['js3'], suf(toks[j - 1])['js3'])
    return f


def tok_label_extra(toks, i):
    fi = suf(toks[i])
    return {'t_js6': fi['js6'], 't_first': fi['first']}


# ---------------------------------------------------------------- arc model
def build_arc_data(rows):
    X, y = [], []
    for r in rows:
        toks, n = r['tokens'], len(r['tokens'])
        for i in range(n - 1):
            gh = r['heads'][i] - 1  # 0-based gold head
            for j in range(i + 1, n):
                X.append(pair_feats(toks, i, j))
                y.append(j == gh)
    return X, np.array(y)


def train_arc(rows, C=1.0, class_weight=None, seed=0):
    X, y = build_arc_data(rows)
    vec = DictVectorizer(sparse=True)
    Xv = vec.fit_transform(X)
    clf = LogisticRegression(C=C, max_iter=600, class_weight=class_weight,
                             random_state=seed)
    clf.fit(Xv, y)
    return vec, clf


def predict_heads(vec, clf, toks):
    n = len(toks)
    heads = [0] * n
    for i in range(n - 1):
        cand = [pair_feats(toks, i, j) for j in range(i + 1, n)]
        proba = clf.predict_proba(vec.transform(cand))[:, 1]
        heads[i] = i + 2 + int(np.argmax(proba))  # 1-indexed
    heads[n - 1] = 0
    return heads


def score_matrix(vec, clf, toks):
    """Log-prob of arc i->j for every rightward pair. Returns n x n matrix
    with -inf where no arc is possible. Index: [dep][head], 0-based."""
    n = len(toks)
    S = np.full((n, n), -1e9)
    for i in range(n - 1):
        cand = [pair_feats(toks, i, j) for j in range(i + 1, n)]
        lp = clf.predict_log_proba(vec.transform(cand))[:, 1]
        for k, j in enumerate(range(i + 1, n)):
            S[i][j] = lp[k]
    return S


def decode(S):
    """Decode right-branching projective tree with root = last token.

    Trick: reverse the sentence.  In reversed coordinates the root becomes
    token 0 and all arcs go right with root as leftmost -> standard
    right-branching Eisner where H[s][t] = subtree rooted at s (leftmost)
    covering [s..t]."""
    n = S.shape[0]
    # reversed score: dep' = n-1-j (original head becomes dep in reversed?
    # original arc dep->head (dep<head). reversed index x = n-1-orig.
    # In reversed sentence, token x corresponds to orig n-1-x.
    # We want a right-branching tree on reversed tokens where parent of x
    # is some y < x ... simplest: build R[x][y] = score of arc x->y in the
    # reversed sentence where an arc means "x is child of y" with y>x.
    # orig arc dep->head  <=>  reversed: x=n-1-head (parent), y=n-1-dep
    # (child); x < y. So reversed parent x, child y, score S[dep][head] =
    # S[n-1-y][n-1-x].
    n_ = n
    NEG = -1e18
    H = np.full((n_, n_), NEG)
    back = {}
    for i in range(n_):
        H[i][i] = 0.0

    def arc(parent_x, child_y):
        # score of parent_x -> child_y in reversed coords
        return S[n_ - 1 - child_y][n_ - 1 - parent_x]

    for width in range(1, n_):
        for s in range(0, n_ - width):
            t = s + width
            best = NEG
            br = None
            for r in range(s + 1, t + 1):
                v = H[s][r - 1] + arc(s, r) + H[r][t]
                if v > best:
                    best = v
                    br = r
            H[s][t] = best
            back[(s, t)] = br

    parent = [-1] * n_  # reversed parent of each reversed token

    def recurse(s, t):
        if s == t:
            return
        r = back[(s, t)]
        parent[r] = s
        recurse(s, r - 1)
        recurse(r, t)

    recurse(0, n_ - 1)
    # reversed token 0 (orig last token) has parent -1 (root). Convert back.
    heads = [0] * n
    for y in range(1, n_):
        x = parent[y]
        orig_head = n_ - 1 - x
        orig_dep = n_ - 1 - y
        heads[orig_dep] = orig_head  # 0-based head
    heads[n - 1] = -1
    return heads


def predict_heads_decode(vec, clf, toks):
    S = score_matrix(vec, clf, toks)
    h0 = decode(S)          # 0-based heads, -1 for root
    return [0 if h == -1 else h + 1 for h in h0]


# --------------------------------------------------------------- label model
def build_label_data(rows):
    X, y = [], []
    for r in rows:
        toks, n = r['tokens'], len(r['tokens'])
        for i in range(n):
            h = r['heads'][i]
            j = -1 if h == 0 else h - 1
            f = pair_feats(toks, i, j)
            f.update(tok_label_extra(toks, i))
            X.append(f)
            y.append(r['labels'][i])
    return X, y


def train_label(rows, C=1.0, seed=0):
    X, y = build_label_data(rows)
    vec = DictVectorizer(sparse=True)
    Xv = vec.fit_transform(X)
    clf = LogisticRegression(C=C, max_iter=600, random_state=seed)
    clf.fit(Xv, y)
    return vec, clf


def predict_labels(vec, clf, toks, heads):
    n = len(toks)
    feats = []
    for i in range(n):
        h = heads[i]
        j = -1 if h == 0 else h - 1
        f = pair_feats(toks, i, j)
        f.update(tok_label_extra(toks, i))
        feats.append(f)
    return list(clf.predict(vec.transform(feats)))


# ------------------------------------------------------------------- driver
class Parser:
    def __init__(self, seeds=(0,)):
        self.seeds = seeds
        self.arc_models = []   # list of (vec, clf)
        self.lab_models = []

    def fit(self, rows, C_arc=1.0, C_lab=1.0, cw=None):
        for s in self.seeds:
            self.arc_models.append(train_arc(rows, C_arc, cw, seed=s))
            self.lab_models.append(train_label(rows, C_lab, seed=s))
        # back-compat single-model handles
        self.arc_vec, self.arc_clf = self.arc_models[0]
        self.lab_vec, self.lab_clf = self.lab_models[0]
        return self

    def predict(self, rows):
        out = []
        for r in rows:
            toks = r['tokens']
            n = len(toks)
            # average arc log-probs over ensemble, then argmax per token
            heads = [0] * n
            for i in range(n - 1):
                cand = [pair_feats(toks, i, j) for j in range(i + 1, n)]
                acc = None
                for vec, clf in self.arc_models:
                    lp = clf.predict_log_proba(vec.transform(cand))[:, 1]
                    acc = lp if acc is None else acc + lp
                heads[i] = i + 2 + int(np.argmax(acc))
            # label: majority vote over ensemble (with prob averaging)
            labels = self._vote_labels(toks, heads)
            out.append({'id': r['id'], 'heads': heads, 'labels': labels})
        return out

    def _vote_labels(self, toks, heads):
        n = len(toks)
        feats = []
        for i in range(n):
            h = heads[i]
            j = -1 if h == 0 else h - 1
            f = pair_feats(toks, i, j)
            f.update(tok_label_extra(toks, i))
            feats.append(f)
        if len(self.lab_models) == 1:
            vec, clf = self.lab_models[0]
            return list(clf.predict(vec.transform(feats)))
        # average class probabilities
        acc = None
        classes = None
        for vec, clf in self.lab_models:
            P = clf.predict_proba(vec.transform(feats))
            acc = P if acc is None else acc + P
            classes = clf.classes_
        idx = np.argmax(acc, axis=1)
        return [classes[k] for k in idx]

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)


def split(rows, seed=42, ndev=480):
    import random
    random.seed(seed)
    idx = list(range(len(rows)))
    random.shuffle(idx)
    dev = set(idx[:ndev])
    return ([r for k, r in enumerate(rows) if k not in dev],
            [r for k, r in enumerate(rows) if k in dev])


def uas(gold_rows, pred_rows):
    pred_map = {p['id']: p for p in pred_rows}
    correct = total = 0
    for g in gold_rows:
        p = pred_map[g['id']]
        for gh, ph in zip(g['heads'], p['heads']):
            total += 1
            correct += (gh == ph)
    return correct / max(total, 1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval', action='store_true', help='holdout evaluation')
    ap.add_argument('--C_arc', type=float, default=1.0)
    ap.add_argument('--C_lab', type=float, default=1.0)
    ap.add_argument('--cw', action='store_true', help='balanced class weight for arc model')
    ap.add_argument('--save', default='solution/model.pkl')
    ap.add_argument('--seeds', type=int, default=1, help='ensemble size')
    args = ap.parse_args()

    rows = load_train()
    seeds = tuple(range(args.seeds))
    if args.eval:
        tr, dv = split(rows)
        p = Parser(seeds).fit(tr, args.C_arc, args.C_lab,
                              'balanced' if args.cw else None)
        pred = p.predict(dv)
        print('UAS:', uas(dv, pred))
        print('LAS:', las(dv, pred))
    else:
        p = Parser(seeds).fit(rows, args.C_arc, args.C_lab,
                              'balanced' if args.cw else None)
        p.save(args.save)
        test = load_test()
        write_submission(p.predict(test), 'outputs/submission.csv')
        print('wrote outputs/submission.csv')
