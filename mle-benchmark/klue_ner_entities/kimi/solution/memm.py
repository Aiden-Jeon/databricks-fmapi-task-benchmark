"""Char-level MEMM tagger for KLUE-NER with Viterbi decoding.

Uses sklearn DictVectorizer + SGDClassifier(log_loss) + perceptron-style
weight averaging trick via partial_fit over epochs.
"""
import sys, os, re, time
import random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import Counter, defaultdict
from common import (TASK_DIR, LABELS, load_csv, parse_entities, entities_to_str,
                    score_f1, extract_rule_spans, build_gazetteer)

TAGS = ['O'] + [f'{p}-{t}' for t in LABELS for p in ('B', 'I')]
TAG2ID = {t: i for i, t in enumerate(TAGS)}
NUM_TAGS = len(TAGS)

# char class
_CTABLE = [0] * 128
for c in '0123456789': _CTABLE[ord(c)] = 1
for c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ': _CTABLE[ord(c)] = 2


def char_class(ch):
    o = ord(ch)
    if o < 128:
        c = _CTABLE[o]
        if c: return c
        if ch == ' ': return 3
        return 4
    if 0xAC00 <= o <= 0xD7A3: return 5
    if 0x3130 <= o <= 0x318F: return 6
    if 0x4E00 <= o <= 0x9FFF: return 7
    if 0x3040 <= o <= 0x30FF: return 8
    if ch == ' ': return 3
    return 4


def tokenize(sent):
    toks = []
    i, n = 0, len(sent)
    while i < n:
        c = char_class(sent[i])
        if c == 3:
            j = i
            while j < n and char_class(sent[j]) == 3: j += 1
            toks.append((i, j, ' '))
            i = j
        elif c == 4:
            j = i
            while j < n and char_class(sent[j]) == 4 and ord(sent[j]) < 128: j += 1
            if j == i: j = i + 1
            toks.append((i, j, 'P'))
            i = j
        else:
            j = i
            while j < n and char_class(sent[j]) == c: j += 1
            toks.append((i, j, 'C'))
            i = j
    return toks


class SentFeats:
    def __init__(self, sent, gaz=None):
        self.sent = sent
        self.n = len(sent)
        self.toks = tokenize(sent)
        self.tok_start = [t[0] for t in self.toks]
        self.tok_end = [t[1] for t in self.toks]
        self.ntok = len(self.toks)
        self.tokid = [0] * self.n
        self.toktype = [0] * self.n  # 0 C,1 P,2 space
        for ti, (s, e, tt) in enumerate(self.toks):
            v = 0 if tt == 'C' else (1 if tt == 'P' else 2)
            for i in range(s, e):
                self.tokid[i] = ti
                self.toktype[i] = v
        # rule spans covering each char
        self.rule = [None] * self.n
        for s, e, t in extract_rule_spans(sent):
            for i in range(s, e):
                self.rule[i] = t
        # gazetteer per-char max match info
        self.gaz_hit = [None] * self.n
        if gaz is not None:
            self._gaz(gaz)

    def _gaz(self, gaz):
        sent = self.sent
        n = self.n
        # build first-char index lazily done outside; gaz is dict expr->type
        maxlen = gaz.get('__maxlen__', 20)
        g = gaz['map']
        for i in range(n):
            best = None
            for L in range(min(maxlen, n - i), 1, -1):
                sub = sent[i:i + L]
                if sub in g:
                    best = (L, g[sub])
                    break
            if best:
                L, t = best
                for k in range(i, i + L):
                    cur = self.gaz_hit[k]
                    if cur is None or cur[1] < L - (k - i):
                        # store (type, remaining, is_first)
                        self.gaz_hit[k] = (t, L - (k - i), k == i)

    def feat_at(self, i, ptag):
        sent = self.sent
        toks = self.toks
        tokid = self.tokid
        n = self.n
        f = ['B', 'pt=' + ptag]
        f.append('c=' + sent[i])
        cc = char_class(sent[i])
        f.append('cc=%d' % cc)
        if i > 0: f.append('c-1=' + sent[i - 1]); f.append('cc-1=%d' % char_class(sent[i - 1]))
        else: f.append('BOS')
        if i < n - 1: f.append('c+1=' + sent[i + 1]); f.append('cc+1=%d' % char_class(sent[i + 1]))
        else: f.append('EOS')
        if i > 1: f.append('c-2=' + sent[i - 2])
        if i < n - 2: f.append('c+2=' + sent[i + 2])
        if i > 0: f.append('c-1c=' + sent[i - 1:i + 1])
        if i < n - 1: f.append('cc+1=' + sent[i:i + 2])
        if i > 1: f.append('c-2c-1=' + sent[i - 2:i])
        if i < n - 2: f.append('c+1c+2=' + sent[i + 1:i + 3])
        # token-level
        ti = tokid[i]
        ts, te, _ = toks[ti]
        ttype = self.toktype[i]
        f.append('tt=%d' % ttype)
        f.append('tpos=%d' % (i - ts))
        f.append('tlen=%d' % min(te - ts, 8))
        f.append('tlrem=%d' % min(te - i, 8))
        if ttype == 0:
            tok = sent[ts:te]
            f.append('tok=' + tok)
            f.append('p1=' + tok[:1]); f.append('s1=' + tok[-1:])
            if len(tok) >= 2:
                f.append('p2=' + tok[:2]); f.append('s2=' + tok[-2:])
            if len(tok) >= 3:
                f.append('p3=' + tok[:3]); f.append('s3=' + tok[-3:])
            if i - ts <= 3:
                f.append('pref=%s' % sent[ts:i + 1])
            if te - i <= 3:
                f.append('suf=%s' % sent[i:te])
        # prev token
        if ti > 0:
            ps, pe, pt = toks[ti - 1]
            if pt == 'C':
                pk = sent[ps:pe]
                f.append('-1s1=' + pk[-1:])
                if len(pk) >= 2: f.append('-1s2=' + pk[-2:])
            else:
                f.append('-1tt=%d' % self.toktype[i - (i - ps)])
        else:
            f.append('-1TOK=BOS')
        # next token
        if ti < self.ntok - 1:
            ns, ne, nt = toks[ti + 1]
            if nt == 'C':
                nk = sent[ns:ne]
                f.append('+1p1=' + nk[:1])
                if len(nk) >= 2: f.append('+1p2=' + nk[:2])
            else:
                f.append('+1tt=%d' % (1 if nt == 'P' else 2))
        else:
            f.append('+1TOK=EOS')
        # context tokens at distance 2
        if ti >= 2:
            s2, e2, t2 = toks[ti - 2]
            if t2 == 'C': f.append('-2s1=' + sent[s2:e2][-1:])
        if ti < self.ntok - 2:
            s2, e2, t2 = toks[ti + 2]
            if t2 == 'C': f.append('+2p1=' + sent[s2:e2][:1])
        # rule
        r = self.rule[i]
        if r: f.append('rule=' + r)
        # gazetteer
        gh = self.gaz_hit[i]
        if gh:
            f.append('gaz=' + gh[0])
            f.append('gazrem=%d' % min(gh[1], 8))
            if gh[2]: f.append('gazfirst')
        return f


# ---------- training data ----------

def build_train_data(rows, gaz):
    """Return list of (SentFeats, [tagids]) """
    data = []
    for row in rows:
        sent = row['sentence']
        n = len(sent)
        tags = ['O'] * n
        for expr, typ in parse_entities(row.get('entities', '')):
            start = 0
            while True:
                idx = sent.find(expr, start)
                if idx < 0:
                    break
                if tags[idx] == 'O':  # don't overwrite
                    tags[idx] = f'B-{typ}'
                    for k in range(idx + 1, idx + len(expr)):
                        tags[k] = f'I-{typ}'
                start = idx + 1
        sf = SentFeats(sent, gaz)
        data.append((sf, [TAG2ID[t] for t in tags]))
    return data


def train_memm(data, n_epochs=5, alpha=0.0001, seed=42, verbose=True):
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import SGDClassifier
    vec = DictVectorizer()
    clf = SGDClassifier(loss='log_loss', penalty='l2', alpha=alpha,
                        max_iter=1, tol=None, random_state=seed, learning_rate='invscaling',
                        eta0=0.5, power_t=0.5)
    rng = random.Random(seed)
    classes = np.arange(NUM_TAGS)
    t0 = time.time()
    for ep in range(n_epochs):
        idx = list(range(len(data)))
        rng.shuffle(idx)
        tot_loss = 0.0
        nb = 0
        for start in range(0, len(idx), 128):
            feats, ys = [], []
            for j in idx[start:start + 128]:
                sf, tags = data[j]
                prev = 'O'
                for i in range(sf.n):
                    feats.append({k: 1 for k in sf.feat_at(i, prev)})
                    ys.append(tags[i])
                    prev = TAGS[tags[i]]
            X = vec.transform(feats) if ep > 0 or nb > 0 else vec.fit_transform(feats)
            if nb == 0 and ep == 0:
                clf.partial_fit(X, ys, classes=classes)
            else:
                clf.partial_fit(X, ys)
            nb += 1
        if verbose:
            print(f"  epoch {ep+1}/{n_epochs} done, {time.time()-t0:.1f}s", flush=True)
    return vec, clf


# ---------- Viterbi ----------

def make_transitions():
    allowed = np.zeros((NUM_TAGS, NUM_TAGS), dtype=bool)
    o = TAG2ID['O']
    for t in LABELS:
        b = TAG2ID[f'B-{t}']; i_ = TAG2ID[f'I-{t}']
        allowed[o, b] = True
        allowed[b, i_] = True
        allowed[i_, i_] = True
        allowed[b, o] = True
        allowed[i_, o] = True
        allowed[b, b] = True
        allowed[i_, b] = True
        for t2 in LABELS:
            if t2 != t:
                allowed[b, TAG2ID[f'B-{t2}']] = True
                allowed[i_, TAG2ID[f'B-{t2}']] = True
    allowed[o, o] = True
    return allowed

ALLOWED = make_transitions()


def viterbi_decode(sf, vec, clf, temp=1.0, o_bias=0.0):
    n = sf.n
    scores = np.zeros((n, NUM_TAGS), dtype=np.float32)
    for i in range(n):
        x = vec.transform([{k: 1 for k in sf.feat_at(i, 'O')}])
        lp = clf.predict_log_proba(x)[0]
        scores[i] = lp
    if o_bias:
        scores[:, TAG2ID['O']] += o_bias
    dp = np.full((n, NUM_TAGS), -1e9, dtype=np.float64)
    bp = np.zeros((n, NUM_TAGS), dtype=np.int8)
    dp[0] = scores[0]
    dp[0, [TAG2ID[f'I-{t}'] for t in LABELS]] = -1e9
    for i in range(1, n):
        for j in range(NUM_TAGS):
            col = dp[i - 1]
            mask = ALLOWED[:, j]
            if not mask.any():
                continue
            best = -1e9; barg = 0
            for k in np.nonzero(mask)[0]:
                v = col[k]
                if v > best:
                    best = v; barg = k
            dp[i, j] = best + scores[i, j]
            bp[i, j] = barg
    last = int(np.argmax(dp[n - 1]))
    path = [0] * n
    path[n - 1] = last
    for i in range(n - 1, 0, -1):
        path[i - 1] = bp[i, path[i]]
    return path


def tags_to_spans(sent, path):
    spans = []
    i, n = 0, len(sent)
    while i < n:
        t = TAGS[path[i]]
        if t.startswith('B-'):
            typ = t[2:]
            j = i + 1
            while j < n and TAGS[path[j]] == f'I-{typ}':
                j += 1
            spans.append((i, j, typ))
            i = j
        else:
            i += 1
    return spans


def predict_rows(rows, vec, clf, gaz, o_bias=0.0, verbose=False):
    out = {}
    for ri, row in enumerate(rows):
        sent = row['sentence']
        sf = SentFeats(sent, gaz)
        path = viterbi_decode(sf, vec, clf, o_bias=o_bias)
        spans = tags_to_spans(sent, path)
        out[row['id']] = [(sent[s:e], t) for s, e, t in spans]
        if verbose and ri % 1000 == 0:
            print(f"    pred {ri}/{len(rows)}", flush=True)
    return out


def evaluate(rows, vec, clf, gaz, o_bias=0.0, verbose=False):
    gold = {r['id']: parse_entities(r['entities']) for r in rows}
    pred = predict_rows(rows, vec, clf, gaz, o_bias=o_bias, verbose=verbose)
    return score_f1(gold, pred)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--alpha', type=float, default=1e-4)
    ap.add_argument('--dev', type=int, default=2000)
    ap.add_argument('--o_bias', type=float, default=0.0)
    ap.add_argument('--full', action='store_true', help='train on all data and predict test')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    rows = load_csv(f"{TASK_DIR}/train.csv")
    rng = random.Random(args.seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)

    if args.full:
        train_rows = rows
        dev_rows = None
    else:
        dev_idx = set(idx[:args.dev])
        train_rows = [rows[i] for i in idx[args.dev:]]
        dev_rows = [rows[i] for i in idx[:args.dev]]

    # gazetteer from train only
    gmap = {}
    cnt = defaultdict(Counter)
    for row in train_rows:
        for e, t in parse_entities(row['entities']):
            cnt[e][t] += 1
    for e, c in cnt.items():
        if len(e) >= 2:
            gmap[e] = c.most_common(1)[0][0]
    gaz = {'map': gmap, '__maxlen__': max((len(e) for e in gmap), default=2)}
    print(f"gazetteer size: {len(gmap)}, maxlen {gaz['__maxlen__']}", flush=True)

    t0 = time.time()
    train_data = build_train_data(train_rows, gaz)
    print(f"train data built: {len(train_data)} sents, {time.time()-t0:.1f}s", flush=True)

    vec, clf = train_memm(train_data, n_epochs=args.epochs, alpha=args.alpha, seed=args.seed)

    if dev_rows is not None:
        t0 = time.time()
        gold = {rr['id']: parse_entities(rr['entities']) for rr in dev_rows}
        pred = predict_rows(dev_rows, vec, clf, gaz, o_bias=args.o_bias, verbose=True)
        p, r, f1 = score_f1(gold, pred)
        print(f"DEV entity micro-F1: {f1:.4f} (P {p:.4f} R {r:.4f})  [{time.time()-t0:.1f}s]", flush=True)
        from collections import defaultdict as dd
        pert = dd(lambda: [0, 0, 0])
        for k in gold:
            gc = Counter(gold[k]); pc = Counter(pred.get(k, []))
            for key in set(gc) | set(pc):
                t = key[1]
                pert[t][0] += min(gc[key], pc[key])
                pert[t][1] += max(0, pc[key] - gc[key])
                pert[t][2] += max(0, gc[key] - pc[key])
        for t in LABELS:
            a, b, c = pert[t]
            pp = a / (a + b) if a + b else 0; rr_ = a / (a + c) if a + c else 0
            ff = 2 * pp * rr_ / (pp + rr_ + 1e-9)
            print(f"  {t}: P {pp:.3f} R {rr_:.3f} F1 {ff:.3f} (tp{a} fp{b} fn{c})")
    else:
        # predict test
        test_rows = load_csv(f"{TASK_DIR}/test.csv")
        pred = predict_rows(test_rows, vec, clf, gaz, o_bias=args.o_bias, verbose=True)
        os.makedirs(f"{TASK_DIR}/outputs", exist_ok=True)
        out_path = f"{TASK_DIR}/outputs/submission.csv"
        import csv as _csv
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            w = _csv.writer(f)
            w.writerow(['id', 'entities'])
            for row in test_rows:
                w.writerow([row['id'], entities_to_str(pred[row['id']])])
        print(f"wrote {out_path} ({len(test_rows)} rows)")
