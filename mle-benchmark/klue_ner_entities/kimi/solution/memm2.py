"""Char-level MEMM tagger for KLUE-NER with Viterbi decoding. Optimized version.

Features extracted once and stored as list-of-strings per char; DictVectorizer
applied per epoch; SGDClassifier(log_loss) via partial_fit.
"""
import sys, os, re, time, random, pickle
import numpy as np
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (TASK_DIR, LABELS, load_csv, parse_entities, entities_to_str,
                    score_f1, extract_rule_spans)  # noqa: F811

TAGS = ['O'] + [f'{p}-{t}' for t in LABELS for p in ('B', 'I')]
TAG2ID = {t: i for i, t in enumerate(TAGS)}
NUM_TAGS = len(TAGS)

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
            toks.append((i, j, 2))
            i = j
        elif c == 4:
            j = i
            while j < n and char_class(sent[j]) == 4 and ord(sent[j]) < 128: j += 1
            if j == i: j = i + 1
            toks.append((i, j, 1))
            i = j
        else:
            j = i
            while j < n and char_class(sent[j]) == c: j += 1
            toks.append((i, j, 0))
            i = j
    return toks


def extract_feats(sent, gaz_map=None, gaz_maxlen=20):
    """Return list of feature-lists (one per char, without prev-tag features)."""
    n = len(sent)
    toks = tokenize(sent)
    ntok = len(toks)
    tokid = [0] * n
    toktype = [0] * n
    for ti, (s, e, tt) in enumerate(toks):
        for i in range(s, e):
            tokid[i] = ti
            toktype[i] = tt
    # rule spans
    rule = [None] * n
    for s, e, t in extract_rule_spans(sent):
        for i in range(s, e):
            rule[i] = t
    # gazetteer per-char info
    gaz_t = [None] * n
    gaz_rem = [0] * n
    gaz_first = [False] * n
    if gaz_map:
        for i in range(n):
            for L in range(min(gaz_maxlen, n - i), 1, -1):
                sub = sent[i:i + L]
                if sub in gaz_map:
                    t = gaz_map[sub]
                    for k in range(i, i + L):
                        if gaz_rem[k] < L - (k - i):
                            gaz_t[k] = t
                            gaz_rem[k] = L - (k - i)
                            gaz_first[k] = (k == i)
                    break
    feats = []
    for i in range(n):
        f = ['B']
        ch = sent[i]
        f.append('c=' + ch)
        cc = char_class(ch)
        f.append('cc=%d' % cc)
        if i > 0:
            f.append('c-1=' + sent[i - 1]); f.append('d1=%d' % char_class(sent[i - 1]))
        else:
            f.append('BOS')
        if i < n - 1:
            f.append('c+1=' + sent[i + 1]); f.append('d2=%d' % char_class(sent[i + 1]))
        else:
            f.append('EOS')
        if i > 1: f.append('c-2=' + sent[i - 2])
        if i < n - 2: f.append('c+2=' + sent[i + 2])
        if i > 0: f.append('g=' + sent[i - 1:i + 1])
        if i < n - 1: f.append('h=' + sent[i:i + 2])
        if i > 1: f.append('i=' + sent[i - 2:i])
        if i < n - 2: f.append('j=' + sent[i + 1:i + 3])
        ti = tokid[i]
        ts, te, tt = toks[ti]
        f.append('tt=%d' % tt)
        f.append('tp=%d' % min(i - ts, 7))
        f.append('tl=%d' % min(te - ts, 8))
        f.append('tr=%d' % min(te - i, 8))
        if tt == 0:
            tok = sent[ts:te]
            f.append('tok=' + tok)
            f.append('p1=' + tok[:1]); f.append('s1=' + tok[-1:])
            if len(tok) >= 2:
                f.append('p2=' + tok[:2]); f.append('s2=' + tok[-2:])
            if len(tok) >= 3:
                f.append('p3=' + tok[:3]); f.append('s3=' + tok[-3:])
            if len(tok) >= 4:
                f.append('p4=' + tok[:4]); f.append('s4=' + tok[-4:])
            if i - ts <= 3:
                f.append('pf=' + sent[ts:i + 1])
            if te - i <= 3:
                f.append('sf=' + sent[i:te])
            # immediate left/right context chars of the token
            if ts > 0:
                f.append('L=' + sent[ts - 1])
                if ts > 1:
                    f.append('L2=' + sent[ts - 2:ts])
            else:
                f.append('LB')
            if te < n:
                f.append('N=' + sent[te])
                if te + 1 < n:
                    f.append('N2=' + sent[te:te + 2])
            else:
                f.append('NB')
            if ts > 0 and sent[ts - 1] == ' ':
                f.append('Lsp')
            if te < n and sent[te] == ' ':
                f.append('Rsp')
        if ti > 0:
            ps, pe, pt = toks[ti - 1]
            if pt == 0:
                pk = sent[ps:pe]
                f.append('k1=' + pk[-1:])
                if len(pk) >= 2: f.append('k2=' + pk[-2:])
            else:
                f.append('kt=%d' % pt)
        else:
            f.append('kB')
        if ti < ntok - 1:
            ns_, ne, nt = toks[ti + 1]
            if nt == 0:
                nk = sent[ns_:ne]
                f.append('l1=' + nk[:1])
                if len(nk) >= 2: f.append('l2=' + nk[:2])
            else:
                f.append('lt=%d' % nt)
        else:
            f.append('lE')
        if ti >= 2:
            s2, e2, t2 = toks[ti - 2]
            if t2 == 0:
                f.append('m1=' + sent[e2 - 1:e2])
                if e2 - s2 >= 2: f.append('m2=' + sent[e2 - 2:e2])
        if ti < ntok - 2:
            s2, e2, t2 = toks[ti + 2]
            if t2 == 0:
                f.append('n1=' + sent[s2:s2 + 1])
                if e2 - s2 >= 2: f.append('n2=' + sent[s2:s2 + 2])
        r = rule[i]
        if r: f.append('R=' + r)
        if gaz_t[i]:
            f.append('G=' + gaz_t[i])
            f.append('Gr=%d' % min(gaz_rem[i], 8))
            if gaz_first[i]: f.append('Gf')
            endi = i + gaz_rem[i]
            if endi < n:
                f.append('Ge=' + sent[endi])
            else:
                f.append('GeB')
        feats.append(f)
    return feats


def build_gold_tags(rows):
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
                if tags[idx] == 'O':
                    tags[idx] = f'B-{typ}'
                    for k in range(idx + 1, idx + len(expr)):
                        tags[k] = f'I-{typ}'
                start = idx + 1
        data.append([TAG2ID[t] for t in tags])
    return data


def make_gazetteer(rows, minlen=2):
    cnt = defaultdict(Counter)
    for row in rows:
        for e, t in parse_entities(row['entities']):
            cnt[e][t] += 1
    gmap = {e: c.most_common(1)[0][0] for e, c in cnt.items() if len(e) >= minlen}
    maxlen = max((len(e) for e in gmap), default=2)
    return gmap, maxlen


class MemmModel:
    def __init__(self, alpha=1e-4, eta0=0.5, seed=42):
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.linear_model import SGDClassifier
        self.vec = DictVectorizer()
        self.clf = SGDClassifier(loss='log_loss', penalty='l2', alpha=alpha,
                                 max_iter=1, tol=None, random_state=seed,
                                 learning_rate='invscaling', eta0=eta0, power_t=0.5)
        self.alpha = alpha

    def _prep(self, all_feats, all_tags, batch_chars=200000):
        """Generator of (X, y) batches cycling through sentences."""
        Xb, yb = [], []
        cnt = 0
        for feats, tags in zip(all_feats, all_tags):
            prev = 'O'
            for i, fl in enumerate(feats):
                fl.append('PT=' + prev)
                Xb.append(fl)
                yb.append(tags[i])
                prev = TAGS[tags[i]]
                cnt += 1
            if cnt >= batch_chars:
                yield Xb, yb
                Xb, yb = [], []
                cnt = 0
        if Xb:
            yield Xb, yb

    def fit(self, all_feats, all_tags, epochs=5, seed=42, verbose=True):
        rng = random.Random(seed)
        classes = np.arange(NUM_TAGS)
        fitted = False
        t0 = time.time()
        for ep in range(epochs):
            order = list(range(len(all_feats)))
            rng.shuffle(order)
            feats_sh = [all_feats[i] for i in order]
            tags_sh = [all_tags[i] for i in order]
            for Xb, yb in self._prep(feats_sh, tags_sh):
                dicts = [{k: 1 for k in fl} for fl in Xb]
                if not fitted:
                    X = self.vec.fit_transform(dicts)
                    self.clf.partial_fit(X, yb, classes=classes)
                    fitted = True
                else:
                    X = self.vec.transform(dicts)
                    self.clf.partial_fit(X, yb)
            if verbose:
                print(f"  epoch {ep+1}/{epochs} {time.time()-t0:.1f}s", flush=True)
            # reset PT features (they were appended in-place)
            for fl_list in all_feats:
                for fl in fl_list:
                    fl.pop()  # remove last 'PT=...'
        return self

    def char_scores(self, feats, o_bias=0.0, beam_bias=0.0, i_bias=0.0):
        n = len(feats)
        dicts = []
        for fl in feats:
            d = {k: 1 for k in fl}
            d['PT=O'] = 1
            dicts.append(d)
        X = self.vec.transform(dicts)
        S = self.clf.predict_log_proba(X).astype(np.float64)
        if o_bias:
            S[:, 0] += o_bias
        if i_bias:
            for t in LABELS:
                S[:, TAG2ID[f'I-{t}']] += i_bias
        if beam_bias:
            for t in LABELS:
                S[:, TAG2ID[f'B-{t}']] += beam_bias
        return S


# Beam search over states (tag, etype) with exact in-entity type consistency.
# etype: -1 when current tag is O, else label index of the open entity.
def _build_trans():
    trans = {}
    for t in range(NUM_TAGS):
        for e in range(-1, len(LABELS)):
            name = TAGS[t]
            opts = []
            if name == 'O':
                opts.append((0, -1))
                for li in range(len(LABELS)):
                    opts.append((TAG2ID[f'B-{LABELS[li]}'], li))
            else:
                cur = LABELS.index(name[2:])
                if cur != e:
                    continue
                itag = TAG2ID[f'I-{LABELS[cur]}']
                opts.append((itag, cur))
                opts.append((0, -1))
                for li in range(len(LABELS)):
                    opts.append((TAG2ID[f'B-{LABELS[li]}'], li))
            trans[(t, e)] = opts
    return trans

TRANS = _build_trans()
STATE0 = [(0, -1)] + [(TAG2ID[f'B-{t}'], li) for li, t in enumerate(LABELS)]


def beam_decode(S, beam=12, min_ent_len=2):
    """Beam search. State = (tag, etype, entlen). Entities shorter than
    min_ent_len chars are disallowed (can't close)."""
    n = S.shape[0]
    # state: (tag, etype, entlen) ; entlen counts chars of current open entity
    cur = {}
    for (t, e) in STATE0:
        cur[(t, e, 1 if e >= 0 else 0)] = S[0, t]
    bps = []
    for i in range(1, n):
        nxt = {}
        bp = {}
        si = S[i]
        for (t, e, el), sc in cur.items():
            for (nt, ne) in TRANS[(t, e)]:
                if ne >= 0:
                    nel = el + 1 if (TAGS[nt].startswith('I-')) else 1
                else:
                    if e >= 0 and el < min_ent_len and TAGS[nt] == 'O':
                        continue  # would close a too-short entity
                    nel = 0
                if e >= 0 and el < min_ent_len and ne < 0:
                    continue
                key = (nt, ne, nel)
                v = sc + si[nt]
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
    # final: disallow open entity shorter than min
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
            spans.append((sent[i:j], typ))
            i = j
        else:
            i += 1
    return spans


def predict_sents(sents, model, gaz_map, gaz_maxlen, o_bias=0.0, beam=12, verbose=False):
    out = []
    for ri, sent in enumerate(sents):
        feats = extract_feats(sent, gaz_map, gaz_maxlen)
        S = model.char_scores(feats, o_bias=o_bias)
        path = beam_decode(S, beam=beam)
        out.append(tags_to_spans(sent, path))
        if verbose and ri % 500 == 0:
            print(f"    pred {ri}/{len(sents)}", flush=True)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--alpha', type=float, default=1e-4)
    ap.add_argument('--eta0', type=float, default=0.5)
    ap.add_argument('--dev', type=int, default=2000)
    ap.add_argument('--o_bias', type=float, default=0.0)
    ap.add_argument('--full', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--save', type=str, default='')
    ap.add_argument('--load', type=str, default='')
    ap.add_argument('--out', type=str, default='')
    ap.add_argument('--tag', type=str, default='')
    args = ap.parse_args()

    rows = load_csv(f"{TASK_DIR}/train.csv")
    rng = random.Random(args.seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    if args.full:
        train_rows = rows
        dev_rows = None
    else:
        train_rows = [rows[i] for i in idx[args.dev:]]
        dev_rows = [rows[i] for i in idx[:args.dev]]

    gaz_map, gaz_maxlen = make_gazetteer(train_rows)
    print(f"[{args.tag}] gazetteer {len(gaz_map)} maxlen {gaz_maxlen}", flush=True)

    t0 = time.time()
    train_feats = [extract_feats(r['sentence'], gaz_map, gaz_maxlen) for r in train_rows]
    train_tags = build_gold_tags(train_rows)
    print(f"[{args.tag}] feats built {time.time()-t0:.1f}s", flush=True)

    if args.load:
        with open(args.load, 'rb') as f:
            model = pickle.load(f)
        print(f"[{args.tag}] model loaded from {args.load}", flush=True)
    else:
        model = MemmModel(alpha=args.alpha, eta0=args.eta0, seed=args.seed)
        model.fit(train_feats, train_tags, epochs=args.epochs, seed=args.seed)
    if args.save:
        with open(args.save, 'wb') as f:
            pickle.dump(model, f)
        print(f"[{args.tag}] model saved to {args.save}", flush=True)

    if dev_rows is not None:
        t0 = time.time()
        sents = [r['sentence'] for r in dev_rows]
        preds = predict_sents(sents, model, gaz_map, gaz_maxlen, o_bias=args.o_bias, verbose=True)
        gold = {r['id']: parse_entities(r['entities']) for r in dev_rows}
        pred = {r['id']: preds[i] for i, r in enumerate(dev_rows)}
        p, rc, f1 = score_f1(gold, pred)
        print(f"[{args.tag}] DEV micro-F1 {f1:.4f} (P {p:.4f} R {rc:.4f}) [{time.time()-t0:.1f}s]", flush=True)
        pert = defaultdict(lambda: [0, 0, 0])
        for k in gold:
            gc = Counter(gold[k]); pc = Counter(pred[k])
            for key in set(gc) | set(pc):
                t = key[1]
                pert[t][0] += min(gc[key], pc[key])
                pert[t][1] += max(0, pc[key] - gc[key])
                pert[t][2] += max(0, gc[key] - pc[key])
        for t in LABELS:
            a, b, c = pert[t]
            pp = a / (a + b) if a + b else 0
            rr = a / (a + c) if a + c else 0
            ff = 2 * pp * rr / (pp + rr + 1e-9)
            print(f"  {t}: P {pp:.3f} R {rr:.3f} F1 {ff:.3f}", flush=True)
        return f1

    if args.out:
        test_rows = load_csv(f"{TASK_DIR}/test.csv")
        sents = [r['sentence'] for r in test_rows]
        preds = predict_sents(sents, model, gaz_map, gaz_maxlen, o_bias=args.o_bias, verbose=True)
        import csv as _csv
        with open(args.out, 'w', newline='', encoding='utf-8') as f:
            w = _csv.writer(f)
            w.writerow(['id', 'entities'])
            for i, row in enumerate(test_rows):
                w.writerow([row['id'], entities_to_str(preds[i])])
        print(f"[{args.tag}] wrote {args.out} ({len(test_rows)} rows)", flush=True)


if __name__ == '__main__':
    main()
