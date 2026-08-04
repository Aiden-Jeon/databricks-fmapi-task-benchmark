"""Character-level Korean NER: averaged structured perceptron with Viterbi decoding.

Pure numpy/pandas implementation (no external pretrained resources).
Trained only on train.csv.
"""
import numpy as np
import unicodedata
from collections import Counter, defaultdict

TYPES = ['PS', 'LC', 'OG', 'DT', 'TI', 'QT']
# label ids: 0 = O, then B-T, I-T for each type
LABELS = ['O'] + [p + '-' + t for t in TYPES for p in ('B', 'I')]
L2I = {l: i for i, l in enumerate(LABELS)}
NL = len(LABELS)

PAD = '\u0000'


# ---------------------------------------------------------------- parsing utils
def parse_ents(s):
    """'text:TYPE|text:TYPE' -> [(text, TYPE), ...]"""
    if not isinstance(s, str) or not s:
        return []
    out = []
    for part in s.split('|'):
        if not part:
            continue
        i = part.rfind(':')
        if i <= 0:
            continue
        out.append((part[:i], part[i + 1:]))
    return out


def ents_to_spans(sent, ents):
    """Align entity list (in order of appearance) to character spans."""
    spans = []
    pos = 0
    for text, ty in ents:
        if ty not in TYPES:
            continue
        i = sent.find(text, pos)
        if i < 0:
            i = sent.find(text)
            if i < 0:
                continue
        spans.append((i, i + len(text), ty))
        pos = i + len(text)
    return spans


def spans_to_labels(n, spans):
    y = np.zeros(n, dtype=np.int8)
    for a, b, ty in spans:
        if a >= n:
            continue
        y[a] = L2I['B-' + ty]
        for k in range(a + 1, min(b, n)):
            y[k] = L2I['I-' + ty]
    return y


def labels_to_spans(y):
    spans = []
    i = 0
    n = len(y)
    while i < n:
        lab = LABELS[y[i]]
        if lab[0] == 'B':
            ty = lab[2:]
            j = i + 1
            while j < n and LABELS[y[j]] == 'I-' + ty:
                j += 1
            spans.append((i, j, ty))
            i = j
        elif lab[0] == 'I':
            # stray I- treated as start of entity
            ty = lab[2:]
            j = i + 1
            while j < n and LABELS[y[j]] == 'I-' + ty:
                j += 1
            spans.append((i, j, ty))
            i = j
        else:
            i += 1
    return spans


def spans_to_str(sent, spans):
    return '|'.join('%s:%s' % (sent[a:b], ty) for a, b, ty in spans)


# ---------------------------------------------------------------- char type
_CT_CACHE = {}


def ctype(ch):
    v = _CT_CACHE.get(ch)
    if v is not None:
        return v
    o = ord(ch)
    if ch == PAD:
        v = 'P'
    elif ch == ' ':
        v = 'S'
    elif 0xAC00 <= o <= 0xD7A3:
        v = 'H'          # hangul syllable
    elif 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F:
        v = 'h'          # hangul jamo
    elif ch.isdigit():
        v = 'D'
    elif 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF:
        v = 'C'          # cjk ideograph
    elif 0x3040 <= o <= 0x30FF:
        v = 'J'          # kana
    elif 'a' <= ch <= 'z':
        v = 'l'
    elif 'A' <= ch <= 'Z':
        v = 'u'
    elif ch in '.,!?;:\'"()[]{}<>~/\\-_=+*&^%$#@`|':
        v = 'p'
    else:
        cat = unicodedata.category(ch)
        v = 'p' if cat.startswith('P') or cat.startswith('S') else 'o'
    _CT_CACHE[ch] = v
    return v


_CHO = 'ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ'
_JUNG = 'ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ'
_JONG = '_ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ'
_JAMO_CACHE = {}


def jamo(ch):
    """(onset, vowel, coda) for a hangul syllable, else ('', '', '')"""
    v = _JAMO_CACHE.get(ch)
    if v is not None:
        return v
    o = ord(ch)
    if 0xAC00 <= o <= 0xD7A3:
        k = o - 0xAC00
        v = (_CHO[k // 588], _JUNG[(k % 588) // 28], _JONG[k % 28])
    else:
        v = ('', '', '')
    _JAMO_CACHE[ch] = v
    return v


_SHAPE_CACHE = {}


def shape(tok):
    v = _SHAPE_CACHE.get(tok)
    if v is not None:
        return v
    out = []
    for ch in tok:
        c = ctype(ch)
        if not out or out[-1] != c:
            out.append(c)
    v = ''.join(out)
    _SHAPE_CACHE[tok] = v
    return v


_NORM_CACHE = {}


def normtok(tok):
    v = _NORM_CACHE.get(tok)
    if v is not None:
        return v
    v = ''.join('0' if c.isdigit() else c for c in tok)
    _NORM_CACHE[tok] = v
    return v


def lenbucket(n):
    if n <= 3:
        return str(n)
    if n <= 5:
        return '4'
    if n <= 8:
        return '6'
    return '9'


# ---------------------------------------------------------------- gazetteer
class Gazetteer:
    def __init__(self, max_len=14, min_count=1):
        self.d = {}
        self.max_len = max_len
        self.min_count = min_count

    def fit(self, sentences, spans_list):
        cnt = defaultdict(Counter)
        text_total = Counter()
        for sent, spans in zip(sentences, spans_list):
            for a, b, ty in spans:
                t = sent[a:b]
                if 1 <= len(t) <= self.max_len:
                    cnt[t][ty] += 1
        # how often does the string occur in the corpus at all (rough)
        for t in cnt:
            text_total[t] = sum(cnt[t].values())
        self.d = {}
        for t, c in cnt.items():
            if text_total[t] < self.min_count:
                continue
            best = c.most_common()
            tot = sum(c.values())
            self.d[t] = [(ty, n, n / tot) for ty, n in best]
        self.lens = sorted({len(t) for t in self.d}, reverse=True)
        return self

    def match(self, sent):
        """returns per position: dict type -> (role, matchlen, purity, cnt)"""
        n = len(sent)
        res = [None] * n
        d = self.d
        lens = self.lens
        for i in range(n):
            hit = None
            for l in lens:
                if i + l > n:
                    continue
                sub = sent[i:i + l]
                e = d.get(sub)
                if e is not None:
                    hit = (l, e)
                    break
            if hit is None:
                continue
            l, e = hit
            for k in range(i, i + l):
                if res[k] is None:
                    res[k] = []
            for ty, c, purity in e[:2]:
                res[i].append((ty, 'B', l, c, purity, i))
                for k in range(i + 1, i + l):
                    res[k].append((ty, 'I', l, c, purity, i))
        return res


# ---------------------------------------------------------------- features
def token_info(sent):
    """for each char: (token, index_in_token, token_len, prev_token, next_token)"""
    n = len(sent)
    toks = []
    starts = []
    i = 0
    while i < n:
        if sent[i] == ' ':
            i += 1
            continue
        j = i
        while j < n and sent[j] != ' ':
            j += 1
        toks.append(sent[i:j])
        starts.append(i)
        i = j
    tok_of = [-1] * n
    for ti, (st, tk) in enumerate(zip(starts, toks)):
        for k in range(st, st + len(tk)):
            tok_of[k] = ti
    return toks, starts, tok_of


def sent_features(sent, gaz_match=None):
    """Return list (len = len(sent)) of lists of feature strings."""
    n = len(sent)
    s = PAD * 3 + sent + PAD * 3
    toks, starts, tok_of = token_info(sent)
    nt = len(toks)
    feats = []
    for i in range(n):
        p = i + 3
        c_3, c_2, c_1, c0, c1, c2, c3 = (s[p - 3], s[p - 2], s[p - 1], s[p],
                                         s[p + 1], s[p + 2], s[p + 3])
        f = [
            '@',
            'a' + c_2, 'b' + c_1, 'c' + c0, 'd' + c1, 'e' + c2,
            'f' + c_3, 'g' + c3,
            'h' + c_2 + c_1, 'i' + c_1 + c0, 'j' + c0 + c1, 'k' + c1 + c2,
            'l' + c_2 + c_1 + c0, 'm' + c_1 + c0 + c1, 'n' + c0 + c1 + c2,
            'o' + c_1 + c1,
            'q' + ctype(c_1) + ctype(c0) + ctype(c1),
            'r' + ctype(c_2) + ctype(c_1) + ctype(c0) + ctype(c1) + ctype(c2),
            'v' + c_1 + c0 + c1 + c2,
            'w' + c_2 + c_1 + c0 + c1,
        ]
        j0 = jamo(c0)
        j_1 = jamo(c_1)
        j1 = jamo(c1)
        f.append('A' + j0[0])
        f.append('B' + j0[2])
        f.append('C' + j_1[2] + '/' + j0[0])
        f.append('D' + j0[2] + '/' + j1[0])
        f.append('E' + j_1[2] + j0[0] + j0[2])
        ti = tok_of[i]
        if ti >= 0:
            tk = toks[ti]
            off = i - starts[ti]
            tl = len(tk)
            f.append('W' + tk)
            f.append('P' + lenbucket(off + 1) + '/' + lenbucket(tl))
            f.append('Q' + str(min(off, 4)))
            f.append('R' + str(min(tl - off - 1, 4)))
            f.append('S' + tk[:1])
            f.append('T' + tk[-1:])
            f.append('U' + tk[-2:])
            f.append('V' + tk[:2])
            f.append('X' + str(min(tl, 9)))
            f.append('Y' + tk + '/' + str(min(off, 4)))
            f.append('M' + shape(tk))
            ntk = normtok(tk)
            f.append('N' + ntk)
            f.append('F' + ntk[-2:])
            if ti > 0:
                ptk = toks[ti - 1]
                f.append('p' + ptk)
                f.append('s' + ptk[-2:])
                f.append('K' + normtok(ptk))
                f.append('J' + shape(ptk))
            else:
                f.append('p^')
            if ti + 1 < nt:
                f.append('t' + toks[ti + 1])
                f.append('u' + toks[ti + 1][:2])
            else:
                f.append('t$')
            f.append('z' + str(min(ti, 3)) + '/' + str(min(nt - ti - 1, 3)))
        else:
            f.append('W_SP')
            tk, off, tl = '', 0, 0
        if gaz_match is not None:
            m = gaz_match[i]
            if m:
                for ty, role, ml, cn, purity, st in m:
                    pb = '2' if purity > 0.9 else ('1' if purity > 0.6 else '0')
                    cb = '2' if cn >= 10 else ('1' if cn >= 3 else '0')
                    f.append('G' + ty + role)
                    f.append('H' + ty + role + lenbucket(ml))
                    f.append('I' + ty + role + pb + cb)
                    lb = '1' if (st == 0 or sent[st - 1] == ' ') else '0'
                    en = st + ml
                    rb = '1' if (en >= n or sent[en] == ' ') else '0'
                    f.append('L' + ty + role + lb + rb)
                    if ti >= 0:
                        f.append('Z' + ty + role + str(min(off, 3)))
            else:
                f.append('G_')
        feats.append(f)
    return feats


class FeatureMap:
    def __init__(self):
        self.d = {}
        self.frozen = False

    def encode(self, feats):
        d = self.d
        idx = []
        cnts = np.empty(len(feats), dtype=np.int32)
        if not self.frozen:
            for k, fl in enumerate(feats):
                for f in fl:
                    j = d.get(f)
                    if j is None:
                        j = len(d)
                        d[f] = j
                    idx.append(j)
                cnts[k] = len(fl)
        else:
            for k, fl in enumerate(feats):
                c = 0
                for f in fl:
                    j = d.get(f)
                    if j is not None:
                        idx.append(j)
                        c += 1
                cnts[k] = c
        return np.asarray(idx, dtype=np.int32), cnts


# ---------------------------------------------------------------- perceptron
NEG = -1e9


def build_constraint():
    """mask[prev, cur] = 0 allowed, NEG forbidden"""
    m = np.zeros((NL + 1, NL), dtype=np.float64)  # last row = START
    for pi, pl in enumerate(LABELS):
        for ci, cl in enumerate(LABELS):
            if cl[0] == 'I' and not (pl == 'B-' + cl[2:] or pl == 'I-' + cl[2:]):
                m[pi, ci] = NEG
    for ci, cl in enumerate(LABELS):
        if cl[0] == 'I':
            m[NL, ci] = NEG
    return m


CONSTRAINT = build_constraint()


class Perceptron:
    def __init__(self, nfeat):
        self.W = np.zeros((nfeat, NL), dtype=np.float64)
        self.T = np.zeros((NL + 1, NL), dtype=np.float64)  # row NL = start
        self.WA = np.zeros((nfeat, NL), dtype=np.float64)
        self.TA = np.zeros((NL + 1, NL), dtype=np.float64)
        self.Wt = np.zeros((nfeat, NL), dtype=np.int32)
        self.Tt = np.zeros((NL + 1, NL), dtype=np.int64)
        self.t = 0

    def emissions(self, idx, offs):
        return np.add.reduceat(self.W[idx], offs, axis=0)

    def decode(self, idx, offs, n, o_bias=0.0):
        E = np.add.reduceat(self.W[idx], offs, axis=0) if len(idx) else np.zeros((n, NL))
        if o_bias:
            E = E.copy()
            E[:, 0] -= o_bias
        T = self.T + CONSTRAINT
        dp = np.empty((n, NL))
        bp = np.zeros((n, NL), dtype=np.int8)
        dp[0] = E[0] + T[NL]
        for i in range(1, n):
            sc = dp[i - 1][:, None] + T[:NL]
            bp[i] = np.argmax(sc, axis=0)
            dp[i] = sc[bp[i], np.arange(NL)] + E[i]
        y = np.empty(n, dtype=np.int8)
        y[n - 1] = int(np.argmax(dp[n - 1]))
        for i in range(n - 1, 0, -1):
            y[i - 1] = bp[i, y[i]]
        return y

    # ---- lazy averaging helpers
    def _upd(self, rows, lab, val):
        if len(rows) == 0:
            return
        W, WA, Wt = self.W, self.WA, self.Wt
        t = self.t
        WA[rows, lab] += (t - Wt[rows, lab]) * W[rows, lab]
        Wt[rows, lab] = t
        W[rows, lab] += val

    def _updT(self, pi, ci, val):
        self.TA[pi, ci] += (self.t - self.Tt[pi, ci]) * self.T[pi, ci]
        self.Tt[pi, ci] = self.t
        self.T[pi, ci] += val

    def update(self, idx, offs, gold, pred):
        self.t += 1
        n = len(gold)
        wrong = np.nonzero(gold != pred)[0]
        if len(wrong) == 0:
            return 0
        ends = np.append(offs[1:], len(idx))
        # emission updates grouped by label
        gmap = defaultdict(list)
        pmap = defaultdict(list)
        for i in wrong:
            seg = idx[offs[i]:ends[i]]
            gmap[int(gold[i])].append(seg)
            pmap[int(pred[i])].append(seg)
        for lab, segs in gmap.items():
            self._upd(np.concatenate(segs), lab, 1.0)
        for lab, segs in pmap.items():
            self._upd(np.concatenate(segs), lab, -1.0)
        # transition updates
        gt = Counter()
        for i in range(n):
            pg = NL if i == 0 else int(gold[i - 1])
            pp = NL if i == 0 else int(pred[i - 1])
            gt[(pg, int(gold[i]))] += 1
            gt[(pp, int(pred[i]))] -= 1
        for (a, b), v in gt.items():
            if v:
                self._updT(a, b, float(v))
        return len(wrong)

    def averaged(self):
        """Non-destructive averaged weights."""
        t = self.t + 1
        Wf = self.WA + (t - self.Wt) * self.W
        Wf /= t
        Tf = (self.TA + (t - self.Tt) * self.T) / t
        return Wf, Tf

    def use_avg(self):
        self.W_raw, self.T_raw = self.W, self.T
        self.W, self.T = self.averaged()

    def use_raw(self):
        self.W, self.T = self.W_raw, self.T_raw
        del self.W_raw, self.T_raw


class Decoder:
    """Lightweight holder for (possibly ensembled) weights."""

    def __init__(self, W, T):
        self.W = W
        self.T = T

    decode = Perceptron.decode


# ---------------------------------------------------------------- eval
def micro_f1(gold_strs, pred_strs):
    tp = fp = fn = 0
    for g, p in zip(gold_strs, pred_strs):
        gc = Counter(parse_ents(g))
        pc = Counter(parse_ents(p))
        inter = sum((gc & pc).values())
        tp += inter
        fp += sum(pc.values()) - inter
        fn += sum(gc.values()) - inter
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1
