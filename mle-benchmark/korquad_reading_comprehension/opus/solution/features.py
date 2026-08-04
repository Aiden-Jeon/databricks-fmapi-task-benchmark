"""Candidate span generation + feature extraction for KorQuAD char-F1 task.

No pretrained models / external data: everything is derived from train.csv.
Approach: extractive QA as candidate-span ranking with gradient boosted trees.
"""
import re
import collections
import numpy as np

# ---------------------------------------------------------------- tokenisation
OPEN_P = set(list("《〈'\"‘“<(「『[≪«＜`（"))
CLOSE_P = set(list("》〉'\"’”>)」』]≫»＞）"))
PUNCT_STRIP = set(list(",.;:!?·…-–—～~*")) | CLOSE_P

PARTICLES = ['이라는', '으로서', '으로써', '에서는', '에게서', '이라고', '이었다', '였다', '이며',
             '에서', '에게', '에는', '으로', '부터', '까지', '라는', '라고', '와의', '과의', '이나',
             '이다', '한다', '에도', '에만', '만을', '로서', '로써', '처럼', '보다', '조차', '마저',
             '밖에', '뿐만', '였고', '이고', '한', '인', '의', '에', '을', '를', '는', '이', '가',
             '은', '로', '와', '과', '도', '만', '나', '라', '며', '고', '서', '께서', '및', '등',
             '등을', '등의', '등이', '등에', '씨', '님', '들', '들의', '들은', '들이', '들을',
             '대로', '치고', '이랑', '랑', '했다', '하다', '하는', '했고', '되었다', '된', '되어',
             '했으며', '이었고']
PARTICLES = sorted(set(PARTICLES), key=len, reverse=True)

# suffix vocab used as a categorical feature
STRIP_VOCAB = ['', '의', '에', '을', '를', '는', '이', '가', '은', '에서', '로', '으로', '와', '과',
               '에는', '에게', '이다.', '부터', '까지', '라는', '도', '이라는', ',', '.', '》', "'",
               '"', '였다.', '이었다.', '이라고', '라고', '인', '한', '에서는']
STRIP_ID = {s: i + 1 for i, s in enumerate(STRIP_VOCAB)}

SENT_END = re.compile(r'(?<=[.!?])\s+|\n+')
WS = re.compile(r'\S+')
HANGUL = re.compile(r'[가-힣]')
DIGIT = re.compile(r'[0-9]')
LATIN = re.compile(r'[A-Za-z]')

SUSPECT_END = ('다', '며', '고', '서', '까', '네', '자', '라', '던', '지', '요', '죠', '까지')

Q_STOP = set(['무엇인가', '무엇인가?', '누구인가', '누구인가?', '무엇은', '것은', '무엇', '누구', '누가',
              '어디', '언제', '얼마', '어떤', '어느', '무슨', '왜', '어떻게', '이름은', '이름은?',
              '이름', '몇', '무엇을', '무엇이', '누구를', '누구의', '어디에', '어디서', '어디인가',
              '있는', '대한', '대해', '된', '한', '및', '가장', '어떠한', '해', '수', '것', '무엇이라고',
              '누구인가요', '무엇인가요', '이라고'])

MAX_EOJ = 4


def split_sentences(ctx):
    spans, prev = [], 0
    for m in SENT_END.finditer(ctx):
        spans.append((prev, m.start()))
        prev = m.end()
    if prev < len(ctx):
        spans.append((prev, len(ctx)))
    out = [(s, e) for s, e in spans if e > s]
    return out if out else [(0, len(ctx))]


def eojeol_spans(ctx):
    return [(m.start(), m.end()) for m in WS.finditer(ctx)]


def _strip_suffixes(tok, max_iter=3):
    """iteratively remove trailing punctuation / particles, yielding prefixes"""
    outs = []
    cur = tok
    for _ in range(max_iter):
        nxt = None
        if cur and cur[-1] in PUNCT_STRIP:
            nxt = cur[:-1]
        else:
            for p in PARTICLES:
                if len(cur) > len(p) and cur.endswith(p):
                    nxt = cur[:-len(p)]
                    break
        if not nxt:
            break
        cur = nxt
        outs.append(cur)
    return outs


def end_variants(ctx, e_s, e_e, maxv=4):
    outs = [e_e]
    tok = ctx[e_s:e_e]
    for pref in _strip_suffixes(tok):
        outs.append(e_s + len(pref))
        if len(outs) >= maxv:
            break
    return outs


def start_variants(ctx, e_s, e_e, maxv=3):
    outs = [e_s]
    for i in range(e_s, e_e - 1):
        if ctx[i] in OPEN_P:
            outs.append(i + 1)
            if len(outs) >= maxv:
                break
    return outs


def question_tokens(q):
    """content tokens of the question (raw eojeol + stemmed prefixes)"""
    toks = []
    for m in WS.finditer(q):
        raw = m.group(0).strip('?!.,')
        if not raw:
            continue
        variants = [raw] + _strip_suffixes(raw)
        best = None
        for v in variants:
            if v in Q_STOP:
                continue
            if len(v) < 2:
                continue
            if not (HANGUL.search(v) or LATIN.search(v) or DIGIT.search(v)):
                continue
            best = v          # keep the shortest (most stemmed) acceptable form
        if best:
            toks.append(best)
    # dedupe, keep order
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# ------------------------------------------------------------------ q type
def q_type(q):
    if '누구' in q or '누가' in q:
        return 1
    if '언제' in q:
        return 2
    if '어디' in q:
        return 3
    if '몇 년' in q or '몇년' in q or '어느 해' in q:
        return 4
    if '몇' in q or '얼마' in q or '수는' in q:
        return 5
    if '이름' in q:
        return 6
    if '왜' in q or '이유' in q:
        return 7
    if '어떻게' in q or '방법' in q:
        return 8
    if '무슨' in q or '어떤' in q or '어느' in q:
        return 9
    if '무엇' in q or '것은' in q:
        return 10
    return 0


N_STATIC = 24


class ContextInfo:
    """Pre-computed, question independent information for one context."""
    __slots__ = ('ctx', 'L', 'sents', 'sent_id', 'ej', 'starts', 'ends', 'static',
                 'sent_of_cand', 'ej_start_set', 'ej_end_set', 'ej_tok_id', 'n')

    def __init__(self, ctx, df, ndoc, max_eoj=MAX_EOJ):
        self.ctx = ctx
        L = len(ctx)
        self.L = L
        self.sents = split_sentences(ctx)
        sid = np.zeros(L + 1, dtype=np.int16)
        for k, (s, e) in enumerate(self.sents):
            sid[s:e] = k
        for s, e in self.sents:
            pass
        sid[L] = len(self.sents) - 1
        self.sent_id = sid
        ej = eojeol_spans(ctx)
        self.ej = ej
        ej_starts = np.array([a for a, b in ej], dtype=np.int32)
        ej_ends = np.array([b for a, b in ej], dtype=np.int32)
        self.ej_start_set = set(ej_starts.tolist())
        self.ej_end_set = set(ej_ends.tolist())

        # word idf per eojeol (use stemmed form)
        ej_idf = np.empty(len(ej), dtype=np.float32)
        maxidf = np.log((ndoc + 1.0) / 1.0)
        for i, (a, b) in enumerate(ej):
            tok = ctx[a:b]
            variants = [tok] + _strip_suffixes(tok)
            v = variants[-1] if len(variants[-1]) >= 2 else tok
            ej_idf[i] = df.get(v, maxidf)

        cands = set()
        n_ej = len(ej)
        for i in range(n_ej):
            for st in start_variants(ctx, *ej[i]):
                for j in range(i, min(n_ej, i + max_eoj)):
                    for en in end_variants(ctx, *ej[j]):
                        if en > st:
                            cands.add((st, en))
        cands = sorted(cands)
        n = len(cands)
        self.n = n
        starts = np.array([c[0] for c in cands], dtype=np.int32)
        ends = np.array([c[1] for c in cands], dtype=np.int32)
        self.starts, self.ends = starts, ends

        F = np.zeros((n, N_STATIC), dtype=np.float32)
        text_count = collections.Counter()
        texts = [ctx[s:e] for s, e in cands]
        for t in texts:
            text_count[t] += 1
        i_ej_start = np.searchsorted(ej_starts, starts, side='right') - 1
        i_ej_end = np.searchsorted(ej_ends, ends, side='left')
        i_ej_end = np.clip(i_ej_end, 0, n_ej - 1)
        sof = sid[starts]
        self.sent_of_cand = sof.astype(np.int32)
        sent_len = np.array([e - s for s, e in self.sents], dtype=np.float32)
        sent_st = np.array([s for s, e in self.sents], dtype=np.float32)

        for k, (s, e) in enumerate(cands):
            t = texts[k]
            tl = e - s
            nsp = t.count(' ')
            ie, je = i_ej_start[k], i_ej_end[k]
            eoj_end = ej[je][1]
            stripped = ctx[e:eoj_end]
            nxt = ctx[e] if e < L else ' '
            prv = ctx[s - 1] if s > 0 else ' '
            F[k, 0] = tl
            F[k, 1] = nsp + 1
            F[k, 2] = s / max(L, 1)
            F[k, 3] = len(self.sents)
            F[k, 4] = (s - sent_st[sof[k]]) / max(sent_len[sof[k]], 1)
            F[k, 5] = 1.0 if e in self.ej_end_set else 0.0
            F[k, 6] = 1.0 if s in self.ej_start_set else 0.0
            F[k, 7] = len(stripped)
            F[k, 8] = 1.0 if DIGIT.search(t) else 0.0
            F[k, 9] = sum(ch.isdigit() for ch in t) / tl
            F[k, 10] = 1.0 if LATIN.search(t) else 0.0
            F[k, 11] = 1.0 if HANGUL.search(t) else 0.0
            F[k, 12] = 1.0 if (t.isdigit() and 3 <= tl <= 4) else 0.0
            F[k, 13] = 1.0 if (prv in OPEN_P and nxt in CLOSE_P) else 0.0
            F[k, 14] = (0 if nxt.isspace() else (1 if HANGUL.match(nxt) else (2 if nxt.isdigit() else 3)))
            F[k, 15] = (0 if prv.isspace() else (1 if HANGUL.match(prv) else (2 if prv.isdigit() else 3)))
            F[k, 16] = 1.0 if t.endswith(SUSPECT_END) else 0.0
            F[k, 17] = text_count[t]
            F[k, 18] = STRIP_ID.get(stripped, 0) if len(stripped) <= 4 else 0
            F[k, 19] = ej_idf[ie:je + 1].min()
            F[k, 20] = ej_idf[ie:je + 1].mean()
            F[k, 21] = je - ie + 1
            F[k, 22] = sent_len[sof[k]]
            F[k, 23] = 1.0 if (prv in OPEN_P or prv == '(') else 0.0
        self.static = F


def build_df(contexts):
    """document frequency (idf) of stemmed eojeols over contexts"""
    dfc = collections.Counter()
    for ctx in contexts:
        toks = set()
        for m in WS.finditer(ctx):
            tok = m.group(0)
            variants = [tok] + _strip_suffixes(tok)
            v = variants[-1] if len(variants[-1]) >= 2 else tok
            toks.add(v)
            toks.add(tok)
        dfc.update(toks)
    n = len(contexts)
    return {k: float(np.log((n + 1.0) / (v + 1.0))) for k, v in dfc.items()}, n


N_QF = 34
N_FEAT = N_STATIC + N_QF


def question_features(ci: ContextInfo, q, dfmap, maxidf, sel=None):
    """features that depend on the question.

    Always computed over *all* candidates of the context (so that the
    group-normalised features are identical at train and test time), then
    optionally sub-selected with `sel`.
    """
    ctx, L = ci.ctx, ci.L
    starts = ci.starts
    ends = ci.ends
    n = len(starts)
    toks = question_tokens(q)
    W = np.zeros(L + 1, dtype=np.float32)
    ntok = len(toks)
    M = np.zeros((max(ntok, 1), L + 1), dtype=np.float32)
    hit_pos = []
    tok_idf = []
    for ti, t in enumerate(toks):
        idf = dfmap.get(t, maxidf)
        tok_idf.append(idf)
        p = ctx.find(t)
        while p >= 0:
            W[p:p + len(t)] += idf
            M[ti, p:p + len(t)] = 1.0
            hit_pos.append((p, p + len(t)))
            p = ctx.find(t, p + 1)
    cw = np.concatenate([[0], np.cumsum(W)]).astype(np.float32)   # len L+2
    cm = np.concatenate([np.zeros((max(ntok, 1), 1), np.float32),
                         np.cumsum(M, axis=1)], axis=1)

    def wsum(lo, hi):
        lo = np.clip(lo, 0, L)
        hi = np.clip(hi, 0, L)
        return cw[hi] - cw[lo]

    def ndistinct(lo, hi):
        if ntok == 0:
            return np.zeros(n, dtype=np.float32)
        lo = np.clip(lo, 0, L)
        hi = np.clip(hi, 0, L)
        d = cm[:, hi] - cm[:, lo]
        return (d > 0).sum(axis=0).astype(np.float32)

    if hit_pos:
        hp = np.array(sorted(set([p for p, _ in hit_pos] + [e - 1 for _, e in hit_pos])), dtype=np.int32)
    else:
        hp = np.array([-10 ** 6], dtype=np.int32)

    il = np.searchsorted(hp, starts, side='left') - 1
    dist_left = np.where(il >= 0, starts - hp[np.clip(il, 0, len(hp) - 1)], 10 ** 4).astype(np.float32)
    ir = np.searchsorted(hp, ends, side='left')
    dist_right = np.where(ir < len(hp), hp[np.clip(ir, 0, len(hp) - 1)] - ends, 10 ** 4).astype(np.float32)
    np.clip(dist_left, 0, 1000, out=dist_left)
    np.clip(dist_right, 0, 1000, out=dist_right)

    tl = (ends - starts).astype(np.float32)
    inside = wsum(starts, ends)
    F = np.zeros((n, N_QF), dtype=np.float32)
    F[:, 0] = inside / tl
    F[:, 1] = inside
    F[:, 2] = wsum(starts - 20, ends + 20) - inside
    F[:, 3] = wsum(starts - 50, ends + 50) - inside
    F[:, 4] = wsum(starts - 120, ends + 120) - inside
    F[:, 5] = dist_left
    F[:, 6] = dist_right
    F[:, 7] = np.minimum(dist_left, dist_right)
    F[:, 8] = ndistinct(starts - 30, ends + 30)
    F[:, 9] = ndistinct(starts - 80, ends + 80)
    F[:, 10] = ndistinct(starts, ends)
    F[:, 11] = ntok

    # sentence level
    ns = len(ci.sents)
    ss = np.array([s for s, e in ci.sents], dtype=np.int32)
    se = np.array([e for s, e in ci.sents], dtype=np.int32)
    sw = (cw[np.clip(se, 0, L)] - cw[np.clip(ss, 0, L)])
    slen = (se - ss).astype(np.float32)
    if ntok:
        sd = ((cm[:, np.clip(se, 0, L)] - cm[:, np.clip(ss, 0, L)]) > 0).sum(axis=0).astype(np.float32)
    else:
        sd = np.zeros(ns, dtype=np.float32)
    score = sw / np.sqrt(slen + 1.0)
    order = np.argsort(-score)
    rank = np.empty(ns, dtype=np.float32)
    rank[order] = np.arange(ns)
    soc = ci.sent_of_cand
    mx = score.max() if ns else 1.0
    F[:, 12] = sw[soc]
    F[:, 13] = (sw / np.maximum(slen, 1))[soc]
    F[:, 14] = sd[soc]
    F[:, 15] = rank[soc]
    F[:, 16] = score[soc] / (mx + 1e-6)
    F[:, 17] = (sd / max(ntok, 1))[soc]

    # char-level overlap with question (answers rarely repeat question words)
    qset = set(q.replace(' ', ''))
    qf = np.fromiter((1.0 if ch in qset else 0.0 for ch in ctx), dtype=np.float32, count=L)
    cq = np.concatenate([[0.0], np.cumsum(qf)]).astype(np.float32)
    F[:, 18] = (cq[ends] - cq[starts]) / tl

    # nearest occurrence of the rarest / last question token
    def token_dist(t):
        if not t:
            return np.full(n, 1000.0, dtype=np.float32)
        pos = []
        p = ctx.find(t)
        while p >= 0:
            pos.append(p)
            p = ctx.find(t, p + 1)
        if not pos:
            return np.full(n, 1000.0, dtype=np.float32)
        pa = np.array(pos, dtype=np.int32)
        mid = ((starts + ends) // 2).astype(np.int32)
        d = np.abs(pa[None, :] - mid[:, None]).min(axis=1).astype(np.float32)
        return np.clip(d, 0, 1000)

    if toks:
        rarest = toks[int(np.argmax(tok_idf))]
        F[:, 19] = token_dist(rarest)
        F[:, 20] = token_dist(toks[-1])
        F[:, 21] = token_dist(toks[0])
        F[:, 22] = max(tok_idf)
    else:
        F[:, 19] = F[:, 20] = F[:, 21] = 1000.0
    F[:, 23] = len(q)
    F[:, 24] = q_type(q)
    F[:, 25] = tl / max(L, 1)

    # ---- group (listwise) normalised features -------------------------------
    tot_idf = float(sum(tok_idf)) if toks else 1.0

    def norm(col, out):
        v = F[:, col]
        mx = v.max()
        F[:, out] = v / (mx + 1e-6)

    norm(2, 26)
    norm(3, 27)
    norm(4, 28)
    w = F[:, 3]
    F[:, 29] = (w - w.mean()) / (w.std() + 1e-6)
    F[:, 30] = w.argsort().argsort() / max(n - 1, 1)
    F[:, 31] = F[:, 3] / (tot_idf + 1e-6)
    F[:, 32] = tot_idf
    d = F[:, 7]
    F[:, 33] = d.argsort().argsort() / max(n - 1, 1)
    if sel is not None:
        F = F[sel]
    return F


def char_f1(pred, gold):
    p = collections.Counter(pred.replace(' ', ''))
    g = collections.Counter(gold.replace(' ', ''))
    inter = sum((p & g).values())
    if inter == 0:
        return 0.0
    pl, gl = sum(p.values()), sum(g.values())
    return 2.0 * inter / (pl + gl)
