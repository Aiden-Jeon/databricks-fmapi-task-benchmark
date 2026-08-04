"""Candidate generation, cheap pre-scoring and rich feature extraction."""
import math
import re
import numpy as np
from collections import Counter

from common import (
    tokens_with_pos, span_variants, split_sentences, content_words,
    normalize_answer, JOSA_SET, char_f1,
)

MAXN = 4          # max whitespace tokens in a candidate
WIN = 90          # proximity window (chars)


# ---------------- question typing ----------------
QTYPE_PATTERNS = [
    ("who", r"누구|누가|어느 사람|인물|이름은|성명|저자|감독|작가|대표|사장|회장|주인공"),
    ("when", r"언제|몇 년|몇년|몇 월|며칠|연도|시기|시점|날짜|기간|몇 시|시간은"),
    ("where", r"어디|어느 곳|장소|지역|위치|나라는|도시"),
    ("howmany", r"얼마|몇 |몇개|몇 개|몇명|규모|수는|수량|비율|금액|가격|인원|횟수|퍼센트|개수"),
    ("what", r"무엇|무슨|어떤|어느|뭐"),
    ("why", r"왜|이유|원인|까닭"),
    ("how", r"어떻게|방법|방식"),
]
QTYPES = [t for t, _ in QTYPE_PATTERNS] + ["other"]
_QRE = [(t, re.compile(p)) for t, p in QTYPE_PATTERNS]


def qtype_vec(q):
    v = [0.0] * len(QTYPES)
    hit = False
    for i, (t, r) in enumerate(_QRE):
        if r.search(q):
            v[i] = 1.0
            hit = True
    if not hit:
        v[-1] = 1.0
    return v


# ---------------- answer surface typing ----------------
_RE_DIGIT = re.compile(r"\d")
_RE_ALLDIGIT = re.compile(r"^[\d,.\s]+$")
_RE_DATE = re.compile(r"\d+\s*(년|월|일|년대|세기|분기)")
_RE_TIME = re.compile(r"\d+\s*(시|분|초)")
_RE_MONEY = re.compile(r"(원|달러|엔|위안|유로|억|만|조|천)")
_RE_COUNT = re.compile(r"\d+\s*(명|개|대|건|곳|회|번|차례|가지|종|권|편|%|퍼센트|배|위)")
_RE_LATIN = re.compile(r"[A-Za-z]")
_RE_PERSONSUF = re.compile(r"(씨|장관|대통령|의원|교수|감독|회장|사장|대표|위원|총리|왕|공|박사|선수|작가)$")
_RE_PLACESUF = re.compile(r"(시|도|군|구|읍|면|동|리|국|주|성|현|섬|산|강|호|역|공항|대학교|병원|센터|점)$")
_RE_ORGSUF = re.compile(r"(사|회사|그룹|은행|공사|협회|단체|위원회|정부|부|청|원|당|팀|재단|연구소|대학|학교)$")
_RE_HANGUL = re.compile(r"[\uac00-\ud7a3]")
_RE_HANJA = re.compile(r"[\u4e00-\u9fff]")


def surface_vec(a):
    """Typed features of a candidate answer string."""
    n = len(a)
    return [
        float(bool(_RE_DIGIT.search(a))),
        float(bool(_RE_ALLDIGIT.match(a))),
        float(bool(_RE_DATE.search(a))),
        float(bool(_RE_TIME.search(a))),
        float(bool(_RE_MONEY.search(a))),
        float(bool(_RE_COUNT.search(a))),
        float(bool(_RE_LATIN.search(a))),
        float(bool(_RE_PERSONSUF.search(a))),
        float(bool(_RE_PLACESUF.search(a))),
        float(bool(_RE_ORGSUF.search(a))),
        float(bool(_RE_HANJA.search(a))),
        float(bool(_RE_HANGUL.search(a))),
        float(n),
        math.log1p(n),
        float(len(a.split())),
        float(a[0].isdigit()),
        float(a[-1].isdigit()),
        float(a[-1] in "%"),
        float(2 <= n <= 4),
        float(n > 20),
    ]


class IdfTable:
    def __init__(self, docs):
        df = Counter()
        for d in docs:
            df.update(set(content_words(d)))
        self.df = df
        self.N = len(docs)

    def __call__(self, w):
        return math.log((self.N + 1.0) / (self.df.get(w, 0) + 1.0))


def build_example(ctx, question, idf, topk=70, gold=None, max_cands=None):
    """Generate candidates for one (context, question) pair.

    Returns (list_of_(s,e), feature_matrix(np.float32), extras dict)
    """
    qwords = content_words(question)
    qw_uniq = []
    seen = set()
    for w in qwords:
        if len(w) >= 2 and w not in seen:
            seen.add(w)
            qw_uniq.append(w)
    qw_set = set(qwords)
    qidf = {w: idf(w) for w in qw_uniq}
    tot_qidf = sum(qidf.values()) + 1e-9

    L = len(ctx)
    # ---- relevance mass along context ----
    rel = np.zeros(L + 1, dtype=np.float32)
    covered = np.zeros(L + 1, dtype=np.float32)   # 1 where a q-word matches
    matched_words = set()
    for w in qw_uniq:
        st = 0
        wi = qidf[w]
        found = False
        while True:
            p = ctx.find(w, st)
            if p < 0:
                break
            found = True
            rel[p] += wi
            covered[p:p + len(w)] = 1.0
            st = p + 1
        if found:
            matched_words.add(w)
    crel = np.concatenate([[0.0], np.cumsum(rel)]).astype(np.float32)
    ccov = np.concatenate([[0.0], np.cumsum(covered)]).astype(np.float32)
    matched_idf = sum(qidf[w] for w in matched_words)

    def relsum(a, b):
        a = max(0, min(a, L))
        b = max(0, min(b, L))
        if b <= a:
            return 0.0
        return float(crel[b] - crel[a])

    def covsum(a, b):
        a = max(0, min(a, L))
        b = max(0, min(b, L))
        if b <= a:
            return 0.0
        return float(ccov[b] - ccov[a])

    # ---- sentence info ----
    sents = split_sentences(ctx)
    sent_of = np.zeros(L + 1, dtype=np.int32)
    sent_score = []
    for k, (s, e) in enumerate(sents):
        sent_of[s:e] = k
        sw = set(content_words(ctx[s:e]))
        inter = qw_set & sw
        sc = sum(idf(w) for w in inter) / tot_qidf
        sent_score.append(sc)
    sent_score = np.array(sent_score, dtype=np.float32)
    ss_order = np.argsort(-sent_score)
    sent_rank = np.empty(len(sents), dtype=np.int32)
    sent_rank[ss_order] = np.arange(len(sents))
    max_ss = float(sent_score.max()) if len(sents) else 0.0

    # ---- candidates ----
    toks = tokens_with_pos(ctx)
    ntok = len(toks)
    cand = {}
    for i in range(ntok):
        for n in range(1, MAXN + 1):
            if i + n > ntok:
                break
            s0 = toks[i][1]
            e0 = toks[i + n - 1][2]
            if e0 - s0 > 40:
                break
            for (s, e) in span_variants(ctx, s0, e0):
                if (s, e) not in cand:
                    cand[(s, e)] = (i, n)
    keys = list(cand.keys())

    # ---- cheap prescore ----
    pre = np.empty(len(keys), dtype=np.float32)
    for k, (s, e) in enumerate(keys):
        left = relsum(s - WIN, s)
        right = relsum(e, e + WIN)
        inside = covsum(s, e)
        span_txt = ctx[s:e]
        self_pen = 0.0
        nt = normalize_answer(span_txt)
        if nt and nt in normalize_answer(question):
            self_pen += 2.0
        pre[k] = (left + right) / tot_qidf - 0.6 * inside / max(1, e - s) - self_pen \
            + 0.4 * sent_score[sent_of[s]]
    order = np.argsort(-pre)
    keep = list(order[:topk])
    if gold is not None:
        # ensure positives are present during training
        gset = set()
        st = 0
        while True:
            p = ctx.find(gold, st)
            if p < 0:
                break
            gset.add((p, p + len(gold)))
            st = p + 1
        kept = set(keys[i] for i in keep)
        for k, key in enumerate(keys):
            if key in gset and key not in kept:
                keep.append(k)
    keep_keys = [keys[i] for i in keep]

    # ---- rich features ----
    qv = qtype_vec(question)
    qlen = len(question)
    nq = len(qw_uniq)
    global_feats = [
        max_ss,
        matched_idf / tot_qidf,
        float(nq),
        math.log1p(L),
        qlen / 50.0,
        float(len(sents)),
    ]
    ctx_freq = Counter()
    rows = []
    for i in keep:
        s, e = keys[i]
        txt = ctx[s:e]
        ctx_freq[txt] += 1
    for i in keep:
        s, e = keys[i]
        ti, n = cand[keys[i]]
        txt = ctx[s:e]
        si = int(sent_of[s])
        s_s, s_e = sents[si]
        f = []
        f.append(float(pre[i]))
        # proximity at multiple scales
        for w in (25, 50, 100, 200):
            f.append((relsum(s - w, s) + relsum(e, e + w)) / tot_qidf)
            f.append(relsum(s - w, s) / tot_qidf)
            f.append(relsum(e, e + w) / tot_qidf)
        f.append(covsum(s, e) / max(1, e - s))
        f.append(float(normalize_answer(txt) in normalize_answer(question)))
        ntxt = normalize_answer(txt)
        f.append(float(any(ntxt == w or w == ntxt for w in qw_uniq)))
        f.append(float(sum(1 for w in qw_uniq if w in txt)))
        # sentence features
        f.append(float(sent_score[si]))
        f.append(float(sent_rank[si]))
        f.append(float(sent_rank[si] == 0))
        f.append(float(si) / max(1, len(sents)))
        f.append((s - s_s) / max(1.0, s_e - s_s))
        f.append(relsum(s_s, s_e) / tot_qidf)
        # global position
        f.append(s / max(1.0, L))
        f.append(float(ti) / max(1, ntok))
        f.append(float(n))
        f.append(float(ctx_freq[txt]))
        f.append(float(ctx.count(txt)))
        # boundary: is span exactly token-aligned?
        f.append(float(e == toks[ti + n - 1][2]))
        f.append(float(s == toks[ti][1]))
        # neighbouring token text hints
        prev_tok = toks[ti - 1][0] if ti > 0 else ""
        next_tok = toks[ti + n][0] if ti + n < ntok else ""
        f.append(float(bool(re.search(r"(는|은|이|가|을|를)$", prev_tok))))
        f.append(float(bool(re.search(r"(했다|이다|였다|한다|된다)", next_tok))))
        f += surface_vec(txt)
        f += qv
        f += global_feats
        rows.append(f)
    X = np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 1), dtype=np.float32)
    return keep_keys, X, {"sents": sents, "max_ss": max_ss,
                          "cov": matched_idf / tot_qidf, "nq": nq}


FEATNAMES = None
