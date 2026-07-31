"""Feature-based extractive QA for KLUE-MRC (CPU only, no pretrained models).

Pipeline:
  1. candidate span generation (whitespace token n-grams + particle stripping)
  2. hand-crafted lexical / type / proximity features built on two
     question<->context matching profiles:
        - eojeol (word) level exact/stem matches, IDF weighted
        - character n-gram longest-match profile (robust to Korean inflection)
  3. stage-1 GBDT regressor: predicts char-F1 of a candidate vs the gold answer
  4. stage-2 GBDT models on example-level features: expected F1 of top-1 span and
     P(unanswerable)  ->  decide between predicting the span or an empty string.
"""
import re
import numpy as np
from collections import Counter

# ---------------------------------------------------------------- metric
_PUNC = set(""""'`~!@#$%^&*()-_=+[]{}\\|;:'",.<>/?“”‘’·…—–「」『』《》〈〉【】""")


def norm_chars(s):
    if s is None:
        return []
    return [ch.lower() for ch in str(s) if not (ch.isspace() or ch in _PUNC)]


def char_f1(pred, gold):
    p, g = norm_chars(pred), norm_chars(gold)
    if len(p) == 0 and len(g) == 0:
        return 1.0
    if len(p) == 0 or len(g) == 0:
        return 0.0
    ns = sum((Counter(p) & Counter(g)).values())
    if ns == 0:
        return 0.0
    prec, rec = ns / len(p), ns / len(g)
    return 2 * prec * rec / (prec + rec)


# ---------------------------------------------------------------- korean helpers
PARTICLES = [
    "으로써", "이라고", "이라는", "에게서", "에서는", "으로는", "으로도", "이라도",
    "라고", "라는", "에서", "으로", "에게", "한테", "보다", "처럼", "까지", "부터",
    "이며", "이라", "에는", "에도", "만큼", "마다", "조차", "밖에", "이나", "이란",
    "라며", "께서", "로써", "로는", "라도", "에서도", "에서만",
    "은", "는", "이", "가", "을", "를", "의", "에", "와", "과", "도", "만", "로",
    "며", "라", "나", "야", "여", "고", "서", "등",
]
PARTICLES = sorted(set(PARTICLES), key=lambda x: -len(x))

_PUNC_STRIP = """"'`~!@#$%^&*()_=+[]{}\\|;:'",.<>/?“”‘’·…—–「」『』《》〈〉【】"""

QWORDS = set("""무엇 무슨 어떤 어느 어디 언제 누구 누가 얼마 몇 왜 어떻게 어떠한 어찌
무엇을 무엇인가 어디인가 언제인가 누구인가 몇인가 뭐 뭔가 것은 것이 곳은 곳이 사람은 사람이
있는가 있나 하는가 인가 일까 할까 인지 였나 했나 하였나 무엇이 어디를 어디서 어디에""".split())

STOP = set("""그리고 그러나 하지만 또한 또 이런 저런 그런 매우 아주 가장 더 덜 이것 저것 그것
있다 없다 한다 된다 하는 되는 있는 없는 대한 대해 위해 통해 따라 관한 관해 이후 이전 당시
때문 경우 정도 중에 중인 함께 모두 다시 바로 이미 아직 만약 즉 등의 등을 등이 및 약""".split())


def strip_particle(tok):
    outs = [tok]
    t = tok.rstrip(_PUNC_STRIP)
    if t != tok and t:
        outs.append(t)
    base = t if t else tok
    for p in PARTICLES:
        if base.endswith(p) and len(base) - len(p) >= 1:
            outs.append(base[: -len(p)])
            break
    return outs


def tokenize(text):
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


_SENT_END = re.compile(r"[.!?]\s+|[.!?](?=[가-힣A-Za-z0-9(‘“\"])|\n|다\.|었다|했다\s")


def sentences(text):
    """Return list of (start, end) char spans of sentences."""
    cuts = [m.end() for m in re.finditer(r"[.!?]\s+|[.!?](?=[가-힣A-Za-z0-9(‘“\"])|\n", text)]
    spans = []
    prev = 0
    for c in cuts:
        if c - prev < 10:
            continue
        spans.append((prev, c))
        prev = c
    if prev < len(text):
        spans.append((prev, len(text)))
    if not spans:
        spans = [(0, len(text))]
    return spans


def stem(tok):
    t = tok.strip(_PUNC_STRIP)
    for p in PARTICLES:
        if t.endswith(p) and len(t) - len(p) >= 2:
            t = t[: -len(p)]
            break
    return t


def q_units(question):
    units = []
    for m in re.finditer(r"\S+", question):
        tok = m.group()
        raw = tok.strip(_PUNC_STRIP)
        if not raw:
            continue
        st = stem(tok)
        for u in {raw, st}:
            if len(u) < 2 or u in QWORDS or u in STOP:
                continue
            units.append(u)
    return sorted(set(units), key=lambda x: -len(x))


# ---------------------------------------------------------------- question type
QT_PATTERNS = [
    ("person", r"누구|누가|사람은|사람이|인물|필자|저자|작가는|감독은|대표는|회장은|사장은|장관은|의원은|씨는|이름은"),
    ("date", r"언제|몇 ?년|몇 ?월|며칠|몇 ?일|몇 ?시|연도|날짜|시기|시점|일자|몇 ?분|무슨 ?요일"),
    ("number", r"얼마|몇|수는|수가|개수|비율|금액|가격|인원|규모|매출|비용|퍼센트|%|배로"),
    ("loc", r"어디|장소|지역|나라|도시|국가|위치|출신|곳은|곳이|도착|출발|어느 ?나라"),
    ("org", r"회사|기업|업체|단체|기관|브랜드|학교|대학|팀은|팀이|정당|부처|은행|그룹|협회|조직"),
    ("title", r"제목|이름|명칭|작품|영화|드라마|책|노래|곡은|곡이|앨범|프로그램|용어|단어|법안|법은"),
    ("reason", r"왜|이유|원인|까닭|배경은|목적"),
    ("how", r"어떻게|방법|방식|수단|과정"),
    ("what", r"무엇|무슨|어떤|어느"),
]
QT_RE = [(n, re.compile(p)) for n, p in QT_PATTERNS]
N_QT = len(QT_RE)


def qtype_vec(question):
    return [1.0 if r.search(question) else 0.0 for _, r in QT_RE]


# ---------------------------------------------------------------- candidate types
RE_DIGIT = re.compile(r"[0-9]")
RE_ALLDIGIT = re.compile(r"^[0-9,.%]+$")
RE_LATIN = re.compile(r"[A-Za-z]")
RE_DATE = re.compile(r"[0-9]\s*(년|월|일|시|분|초|세기|년대)")
RE_UNIT = re.compile(r"(명|원|개|%|억|만|천|건|대|권|회|위|세|살|번|km|kg|m|cm|t|톤|퍼센트|점|배|호|가지|kW|㎡|평|리터|달러|엔|위안|유로)$")
RE_PERSON_SUF = re.compile(r"(씨|군|양|옹|교수|의원|장관|대표|사장|회장|감독|선수|기자|작가|박사|총장|위원장|시장|지사|대통령|총리|왕|공주|백작|경)$")
RE_LOC_SUF = re.compile(r"(시|도|군|구|읍|면|동|리|국|주|성|역|산|강|공항|항|반도|대륙|해|섬)$")
RE_ORG_SUF = re.compile(r"(사|회사|그룹|전자|은행|공사|청|부|원|회|단|협회|학교|대학교|대학|대|팀|연구소|센터|공단|위원회|조합|노조|당|재단|본부|처|실|과)$")
RE_QUOTE = re.compile(r"[‘“'\"《〈「『]")
RE_QUOTE_E = re.compile(r"[’”'\"》〉」』]")
RE_VERBEND = re.compile(r"(다|다\.|했다|한다|이다|였다|된다|하는|되는|있는|없는|라고|면서|으로|에서|까지)$")


def _hangul_ratio(s):
    if not s:
        return 0.0
    n = sum(1 for c in s if "\uac00" <= c <= "\ud7a3")
    return n / len(s)


def gen_candidates(ctx, max_ngram=3, max_len=40):
    toks = tokenize(ctx)
    seen = set()
    out = []
    n = len(toks)
    for i in range(n):
        s0, e0 = toks[i]
        lstarts = {s0}
        st = s0
        while st < e0 and ctx[st] in _PUNC_STRIP:
            st += 1
            lstarts.add(st)
        for j in range(i, min(i + max_ngram, n)):
            e_tok = ctx[toks[j][0]:toks[j][1]]
            for variant in strip_particle(e_tok):
                e = toks[j][0] + len(variant)
                for s in lstarts:
                    if e - s <= 0 or e - s > max_len or (s, e) in seen:
                        continue
                    seen.add((s, e))
                    out.append((s, e))
    return out


def build_idf(texts):
    df = Counter()
    for t in texts:
        seen = set()
        for m in re.finditer(r"\S+", t):
            u = stem(m.group())
            if len(u) >= 2:
                seen.add(u)
        df.update(seen)
    return df, len(texts)


class IDF:
    def __init__(self, df, N):
        self.df = df
        self.N = N

    def __call__(self, u):
        return float(np.log((self.N + 1) / (self.df.get(u, 0) + 1)))


def _cum(a):
    return np.concatenate([[0.0], np.cumsum(a, dtype=np.float64)])


def _dist_profile(mask, L):
    if mask.any():
        ar = np.arange(L)
        prev = np.maximum.accumulate(np.where(mask, ar, -10 ** 6))
        nxt = np.minimum.accumulate(np.where(mask, ar, 10 ** 6)[::-1])[::-1]
        return np.minimum(ar - prev, nxt - ar).astype(np.float32)
    return np.full(L, 1e4, dtype=np.float32)


MAXSUB = 9


def match_profile(ctx, q):
    """char n-gram longest-match profile: mlen[p] = longest question substring
    matching context at position p (2..MAXSUB, 0 if none)."""
    L = len(ctx)
    mlen = np.zeros(L, dtype=np.float32)
    qcov = np.zeros(len(q), dtype=np.float32)
    qn = len(q)
    for ln in range(2, MAXSUB + 1):
        if qn < ln:
            break
        subs = {}
        for i in range(qn - ln + 1):
            sub = q[i:i + ln]
            if sub.strip() == "" or "?" in sub:
                continue
            subs.setdefault(sub, []).append(i)
        for sub, qpos in subs.items():
            start = 0
            found = False
            hits = []
            while True:
                p = ctx.find(sub, start)
                if p < 0:
                    break
                found = True
                hits.append(p)
                start = p + 1
            if len(hits) <= 6:
                for p in hits:
                    seg = mlen[p:p + ln]
                    np.maximum(seg, ln, out=seg)
            if found:
                for i in qpos:
                    seg = qcov[i:i + ln]
                    np.maximum(seg, ln, out=seg)
    return mlen, qcov


class SentSim:
    """global char-ngram tf-idf model used for question<->sentence similarity"""

    def __init__(self, vec):
        self.vec = vec

    @staticmethod
    def fit(texts, n=6000, seed=0):
        from sklearn.feature_extraction.text import TfidfVectorizer
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(texts), min(n, len(texts)), replace=False)
        sample = [texts[i] for i in idx]
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=4,
                              max_features=400000, sublinear_tf=True, dtype=np.float32)
        vec.fit(sample)
        return SentSim(vec)

    def sims(self, q, sent_texts):
        M = self.vec.transform([q] + sent_texts)
        qv = M[0]
        S = M[1:]
        return np.asarray((S @ qv.T).todense()).ravel()


def example_features(ctx, q, idf, cands=None, ssim=None):
    L = len(ctx)
    if cands is None:
        cands = gen_candidates(ctx)
    units = q_units(q)
    qt = qtype_vec(q)
    qchars = set(norm_chars(q))

    # ---- eojeol level match weights
    mw = np.zeros(L, dtype=np.float32)
    inq = np.zeros(L, dtype=np.float32)
    tot_w = matched_w = 0.0
    n_matched = 0
    for u in units:
        w = idf(u) * min(len(u), 6) / 6.0
        tot_w += w
        hits = [m.start() for m in re.finditer(re.escape(u), ctx)]
        if hits:
            n_matched += 1
            matched_w += w
            ww = w / (1.0 + 0.3 * (len(hits) - 1))
            for h in hits:
                mw[h:h + len(u)] += ww
                inq[h:h + len(u)] = 1.0
    # ---- char ngram longest match profile
    mlen, qcov = match_profile(ctx, q)
    strong = (mlen >= 4).astype(np.float32)
    mpow = np.clip(mlen - 1.0, 0, None) ** 1.6

    cum_mw, cum_inq = _cum(mw), _cum(inq)
    cum_mp, cum_st = _cum(mpow), _cum(strong)
    d_mw = _dist_profile(mw > 0, L)
    d_st = _dist_profile(mlen >= 4, L)
    d_st5 = _dist_profile(mlen >= 6, L)

    def wsum(cum, a, b):
        a = 0 if a < 0 else (L if a > L else a)
        b = 0 if b < 0 else (L if b > L else b)
        return float(cum[b] - cum[a]) if b > a else 0.0

    sents = sentences(ctx)
    ns = len(sents)
    s_sc = np.empty(ns, dtype=np.float32)
    s_sc2 = np.empty(ns, dtype=np.float32)
    for si, (a, b) in enumerate(sents):
        ln = max(b - a, 1)
        s_sc[si] = wsum(cum_mp, a, b) / np.sqrt(ln)
        s_sc2[si] = wsum(cum_mw, a, b) / np.sqrt(ln)
    if ssim is not None:
        cos = ssim.sims(q, [ctx[a:b] for a, b in sents])
        cos = np.asarray(cos, dtype=np.float32)
        s_sc = s_sc / max(float(s_sc.max()), 1e-6) + 2.0 * cos / max(float(cos.max()), 1e-6)
    else:
        cos = np.zeros(ns, dtype=np.float32)
    order = np.argsort(-s_sc)
    s_rank = np.empty(ns, dtype=np.int32)
    s_rank[order] = np.arange(ns)
    best_s = float(s_sc.max())
    second_s = float(s_sc[order[1]]) if ns > 1 else 0.0
    sent_of = np.zeros(L + 1, dtype=np.int32)
    for si, (a, b) in enumerate(sents):
        sent_of[a:b] = si
    sent_of[L] = ns - 1

    qcov_frac = float((qcov >= 3).mean()) if len(q) else 0.0
    qcov4 = float((qcov >= 5).mean()) if len(q) else 0.0
    cov_frac = n_matched / max(len(units), 1)
    covw_frac = matched_w / max(tot_w, 1e-6)
    ex_base = [cov_frac, covw_frac, float(len(units)), float(len(q)), float(L),
               best_s, second_s, best_s - second_s, float(ns), qcov_frac, qcov4]

    rows = []
    for (s, e) in cands:
        txt = ctx[s:e]
        ln = e - s
        ntok = txt.count(" ") + 1
        si = int(sent_of[s])
        sa, sb = sents[si]
        f = [
            # --- candidate surface / type
            ln, ntok, float(bool(RE_DIGIT.search(txt))), float(bool(RE_ALLDIGIT.match(txt))),
            float(bool(RE_LATIN.search(txt))), float(bool(RE_DATE.search(txt))),
            float(bool(RE_UNIT.search(txt))), float(bool(RE_PERSON_SUF.search(txt))),
            float(bool(RE_LOC_SUF.search(txt))), float(bool(RE_ORG_SUF.search(txt))),
            float(bool(RE_VERBEND.search(txt))),
            float(bool(RE_QUOTE.search(ctx[max(0, s - 1):s]))),
            float(bool(RE_QUOTE_E.search(ctx[e:e + 1]))),
            _hangul_ratio(txt), idf(stem(txt.split()[0])) if txt.split() else 0.0,
            float(ctx.count(txt)), float(txt in q),
            len(set(norm_chars(txt)) & qchars) / max(len(set(norm_chars(txt))), 1),
            # --- how much of the candidate is itself matched by the question (should be low)
            wsum(cum_inq, s, e) / max(ln, 1),
            wsum(cum_mp, s, e) / max(ln, 1),
            wsum(cum_st, s, e) / max(ln, 1),
            float(mlen[s:e].max()) if ln else 0.0,
            # --- eojeol proximity
            wsum(cum_mw, s - 60, s), wsum(cum_mw, e, e + 60),
            wsum(cum_mw, s - 20, s) + wsum(cum_mw, e, e + 20),
            wsum(cum_mw, s - 150, s + 150),
            wsum(cum_mw, s - 60, s) - wsum(cum_mw, e, e + 60),
            # --- char-ngram proximity
            wsum(cum_mp, s - 30, s), wsum(cum_mp, e, e + 30),
            wsum(cum_mp, s - 80, s), wsum(cum_mp, e, e + 80),
            wsum(cum_mp, s - 200, s + 200),
            wsum(cum_st, s - 30, s), wsum(cum_st, e, e + 30),
            wsum(cum_st, s - 100, s + 100),
            (wsum(cum_mp, s - 80, s) - wsum(cum_mp, e, e + 80)),
            # --- distances
            min(float(d_mw[max(s - 1, 0)]), 500.0), min(float(d_mw[min(e, L - 1)]), 500.0),
            min(float(d_st[max(s - 1, 0)]), 500.0), min(float(d_st[min(e, L - 1)]), 500.0),
            min(float(d_st5[max(s - 1, 0)]), 500.0), min(float(d_st5[min(e, L - 1)]), 500.0),
            # --- position
            s / max(L, 1), (s - sa) / max(sb - sa, 1), ln / max(sb - sa, 1),
            float(e >= sb - 3), float(s <= sa + 1),
            # --- sentence
            float(s_sc[si]), float(s_rank[si]), float(s_sc[si]) - best_s,
            float(s_sc2[si]), float(sb - sa),
            wsum(cum_mp, sa, sb) / max(sb - sa, 1),
            float(cos[si]), float(cos.max()), float(cos[si] - cos.max()),
            float(cos[si - 1]) if si > 0 else 0.0,
            float(cos[si + 1]) if si + 1 < ns else 0.0,
            float(s_sc[si - 1]) if si > 0 else 0.0,
            float(s_sc[si + 1]) if si + 1 < ns else 0.0,
        ] + ex_base + qt
        f += [
            qt[0] * float(bool(RE_PERSON_SUF.search(txt))),
            qt[1] * float(bool(RE_DATE.search(txt))),
            qt[2] * float(bool(RE_DIGIT.search(txt))),
            qt[3] * float(bool(RE_LOC_SUF.search(txt))),
            qt[4] * float(bool(RE_ORG_SUF.search(txt))),
            qt[5] * float(bool(RE_QUOTE.search(ctx[max(0, s - 1):s]))),
            qt[1] * float(bool(RE_DIGIT.search(txt))),
            qt[2] * float(bool(RE_UNIT.search(txt))),
            qt[6] * ln, qt[7] * ln,
        ]
        rows.append(f)
    X = np.asarray(rows, dtype=np.float32)
    ex = dict(cos_max=float(cos.max()) if ns else 0.0,
              cos_mean=float(cos.mean()) if ns else 0.0, cov_frac=cov_frac, covw_frac=covw_frac, n_units=len(units), qlen=len(q),
              clen=L, best_sent=best_s, n_sents=ns, qt=qt, qcov=qcov_frac, qcov4=qcov4,
              second_sent=second_s)
    return cands, X, ex
