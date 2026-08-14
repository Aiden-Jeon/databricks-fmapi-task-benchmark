"""Extractive MRC pipeline (classical ML, no pretrained weights)."""
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import (normalize_text, strip_ws, split_sentences, char_f1,
                    content_words)
from features import qtype, question_keywords

_HANGUL_SYL = re.compile(r"[가-힣]+")
_HAS_DIGIT = re.compile(r"\d")
_DATE_PAT = re.compile(
    r"(\d{4}\s*년|\d+\s*월|\d+\s*일|\d+년대|\d{4}|\d+\s*세기|어제|오늘|내일|작년|올해|금년)")
_NUM_WORD = re.compile(
    r"(한|두|세|네|다섯|여섯|일곱|여덟|아홉|열|스무|서른|백|천|만|억)\s*"
    r"(명|개|번|년|월|일|가지|곳|차례|회|종류|권|마리|대|척|채|배|%|원|달러|점|살)")


def make_char_vectorizer():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                           min_df=1, sublinear_tf=True, norm="l2")


class SentenceRetriever:
    def __init__(self):
        self.vec = make_char_vectorizer()

    def rank(self, context, question, top_k=6):
        sents = split_sentences(context)
        if not sents:
            return []
        texts = [s for _, s in sents]
        q = normalize_text(question)
        try:
            X = self.vec.fit_transform(texts + [q])
            sims = (X[-1] @ X[:-1].T).toarray()[0]
        except Exception:
            sims = np.zeros(len(texts))
        kws = [k for k in question_keywords(question) if len(strip_ws(k)) >= 2]
        for i, t in enumerate(texts):
            ct = strip_ws(t)
            bonus = 0.0
            for kw in kws:
                if kw in ct:
                    bonus += 0.03 * min(len(strip_ws(kw)), 5)
            sims[i] += bonus
        order = np.argsort(-sims)
        out = []
        for i in order[: max(top_k, 1)]:
            out.append((int(i), sents[i][0], sents[i][1], float(sims[i])))
        return out


def enumerate_candidates(sentence, sent_start, max_tok_n=7, max_char_len=70):
    toks = [(m.start(), m.end()) for m in re.finditer(r"\S+", sentence)]
    n = len(toks)
    cands = []
    for i in range(n):
        for j in range(i, min(i + max_tok_n, n)):
            s = toks[i][0]; e = toks[j][1]
            text = sentence[s:e]
            if not strip_ws(text) or len(text) > max_char_len:
                continue
            cands.append((sent_start + s, sent_start + e, text))
    for m in _HANGUL_SYL.finditer(sentence):
        t = m.group(0)
        if 1 <= len(t) <= 9:
            cands.append((sent_start + m.start(), sent_start + m.end(), t))
    return cands


QTYPE_IDX = {t: i for i, t in enumerate(
    ["NUM", "DATE", "PERSON", "PLACE", "ORG", "TITLE", "DEF", "OTHER"])}


def _num_match(text):
    return bool(_HAS_DIGIT.search(text)) or bool(_NUM_WORD.search(text))


def token_type_match(text, qt):
    t = text.strip()
    s = 0.0
    has_num = _num_match(t)
    if qt == "NUM":
        s += 1.0 if has_num else -0.6
    if qt == "DATE":
        if _DATE_PAT.search(t):
            s += 1.0
        elif has_num:
            s += 0.2
        else:
            s -= 0.6
    if qt == "PERSON":
        if _HANGUL_SYL.fullmatch(strip_ws(t)) and 2 <= len(strip_ws(t)) <= 4 \
                and not has_num:
            s += 0.6
        if has_num:
            s -= 0.6
    return s


def candidate_features(cand_text, sentence, question, sent_sim, sent_rank,
                       cand_start_in_sent, qt):
    cws = strip_ws(cand_text)
    q_kws = question_keywords(question)
    s_ws = strip_ws(sentence)
    cand_toks = set(content_words(cand_text))
    q_toks = set(q_kws)
    q_overlap = len(cand_toks & q_toks) / max(len(cand_toks), 1)
    sent_support = sum(1 for kw in q_kws if kw in s_ws) / max(len(q_kws), 1)
    rel_pos = cand_start_in_sent / max(len(sentence), 1)
    tok_len = len(cand_text.split())
    char_len = len(cws)

    end_of_cand = cand_start_in_sent + len(cand_text)
    tail = sentence[end_of_cand: end_of_cand + 5]
    head = sentence[max(0, cand_start_in_sent - 5):cand_start_in_sent]
    ends_noun = 1.0 if re.match(
        r"^\s*(은|는|이|가|을|를|에|의|로|으로|와|과|도|다|였다|이다|했다|입니다)",
        tail) else 0.0
    # preceding verb like '총/약/무려' for numbers, '성은/이름은' etc.
    pre_num = 1.0 if re.search(r"(총|약|무려|대략)\s*$", head) else 0.0
    pre_cop = 1.0 if re.search(r"(은|는|이|가)\s*$", head) else 0.0

    contains_qword = 1.0 if any(
        w in cand_text for w in ("무엇", "누구", "어디", "언제", "왜", "몇")) else 0.0
    tmatch = token_type_match(cand_text, qt)
    punct = 1.0 if re.search(r"[.!?\"'“”‘’<>《》〈〉]", cand_text) else 0.0
    starts_part = 1.0 if re.match(
        r"^(은|는|이|가|을|를|에|의|로|으로|와|과|도|만|부터|까지|에서)",
        cand_text.strip()) else 0.0
    ends_part = 1.0 if re.search(
        r"(은|는|이|가|을|를|에|의|로|으로|와|과|도|만|부터|까지|에서)$",
        cand_text.strip()) else 0.0

    # length sweet spot: most answers 1-12 chars
    len_score = -abs(char_len - 5.0) / 10.0

    feats = [
        sent_sim,
        1.0 / (1.0 + sent_rank),
        q_overlap,
        sent_support,
        rel_pos,
        float(tok_len),
        float(char_len),
        len_score,
        ends_noun,
        pre_num,
        pre_cop,
        contains_qword,
        tmatch,
        punct,
        starts_part,
        ends_part,
        float(_num_match(cand_text)),
        1.0 if _HANGUL_SYL.fullmatch(cws) else 0.0,
        float(bool(_DATE_PAT.search(cand_text))),
    ]
    qt_vec = [0.0] * len(QTYPE_IDX)
    if qt in QTYPE_IDX:
        qt_vec[QTYPE_IDX[qt]] = 1.0
    feats.extend(qt_vec)
    return feats


class SpanScorer:
    def __init__(self, C=1.0, max_iter=400):
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")

    def fit(self, X, y, sample_weight=None):
        Xs = self.scaler.fit_transform(np.asarray(X))
        self.clf.fit(Xs, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        Xs = self.scaler.transform(np.asarray(X))
        return self.clf.predict_proba(Xs)[:, 1]


def build_training_samples(df, top_k=6, neg_cap=40, pos_thresh=0.55, seed=0):
    from collections import defaultdict
    retriever = SentenceRetriever()
    rng = np.random.RandomState(seed)
    X, y, w, groups = [], [], [], []
    for gi, (_, row) in enumerate(df.iterrows()):
        ctx = row["context"]; q = row["question"]
        gold = row.get("answer", "") or ""
        qt = qtype(q)
        ranked = retriever.rank(ctx, q, top_k=top_k)
        cands = []
        for rank, (si, sstart, sent, sim) in enumerate(ranked):
            for (gs, ge, text) in enumerate_candidates(sent, sstart):
                feats = candidate_features(text, sent, q, sim, rank,
                                           gs - sstart, qt)
                f1 = char_f1(text, gold) if gold else 0.0
                cands.append((feats, f1, text))
        if not cands:
            continue
        f1s = np.array([c[1] for c in cands])
        if gold:
            pos_idx = [i for i, c in enumerate(cands) if c[1] >= pos_thresh]
            if not pos_idx:
                pos_idx = [int(np.argmax(f1s))]
            neg_idx = [i for i, c in enumerate(cands) if c[1] < 0.2]
            rng.shuffle(neg_idx)
            neg_idx = neg_idx[:neg_cap]
            for i in pos_idx:
                X.append(cands[i][0]); y.append(1)
                w.append(0.5 + cands[i][1]); groups.append(gi)
            for i in neg_idx:
                X.append(cands[i][0]); y.append(0); w.append(1.0); groups.append(gi)
        else:
            # unanswerable: sample some negatives, no positives
            take = list(range(len(cands)))[:neg_cap]
            for i in take:
                X.append(cands[i][0]); y.append(0); w.append(0.8); groups.append(gi)
    return np.array(X), np.array(y), np.array(w), np.array(groups)


def predict_row(scorer, retriever, context, question, top_k=6):
    qt = qtype(question)
    ranked = retriever.rank(context, question, top_k=top_k)
    cands = []
    for rank, (si, sstart, sent, sim) in enumerate(ranked):
        for (gs, ge, text) in enumerate_candidates(sent, sstart):
            feats = candidate_features(text, sent, question, sim, rank,
                                       gs - sstart, qt)
            cands.append((feats, text))
    if not cands:
        return "", 0.0, []
    X = np.array([c[0] for c in cands])
    probs = scorer.predict_proba(X)
    best_i = int(np.argmax(probs))
    return cands[best_i][1], float(probs[best_i]), list(
        zip([c[1] for c in cands], probs))
