"""Deterministic extractive MRC: no training, weighted span scoring.

score(span) = w_sim*sentence_sim + w_type*type_match + w_sup*sent_support
              + w_len*len_prior + w_pos*position + morphological bonuses
Then threshold for unanswerable decision.
"""
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from common import normalize_text, strip_ws, split_sentences, char_f1, content_words
from features import qtype, question_keywords

_HANGUL = re.compile(r"[가-힣]+")
_DIGIT = re.compile(r"\d")
_NUMW = re.compile(r"(한|두|세|네|다섯|여섯|일곱|여덟|아홉|열|스무|서른|백|천|만|억)"
                   r"\s*(명|개|번|년|월|일|가지|곳|차례|회|종류|권|마리|대|척|채|배|%|원|달러|점|살)")
_DATE = re.compile(r"(\d{4}\s*년|\d+\s*월|\d+\s*일|\d+년대|\d{4}|\d+\s*세기|어제|오늘|내일|작년|올해)")
_PART_END = ("은", "는", "이", "가", "을", "를", "에", "의", "로", "으로", "와",
             "과", "도", "만", "에서", "부터", "까지")


def has_num(t):
    return bool(_DIGIT.search(t)) or bool(_NUMW.search(t))


def type_match(text, qt):
    t = strip_ws(text)
    if not t:
        return -1.0
    hn = has_num(t)
    if qt == "NUM":
        return 1.0 if hn else -0.8
    if qt == "DATE":
        if _DATE.search(text):
            return 1.0
        return 0.2 if hn else -0.8
    if qt == "PERSON":
        if _HANGUL.fullmatch(t) and 2 <= len(t) <= 4 and not hn:
            return 0.6
        return -0.4 if hn else 0.0
    if qt in ("PLACE", "ORG"):
        return -0.3 if hn else 0.1
    return 0.0


class HeuristicMRC:
    def __init__(self, top_k=5, max_tok=7, max_char=60, w=None):
        self.top_k = top_k
        self.max_tok = max_tok
        self.max_char = max_char
        self.w = w or {}
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                   min_df=1, sublinear_tf=True, norm="l2")

    def _rank_sents(self, context, question):
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
            b = 0.0
            for kw in kws:
                if kw in ct:
                    b += 0.03 * min(len(strip_ws(kw)), 5)
            sims[i] += b
        order = np.argsort(-sims)[: self.top_k]
        return [(int(i), sents[i][0], sents[i][1], float(sims[i])) for i in order]

    def _cands(self, sentence, sstart):
        toks = [(m.start(), m.end()) for m in re.finditer(r"\S+", sentence)]
        n = len(toks)
        out = []
        for i in range(n):
            for j in range(i, min(i + self.max_tok, n)):
                s, e = toks[i][0], toks[j][1]
                text = sentence[s:e]
                if strip_ws(text) and len(text) <= self.max_char:
                    out.append((sstart + s, sstart + e, text, s, e))
        return out

    def predict(self, context, question, return_all=False):
        w = self.w
        qt = qtype(question)
        q_kws = question_keywords(question)
        ranked = self._rank_sents(context, question)
        best = ("", -1e9, 0.0)
        allsc = []
        for rank, (si, sstart, sent, sim) in enumerate(ranked):
            s_ws = strip_ws(sent)
            support = sum(1 for kw in q_kws if kw in s_ws) / max(len(q_kws), 1)
            for (gs, ge, text, ls, le) in self._cands(sent, sstart):
                cws = strip_ws(text)
                if not cws:
                    continue
                tm = type_match(text, qt)
                char_len = len(cws)
                # length prior: favor 1..12 chars, peak ~4
                lenp = -abs(char_len - 4.0) * 0.08
                tok_len = len(text.split())
                # morphological cleanliness
                clean = 0.0
                if text.strip().endswith(_PART_END):
                    clean -= 0.5
                if re.search(r"[.!?\"'“”‘’<>《》〈〉]", text):
                    clean -= 0.6
                if any(x in text for x in ("무엇", "누구", "어디", "언제", "왜", "몇")):
                    clean -= 0.7
                # ends with noun-followed-by-particle in sentence (span is a subject/object)
                tail = sent[le: le + 4]
                if re.match(r"^\s*(은|는|이|가|을|를|에|의|로|으로|다|이다|였다|했다)", tail):
                    clean += 0.15
                score = (w.get("sim", 1.0) * sim
                         + w.get("type", 0.5) * tm
                         + w.get("sup", 0.4) * support
                         + w.get("len", 1.0) * lenp
                         + w.get("clean", 1.0) * clean
                         + w.get("rank", 0.1) * (1.0 / (1 + rank))
                         - w.get("tokpen", 0.03) * tok_len)
                allsc.append((score, text, sim))
                if score > best[1]:
                    best = (text, score, sim)
        if return_all:
            return best[0], best[1], allsc
        return trim_answer(best[0]), best[1]


_TRAIL_JUNK = re.compile(
    r"[\s.,!?\"'“”‘’()<>《》〈〉:;~…\-—]+$")
_TRAIL_PART = re.compile(
    r"(에서|부터|까지|으로|에게|한테|처럼|보다|엔|인|은|는|이|가|을|를|에|의|로|와|과|도|만|랑|나|이나|하고|였다|이다|했다|입니다|였|이었|되었|인|됐|였다는|라는)$")


def trim_answer(text: str) -> str:
    """Trim trailing particles/punctuation/verb-endings to match gold style."""
    t = text.strip()
    prev = None
    while prev != t:
        prev = t
        t = _TRAIL_JUNK.sub("", t)
        t = _TRAIL_PART.sub("", t)
        t = t.strip()
    return t


def evaluate(df, model, threshold, max_n=None):
    f1s = []
    n = 0
    for _, row in df.iterrows():
        if max_n and n >= max_n:
            break
        gold = row.get("answer", "") or ""
        text, score = model.predict(row["context"], row["question"])
        pred = text if score >= threshold else ""
        f1s.append(char_f1(pred, gold))
        n += 1
    return float(np.mean(f1s)), f1s
