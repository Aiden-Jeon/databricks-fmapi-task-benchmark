"""Baseline v2 - fast sklearn-based span extraction for KLUE-MRC.

Approach:
1. TF-IDF (char trigrams) over contexts and questions.
2. For each test question, rank context sentences by char-trigram cosine to the question.
3. Within top sentences, slide candidate spans of learned lengths and score by
   word/char overlap with question keywords. Pick best span.
4. Predict empty if best score below threshold (learned from train answerable/unanswerable).
"""
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

TASK_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = TASK_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def pre(s: str) -> str:
    return s.replace(" ", "")


def sent_split(ctx: str):
    parts = re.split(r"(?<=[.!?;])\s+|\n+", ctx)
    return [p.strip() for p in parts if p.strip()]


def main():
    t0 = time.time()
    train = pd.read_csv(TASK_DIR / "train.csv", keep_default_na=False)
    test = pd.read_csv(TASK_DIR / "test.csv", keep_default_na=False)
    print("read", train.shape, test.shape, "time", time.time() - t0, flush=True)

    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 3), preprocessor=pre,
        min_df=2, max_features=30000, norm="l2",
    )
    alltexts = (
        list(train["context"]) + list(train["question"])
        + list(test["context"]) + list(test["question"])
    )
    X = vec.fit_transform(alltexts)
    n_tr = len(train); n_te = len(test)
    Xc_tr = X[:n_tr]; Xq_tr = X[n_tr:2 * n_tr]
    Xc_te = X[2 * n_tr:2 * n_tr + n_te]; Xq_te = X[2 * n_tr + n_te:]
    print("tfidf", X.shape, "time", time.time() - t0, flush=True)

    # Question-context similarity for unanswerable detection
    tr_sim = np.asarray(Xq_tr.multiply(Xc_tr).sum(axis=1)).ravel()
    ans = (train["answer"].str.strip() != "").values
    # Find best threshold
    best_thr, best_f1 = 0.0, -1.0
    for thr in np.linspace(0, 0.5, 101):
        pred = tr_sim >= thr
        tp = ((pred) & (ans)).sum(); fp = ((pred) & (~ans)).sum(); fn = ((~pred) & (ans)).sum()
        if tp + fp == 0 or tp + fn == 0:
            continue
        p = tp / (tp + fp); r = tp / (tp + fn)
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    print("answerable thr=%.3f f1=%.4f" % (best_thr, best_f1), flush=True)

    # Learn answer length distribution (chars) from train non-empty
    nonempty = train[train["answer"].str.strip() != ""].copy()
    nonempty["alen"] = nonempty["answer"].str.replace(" ", "", regex=False).str.len()
    len_dist = nonempty["alen"].value_counts()
    total_len = len_dist.sum()
    # Candidate window sizes: top lengths covering 99%
    cum = (len_dist / total_len).cumsum()
    top_lens = sorted(set([int(L) for L in cum[cum < 0.99].index] + list(range(1, 16))))
    print("top lens", top_lens, flush=True)

    # Word-level question vocabulary for overlap scoring
    def tokens(s):
        return re.findall(r"[\w가-힣]+", s)

    # Question keyword set: tokens that are not common stopwords
    stopwords = set("은 는 이 가 의 에 서 의 에 를 를 로 으로 와 과 도 하고 및 및 에서 되다 있다 없다 이다 그 이 이런 저런 무엇 누구 어디 언제 어떤 어떻게 왜 얼마 몇 무슨 인가 인가요 습니까 합니까 했습니까 했는가 인지 것 수 등 및 때문 위해 통해 대한 대해 대해 대해 위해 때문 중 중 ".split())
    def qkws(q):
        return set(t for t in tokens(q) if t not in stopwords and len(t) > 1)

    # Precompute question keyword sets for train (for scoring calibration) - not strictly needed
    # For span scoring we use: overlap(question_kw, span_tokens) / max(len(span_tokens),1)
    # plus a small bonus for spans that begin/end at non-punct boundaries.

    preds = []
    t1 = time.time()
    for i, row in enumerate(test.itertuples(index=False)):
        q = row.question
        ctx = row.context
        # overall q-c similarity
        qv = Xq_te[i]
        cv = Xc_te[i]
        sim = float(qv.multiply(cv).sum())
        if sim < best_thr:
            preds.append("")
            continue
        sents = sent_split(ctx)
        if not sents:
            preds.append("")
            continue
        # Rank sentences by char-trigram cosine to question
        s_vecs = vec.transform([pre(s) for s in sents])
        q_vec = vec.transform([pre(q)])
        s_sims = np.asarray(s_vecs.multiply(q_vec).sum(axis=1)).ravel()
        order = np.argsort(-s_sims)
        top_sents = [sents[i] for i in order[:3]]
        qkw = qkws(q)
        best_span = ""
        best_score = -1.0
        for s in top_sents:
            if not s:
                continue
            L = len(s)
            s_toks = tokens(s)
            for w in top_lens:
                if w > L:
                    continue
                step = 1 if L < 80 else 2
                for st in range(0, L - w + 1, step):
                    cand = s[st:st + w]
                    # Skip if includes whitespace inside (we want contiguous phrase)
                    if " " in cand:
                        # allow but penalize
                        pass
                    # trim leading/trailing spaces
                    cs = cand.strip()
                    if not cs:
                        continue
                    # Skip if starts/ends with punctuation
                    if cs[0] in ".,?!;:()[]\"'·" or cs[-1] in ".,?!;:()[]\"'·":
                        continue
                    ct = tokens(cs)
                    if not ct:
                        continue
                    overlap = len(qkw & set(ct))
                    # length prior
                    lp = len_dist.get(w, 0) / total_len
                    score = (overlap / max(len(ct), 1)) + 0.05 * lp * 50
                    if score > best_score:
                        best_score = score
                        best_span = cs
        best_span = best_span.strip()
        best_span = re.sub(r"^[.,?!;:()\[\]\'\"·\s]+", "", best_span)
        best_span = re.sub(r"[.,?!;:()\[\]\'\"·\s]+$", "", best_span)
        preds.append(best_span)
        if (i + 1) % 1000 == 0:
            print("processed %d/%d t=%.1fs" % (i + 1, len(test), time.time() - t1), flush=True)

    out = pd.DataFrame({"id": test["id"], "answer": preds})
    out.to_csv(OUT_DIR / "submission.csv", index=False)
    print("wrote submission rows", len(out), "total time", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
