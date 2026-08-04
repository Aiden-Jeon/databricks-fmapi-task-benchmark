"""Article-grouped validation harness.

Splits train into train/val by article (context) groups so no context leaks.
Trains a model on the train portion, evaluates char F1 on the val portion.
Used to tune the solution before producing the test submission.
"""
import re
import sys
import time
import collections
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit

from metric import char_f1, evaluate

TASK_DIR = Path(__file__).resolve().parent.parent


def pre(s):
    return s.replace(" ", "")


def sent_split(ctx):
    parts = re.split(r"(?<=[.!?;])\s+|\n+", ctx)
    return [p.strip() for p in parts if p.strip()]


def tokens(s):
    return re.findall(r"[\w가-힣]+", s)


STOP = set("은 는 이 가 의 에 서 의 에 를 를 로 으로 와 과 도 하고 및 및 에서 되다 있다 없다 이다 그 이 이런 저런 무엇 누구 어디 언제 어떤 어떻게 왜 얼마 몇 무슨 인가 인가요 습니까 합니까 했습니까 했는가 인지 것 수 등 및 때문 위해 통해 대한 대해 위해 때문 중 중".split())


def qkws(q):
    return set(t for t in tokens(q) if t not in STOP and len(t) > 1)


def predict_one(q, ctx, qv, cv, vec, sent_cache, top_lens, len_dist, total_len, ans_thr=0.0, top_sents_k=3):
    sim = float(qv.multiply(cv).sum())
    if sim < ans_thr:
        return ""
    sents = sent_split(ctx)
    if not sents:
        return ""
    if sent_cache is not None and ctx in sent_cache:
        s_vecs, s_toks_list = sent_cache[ctx]
    else:
        s_vecs = vec.transform([pre(s) for s in sents])
        s_toks_list = [tokens(s) for s in sents]
        if sent_cache is not None:
            sent_cache[ctx] = (s_vecs, s_toks_list)
    q_vec = vec.transform([pre(q)])
    s_sims = np.asarray(s_vecs.multiply(q_vec).sum(axis=1)).ravel()
    order = np.argsort(-s_sims)
    top_sents = [sents[i] for i in order[:top_sents_k]]
    top_toks = [s_toks_list[i] for i in order[:top_sents_k]]
    qkw = qkws(q)
    best_span = ""
    best_score = -1.0
    for s, stoks in zip(top_sents, top_toks):
        if not s:
            continue
        L = len(s)
        for w in top_lens:
            if w > L:
                continue
            step = 1 if L < 80 else 2
            for st in range(0, L - w + 1, step):
                cand = s[st:st + w]
                cs = cand.strip()
                if not cs:
                    continue
                if cs[0] in ".,?!;:()[]\"'·" or cs[-1] in ".,?!;:()[]\"'·":
                    continue
                ct = tokens(cs)
                if not ct:
                    continue
                cset = set(ct)
                overlap = len(qkw & cset)
                lp = len_dist.get(w, 0) / total_len
                score = (overlap / max(len(ct), 1)) + 0.05 * lp * 50
                if score > best_score:
                    best_score = score
                    best_span = cs
    best_span = best_span.strip()
    best_span = re.sub(r"^[.,?!;:()\[\]\'\"·\s]+", "", best_span)
    best_span = re.sub(r"[.,?!;:()\[\]\'\"·\s]+$", "", best_span)
    return best_span


def run_split(train_df, val_df, vec, ans_thr=0.0):
    # Build answerable threshold from train_df
    n_tr = len(train_df)
    Xc_tr = vec.transform([pre(c) for c in train_df["context"]])
    Xq_tr = vec.transform([pre(q) for q in train_df["question"]])
    tr_sim = np.asarray(Xq_tr.multiply(Xc_tr).sum(axis=1)).ravel()
    ans = (train_df["answer"].str.strip() != "").values
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
    # Use the provided ans_thr or override
    thr = ans_thr if ans_thr > 0 else best_thr
    print("  ans thr=%.3f f1=%.4f" % (thr, best_f1), flush=True)

    nonempty = train_df[train_df["answer"].str.strip() != ""].copy()
    nonempty["alen"] = nonempty["answer"].str.replace(" ", "", regex=False).str.len()
    len_dist = nonempty["alen"].value_counts()
    total_len = len_dist.sum()
    cum = (len_dist / total_len).cumsum()
    top_lens = sorted(set([int(L) for L in cum[cum < 0.99].index] + list(range(1, 16))))

    sent_cache = {}
    preds = []
    Xc_val = vec.transform([pre(c) for c in val_df["context"]])
    Xq_val = vec.transform([pre(q) for q in val_df["question"]])
    t0 = time.time()
    for i in range(len(val_df)):
        q = val_df.iloc[i]["question"]; ctx = val_df.iloc[i]["context"]
        p = predict_one(q, ctx, Xq_val[i], Xc_val[i], vec, sent_cache, top_lens, len_dist, total_len, ans_thr=thr)
        preds.append(p)
        if (i + 1) % 1000 == 0:
            print("  val %d/%d t=%.1fs" % (i + 1, len(val_df), time.time() - t0), flush=True)
    golds = val_df["answer"].tolist()
    f1 = evaluate(preds, golds)
    print("  VAL char F1 = %.4f" % f1, flush=True)
    return f1, preds


def main():
    train = pd.read_csv(TASK_DIR / "train.csv", keep_default_na=False)
    print("train", train.shape, flush=True)
    # group by context
    groups = train["context"].astype("category").cat.codes
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, va_idx = next(gss.split(train, groups=groups))
    train_df = train.iloc[tr_idx].reset_index(drop=True)
    val_df = train.iloc[va_idx].reset_index(drop=True)
    print("train/val", len(train_df), len(val_df), flush=True)

    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 3), preprocessor=pre,
        min_df=2, max_features=30000, norm="l2",
    )
    vec.fit(list(train_df["context"]) + list(train_df["question"]) + list(val_df["context"]) + list(val_df["question"]))
    print("vec fit done", flush=True)

    run_split(train_df, val_df, vec, ans_thr=0.0)


if __name__ == "__main__":
    main()
