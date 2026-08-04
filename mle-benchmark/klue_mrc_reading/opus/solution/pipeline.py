"""Full pipeline: candidate ranker + answerability classifier for KLUE-MRC.

Stage 1  span ranker : HistGradientBoostingRegressor, target = char-F1(candidate, gold)
Stage 2  answerability: HistGradientBoostingClassifier on ranker score statistics
"""
import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import char_f1                      # noqa: E402
from features import IdfTable, build_example, qtype_vec   # noqa: E402

RS = 42
NPROC = int(os.environ.get("NPROC", "4"))
HARD_NEG = 100
RAND_NEG = 110

_G = {}


def _init(idf):
    _G["idf"] = idf


# ------------------------------------------------------------------ training rows
def _rows_one(args):
    ctx, q, gold = args
    idf = _G["idf"]
    keys, X, extra = build_example(ctx, q, idf, topk=HARD_NEG, gold=gold)
    if X.shape[0] == 0:
        return None
    y = np.array([char_f1(ctx[s:e], gold) for s, e in keys], dtype=np.float32)
    # extra uniform-random negatives so the model also sees "easy" candidates
    keys2, X2, _ = build_example(ctx, q, idf, topk=10 ** 9)
    if X2.shape[0] > 0:
        rng = np.random.default_rng(abs(hash(q)) % (2 ** 31))
        idx = rng.choice(X2.shape[0], size=min(RAND_NEG, X2.shape[0]), replace=False)
        y2 = np.array([char_f1(ctx[s:e], gold) for s, e in (keys2[i] for i in idx)],
                      dtype=np.float32)
        X = np.vstack([X, X2[idx]])
        y = np.concatenate([y, y2])
    return X, y


def make_ranker_data(df, idf):
    args = [(r.context, r.question, r.answer) for r in df.itertuples()]
    Xs, ys = [], []
    with Pool(NPROC, initializer=_init, initargs=(idf,)) as p:
        for out in p.imap_unordered(_rows_one, args, chunksize=16):
            if out is not None:
                Xs.append(out[0])
                ys.append(out[1])
    return np.vstack(Xs), np.concatenate(ys)


# ------------------------------------------------------------------ scoring
def _score_one(args):
    ctx, q = args
    idf = _G["idf"]
    model = _G["model"]
    keys, X, extra = build_example(ctx, q, idf, topk=10 ** 9)
    if X.shape[0] == 0:
        return "", np.zeros(12, dtype=np.float32)
    sc = model.predict(X)
    order = np.argsort(-sc)
    best = keys[order[0]]
    top = sc[order[:20]]
    pad = np.pad(top, (0, max(0, 20 - len(top))), constant_values=top[-1])
    # de-duplicated distinct-string top scores
    seen, dist = set(), []
    for i in order:
        s, e = keys[i]
        t = ctx[s:e]
        if t not in seen:
            seen.add(t)
            dist.append(sc[i])
        if len(dist) >= 5:
            break
    dist = np.pad(np.array(dist, dtype=np.float32), (0, max(0, 5 - len(dist))),
                  constant_values=dist[-1] if dist else 0.0)
    stats = np.array([
        pad[0], pad[1], pad[2], pad[4], pad[9], pad[19],
        pad[0] - pad[1], pad[0] - pad[4], pad[0] - dist[1],
        float(sc.mean()), float(sc.std()), float(np.percentile(sc, 99)),
    ], dtype=np.float32)
    meta = np.array([extra["max_ss"], extra["cov"], extra["nq"],
                     np.log1p(len(ctx)), len(q) / 50.0] + qtype_vec(q),
                    dtype=np.float32)
    return ctx[best[0]:best[1]], np.concatenate([stats, meta])


def _init2(idf, model):
    _G["idf"] = idf
    _G["model"] = model


def score_df(df, idf, model):
    args = [(r.context, r.question) for r in df.itertuples()]
    preds, stats = [], []
    with Pool(NPROC, initializer=_init2, initargs=(idf, model)) as p:
        for t, s in p.imap(_score_one, args, chunksize=16):
            preds.append(t)
            stats.append(s)
    return preds, np.vstack(stats)
