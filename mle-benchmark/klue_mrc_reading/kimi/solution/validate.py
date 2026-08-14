"""Holdout validation to tune threshold and measure char-F1."""
import sys
import time
import numpy as np
import pandas as pd

from common import char_f1
from pipeline import (SentenceRetriever, SpanScorer, build_training_samples,
                      predict_row)


def run(n_train=3000, n_val=800, top_k=6, seed=0, quiet=False):
    t0 = time.time()
    df = pd.read_csv("../train.csv").fillna({"answer": ""})
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(df))
    train_df = df.iloc[idx[:n_train]].reset_index(drop=True)
    val_df = df.iloc[idx[n_train:n_train + n_val]].reset_index(drop=True)
    if not quiet:
        print(f"train {len(train_df)} val {len(val_df)}", flush=True)

    X, y, w, _ = build_training_samples(train_df, top_k=top_k, seed=seed)
    if not quiet:
        print(f"span training set: X={X.shape} pos={y.sum()} "
              f"({time.time()-t0:.0f}s)", flush=True)

    scorer = SpanScorer().fit(X, y, sample_weight=w)
    if not quiet:
        print(f"scorer trained ({time.time()-t0:.0f}s)", flush=True)

    retriever = SentenceRetriever()
    records = []
    for _, row in val_df.iterrows():
        gold = row["answer"]
        pred_text, score, _ = predict_row(scorer, retriever, row["context"],
                                          row["question"], top_k=top_k)
        records.append((gold, pred_text, score))
    if not quiet:
        print(f"val predicted ({time.time()-t0:.0f}s)", flush=True)

    best_th, best_f1 = 0.0, -1.0
    for th in np.arange(0.0, 0.98, 0.02):
        f1s = [char_f1(p if s >= th else "", g) for g, p, s in records]
        m = np.mean(f1s)
        if m > best_f1:
            best_f1, best_th = m, th
    if not quiet:
        print(f"best threshold={best_th:.2f} val char-F1={best_f1:.4f}")
        for g, p, s in records[:15]:
            pred = p if s >= best_th else ""
            print(f"  gold={g!r:28s} pred={pred!r:28s} score={s:.3f} "
                  f"f1={char_f1(pred, g):.2f}")
    return best_th, best_f1, scorer, records


if __name__ == "__main__":
    run()
