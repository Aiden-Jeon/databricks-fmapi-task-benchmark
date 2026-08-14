"""Train on full train.csv and write outputs/submission.csv."""
import sys
import time
import numpy as np
import pandas as pd

from common import char_f1
from pipeline import (SentenceRetriever, SpanScorer, build_training_samples,
                      predict_row)

TOP_K = 6
THRESHOLD = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
N_TRAIN = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
OUT = sys.argv[3] if len(sys.argv) > 3 else "../outputs/submission.csv"


def main():
    t0 = time.time()
    df = pd.read_csv("../train.csv").fillna({"answer": ""})
    test = pd.read_csv("../test.csv")
    df = df.iloc[:N_TRAIN]
    print(f"train on {len(df)}, test {len(test)}", flush=True)

    X, y, w, _ = build_training_samples(df, top_k=TOP_K, seed=0)
    print(f"built X={X.shape} ({time.time()-t0:.0f}s)", flush=True)
    scorer = SpanScorer().fit(X, y, sample_weight=w)
    print(f"trained ({time.time()-t0:.0f}s)", flush=True)

    retriever = SentenceRetriever()
    preds = []
    for i, row in test.iterrows():
        text, score, _ = predict_row(scorer, retriever, row["context"],
                                     row["question"], top_k=TOP_K)
        pred = text if score >= THRESHOLD else ""
        preds.append(pred)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(test)} ({time.time()-t0:.0f}s)", flush=True)

    sub = pd.DataFrame({"id": test["id"], "answer": preds})
    sub["answer"] = sub["answer"].fillna("")
    sub.to_csv(OUT, index=False)
    print(f"wrote {OUT} rows={len(sub)} empty={(sub.answer == '').mean():.3f} "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
