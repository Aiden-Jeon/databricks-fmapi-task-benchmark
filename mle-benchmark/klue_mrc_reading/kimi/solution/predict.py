"""Generate submission using the deterministic heuristic MRC model."""
import sys
import time
import numpy as np
import pandas as pd

from heuristic import HeuristicMRC

# Tuned on holdout (see tune_heuristic.py / report).
WEIGHTS = {"sim": 1.0, "type": 0.5, "sup": 0.4, "len": 1.0, "clean": 1.0,
           "rank": 0.1, "tokpen": 0.03}
THRESHOLD = float(sys.argv[1]) if len(sys.argv) > 1 else 1.7
TOP_K = int(sys.argv[2]) if len(sys.argv) > 2 else 5
OUT = sys.argv[3] if len(sys.argv) > 3 else "../outputs/submission.csv"


def main():
    t0 = time.time()
    test = pd.read_csv("../test.csv")
    model = HeuristicMRC(top_k=TOP_K, w=WEIGHTS)
    preds = []
    for i, row in test.iterrows():
        text, score = model.predict(row["context"], row["question"])
        preds.append(text if score >= THRESHOLD else "")
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(test)} ({time.time()-t0:.0f}s)", flush=True)
    sub = pd.DataFrame({"id": test["id"], "answer": preds})
    sub["answer"] = sub["answer"].fillna("")
    sub.to_csv(OUT, index=False)
    print(f"wrote {OUT} rows={len(sub)} empty={(sub.answer == '').mean():.3f} "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
