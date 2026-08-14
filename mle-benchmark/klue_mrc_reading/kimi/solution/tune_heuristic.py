"""Grid-search heuristic weights + threshold on a holdout."""
import time
import numpy as np
import pandas as pd
from common import char_f1
from heuristic import HeuristicMRC


def main():
    t0 = time.time()
    df = pd.read_csv("../train.csv").fillna({"answer": ""})
    rng = np.random.RandomState(1)
    idx = rng.permutation(len(df))
    val = df.iloc[idx[:600]].reset_index(drop=True)

    weight_sets = [
        {"sim": 1.0, "type": 0.5, "sup": 0.4, "len": 1.0, "clean": 1.0, "rank": 0.1, "tokpen": 0.03},
        {"sim": 1.0, "type": 0.8, "sup": 0.3, "len": 0.8, "clean": 1.0, "rank": 0.1, "tokpen": 0.02},
        {"sim": 1.2, "type": 0.6, "sup": 0.5, "len": 1.2, "clean": 1.0, "rank": 0.15, "tokpen": 0.03},
        {"sim": 1.0, "type": 1.0, "sup": 0.4, "len": 1.0, "clean": 1.2, "rank": 0.1, "tokpen": 0.04},
        {"sim": 1.5, "type": 0.6, "sup": 0.6, "len": 1.0, "clean": 1.0, "rank": 0.2, "tokpen": 0.03},
    ]

    best = (0.0, None, None)
    for w in weight_sets:
        m = HeuristicMRC(top_k=5, w=w)
        recs = []
        for _, row in val.iterrows():
            t, s = m.predict(row["context"], row["question"])
            recs.append((row["answer"], t, s))
        for th in np.arange(-1.0, 2.0, 0.1):
            f1s = [char_f1(p if s >= th else "", g) for g, p, s in recs]
            f = float(np.mean(f1s))
            if f > best[0]:
                best = (f, dict(w), round(float(th), 2))
        print(f"  done w; running best F1={best[0]:.4f} ({time.time()-t0:.0f}s)",
              flush=True)
    print("BEST F1=%.4f th=%s w=%s" % (best[0], best[2], best[1]))


if __name__ == "__main__":
    main()
