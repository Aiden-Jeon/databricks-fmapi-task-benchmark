"""KLUE-NLI (t6_klue_nli) — final solution.

    python solution/run.py                 # fit on train.csv -> outputs/submission.csv
    python solution/run.py --validate      # held-out evaluation of the same pipeline

Pipeline
  1. TF-IDF (word + char) over premise / hypothesis / the novel and shared parts
     of the hypothesis, plus 31 dense overlap-negation-length features.
  2. Multinomial logistic regression (C=0.5)  ->  p(label | premise, hypothesis).
  3. Structured decoding per premise group, using the empirical label-multiset
     prior estimated on train.csv (see solution/decoder.py).

Everything is fit on train.csv only; no external data, no pretrained weights.
Runtime: ~2 min on 4 CPU cores.
"""
import argparse
import collections
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feats import FeatureBuilder, LABELS, L2I  # noqa: E402
from decoder import fit_prior, decode  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C_MC = 0.5          # logistic regression regularisation
W_PRIOR = 3.0       # weight of the group-structure prior
ALPHA = 0.5         # Laplace smoothing of the multiset prior
SEED = 42


def fit_predict(df_fit, df_pred):
    fb = FeatureBuilder()
    X = fb.fit_transform(df_fit)
    Xp = fb.transform(df_pred)
    y = df_fit.label.map(L2I).values
    mc = LogisticRegression(C=C_MC, max_iter=2000, n_jobs=-1).fit(X, y)
    logp = np.log(np.clip(mc.predict_proba(Xp), 1e-9, 1.0))
    prior = fit_prior(df_fit.premise.values, y, alpha=ALPHA)
    pred = decode(df_pred.premise.values, logp, df_fit.premise.values, y,
                  prior, w_prior=W_PRIOR)
    return pred, logp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    train = pd.read_csv(f"{ROOT}/train.csv")

    if args.validate:
        y = train.label.map(L2I).values
        itr, iva = train_test_split(np.arange(len(train)), test_size=5000,
                                    random_state=SEED, stratify=y)
        dtr = train.iloc[itr].reset_index(drop=True)
        dva = train.iloc[iva].reset_index(drop=True)
        pred, logp = fit_predict(dtr, dva)
        yva = y[iva]
        print(f"row-independent argmax accuracy : {(logp.argmax(1) == yva).mean():.4f}")
        print(f"structured decoding accuracy    : {(pred == yva).mean():.4f}")
        # the held-out split has a different mix of group cases than test.csv,
        # so also report the accuracy re-weighted to the test.csv case mix
        known = collections.defaultdict(list)
        for p, l in zip(dtr.premise.values, y[itr]):
            known[p].append(l)
        gb = collections.Counter(dva.premise.values)
        W = {(2, 2, 1): 3084, (1, 1, 2): 1588, (0, 0, 3): 159, (1, 2, 1): 96,
             (1, 1, 1): 51, (0, 0, 2): 18, (0, 0, 1): 1}
        st = collections.defaultdict(lambda: [0, 0])
        for i, p in enumerate(dva.premise.values):
            k = (len(set(known.get(p, []))), len(known.get(p, [])), gb[p])
            st[k][0] += 1
            st[k][1] += int(pred[i] == yva[i])
        est = (sum(W[k] * st[k][1] / st[k][0] for k in W if k in st)
               / sum(W[k] for k in W if k in st))
        print(f"re-weighted to test case mix    : {est:.4f}")
        for k in sorted(st, key=lambda k: -st[k][0]):
            print(f"   known_distinct={k[0]} known_n={k[1]} group_test_n={k[2]}"
                  f"  n={st[k][0]:5d}  acc={st[k][1]/st[k][0]:.3f}")
    else:
        test = pd.read_csv(f"{ROOT}/test.csv")
        pred, logp = fit_predict(train, test)
        os.makedirs(f"{ROOT}/outputs", exist_ok=True)
        sub = pd.DataFrame({"id": test.id.values, "label": [LABELS[i] for i in pred]})
        sub.to_csv(f"{ROOT}/outputs/submission.csv", index=False)
        assert len(sub) == len(test) and sub.id.is_unique
        assert sub.label.isin(LABELS).all()
        print(sub.label.value_counts().to_string())
        print(f"wrote {ROOT}/outputs/submission.csv")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
