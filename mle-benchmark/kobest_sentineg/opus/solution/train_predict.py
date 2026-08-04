"""KoBEST SentiNeg — final training / prediction script.

Approach
--------
No pretrained weights or external data are available, so the model is a
character-level TF-IDF ensemble over two text views of every sentence:

  * ``norm`` : NFC-normalised text, repeated chars collapsed, whitespace squeezed
  * ``jamo`` : the same text with Hangul syllables decomposed into jamo
               (cho/jung/jong), which lets n-grams capture Korean negation
               morphology such as ``-지 않-`` / ``안-`` / ``없-`` across the
               inflected endings.

Five diverse linear/kernel classifiers (RBF-SVM, ridge, logistic regression,
linear SVM, NB-SVM) are trained on the shared feature space and their
standardised decision scores are averaged.  Blend weights were chosen by a
greedy forward search and confirmed on 4x repeated stratified 5-fold OOF
predictions (20 folds):

    blend            0.9613 +- 0.0009
    best single (G_rbf)  0.9593 +- 0.0006
    char-TFIDF LR baseline  0.9466

Usage:  python train_predict.py    (run from the ``solution`` directory)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import Featurizer, make_views, model_zoo, decision  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)

# weights from the blend search in experiment.py (OOF accuracy 0.9613)
BLEND = {"G_rbf": 2, "E_ridge": 2, "I_lr_wb": 1, "D_svc": 1, "J_nbsvm": 1}


def zscore(v):
    return (v - v.mean()) / (v.std() + 1e-12)


def main():
    tr = pd.read_csv(os.path.join(TASK, "train.csv"))
    te = pd.read_csv(os.path.join(TASK, "test.csv"))

    vtr = make_views(tr.sentence.values)
    vte = make_views(te.sentence.values)
    y = tr.label.values

    zoo = model_zoo()
    total = np.zeros(len(te))
    wsum = 0.0
    for name, w in BLEND.items():
        blocks, fac = zoo[name]
        f = Featurizer(blocks)
        X = f.fit_transform(vtr)
        T = f.transform(vte)
        est = fac()
        est.fit(X, y)
        s = decision(est, T)
        total += w * zscore(s)
        wsum += w
        print(f"{name:9s} w={w}  n_features={X.shape[1]:6d}  "
              f"pos_rate={(s > 0).mean():.3f}", flush=True)

    score = total / wsum
    pred = (score > 0).astype(int)

    out_dir = os.path.join(TASK, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    sub = pd.DataFrame({"id": te.id.values, "label": pred})
    sub.to_csv(os.path.join(out_dir, "submission.csv"), index=False)
    print(f"wrote {len(sub)} rows; label counts:\n{sub.label.value_counts().to_string()}")


if __name__ == "__main__":
    main()
