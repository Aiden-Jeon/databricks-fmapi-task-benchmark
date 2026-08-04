"""Final training + prediction for t21_kmmlu (4-choice Korean professional MCQ).

Approach (no pretrained models / no internet):
  * Hand-crafted option-level features (position, length, numeric-ordering,
    question/option and option/option character-n-gram overlap, Korean lexical
    cues interacted with question polarity, ...).
  * Unsupervised LSA (char tf-idf + SVD on train+test text) question/option similarity.
  * Retrieval features: similarity of each option to the correct answers vs the
    distractors of the most similar training questions (leave-one-out on train).
  * Bagged HistGradientBoosting binary "is this the correct option" model, scores
    z-normalised within each question, blended with tf-idf logistic-regression
    models on option text only.  argmax over the 4 options gives the label.

Usage:  python run.py            (from the solution/ directory)
"""
import os
import numpy as np
import pandas as pd

from feats import build_features
from pipeline import (TextSpace, lsa_sim, retrieval_feats, hgb_scores, text_lr_scores,
                      opt_texts, zgroup)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
W_CHAR, W_WORD = 0.05, 0.08          # small weights: stable region of a 3x5-fold CV sweep


def main():
    tr = pd.read_csv(os.path.join(ROOT, "train.csv"))
    te = pd.read_csv(os.path.join(ROOT, "test.csv"))
    n, m = len(tr), len(te)
    y = tr.label.values - 1
    ybin = np.zeros(n * 4, int)
    ybin[np.arange(n) * 4 + y] = 1

    qtr = tr.question.astype(str).tolist()
    qte = te.question.astype(str).tolist()
    otr, ote = opt_texts(tr), opt_texts(te)

    space = TextSpace(qtr + otr + qte + ote)

    Xtr, _, _ = build_features(tr)
    Xte, _, _ = build_features(te)
    Xtr = np.hstack([Xtr, lsa_sim(space, qtr, otr)[:, None],
                     retrieval_feats(space, qtr, otr, y, qtr, otr, exclude_self=True)])
    Xte = np.hstack([Xte, lsa_sim(space, qte, ote)[:, None],
                     retrieval_feats(space, qtr, otr, y, qte, ote)])
    print("features:", Xtr.shape, Xte.shape, flush=True)

    s_hgb = hgb_scores(Xtr, ybin, Xte)
    s_char, s_word = text_lr_scores(otr, ybin, ote)
    score = zgroup(s_hgb) + W_CHAR * zgroup(s_char) + W_WORD * zgroup(s_word)

    pred = score.reshape(m, 4).argmax(1) + 1
    out = pd.DataFrame({"id": te.id.values, "label": pred.astype(int)})
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    path = os.path.join(ROOT, "outputs", "submission.csv")
    out.to_csv(path, index=False)
    print("wrote", path)
    print(out.label.value_counts(normalize=True).sort_index())


if __name__ == "__main__":
    main()
