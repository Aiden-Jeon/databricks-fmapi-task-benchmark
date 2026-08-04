"""Final submission: score ensemble of the two ranker rounds.

model_v1 = ranker trained with random negatives
model_v2 = ranker trained with random + hard negatives (mined with v1)
val char-F1: v1 0.5499, v2 0.5380, 0.6*v1+0.4*v2 -> 0.5684
"""
import os, sys, pickle, numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run as R

W = float(os.environ.get('W', 0.6))


class Ens:
    def __init__(self, ms, w):
        self.ms, self.w = ms, w

    def predict(self, X):
        return self.w * self.ms[0].predict(X) + (1 - self.w) * self.ms[1].predict(X)


def main():
    tr, te = R.load()
    dfmap, ndoc, maxidf = R.get_df(tr)
    m1 = pickle.load(open(os.path.join(R.WORK, 'model_v1.pkl'), 'rb'))
    m2 = pickle.load(open(os.path.join(R.WORK, 'model_v2.pkl'), 'rb'))
    res, _ = R.predict_frame(te, Ens([m1, m2], W), dfmap, ndoc, maxidf, log_every=1000)
    sub = pd.DataFrame({'id': te.id, 'answer': [res[i] for i in te.id]})
    tmp = os.path.join(R.OUT, 'submission_ens.csv')
    sub.to_csv(tmp, index=False)
    assert len(sub) == len(te) and set(sub.id) == set(te.id) and sub.id.duplicated().sum() == 0
    assert (sub.answer.astype(str).str.len() > 0).all()
    os.replace(tmp, os.path.join(R.OUT, 'submission.csv'))
    print('wrote submission.csv', sub.shape)


if __name__ == '__main__':
    main()


