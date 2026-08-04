"""KoBEST WiC — full solution.

Two complementary, purely train.csv-derived models are combined:

1. TEXT MODEL  — logistic regression on surface/structural features of the two
   contexts plus TF-IDF / LSA similarity features (char and word n-grams, the
   vector space is fit on all available contexts, no labels involved).

2. COUNT MODEL — a per-word exchangeable Bayesian model.  The benchmark is built
   per target word: every word contributes a small set of pairs whose
   positive/negative counts are close to balanced (clearly under-dispersed
   w.r.t. a binomial).  The distribution of the number of positives per word is
   estimated from words whose whole pair set lies inside train.csv, and is then
   used, together with the labelled pairs of the same word, to form a posterior
   over the labels of the remaining pairs of that word.

The text model supplies the likelihood ratio that the count model conditions on,
so the final prediction uses both the context evidence and the per-word design
prior.  Everything is estimated from train.csv; test.csv contributes only its
(unlabelled) texts and its word grouping.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import struct_frame, SimSpace, sim_frame          # noqa: E402
from count_model import fit_prior_em, predict as count_predict   # noqa: E402

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier

SEED = 42
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def text_model():
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                         LogisticRegression(C=0.3, max_iter=5000))


def build_features(tr, te):
    ctx = pd.concat([tr.context_1, tr.context_2, te.context_1, te.context_2]).tolist()
    space = SimSpace(ctx, n_comp=180, seed=SEED)
    Xtr = pd.concat([struct_frame(tr), sim_frame(tr, space)[0]], axis=1)
    Xte = pd.concat([struct_frame(te), sim_frame(te, space)[0]], axis=1)
    return Xtr, Xte


def oof_text_probs(X, y, n_splits=5, seed=SEED):
    oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    for a, b in skf.split(X, y):
        m = text_model()
        m.fit(X.iloc[a], y[a])
        oof[b] = m.predict_proba(X.iloc[b])[:, 1]
    return oof


def shrink(p, w):
    """Shrink probabilities toward 0.5 (tempering of the likelihood ratio)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lo = np.log(p / (1 - p)) * w
    return 1 / (1 + np.exp(-lo))


def validate(tr, te, Xtr, y, oof):
    """Faithful simulation: words absent from test.csv have their complete
    original pair set inside train.csv, so splitting their rows reproduces the
    real train/test situation (T known, prior fit on the retained part)."""
    comp_mask = ~tr.word.isin(set(te.word)).values
    comp = tr[comp_mask].reset_index(drop=True)
    p_comp = oof[comp_mask]
    Tc = comp.word.value_counts()
    print(f"[validation] complete words={comp.word.nunique()} rows={len(comp)}")
    best = None
    for w in [0.0, 0.6, 1.0, 1.25, 1.5, 2.0]:
        accs, accs_txt, accs_cnt = [], [], []
        for seed in range(40):
            rs = np.random.RandomState(seed)
            idx = rs.permutation(len(comp))
            cut = int(len(comp) * 0.8)
            ia, ib = idx[:cut], idx[cut:]
            A, B = comp.iloc[ia], comp.iloc[ib]
            prior = fit_prior_em(A, Tc)
            yb = B.label.values
            pt = p_comp[ib]
            pc = count_predict(A, B, Tc, prior, text_p=shrink(pt, w))
            accs.append(((pc > 0.5).astype(int) == yb).mean())
            accs_txt.append(((pt > 0.5).astype(int) == yb).mean())
            if w == 0.0:
                accs_cnt.append(accs[-1])
        m, s = np.mean(accs), np.std(accs)
        print(f"[validation] text_weight={w:<4}  combined={m:.4f} +-{s:.4f}"
              f"   (text alone={np.mean(accs_txt):.4f})")
        if w > 0 and (best is None or m > best[1] + 1e-4):
            best = (w, m)
    print(f"[validation] best text_weight={best[0]} acc={best[1]:.4f}")
    return best[0]


def main():
    tr = pd.read_csv(os.path.join(ROOT, 'train.csv'))
    te = pd.read_csv(os.path.join(ROOT, 'test.csv'))
    y = tr.label.values
    print(f"train={tr.shape} test={te.shape}")

    Xtr, Xte = build_features(tr, te)
    print(f"features: {Xtr.shape[1]}")

    oof = oof_text_probs(Xtr, y)
    print(f"[text model] OOF accuracy = {((oof > 0.5).astype(int) == y).mean():.4f}")

    w = validate(tr, te, Xtr, y, oof)

    # ---- fit on everything and predict test ----
    m = text_model()
    m.fit(Xtr, y)
    p_text = m.predict_proba(Xte)[:, 1]

    total = tr.word.value_counts().add(te.word.value_counts(), fill_value=0)
    prior = fit_prior_em(tr, total)
    p_final = count_predict(tr, te, total, prior, text_p=shrink(p_text, w))
    p_final = np.where(np.isnan(p_final), p_text, p_final)

    pred = (p_final > 0.5).astype(int)
    print(f"[submission] positive rate = {pred.mean():.3f}; "
          f"agreement with text-only = {(pred == (p_text > 0.5)).mean():.3f}")

    out = os.path.join(ROOT, 'outputs')
    os.makedirs(out, exist_ok=True)
    sub = pd.DataFrame({'id': te.id.values, 'label': pred})
    sub.to_csv(os.path.join(out, 'submission.csv'), index=False)

    ss = pd.read_csv(os.path.join(ROOT, 'sample_submission.csv'))
    assert list(sub.columns) == list(ss.columns)
    assert len(sub) == len(te) and sub.id.is_unique
    assert set(sub.id) == set(te.id)
    assert sub.label.isin([0, 1]).all()
    print("[submission] written and validated:", os.path.join(out, 'submission.csv'))


if __name__ == '__main__':
    main()
