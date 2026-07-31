"""KoBEST WiC — final reproducible pipeline.

Approach
--------
No pretrained models / external data are available, so the solution combines
(a) unsupervised text-similarity features between the two contexts (character
    and word TF-IDF cosine, LSA cosine, PPMI-embedding cosine, all also
    normalised *within* the target word), plus surface / collocation features;
(b) per-word label-prior features that exploit the dataset construction: the
    pairs sharing one target word are built to contain a *mix* of same-sense
    and different-sense examples, so the labels of the other pairs of the same
    word are informative (finite-population, negatively correlated).  These are
    computed leave-one-out on train and fold-honestly inside CV.

Models: blend of L2 logistic regressions (3 strengths) + gradient boosting.

Usage:  python run.py          (writes ../outputs/submission.csv)
"""
import numpy as np, pandas as pd, os, time
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from content import build_content
import prior as P

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SEEDS = [42, 7, 2024]

MODELS = {
    'lr0.01': lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=4000)),
    'lr0.02': lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.02, max_iter=4000)),
    'lr0.05': lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=4000)),
    'gb': lambda: GradientBoostingClassifier(n_estimators=300, learning_rate=0.025,
                                             max_depth=3, subsample=0.8, random_state=0),
}


def main():
    t0 = time.time()
    tr, te, Ctr, Cte = build_content()
    y = tr.label.values
    w = tr.word.values
    tot = pd.concat([tr.word, te.word]).value_counts().to_dict()

    simcols = [c for c in Ctr.columns if c[:2] in ('s_', 'z_', 'p_')]
    CORE = simcols + ['jac_tok', 'ovl_tok', 'len_diff', 'len_max', 'ntk_diff', 'pos_diff',
                      'nxt0_1same', 'nxt1_same', 'vmark_agree', 'w_count']

    def pri(lab_idx, q_words, q_self=None):
        dist = P.fit_conditional(tot, w[lab_idx], y[lab_idx])
        F = P.prior_features(tot, w[lab_idx], y[lab_idx], q_words, q_self, dist)
        emp = P.empirical_table(tot, w[lab_idx], y[lab_idx])
        F['pr_emp'] = P.empirical_apply(emp, F)
        return F

    # ---------------- cross-validation ----------------
    oof = {k: np.zeros(len(y)) for k in MODELS}
    for seed in SEEDS:
        for trn, val in StratifiedKFold(5, shuffle=True, random_state=seed).split(Ctr, y):
            Xa = pd.concat([Ctr.iloc[trn][CORE].reset_index(drop=True),
                            pri(trn, w[trn], y[trn])], axis=1)
            Xb = pd.concat([Ctr.iloc[val][CORE].reset_index(drop=True),
                            pri(trn, w[val])], axis=1)
            for k, f in MODELS.items():
                m = f(); m.fit(Xa, y[trn])
                oof[k][val] += m.predict_proba(Xb)[:, 1] / len(SEEDS)
    for k, o in oof.items():
        print('%-8s acc=%.4f auc=%.4f' % (k, accuracy_score(y, o > .5), roc_auc_score(y, o)))
    ens = np.mean([oof[k] for k in MODELS], axis=0)
    thr_q = float(np.quantile(ens, 1 - y.mean()))
    print('ENSEMBLE acc@0.5=%.4f  acc@q=%.4f (thr=%.4f)  auc=%.4f'
          % (accuracy_score(y, ens > .5), accuracy_score(y, ens > thr_q), thr_q,
             roc_auc_score(y, ens)))

    # ---------------- full fit + prediction ----------------
    allidx = np.arange(len(y))
    Xa = pd.concat([Ctr[CORE].reset_index(drop=True), pri(allidx, w, y)], axis=1)
    Xb = pd.concat([Cte[CORE].reset_index(drop=True), pri(allidx, te.word.values)], axis=1)
    Xb = Xb[Xa.columns]
    preds = []
    for k, f in MODELS.items():
        m = f(); m.fit(Xa, y)
        preds.append(m.predict_proba(Xb)[:, 1])
    pte = np.mean(preds, axis=0)
    lab = (pte > 0.5).astype(int)
    os.makedirs(os.path.join(ROOT, 'outputs'), exist_ok=True)
    pd.DataFrame({'id': te.id.values, 'label': lab}).to_csv(
        os.path.join(ROOT, 'outputs', 'submission.csv'), index=False)
    np.save(os.path.join(HERE, 'test_proba.npy'), pte)
    np.save(os.path.join(HERE, 'oof_ens.npy'), ens)
    print('submission written: n=%d  pos_rate=%.3f  (%.0fs)'
          % (len(lab), lab.mean(), time.time() - t0))


if __name__ == '__main__':
    main()
