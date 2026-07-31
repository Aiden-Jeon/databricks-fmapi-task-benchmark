import numpy as np, pandas as pd, time, pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, \
    HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from content import build_content
import prior as P

t0 = time.time()
tr, te, Ctr, Cte = build_content()
y = tr.label.values
w = tr.word.values
tot = pd.concat([tr.word, te.word]).value_counts().to_dict()
SEEDS = [42, 7, 2024]
SIMCOLS = [c for c in Ctr.columns if c[:2] in ('s_', 'z_', 'p_')]
CORE = SIMCOLS + ['jac_tok', 'ovl_tok', 'len_diff', 'len_max', 'ntk_diff', 'pos_diff',
                  'nxt0_1same', 'nxt1_same', 'vmark_agree', 'w_count']


def pri(lab_idx, q_words, q_self=None):
    dist = P.fit_conditional(tot, w[lab_idx], y[lab_idx])
    F = P.prior_features(tot, w[lab_idx], y[lab_idx], q_words, q_self, dist)
    emp = P.empirical_table(tot, w[lab_idx], y[lab_idx])
    F['pr_emp'] = P.empirical_apply(emp, F)
    return F


MODELS = {
    'lr0.01': lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=4000)),
    'lr0.02': lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.02, max_iter=4000)),
    'lr0.05': lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=4000)),
    'lr0.1': lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=4000)),
    'gb': lambda: GradientBoostingClassifier(n_estimators=300, learning_rate=0.025,
                                             max_depth=3, subsample=0.8, random_state=0),
    'et': lambda: ExtraTreesClassifier(n_estimators=900, min_samples_leaf=6, n_jobs=4,
                                       random_state=0),
    'hgb': lambda: HistGradientBoostingClassifier(max_iter=350, learning_rate=0.025,
                                                  max_leaf_nodes=6, min_samples_leaf=50,
                                                  l2_regularization=5.0, random_state=0),
}
oof = {k: np.zeros(len(y)) for k in MODELS}
for seed in SEEDS:
    for trn, val in StratifiedKFold(5, shuffle=True, random_state=seed).split(Ctr, y):
        Xa = pd.concat([Ctr.iloc[trn][CORE].reset_index(drop=True), pri(trn, w[trn], y[trn])], axis=1)
        Xb = pd.concat([Ctr.iloc[val][CORE].reset_index(drop=True), pri(trn, w[val])], axis=1)
        for k, f in MODELS.items():
            m = f(); m.fit(Xa, y[trn])
            oof[k][val] += m.predict_proba(Xb)[:, 1] / len(SEEDS)
    print('seed done %.0fs' % (time.time() - t0), flush=True)
for k, o in oof.items():
    print('%-8s acc=%.4f auc=%.4f' % (k, accuracy_score(y, o > .5), roc_auc_score(y, o)))
import itertools
best = []
keys = list(MODELS)
for r in range(1, 5):
    for combo in itertools.combinations(keys, r):
        o = np.mean([oof[k] for k in combo], axis=0)
        best.append((roc_auc_score(y, o), accuracy_score(y, o > .5), combo))
best.sort(reverse=True)
for b in best[:15]:
    print('auc=%.4f acc=%.4f %s' % b)
with open('oof7.pkl', 'wb') as f:
    pickle.dump(oof, f)
print('%.0fs' % (time.time() - t0))
