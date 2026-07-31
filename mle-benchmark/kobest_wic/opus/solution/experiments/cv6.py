"""Feature-subset / hyper-parameter sweep for the linear model + ensembles."""
import numpy as np, pandas as pd, time
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
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


def pri(lab_idx, q_words, q_self=None):
    dist = P.fit_conditional(tot, w[lab_idx], y[lab_idx])
    F = P.prior_features(tot, w[lab_idx], y[lab_idx], q_words, q_self, dist)
    emp = P.empirical_table(tot, w[lab_idx], y[lab_idx])
    F['pr_emp'] = P.empirical_apply(emp, F)
    return F


SIMCOLS = [c for c in Ctr.columns if c[:2] in ('s_', 'z_', 'p_')]
CORE = SIMCOLS + ['jac_tok', 'ovl_tok', 'len_diff', 'len_max', 'ntk_diff', 'pos_diff',
                  'nxt0_1same', 'nxt1_same', 'vmark_agree', 'w_count']
SETS = {
    'all': list(Ctr.columns),
    'core': CORE,
    'sim': SIMCOLS,
    'simlite': [c for c in SIMCOLS if 'lsa_char' in c or c.endswith('char')],
}
PRISETS = {
    'full': None,
    'lite': ['pr_n1', 'pr_n0', 'pr_diff', 'pr_k', 'pr_tot', 'pr_emp'],
    'emp': ['pr_emp', 'pr_k', 'pr_tot'],
    'cond': ['pr_cond', 'pr_k', 'pr_tot'],
}


def evaluate(cols, pcols, model_fn, seeds=SEEDS):
    oof = np.zeros(len(y))
    for seed in seeds:
        for trn, val in StratifiedKFold(5, shuffle=True, random_state=seed).split(Ctr, y):
            Pa, Pb = pri(trn, w[trn], y[trn]), pri(trn, w[val])
            if pcols:
                Pa, Pb = Pa[pcols], Pb[pcols]
            Xa = pd.concat([Ctr.iloc[trn][cols].reset_index(drop=True), Pa], axis=1)
            Xb = pd.concat([Ctr.iloc[val][cols].reset_index(drop=True), Pb], axis=1)
            m = model_fn()
            m.fit(Xa, y[trn])
            oof[val] += m.predict_proba(Xb)[:, 1] / len(seeds)
    return oof


rows = []
for sname, cols in SETS.items():
    for pname, pcols in PRISETS.items():
        for C in (0.05, 0.2, 0.5):
            o = evaluate(cols, pcols, lambda: make_pipeline(
                StandardScaler(), LogisticRegression(C=C, max_iter=4000)))
            acc = accuracy_score(y, o > .5)
            thr = np.quantile(o, 1 - y.mean())
            rows.append((sname, pname, f'lr{C}', acc, accuracy_score(y, o > thr),
                         roc_auc_score(y, o)))
            print(rows[-1], '%.0fs' % (time.time() - t0), flush=True)
R = pd.DataFrame(rows, columns=['feats', 'prior', 'model', 'acc', 'acc_thr', 'auc'])
R.to_csv('sweep_lr.csv', index=False)
print(R.sort_values('auc', ascending=False).head(12).to_string())
