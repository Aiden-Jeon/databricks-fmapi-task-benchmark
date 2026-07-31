import pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import re

RS = 0
tr = pd.read_csv('train.csv'); tr['question'] = tr['question'].str.strip()

def build_pairs(df):
    rows = []
    for _, r in df.iterrows():
        q = r['question']
        rows.append({'gid': r['id'], 'q': q, 'premise': r['premise'], 'alt': r['alternative_1'], 'pos': 0, 'y': 1 if r['label'] == 0 else 0})
        rows.append({'gid': r['id'], 'q': q, 'premise': r['premise'], 'alt': r['alternative_2'], 'pos': 1, 'y': 1 if r['label'] == 1 else 0})
    return pd.DataFrame(rows)

pdf = build_pairs(tr)
pdf['text'] = (pdf['q'].astype(str) + ' ' + pdf['premise'].astype(str) + ' ' + pdf['alt'].astype(str))

# ---- group-fold setup ----
gids = pdf['gid'].values
unique_gids = pd.unique(gids)
skf_seed = np.random.RandomState(RS)
perm = skf_seed.permutation(len(unique_gids))
fold_assign = np.zeros(len(unique_gids), dtype=int)
for k in range(5):
    fold_assign[perm[k::5]] = k
gid_to_fold = dict(zip(unique_gids, fold_assign))
sample_fold = np.array([gid_to_fold[g] for g in gids])
row_for_gid = tr.set_index('id')

def evaluate(proba_by_idx_per_fold):
    """proba_by_idx_per_fold: dict fold -> array of proba for test pairs in order of te_idx"""
    all_preds = {}
    for k in range(5):
        te_idx = np.where(sample_fold == k)[0]
        proba = proba_by_idx_per_fold[k]
        test_gids = gids[te_idx]
        idx_by_gid = defaultdict(list)
        for i, g in enumerate(test_gids):
            idx_by_gid[g].append(i)
        for g, idxs in idx_by_gid.items():
            p1 = proba[idxs[0]]; p2 = proba[idxs[1]]
            all_preds[g] = 0 if p1 >= p2 else 1
    correct = sum(1 for g, p in all_preds.items() if p == row_for_gid.loc[g, 'label'])
    return correct / len(all_preds)

def run_model(build_pipe, name):
    proba_per_fold = {}
    for k in range(5):
        tr_idx = np.where(sample_fold != k)[0]
        te_idx = np.where(sample_fold == k)[0]
        pipe = build_pipe()
        pipe.fit(pdf['text'].iloc[tr_idx], pdf['y'].iloc[tr_idx])
        proba_per_fold[k] = pipe.predict_proba(pdf['text'].iloc[te_idx])[:, 1]
    acc = evaluate(proba_per_fold)
    print(f"{name}: acc={acc:.4f}")
    return proba_per_fold

probas = {}
# Config A: logreg word+char
probas['A'] = run_model(lambda: Pipeline([('f', FeatureUnion([('w', TfidfVectorizer(sublinear_tf=True, ngram_range=(1,2), min_df=2)), ('c', TfidfVectorizer(sublinear_tf=True, analyzer='char_wb', ngram_range=(2,4), min_df=2))])), ('c', LogisticRegression(C=4.0, max_iter=3000, random_state=RS))]), 'A_logreg_wc')
# Config B: logreg C=2
probas['B'] = run_model(lambda: Pipeline([('f', FeatureUnion([('w', TfidfVectorizer(sublinear_tf=True, ngram_range=(1,2), min_df=2)), ('c', TfidfVectorizer(sublinear_tf=True, analyzer='char_wb', ngram_range=(2,4), min_df=2))])), ('c', LogisticRegression(C=2.0, max_iter=3000, random_state=RS))]), 'B_logreg_c2')
# Config C: logreg char only (2,5)
probas['C'] = run_model(lambda: Pipeline([('f', TfidfVectorizer(sublinear_tf=True, analyzer='char_wb', ngram_range=(2,5), min_df=2)), ('c', LogisticRegression(C=4.0, max_iter=3000, random_state=RS))]), 'C_char_only')
# Config D: NB
probas['D'] = run_model(lambda: Pipeline([('f', FeatureUnion([('w', TfidfVectorizer(sublinear_tf=True, ngram_range=(1,2), min_df=2)), ('c', TfidfVectorizer(sublinear_tf=True, analyzer='char_wb', ngram_range=(2,4), min_df=2))])), ('c', ComplementNB())]), 'D_cnb')
# Config E: word (1,3) + char (2,4)
probas['E'] = run_model(lambda: Pipeline([('f', FeatureUnion([('w', TfidfVectorizer(sublinear_tf=True, ngram_range=(1,3), min_df=2)), ('c', TfidfVectorizer(sublinear_tf=True, analyzer='char_wb', ngram_range=(2,4), min_df=2))])), ('c', LogisticRegression(C=4.0, max_iter=3000, random_state=RS))]), 'E_word13')

# Ensemble (average proba)
ens_per_fold = {}
for k in range(5):
    ens_per_fold[k] = np.mean([probas[key][k] for key in probas], axis=0)
print('Ensemble:', f"{evaluate(ens_per_fold):.4f}")

# Try weighted ensembles
keys = list(probas.keys())
for wA in [1,2,3]:
    for wB in [0,1,2]:
        for wC in [0,1,2]:
            weights = np.array([wA, wB, wC, 1, 1])
            if weights.sum()==0: continue
            ens = {}
            for k in range(5):
                ens[k] = np.average([probas[keys[i]][k] for i in range(len(keys))], axis=0, weights=weights)
            acc = evaluate(ens)
            if acc > 0.60:
                print(f"weights {weights}: {acc:.4f}")
