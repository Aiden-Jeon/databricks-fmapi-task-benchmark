import pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score

RS = 0
tr = pd.read_csv('train.csv'); tr['question'] = tr['question'].str.strip()

# Build pair dataset: each row -> 2 pairs. y=1 means this alt is the correct one.
rows = []
for _, r in tr.iterrows():
    q = r['question']
    rows.append({'gid': r['id'], 'q': q, 'premise': r['premise'], 'alt': r['alternative_1'], 'y': 1 if r['label'] == 0 else 0})
    rows.append({'gid': r['id'], 'q': q, 'premise': r['premise'], 'alt': r['alternative_2'], 'y': 1 if r['label'] == 1 else 0})
pdf = pd.DataFrame(rows)

# text = premise + question + alt
pdf['text'] = pdf['q'].astype(str) + ' ' + pdf['premise'].astype(str) + ' ' + pdf['alt'].astype(str)

word_vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2)
char_vec = TfidfVectorizer(sublinear_tf=True, analyzer='char_wb', ngram_range=(2, 4), min_df=2)
union = FeatureUnion([('w', word_vec), ('c', char_vec)])
pipe = Pipeline([('f', union), ('c', LogisticRegression(C=4.0, max_iter=2000, random_state=RS))])

# CV at group level: ensure both pairs of a group are in same fold
gids = pdf['gid'].values
unique_gids = pd.unique(gids)
fold_assign = np.zeros(len(unique_gids), dtype=int)
skf_seed = np.random.RandomState(RS)
perm = skf_seed.permutation(len(unique_gids))
for k in range(5):
    fold_assign[perm[k::5]] = k
gid_to_fold = dict(zip(unique_gids, fold_assign))
sample_fold = np.array([gid_to_fold[g] for g in gids])

accs = []
for k in range(5):
    tr_idx = np.where(sample_fold != k)[0]
    te_idx = np.where(sample_fold == k)[0]
    pipe.fit(pdf['text'].iloc[tr_idx], pdf['y'].iloc[tr_idx])
    proba = pipe.predict_proba(pdf['text'].iloc[te_idx])[:, 1]
    # group by gid, compare proba of alt1 vs alt2
    test_gids = gids[te_idx]
    # rebuild order: for each gid in test, find the two indices
    from collections import defaultdict
    idx_by_gid = defaultdict(list)
    for i, g in enumerate(test_gids):
        idx_by_gid[g].append(i)
    preds = {}
    for g, idxs in idx_by_gid.items():
        # idxs in order alt1, alt2 (since we built pairs in that order)
        p1 = proba[idxs[0]]
        p2 = proba[idxs[1]]
        preds[g] = 0 if p1 >= p2 else 1
    # compare with original labels
    row_for_gid = tr.set_index('id')
    correct = 0
    for g, p in preds.items():
        if p == row_for_gid.loc[g, 'label']:
            correct += 1
    accs.append(correct / len(preds))
print('CV acc (pair approach):', accs, 'mean', np.mean(accs))
