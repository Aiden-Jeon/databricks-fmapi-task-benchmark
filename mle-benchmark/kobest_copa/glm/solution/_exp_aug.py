import pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
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

def make_text(row):
    return f"{row['premise']} {row['alt']}"

pdf['text'] = pdf.apply(make_text, axis=1)

# ---- Swap augmentation: for each group, add swapped position to remove bias ----
# Build augmented pair set: for each original pair, also add the same text with pos flipped but y kept
def augment(pdf):
    aug = pdf.copy()
    aug['pos'] = 1 - aug['pos']
    return pd.concat([pdf, aug], ignore_index=True)

pdf_aug = augment(pdf)

word_vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2)
char_vec = TfidfVectorizer(sublinear_tf=True, analyzer='char_wb', ngram_range=(2, 4), min_df=2)
union = FeatureUnion([('w', word_vec), ('c', char_vec)])
clf = LogisticRegression(C=4.0, max_iter=3000, random_state=RS)
pipe = Pipeline([('f', union), ('c', clf)])

gids = pdf['gid'].values
unique_gids = pd.unique(gids)
skf_seed = np.random.RandomState(RS)
perm = skf_seed.permutation(len(unique_gids))
fold_assign = np.zeros(len(unique_gids), dtype=int)
for k in range(5):
    fold_assign[perm[k::5]] = k
gid_to_fold = dict(zip(unique_gids, fold_assign))
sample_fold = np.array([gid_to_fold[g] for g in gids])

accs = []
row_for_gid = tr.set_index('id')
for k in range(5):
    # train on pairs whose group fold != k (using augmented)
    tr_mask_aug = np.array([gid_to_fold[g] != k for g in pdf_aug['gid'].values])
    te_idx = np.where(sample_fold == k)[0]
    pipe.fit(pdf_aug['text'].iloc[tr_mask_aug], pdf_aug['y'].iloc[tr_mask_aug])
    proba = pipe.predict_proba(pdf['text'].iloc[te_idx])[:, 1]
    test_gids = gids[te_idx]
    idx_by_gid = defaultdict(list)
    for i, g in enumerate(test_gids):
        idx_by_gid[g].append(i)
    correct = 0
    for g, idxs in idx_by_gid.items():
        p1 = proba[idxs[0]]; p2 = proba[idxs[1]]
        pred = 0 if p1 >= p2 else 1
        if pred == row_for_gid.loc[g, 'label']:
            correct += 1
    accs.append(correct / len(idx_by_gid))
print('CV acc (augmented pair):', accs, 'mean', np.mean(accs))
