import pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from collections import defaultdict

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

def make_text(row, mode='concat'):
    if mode == 'concat':
        return f"{row['q']} {row['premise']} {row['alt']}"
    if mode == 'pair':
        return f"{row['premise']} {row['alt']}"

def run_cv(text_mode='concat', C=4.0, ngram_word=(1,2), ngram_char=(2,4), model='logreg', extra_feats=False):
    pdf['text'] = pdf.apply(lambda r: make_text(r, text_mode), axis=1)
    word_vec = TfidfVectorizer(sublinear_tf=True, ngram_range=ngram_word, min_df=2)
    char_vec = TfidfVectorizer(sublinear_tf=True, analyzer='char_wb', ngram_range=ngram_char, min_df=2)
    union = FeatureUnion([('w', word_vec), ('c', char_vec)])
    if model == 'logreg':
        clf = LogisticRegression(C=C, max_iter=3000, random_state=RS)
    else:
        clf = LinearSVC(C=C, random_state=RS)
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
        tr_idx = np.where(sample_fold != k)[0]
        te_idx = np.where(sample_fold == k)[0]
        pipe.fit(pdf['text'].iloc[tr_idx], pdf['y'].iloc[tr_idx])
        if model == 'logreg':
            proba = pipe.predict_proba(pdf['text'].iloc[te_idx])[:, 1]
        else:
            dec = pipe.decision_function(pdf['text'].iloc[te_idx])
            proba = (dec + 1) / 2
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
    return np.array(accs)

configs = [
    ('concat', 4.0, (1,2), (2,4), 'logreg'),
    ('concat', 4.0, (1,2), (2,4), 'svm'),
    ('pair', 4.0, (1,2), (2,4), 'logreg'),
    ('concat', 4.0, (1,3), (2,5), 'logreg'),
    ('concat', 2.0, (1,2), (2,4), 'logreg'),
    ('concat', 8.0, (1,2), (2,4), 'logreg'),
    ('concat', 4.0, (1,1), (2,4), 'logreg'),
    ('concat', 4.0, (1,2), (3,5), 'logreg'),
]
for cfg in configs:
    accs = run_cv(*cfg)
    print(f"{cfg} -> mean {accs.mean():.4f}  folds {accs}")
