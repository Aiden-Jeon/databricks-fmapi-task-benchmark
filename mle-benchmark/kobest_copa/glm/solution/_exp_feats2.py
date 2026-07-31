import pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from collections import defaultdict
from scipy.sparse import hstack, csr_matrix
import re

RS = 0
tr = pd.read_csv('train.csv'); tr['question'] = tr['question'].str.strip()

def overlap_features(premise, alt):
    pw = set(re.findall(r'\w+', premise)); aw = set(re.findall(r'\w+', alt))
    if not pw or not aw: return 0.0
    return len(pw & aw) / len(pw | aw)

def char_overlap(premise, alt):
    pc = set(premise.replace(' ','')); ac = set(alt.replace(' ',''))
    if not pc or not ac: return 0.0
    return len(pc & ac) / len(pc | ac)

def char_bigram_overlap(premise, alt):
    p = premise.replace(' ',''); a = alt.replace(' ','')
    pb = set(p[i:i+2] for i in range(len(p)-1))
    ab = set(a[i:i+2] for i in range(len(a)-1))
    if not pb or not ab: return 0.0
    return len(pb & ab) / len(pb | ab)

def has_negation(s):
    return 1.0 if re.search(r'안|못|없|아니|지 않|지못|않|못하', s) else 0.0

def ends_past(s):  # past tense markers
    return 1.0 if re.search(r'었|았|였|했다|졌다|냈다', s) else 0.0

def ends_present(s):
    return 1.0 if re.search(r'(?:[^었았였])다\.?$', s) else 0.0

def len_diff(premise, alt):
    return (len(premise) - len(alt)) / 30.0

def share_verb_ending(premise, alt):
    # check if both end with similar final particle
    pm = re.search(r'(.{1,3})다\.?$', premise)
    am = re.search(r'(.{1,3})다\.?$', alt)
    if pm and am:
        return 1.0 if pm.group(1)[-1]==am.group(1)[-1] else 0.0
    return 0.0

def build_pairs(df):
    rows = []
    for _, r in df.iterrows():
        q = r['question']
        for pos, alt, y in [(0, r['alternative_1'], 1 if r['label']==0 else 0),
                           (1, r['alternative_2'], 1 if r['label']==1 else 0)]:
            rows.append({
                'gid': r['id'], 'q': q, 'premise': r['premise'], 'alt': alt, 'pos': pos, 'y': y,
                'ov': overlap_features(r['premise'], alt),
                'cov': char_overlap(r['premise'], alt),
                'cbov': char_bigram_overlap(r['premise'], alt),
                'neg_alt': has_negation(alt),
                'neg_pre': has_negation(r['premise']),
                'past_alt': ends_past(alt),
                'past_pre': ends_past(r['premise']),
                'pres_alt': ends_present(alt),
                'len_diff': len_diff(r['premise'], alt),
                'share_end': share_verb_ending(r['premise'], alt),
            })
    return pd.DataFrame(rows)

pdf = build_pairs(tr)
pdf['text'] = (pdf['q'].astype(str) + ' ' + pdf['premise'].astype(str) + ' ' + pdf['alt'].astype(str))

gids = pdf['gid'].values
unique_gids = pd.unique(gids)
perm = np.random.RandomState(RS).permutation(len(unique_gids))
fold_assign = np.zeros(len(unique_gids), dtype=int)
for k in range(5):
    fold_assign[perm[k::5]] = k
gid_to_fold = dict(zip(unique_gids, fold_assign))
sample_fold = np.array([gid_to_fold[g] for g in gids])
row_for_gid = tr.set_index('id')

def evaluate_fold(k, proba):
    te_idx = np.where(sample_fold == k)[0]
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
    return correct / len(idx_by_gid)

extra_cols = ['ov','cov','cbov','neg_alt','neg_pre','past_alt','past_pre','pres_alt','len_diff','share_end']

# sweep feature subsets
from itertools import combinations
word_vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1,3), min_df=2)
char_vec = TfidfVectorizer(sublinear_tf=True, analyzer='char_wb', ngram_range=(2,4), min_df=2)

def run_with_extras(extra_list, C=4.0):
    accs = []
    for k in range(5):
        tr_idx = np.where(sample_fold != k)[0]
        te_idx = np.where(sample_fold == k)[0]
        wv = word_vec.fit(pdf['text'].iloc[tr_idx])
        cv = char_vec.fit(pdf['text'].iloc[tr_idx])
        Xtr = hstack([wv.transform(pdf['text'].iloc[tr_idx]), cv.transform(pdf['text'].iloc[tr_idx])]).tocsr()
        Xte = hstack([wv.transform(pdf['text'].iloc[te_idx]), cv.transform(pdf['text'].iloc[te_idx])]).tocsr()
        if extra_list:
            etr = pdf[extra_list].iloc[tr_idx].values.astype(float)
            ete = pdf[extra_list].iloc[te_idx].values.astype(float)
            Xtr = hstack([Xtr, csr_matrix(etr)]).tocsr()
            Xte = hstack([Xte, csr_matrix(ete)]).tocsr()
        clf = LogisticRegression(C=C, max_iter=3000, random_state=RS)
        clf.fit(Xtr, pdf['y'].iloc[tr_idx])
        proba = clf.predict_proba(Xte)[:,1]
        accs.append(evaluate_fold(k, proba))
    return np.array(accs)

base = run_with_extras([])
print('ngram only:', base.mean(), base)
full = run_with_extras(extra_cols)
print('all extra:', full.mean(), full)
# add pos
full2 = run_with_extras(extra_cols+['pos'])
print('all+pos:', full2.mean(), full2)

# single feature contributions
for c in extra_cols:
    accs = run_with_extras([c])
    print(f'+{c}: {accs.mean():.4f}')
