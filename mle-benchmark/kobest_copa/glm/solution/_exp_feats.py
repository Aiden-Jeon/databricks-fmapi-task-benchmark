import pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from collections import defaultdict
import re

RS = 0
tr = pd.read_csv('train.csv'); tr['question'] = tr['question'].str.strip()

# ---- Handcrafted features ----
# Korean often uses spaces; words are decent tokens. Char n-grams catch endings.
# Key insight for COPA: 
#   원인(cause): premise is EFFECT, alt is CAUSE. alt should logically precede/cause premise.
#   결과(effect): premise is CAUSE, alt is EFFECT. alt should logically follow from premise.
# Hard to model causality direction without semantics. Use lexical/surface features.

def char_jamo(s):
    # simple: just return the string; char_wb handles it
    return s

def overlap_features(premise, alt):
    pw = set(re.findall(r'\w+', premise))
    aw = set(re.findall(r'\w+', alt))
    if not pw or not aw: return 0.0
    return len(pw & aw) / len(pw | aw)

def char_overlap(premise, alt):
    pc = set(premise); ac = set(alt)
    if not pc or not ac: return 0.0
    return len(pc & ac) / len(pc | ac)

def has_negation(s):
    return 1.0 if re.search(r'안|못|없|아니|지 않|지못|지 않', s) else 0.0

def starts_with_action(s):
    # heuristic: ends with verb-ending 다/었다/한다/졌다 etc
    return 1.0 if re.search(r'다\.?$', s) else 0.0

# Build pair dataset with rich text + features
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
                'neg_alt': has_negation(alt),
                'neg_pre': has_negation(r['premise']),
            })
    return pd.DataFrame(rows)

pdf = build_pairs(tr)
pdf['text'] = (pdf['q'].astype(str) + ' ' + pdf['premise'].astype(str) + ' ' + pdf['alt'].astype(str))

# group fold
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

def evaluate_all(proba_per_fold):
    all_preds = {}
    for k in range(5):
        te_idx = np.where(sample_fold == k)[0]
        proba = proba_per_fold[k]
        test_gids = gids[te_idx]
        idx_by_gid = defaultdict(list)
        for i, g in enumerate(test_gids):
            idx_by_gid[g].append(i)
        for g, idxs in idx_by_gid.items():
            p1 = proba[idxs[0]]; p2 = proba[idxs[1]]
            all_preds[g] = 0 if p1 >= p2 else 1
    correct = sum(1 for g, p in all_preds.items() if p == row_for_gid.loc[g, 'label'])
    return correct / len(all_preds)

# Try: combine n-gram proba with handcrafted features as a second-stage classifier
from scipy.sparse import hstack, csr_matrix

word_vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1,3), min_df=2)
char_vec = TfidfVectorizer(sublinear_tf=True, analyzer='char_wb', ngram_range=(2,4), min_df=2)

# First get n-gram features via CV, then add handcrafted
accs_ng = []
accs_full = []
extra_cols = ['ov','cov','neg_alt','neg_pre']
for k in range(5):
    tr_idx = np.where(sample_fold != k)[0]
    te_idx = np.where(sample_fold == k)[0]
    wv = word_vec.fit(pdf['text'].iloc[tr_idx])
    cv = char_vec.fit(pdf['text'].iloc[tr_idx])
    Xtr_w = wv.transform(pdf['text'].iloc[tr_idx])
    Xte_w = wv.transform(pdf['text'].iloc[te_idx])
    Xtr_c = cv.transform(pdf['text'].iloc[tr_idx])
    Xte_c = cv.transform(pdf['text'].iloc[te_idx])
    from scipy.sparse import hstack
    Xtr = hstack([Xtr_w, Xtr_c]).tocsr()
    Xte = hstack([Xte_w, Xte_c]).tocsr()
    clf = LogisticRegression(C=4.0, max_iter=3000, random_state=RS)
    clf.fit(Xtr, pdf['y'].iloc[tr_idx])
    proba = clf.predict_proba(Xte)[:,1]
    accs_ng.append(evaluate_fold(k, proba))
    
    # add extra
    extra_tr = pdf[extra_cols].iloc[tr_idx].values.astype(float)
    extra_te = pdf[extra_cols].iloc[te_idx].values.astype(float)
    Xtr2 = hstack([Xtr, csr_matrix(extra_tr)]).tocsr()
    Xte2 = hstack([Xte, csr_matrix(extra_te)]).tocsr()
    clf2 = LogisticRegression(C=4.0, max_iter=3000, random_state=RS)
    clf2.fit(Xtr2, pdf['y'].iloc[tr_idx])
    proba2 = clf2.predict_proba(Xte2)[:,1]
    accs_full.append(evaluate_fold(k, proba2))

print('ngram only:', np.mean(accs_ng), accs_ng)
print('ngram + extra:', np.mean(accs_full), accs_full)

# Check: do handcrafted features alone have signal?
from sklearn.metrics import roc_auc_score
for c in extra_cols:
    vals = pdf[c].values
    try:
        auc = roc_auc_score(pdf['y'], vals)
        print(f'{c} AUC={auc:.3f}')
    except: pass
