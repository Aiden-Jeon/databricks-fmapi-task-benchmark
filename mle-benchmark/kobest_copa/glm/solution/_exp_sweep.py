import pandas as pd, numpy as np, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from scipy.sparse import hstack, csr_matrix
RS = 0

tr = pd.read_csv('train.csv'); tr['question'] = tr['question'].str.strip()

def cbov(p, a):
    p = p.replace(' ', ''); a = a.replace(' ', '')
    pb = set(p[i:i+2] for i in range(len(p)-1)); ab = set(a[i:i+2] for i in range(len(a)-1))
    return len(pb & ab) / len(pb | ab) if pb and ab else 0.0
def ov(p, a):
    pw = set(re.findall(r'\w+', p)); aw = set(re.findall(r'\w+', a))
    return len(pw & aw) / len(pw | aw) if pw and aw else 0.0
def hasn(s): return 1.0 if re.search(r'안|못|없|아니|않', s) else 0.0
def past(s): return 1.0 if re.search(r'었|았|였|졌다|했다', s) else 0.0
def present(s): return 1.0 if re.search(r'[^었았였]다\.?$', s) else 0.0
def lend(p, a): return (len(p) - len(a)) / 30.0

rows = []
for _, r in tr.iterrows():
    q = r['question']
    for pos, alt, y in [(0, r['alternative_1'], 1 if r['label'] == 0 else 0),
                        (1, r['alternative_2'], 1 if r['label'] == 1 else 0)]:
        rows.append({'gid': r['id'], 'q': q, 'premise': r['premise'], 'alt': alt, 'pos': pos, 'y': y,
            'ov': ov(r['premise'], alt), 'cbov': cbov(r['premise'], alt),
            'neg_alt': hasn(alt), 'neg_pre': hasn(r['premise']),
            'past_alt': past(alt), 'past_pre': past(r['premise']),
            'pres_alt': present(alt), 'len_diff': lend(r['premise'], alt)})
pdf = pd.DataFrame(rows)
pdf['text'] = (pdf['q'].astype(str) + ' ' + pdf['premise'].astype(str) + ' ' + pdf['alt'].astype(str))

gids = pdf['gid'].values; ug = pd.unique(gids)
perm = np.random.RandomState(RS).permutation(len(ug))
fa = np.zeros(len(ug), dtype=int)
for k in range(5): fa[perm[k::5]] = k
g2f = dict(zip(ug, fa)); sf = np.array([g2f[g] for g in gids]); rfg = tr.set_index('id')

def evf(k, pr):
    ti = np.where(sf == k)[0]; tg = gids[ti]; d = defaultdict(list)
    for i, g in enumerate(tg): d[g].append(i)
    c = 0
    for g, idx in d.items():
        if (0 if pr[idx[0]] >= pr[idx[1]] else 1) == rfg.loc[g, 'label']: c += 1
    return c / len(d)

extras = ['ov', 'cbov', 'neg_alt', 'neg_pre', 'past_alt', 'past_pre', 'pres_alt', 'len_diff']

def build_X(tri, tei, ngram_w=(1,3), ngram_c=(2,4), use_extras=True, use_pos=False, scale_extras=False):
    wv = TfidfVectorizer(sublinear_tf=True, ngram_range=ngram_w, min_df=2).fit(pdf['text'].iloc[tri])
    cv = TfidfVectorizer(sublinear_tf=True, analyzer='char_wb', ngram_range=ngram_c, min_df=2).fit(pdf['text'].iloc[tri])
    Xtr = hstack([wv.transform(pdf['text'].iloc[tri]), cv.transform(pdf['text'].iloc[tri])]).tocsr()
    Xte = hstack([wv.transform(pdf['text'].iloc[tei]), cv.transform(pdf['text'].iloc[tei])]).tocsr()
    if use_extras or use_pos:
        cols = list(extras) if use_extras else []
        if use_pos: cols = cols + ['pos']
        Etr = pdf[cols].iloc[tri].values.astype(float)
        Ete = pdf[cols].iloc[tei].values.astype(float)
        if scale_extras:
            sc = StandardScaler(with_mean=False).fit(Etr)
            Etr = sc.transform(Etr); Ete = sc.transform(Ete)
        Xtr = hstack([Xtr, csr_matrix(Etr)]).tocsr()
        Xte = hstack([Xte, csr_matrix(Ete)]).tocsr()
    return Xtr, Xte

def run(cfg_name, C=4.0, **kwargs):
    accs = []
    for k in range(5):
        tri = np.where(sf != k)[0]; tei = np.where(sf == k)[0]
        Xtr, Xte = build_X(tri, tei, **kwargs)
        c = LogisticRegression(C=C, max_iter=3000, random_state=RS); c.fit(Xtr, pdf['y'].iloc[tri])
        accs.append(evf(k, c.predict_proba(Xte)[:, 1]))
    print(f"{cfg_name}: {np.mean(accs):.4f} {accs}", flush=True)
    return np.array(accs)

run('base_ng', use_extras=False)
run('extras', use_extras=True)
run('extras+pos', use_extras=True, use_pos=True)
run('extras_scaled', use_extras=True, scale_extras=True)
run('pos_only', use_extras=False, use_pos=True)
run('cw12', use_extras=True, ngram_c=(1,4))
run('cw35', use_extras=True, ngram_c=(3,5))
run('C2', use_extras=True, C=2.0)
run('C8', use_extras=True, C=8.0)
run('C1', use_extras=True, C=1.0)
print('DONE', flush=True)
