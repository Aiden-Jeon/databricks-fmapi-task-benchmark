"""Content (text-similarity) features for KoBEST WiC pairs."""
import numpy as np, pandas as pd, re, os, pickle
from collections import Counter
import build as B
from feats import Rep, window, tokens, jac, ovl, eojeol_with_target, split_marked, BR
import feats2 as F2

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'content_cache.pkl')


def build_content(use_cache=True):
    if use_cache and os.path.exists(CACHE):
        with open(CACHE, 'rb') as f:
            return pickle.load(f)
    tr, te = B.load()
    ct = B.build_context_table(tr, te)
    texts = ct.text.tolist()
    i1, i2 = B.pair_index(ct)
    N = len(texts)

    rep_full = Rep(texts)
    rep_w20 = Rep([window(t, 20) for t in texts])

    vocab, E = F2.build_ppmi_emb(texts, dim=150)
    df_cnt = Counter()
    for t in texts:
        for w in set(tokens(t)):
            df_cnt[w] += 1
    idf = {w: np.log(N / (1 + c)) for w, c in df_cnt.items()}
    rep_full.reps['ppmi'] = F2.emb_context_vectors(texts, vocab, E, idf)
    rep_w20.reps['ppmi'] = F2.emb_context_vectors([window(t, 20) for t in texts], vocab, E, idf)

    SIMKEYS = [('full', k) for k in ('char', 'word', 'lsa_char', 'lsa_word', 'ppmi')] + \
              [('w20', k) for k in ('char', 'lsa_char', 'ppmi')]
    REPS = {'full': rep_full, 'w20': rep_w20}

    def getR(tag, key):
        return REPS[tag].reps[key]

    def simvals(tag, key, a, b):
        R = getR(tag, key)
        if hasattr(R, 'multiply'):
            return np.asarray(R[a].multiply(R[b]).sum(axis=1)).ravel()
        return np.einsum('ij,ij->i', R[a], R[b])

    wgroups = {w: np.array(g.cid.tolist()) for w, g in ct.groupby('word')}
    wstats = {}
    for tag, key in SIMKEYS:
        R = getR(tag, key)
        st = {}
        for w, ids in wgroups.items():
            if len(ids) < 3:
                st[w] = (0.0, 1.0, np.array([]))
                continue
            A = R[ids]
            S = (A @ A.T)
            if hasattr(S, 'toarray'):
                S = S.toarray()
            iu = np.triu_indices(len(ids), 1)
            v = np.asarray(S)[iu]
            st[w] = (v.mean(), v.std() + 1e-9, np.sort(v))
        wstats[(tag, key)] = st

    nxt1 = [F2.clean(F2.next_eojeol(t, 1)) for t in texts]
    nxt0 = [F2.clean(eojeol_with_target(t)[2]) for t in texts]
    prv1 = [F2.clean(F2.prev_eojeol(t, 1)) for t in texts]
    nxt1s = [t[:2] for t in nxt1]
    prv1s = [t[:2] for t in prv1]
    tgts = [split_marked(t)[1] for t in texts]
    toks = [tokens(t) for t in texts]
    lens = np.array([len(t) for t in texts], float)
    posn = np.array([(BR.search(t).start() / max(1, len(t))) if BR.search(t) else .5
                     for t in texts])
    has_ell = np.array([float('…' in t or '...' in t) for t in texts])
    has_hanja = np.array([float(bool(re.search(r'[\u4e00-\u9fff]', t))) for t in texts])
    ends_da = np.array([float(t.rstrip().endswith('다.')) for t in texts])
    ntk = np.array([len(x) for x in toks], float)
    vmark = np.array([[float(v in nxt0[i] + nxt1[i][:2]) for v in F2.VERB_MARK]
                      for i in range(N)])
    cnt_all = pd.concat([tr.word, te.word]).value_counts()

    def pair_feats(words, a, b):
        F = {}
        for tag, key in SIMKEYS:
            v = simvals(tag, key, a, b)
            F[f's_{tag}_{key}'] = v
            st = wstats[(tag, key)]
            mu = np.array([st[w][0] for w in words])
            sd = np.array([st[w][1] for w in words])
            F[f'z_{tag}_{key}'] = (v - mu) / sd
            F[f'p_{tag}_{key}'] = np.array(
                [(np.searchsorted(st[w][2], vv) / max(1, len(st[w][2]))) if len(st[w][2])
                 else 0.5 for w, vv in zip(words, v)])
        F['jac_tok'] = np.array([jac(toks[x], toks[z]) for x, z in zip(a, b)])
        F['ovl_tok'] = np.array([ovl(toks[x], toks[z]) for x, z in zip(a, b)])
        F['len_min'] = np.minimum(lens[a], lens[b])
        F['len_max'] = np.maximum(lens[a], lens[b])
        F['len_diff'] = np.abs(lens[a] - lens[b])
        F['ntk_min'] = np.minimum(ntk[a], ntk[b])
        F['ntk_diff'] = np.abs(ntk[a] - ntk[b])
        F['pos_diff'] = np.abs(posn[a] - posn[b])
        F['pos_min'] = np.minimum(posn[a], posn[b])
        F['ell_sum'] = has_ell[a] + has_ell[b]
        F['ell_diff'] = np.abs(has_ell[a] - has_ell[b])
        F['hanja_sum'] = has_hanja[a] + has_hanja[b]
        F['hanja_diff'] = np.abs(has_hanja[a] - has_hanja[b])
        F['da_sum'] = ends_da[a] + ends_da[b]
        F['da_diff'] = np.abs(ends_da[a] - ends_da[b])
        F['tgt_same'] = np.array([float(tgts[x] == tgts[z]) for x, z in zip(a, b)])
        F['tgt_len'] = np.array([len(tgts[x]) for x in a], float)
        F['nxt0_same'] = np.array([float(nxt0[x] == nxt0[z]) for x, z in zip(a, b)])
        F['nxt0_1same'] = np.array([float(nxt0[x][:1] == nxt0[z][:1]) for x, z in zip(a, b)])
        F['nxt0_empty'] = np.array([float(nxt0[x] == '') + float(nxt0[z] == '')
                                    for x, z in zip(a, b)])
        F['nxt1_same'] = np.array([float(nxt1s[x] == nxt1s[z] and nxt1s[x] != '')
                                   for x, z in zip(a, b)])
        F['prv1_same'] = np.array([float(prv1s[x] == prv1s[z] and prv1s[x] != '')
                                   for x, z in zip(a, b)])
        F['prv_empty'] = np.array([float(prv1[x] == '') + float(prv1[z] == '')
                                   for x, z in zip(a, b)])
        F['vmark_agree'] = (vmark[a] * vmark[b]).sum(axis=1)
        F['vmark_dis'] = np.abs(vmark[a] - vmark[b]).sum(axis=1)
        for j, v in enumerate(F2.VERB_MARK):
            F[f'vm_{v}'] = vmark[a][:, j] + vmark[b][:, j]
        F['w_count'] = np.array([cnt_all[w] for w in words], float)
        F['w_len'] = np.array([len(w) for w in words], float)
        return pd.DataFrame(F)

    a_tr = np.array([i1[('train', k)] for k in range(len(tr))])
    b_tr = np.array([i2[('train', k)] for k in range(len(tr))])
    a_te = np.array([i1[('test', k)] for k in range(len(te))])
    b_te = np.array([i2[('test', k)] for k in range(len(te))])
    Xtr = pair_feats(tr.word.values, a_tr, b_tr)
    Xte = pair_feats(te.word.values, a_te, b_te)
    out = (tr, te, Xtr, Xte)
    with open(CACHE, 'wb') as f:
        pickle.dump(out, f)
    return out
