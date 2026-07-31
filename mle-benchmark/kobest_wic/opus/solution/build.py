"""Build pair-level feature matrix for KoBEST WiC."""
import numpy as np
import pandas as pd
import re
from feats import (Rep, split_marked, tokens, eojeol_with_target, jac, ovl, window,
                   mask_text, BR)


import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load():
    tr = pd.read_csv(os.path.join(ROOT, 'train.csv'))
    te = pd.read_csv(os.path.join(ROOT, 'test.csv'))
    return tr, te


def build_context_table(tr, te):
    rows = []
    for df, part in ((tr, 'train'), (te, 'test')):
        for k, r in df.iterrows():
            rows.append((part, k, 1, r.word, r.context_1))
            rows.append((part, k, 2, r.word, r.context_2))
    ct = pd.DataFrame(rows, columns=['part', 'row', 'side', 'word', 'text'])
    ct['cid'] = np.arange(len(ct))
    return ct


def pair_index(ct):
    i1 = ct[(ct.side == 1)].set_index(['part', 'row']).cid
    i2 = ct[(ct.side == 2)].set_index(['part', 'row']).cid
    return i1, i2


def make_features(tr, te, ct):
    texts = ct.text.tolist()
    rep_full = Rep(texts)
    rep_w8 = Rep([window(t, 8) for t in texts])
    rep_w20 = Rep([window(t, 20) for t in texts])

    toks = [tokens(t) for t in texts]
    ejs = [eojeol_with_target(t) for t in texts]
    tgts = [split_marked(t)[1] for t in texts]
    lens = np.array([len(t) for t in texts], dtype=float)
    posn = np.array([BR.search(t).start() / max(1, len(t)) if BR.search(t) else 0.5
                     for t in texts])

    i1, i2 = pair_index(ct)
    feats = {}
    order = []
    for part, df in (('train', tr), ('test', te)):
        idx = [(part, k) for k in range(len(df))]
        order.append((part, np.array([i1[x] for x in idx]), np.array([i2[x] for x in idx])))

    out = {}
    for part, a, b in order:
        F = {}
        for tag, rep in (('full', rep_full), ('w8', rep_w8), ('w20', rep_w20)):
            for key in ('char', 'word', 'lsa_char', 'lsa_word'):
                F[f'{tag}_{key}'] = rep.sim(key, a, b)
        F['jac_tok'] = np.array([jac(toks[x], toks[y]) for x, y in zip(a, b)])
        F['ovl_tok'] = np.array([ovl(toks[x], toks[y]) for x, y in zip(a, b)])
        F['ntok_min'] = np.array([min(len(toks[x]), len(toks[y])) for x, y in zip(a, b)], float)
        F['ntok_diff'] = np.array([abs(len(toks[x]) - len(toks[y])) for x, y in zip(a, b)], float)
        F['len_min'] = np.minimum(lens[a], lens[b])
        F['len_max'] = np.maximum(lens[a], lens[b])
        F['len_diff'] = np.abs(lens[a] - lens[b])
        F['pos_diff'] = np.abs(posn[a] - posn[b])
        F['tgt_same'] = np.array([float(tgts[x] == tgts[y]) for x, y in zip(a, b)])
        F['tgt_len'] = np.array([len(tgts[x]) for x in a], float)
        post = [ejs[x][2] for x in range(len(texts))]
        pre = [ejs[x][1] for x in range(len(texts))]
        F['post_same'] = np.array([float(post[x] == post[y]) for x, y in zip(a, b)])
        F['post_empty'] = np.array([float(post[x] == '') + float(post[y] == '')
                                    for x, y in zip(a, b)])
        F['pre_same'] = np.array([float(pre[x] == pre[y]) for x, y in zip(a, b)])
        F['pre_nonempty'] = np.array([float(pre[x] != '') + float(pre[y] != '')
                                      for x, y in zip(a, b)])
        F['post1_same'] = np.array([float(post[x][:1] == post[y][:1]) for x, y in zip(a, b)])
        F['ej_jac'] = np.array([jac(list(ejs[x][0]), list(ejs[y][0])) for x, y in zip(a, b)])
        out[part] = pd.DataFrame(F)
    return out, rep_full, rep_w20


def word_features(tr, te):
    """Word-level features (no label leakage: OOF prior handled separately)."""
    cnt = pd.concat([tr.word, te.word]).value_counts()
    res = {}
    for part, df in (('train', tr), ('test', te)):
        F = pd.DataFrame(index=range(len(df)))
        F['w_count'] = df.word.map(cnt).values
        F['w_len'] = df.word.str.len().values
        F['w_in_train'] = df.word.isin(set(tr.word)).astype(float).values
        F['w_train_count'] = df.word.map(tr.word.value_counts()).fillna(0).values
        res[part] = F.reset_index(drop=True)
    return res
