"""KorQuAD char-F1: candidate-span ranking with gradient boosted trees.

usage: python run.py build   # build training matrices  -> work/
       python run.py train   # fit model, validate      -> work/
       python run.py predict # write outputs/submission.csv
"""
import os
import sys
import time
import pickle
import collections
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import (ContextInfo, build_df, question_features, char_f1, N_FEAT)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(ROOT, 'work')
OUT = os.path.join(ROOT, 'outputs')
os.makedirs(WORK, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

SEED = 0
N_NEG = 40
POS_TH = 0.35


def load():
    tr = pd.read_csv(os.path.join(ROOT, 'train.csv'))
    te = pd.read_csv(os.path.join(ROOT, 'test.csv'))
    tr['answer'] = tr['answer'].astype(str)
    return tr, te


def get_df(tr):
    p = os.path.join(WORK, 'df.pkl')
    if os.path.exists(p):
        with open(p, 'rb') as f:
            return pickle.load(f)
    ctxs = tr.context.drop_duplicates().tolist()
    dfmap, ndoc = build_df(ctxs)
    maxidf = float(np.log((ndoc + 1.0) / 1.0))
    with open(p, 'wb') as f:
        pickle.dump((dfmap, ndoc, maxidf), f)
    return dfmap, ndoc, maxidf


def doc_of(i):
    return str(i).split('-')[0]


def build_training(tr, dfmap, ndoc, maxidf, contexts, tag, n_neg=N_NEG, seed=SEED,
                   hard=None):
    rng = np.random.default_rng(seed)
    sub = tr[tr.context.isin(set(contexts))]
    Xs, ys, qg = [], [], []
    t0 = time.time()
    for ci_i, (ctx, g) in enumerate(sub.groupby('context', sort=False)):
        ci = ContextInfo(ctx, dfmap, ndoc)
        allidx = np.arange(ci.n)
        for row in g.itertuples():
            gold = row.answer
            gi = ctx.find(gold)
            gj = gi + len(gold)
            ov = np.where((ci.ends > gi) & (ci.starts < gj))[0]
            f1o = np.array([char_f1(ctx[ci.starts[k]:ci.ends[k]], gold) for k in ov],
                           dtype=np.float32)
            keep = f1o >= POS_TH
            pos, posf = ov[keep], f1o[keep]
            if len(pos) > 8:
                o = np.argsort(-posf)[:8]
                pos, posf = pos[o], posf[o]
            mask = np.ones(ci.n, dtype=bool)
            mask[pos] = False
            pool = allidx[mask]
            k = min(n_neg, len(pool))
            neg = rng.choice(pool, size=k, replace=False) if k else pool
            if hard is not None:
                hn = hard.get(row.id)
                if hn is not None:
                    hn = np.array([h for h in hn if mask[h]], dtype=np.int64)
                    neg = np.unique(np.concatenate([neg, hn]))
            negf = np.array([char_f1(ctx[ci.starts[k2]:ci.ends[k2]], gold) for k2 in neg],
                            dtype=np.float32)
            sel = np.concatenate([pos, neg]).astype(np.int64)
            y = np.concatenate([posf, negf]).astype(np.float32)
            F = np.hstack([ci.static[sel],
                           question_features(ci, row.question, dfmap, maxidf, sel)])
            Xs.append(F.astype(np.float32))
            ys.append(y)
            qg.append(np.full(len(sel), len(qg), dtype=np.int32))
        if ci_i % 500 == 0:
            print(f'  ctx {ci_i} rows={sum(len(a) for a in ys)} '
                  f'{time.time()-t0:.0f}s', flush=True)
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    grp = np.concatenate(qg)
    np.save(os.path.join(WORK, f'X_{tag}.npy'), X)
    np.save(os.path.join(WORK, f'y_{tag}.npy'), y)
    np.save(os.path.join(WORK, f'g_{tag}.npy'), grp)
    print('built', tag, X.shape, 'in', time.time() - t0, 's')
    return X, y, grp


def predict_frame(frame, model, dfmap, ndoc, maxidf, gold_map=None, topk_hard=0,
                  batch_q=64, log_every=200):
    """score every candidate of every question, return best span per id"""
    res = {}
    hard = {}
    t0 = time.time()
    cnt = 0
    for ctx, g in frame.groupby('context', sort=False):
        ci = ContextInfo(ctx, dfmap, ndoc)
        Fs, ids, offs = [], [], []
        for row in g.itertuples():
            F = np.hstack([ci.static,
                           question_features(ci, row.question, dfmap, maxidf, None)])
            Fs.append(F.astype(np.float32))
            ids.append(row.id)
        big = np.vstack(Fs)
        sc = model.predict(big)
        p = 0
        for qi, qid in enumerate(ids):
            s = sc[p:p + ci.n]
            p += ci.n
            b = int(np.argmax(s))
            res[qid] = ctx[ci.starts[b]:ci.ends[b]]
            if topk_hard:
                idx = np.argpartition(-s, min(topk_hard, ci.n - 1))[:topk_hard]
                hard[qid] = idx.tolist()
        cnt += len(ids)
        if log_every and (cnt // log_every) != ((cnt - len(ids)) // log_every):
            print(f'  scored {cnt} q  {time.time()-t0:.0f}s', flush=True)
    return res, hard


def evaluate(res, frame):
    f1 = [char_f1(res[r.id], r.answer) for r in frame.itertuples()]
    em = [float(res[r.id] == r.answer) for r in frame.itertuples()]
    return float(np.mean(f1)), float(np.mean(em))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    tr, te = load()
    dfmap, ndoc, maxidf = get_df(tr)
    docs = tr.id.map(doc_of)
    udocs = np.array(sorted(docs.unique()))
    rng = np.random.default_rng(42)
    rng.shuffle(udocs)
    n_val = max(1, int(0.06 * len(udocs)))
    val_docs = set(udocs[:n_val].tolist())
    va = tr[docs.isin(val_docs)]
    trn = tr[~docs.isin(val_docs)]
    print('train q', len(trn), 'val q', len(va), 'train ctx', trn.context.nunique())

    if mode in ('build', 'all'):
        ctxs = trn.context.drop_duplicates().tolist()
        lim = int(os.environ.get('NCTX', len(ctxs)))
        ctxs = ctxs[:lim]
        nsh = int(os.environ.get('NSHARD', 1))
        sh = int(os.environ.get('SHARD', 0))
        tag = 'trn' if nsh == 1 else f'trn{sh}'
        build_training(trn, dfmap, ndoc, maxidf, ctxs[sh::nsh], tag)
    if mode == 'merge':
        nsh = int(os.environ.get('NSHARD', 3))
        X = np.vstack([np.load(os.path.join(WORK, f'X_trn{i}.npy')) for i in range(nsh)])
        y = np.concatenate([np.load(os.path.join(WORK, f'y_trn{i}.npy')) for i in range(nsh)])
        np.save(os.path.join(WORK, 'X_trn.npy'), X)
        np.save(os.path.join(WORK, 'y_trn.npy'), y)
        print('merged', X.shape)
    if mode in ('train', 'all'):
        from sklearn.ensemble import HistGradientBoostingRegressor
        X = np.load(os.path.join(WORK, 'X_trn.npy'))
        y = np.load(os.path.join(WORK, 'y_trn.npy'))
        print('X', X.shape)
        m = HistGradientBoostingRegressor(
            max_iter=int(os.environ.get('ITERS', 400)), learning_rate=0.1,
            max_leaf_nodes=63, min_samples_leaf=40, l2_regularization=1.0,
            early_stopping=False, random_state=SEED)
        t0 = time.time()
        m.fit(X, y)
        print('fit', time.time() - t0)
        with open(os.path.join(WORK, 'model.pkl'), 'wb') as f:
            pickle.dump(m, f)
    if mode == 'mine':
        with open(os.path.join(WORK, 'model.pkl'), 'rb') as f:
            m = pickle.load(f)
        nsh = int(os.environ.get('NSHARD', 1))
        sh = int(os.environ.get('SHARD', 0))
        ctxs = trn.context.drop_duplicates().tolist()[sh::nsh]
        part = trn[trn.context.isin(set(ctxs))]
        _, hard = predict_frame(part, m, dfmap, ndoc, maxidf, topk_hard=12,
                                log_every=2000)
        with open(os.path.join(WORK, f'hard{sh}.pkl'), 'wb') as f:
            pickle.dump(hard, f)
        print('mined', len(hard))
    if mode == 'build2':
        nsh = int(os.environ.get('NSHARD', 1))
        sh = int(os.environ.get('SHARD', 0))
        hard = {}
        for i in range(int(os.environ.get('NHARD', 3))):
            p = os.path.join(WORK, f'hard{i}.pkl')
            if os.path.exists(p):
                with open(p, 'rb') as f:
                    hard.update(pickle.load(f))
        print('hard negs for', len(hard), 'questions')
        ctxs = trn.context.drop_duplicates().tolist()
        build_training(trn, dfmap, ndoc, maxidf, ctxs[sh::nsh],
                       'trn' if nsh == 1 else f'trn{sh}',
                       n_neg=int(os.environ.get('NNEG', N_NEG)), hard=hard)
    if mode in ('val', 'all'):
        with open(os.path.join(WORK, 'model.pkl'), 'rb') as f:
            m = pickle.load(f)
        vs = va if os.environ.get('NVAL') is None else va.head(int(os.environ['NVAL']))
        res, _ = predict_frame(vs, m, dfmap, ndoc, maxidf)
        f1, em = evaluate(res, vs)
        print(f'VAL char-F1 {f1:.4f}  EM {em:.4f}  (n={len(vs)})')
    if mode in ('predict', 'all'):
        with open(os.path.join(WORK, 'model.pkl'), 'rb') as f:
            m = pickle.load(f)
        res, _ = predict_frame(te, m, dfmap, ndoc, maxidf)
        sub = pd.DataFrame({'id': te.id, 'answer': [res[i] for i in te.id]})
        sub.to_csv(os.path.join(OUT, 'submission.csv'), index=False)
        print('wrote', os.path.join(OUT, 'submission.csv'), sub.shape)


if __name__ == '__main__':
    main()
