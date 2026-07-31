"""End-to-end training + inference. Usage: python run.py [stage]

stages: a (candidate feature sampling) -> b (cross-fit scoring) -> c (test predict)
Run `python run.py all` to do everything.
"""
import os, sys, time, pickle
import numpy as np
import pandas as pd
import multiprocessing as mp
import qa_lib as Q

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)
os.makedirs(os.path.join(TASK, "outputs"), exist_ok=True)
SEED = 0
NEG_PER_EX = 45
NFOLD = 2

_G = {}


def _init(idf, models, ssim):
    """initializer for spawned workers"""
    _G["idf"] = idf
    _G["models"] = models
    _G["ssim"] = ssim


def make_pool(nproc, idf, models=None, ssim=None):
    ctx = mp.get_context("spawn")
    return ctx.Pool(nproc, initializer=_init, initargs=(idf, models, ssim))


def init_ssim(tr, te):
    p = os.path.join(CACHE, "ssim.pkl")
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    ss = Q.SentSim.fit(list(tr.context) + list(te.context))
    with open(p, "wb") as f:
        pickle.dump(ss, f)
    return ss


def load_data():
    tr = pd.read_csv(os.path.join(TASK, "train.csv"))
    te = pd.read_csv(os.path.join(TASK, "test.csv"))
    tr["ans"] = tr["answer"].fillna("")
    return tr, te


def init_idf(tr, te):
    p = os.path.join(CACHE, "idf.pkl")
    if os.path.exists(p):
        with open(p, "rb") as f:
            df, N = pickle.load(f)
    else:
        df, N = Q.build_idf(list(tr.context) + list(te.context))
        with open(p, "wb") as f:
            pickle.dump((df, N), f)
    return Q.IDF(df, N)


# ------------------------------------------------------------------ pass A
def _work_a(arg):
    i, ctx, q, ans = arg
    idf = _G["idf"]
    rng = np.random.RandomState(i + 1234)
    cands, X, ex = Q.example_features(ctx, q, idf, ssim=_G.get("ssim"))
    if ans:
        y = np.array([Q.char_f1(ctx[s:e], ans) for s, e in cands], dtype=np.float32)
    else:
        y = np.zeros(len(cands), dtype=np.float32)
    def samp(idx, k):
        if len(idx) <= k:
            return idx
        return rng.choice(idx, k, replace=False)

    hi = np.flatnonzero(y > 0.45)
    mid = np.flatnonzero((y > 0.05) & (y <= 0.45))
    lo = np.flatnonzero(y <= 0.05)
    idxs = np.concatenate([samp(hi, 25), samp(mid, 20), samp(lo, NEG_PER_EX)]).astype(np.int64)
    return X[idxs], y[idxs], np.full(len(idxs), i, dtype=np.int32)


def pass_a(tr, idf, ssim, nproc=4):
    out = os.path.join(CACHE, "stage1_rows.npz")
    if os.path.exists(out):
        d = np.load(out)
        return d["X"], d["y"], d["g"]
    args = [(i, r.context, r.question, r.ans) for i, r in enumerate(tr.itertuples())]
    t = time.time()
    Xs, ys, gs = [], [], []
    folds_a = np.load(os.path.join(CACHE, "folds_a.npy"))
    with make_pool(nproc, idf, None, ssim) as pool:
        for n, (X, y, g) in enumerate(pool.imap(_work_a, args, chunksize=32)):
            Xs.append(X); ys.append(y); gs.append(g)
            if n % 2000 == 0:
                print("  A %d/%d %.0fs" % (n, len(args), time.time() - t), flush=True)
    X = np.concatenate(Xs); y = np.concatenate(ys); g = np.concatenate(gs)
    np.savez_compressed(out, X=X, y=y, g=g)
    print("pass A done", X.shape, "%.0fs" % (time.time() - t))
    return X, y, g


def train_stage1(X, y, g, folds):
    from sklearn.ensemble import HistGradientBoostingRegressor
    models = []
    for f in range(NFOLD):
        m = HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.07, max_leaf_nodes=63,
            min_samples_leaf=40, l2_regularization=1.0, early_stopping=False,
            random_state=SEED)
        sel = folds[g] != f
        t = time.time()
        m.fit(X[sel], y[sel])
        print("  stage1 fold%d fit %.0fs on %d rows" % (f, time.time() - t, sel.sum()), flush=True)
        models.append(m)
    return models


# ------------------------------------------------------------------ pass B/C
def _score_example(ctx, q, ans, models, idf, topn=8, ssim=None):
    cands, X, ex = Q.example_features(ctx, q, idf, ssim=ssim)
    if len(cands) == 0:
        return None
    s = np.zeros(len(cands))
    for m in models:
        s += m.predict(X)
    s /= len(models)
    order = np.argsort(-s)
    top = order[:topn]
    t0 = top[0]
    s0, e0 = cands[t0]
    txt = ctx[s0:e0]
    # competition: best score among candidates not overlapping top1
    nonov = [k for k in order[:200] if cands[k][1] <= s0 or cands[k][0] >= e0]
    comp = s[nonov[0]] if nonov else 0.0
    # best score in a different sentence
    feats = [s[k] for k in top] + [0.0] * max(0, topn - len(top))
    scores_sorted = s[order]
    f = list(feats[:topn]) + [
        float(s.max()), float(np.mean(scores_sorted[:20])), float(np.mean(scores_sorted[:50])),
        float(s.mean()), float(np.std(s)), float(comp), float(s.max() - comp),
        float((s > 0.5).sum()), float((s > 0.3).sum()), float((s > 0.15).sum()),
        float(len(cands)),
        ex["cov_frac"], ex["covw_frac"], float(ex["n_units"]), float(ex["qlen"]),
        float(ex["clen"]), ex["best_sent"], float(ex["n_sents"]),
        ex["qcov"], ex["qcov4"], ex["second_sent"], ex["best_sent"] - ex["second_sent"],
        ex["cos_max"], ex["cos_mean"],
        float(len(txt)), float(txt.count(" ") + 1),
        float(bool(Q.RE_DIGIT.search(txt))), float(bool(Q.RE_DATE.search(txt))),
        float(s0) / max(len(ctx), 1),
    ] + ex["qt"]
    rec = dict(feat=np.asarray(f, dtype=np.float32), text=txt, top1=float(s[t0]))
    if ans is not None:
        rec["f1"] = Q.char_f1(txt, ans)
        rec["unans"] = 1.0 if ans == "" else 0.0
        # oracle among top candidates
        rec["f1_top"] = max(Q.char_f1(ctx[cands[k][0]:cands[k][1]], ans) for k in top)
    return rec


def _work_b(arg):
    i, ctx, q, ans, fold = arg
    # model `fold` was trained on every fold EXCEPT `fold`; use it for this example
    models = [_G["models"][fold]]
    r = _score_example(ctx, q, ans, models, _G["idf"], ssim=_G.get("ssim"))
    r["i"] = i
    return r


def pass_b(tr, idf, models, folds, ssim, nproc=4):
    out = os.path.join(CACHE, "stage2_rows.pkl")
    if os.path.exists(out):
        with open(out, "rb") as f:
            return pickle.load(f)
    args = [(i, r.context, r.question, r.ans, int(folds[i]))
            for i, r in enumerate(tr.itertuples())]
    t = time.time()
    recs = []
    with make_pool(nproc, idf, models, ssim) as pool:
        for n, r in enumerate(pool.imap(_work_b, args, chunksize=16)):
            recs.append(r)
            if n % 2000 == 0:
                print("  B %d/%d %.0fs" % (n, len(args), time.time() - t), flush=True)
    recs.sort(key=lambda r: r["i"])
    with open(out, "wb") as f:
        pickle.dump(recs, f)
    print("pass B done %.0fs" % (time.time() - t))
    return recs


def _work_c(arg):
    i, ctx, q = arg
    r = _score_example(ctx, q, None, _G["models"], _G["idf"], ssim=_G.get("ssim"))
    r["i"] = i
    return r


def pass_c(te, idf, models, ssim, nproc=4):
    out = os.path.join(CACHE, "test_rows.pkl")
    if os.path.exists(out):
        with open(out, "rb") as f:
            return pickle.load(f)
    args = [(i, r.context, r.question) for i, r in enumerate(te.itertuples())]
    t = time.time()
    recs = []
    with make_pool(nproc, idf, models, ssim) as pool:
        for n, r in enumerate(pool.imap(_work_c, args, chunksize=16)):
            recs.append(r)
            if n % 1000 == 0:
                print("  C %d/%d %.0fs" % (n, len(args), time.time() - t), flush=True)
    recs.sort(key=lambda r: r["i"])
    with open(out, "wb") as f:
        pickle.dump(recs, f)
    print("pass C done %.0fs" % (time.time() - t))
    return recs


# ------------------------------------------------------------------ main
def main(stage="all"):
    tr, te = load_data()
    idf = init_idf(tr, te)
    ssim = init_ssim(tr, te)
    rng = np.random.RandomState(SEED)
    folds = rng.randint(0, NFOLD, size=len(tr))
    np.save(os.path.join(CACHE, "folds.npy"), folds)
    # fold ids aligned with the answerable-only subset used for stage 1
    folds_a = folds[(tr.ans != "").values]
    np.save(os.path.join(CACHE, "folds_a.npy"), folds_a)

    tr_ans = tr[tr.ans != ""].reset_index(drop=True)
    X, y, g = pass_a(tr_ans, idf, ssim)
    print("stage1 rows", X.shape, "pos frac", (y > 0.5).mean())
    mp = os.path.join(CACHE, "stage1_models.pkl")
    if os.path.exists(mp):
        with open(mp, "rb") as f:
            models = pickle.load(f)
    else:
        models = train_stage1(X, y, g, folds_a)
        with open(mp, "wb") as f:
            pickle.dump(models, f)
    if stage == "a":
        return
    NB = int(os.environ.get("NB", 3400))
    recs = pass_b(tr.iloc[:NB].reset_index(drop=True), idf, models, folds[:NB], ssim)
    f1top1 = np.array([r["f1"] for r in recs])
    unans = np.array([r["unans"] for r in recs])
    print("mean f1 of top1 (always answer): %.4f" % f1top1.mean())
    print("  on answerable only: %.4f" % f1top1[unans == 0].mean())
    print("  oracle top8 answerable: %.4f" % np.array([r["f1_top"] for r in recs])[unans == 0].mean())
    if stage == "b":
        return
    trecs = pass_c(te, idf, models, ssim)
    print("test recs", len(trecs))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
