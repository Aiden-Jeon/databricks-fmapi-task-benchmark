"""Fast dev loop on a subset: sanity-check features / stage-1 ranker quality."""
import sys, time
import numpy as np, pandas as pd
import qa_lib as Q
import run as R


def main():
    NTR = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    NVA = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    tr, te = R.load_data()
    idf = R.init_idf(tr, te)
    ssim = R.init_ssim(tr, te)
    sub = tr.iloc[:NTR]
    sub = sub[sub.ans != '']
    val = tr.iloc[NTR:NTR + NVA]

    t = time.time()
    args = [(i, r.context, r.question, r.ans) for i, r in enumerate(sub.itertuples())]
    with R.make_pool(4, idf, None, ssim) as p:
        res = list(p.imap(R._work_a, args, chunksize=32))
    X = np.concatenate([a for a, b, c in res]); y = np.concatenate([b for a, b, c in res])
    print("featurize %.0fs rows %s" % (time.time() - t, X.shape), flush=True)

    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_leaf_nodes=63,
                                      min_samples_leaf=30, early_stopping=False, random_state=0)
    t = time.time(); m.fit(X, y); print("fit %.0fs" % (time.time() - t), flush=True)

    t = time.time()
    vargs = [(i, r.context, r.question, r.ans, 1) for i, r in enumerate(val.itertuples())]
    with R.make_pool(4, idf, [m, m], ssim) as p:
        vrecs = list(p.imap(R._work_b, vargs, chunksize=16))
    print("score %.0fs" % (time.time() - t), flush=True)
    f1 = np.array([r["f1"] for r in vrecs])
    un = np.array([r["unans"] for r in vrecs])
    ftop = np.array([r["f1_top"] for r in vrecs])
    top1 = np.array([r["top1"] for r in vrecs])
    print("always-answer F1 = %.4f | answerable-only %.4f | oracle-top8 %.4f | unans frac %.3f"
          % (f1.mean(), f1[un == 0].mean(), ftop[un == 0].mean(), un.mean()))
    print("always-empty F1 = %.4f" % un.mean())
    for th in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        pred = np.where(top1 >= th, f1, un)
        print("  thr top1>=%.2f -> %.4f (answer frac %.2f)" % (th, pred.mean(), (top1 >= th).mean()))
    print("top1 score: ans %.3f unans %.3f" % (top1[un == 0].mean(), top1[un == 1].mean()))
    import pickle
    with open("cache/dev_vrecs.pkl", "wb") as f:
        pickle.dump((vrecs, val.reset_index(drop=True)), f)


if __name__ == "__main__":
    main()
