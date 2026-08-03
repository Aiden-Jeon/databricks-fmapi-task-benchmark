import sys, time, itertools, pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from common import load, norm, macro_f1, tune_thresholds, decide, NL
from exp import oof, get_model, scores
import common

VECS = {
    "cw25_w12": lambda: FeatureUnion([
        ("cw", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=400000)),
        ("w", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True))]),
    "cw26_w12": lambda: FeatureUnion([
        ("cw", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6), min_df=2, sublinear_tf=True, max_features=600000)),
        ("w", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True))]),
    "cw15_w12": lambda: FeatureUnion([
        ("cw", TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), min_df=1, sublinear_tf=True, max_features=600000)),
        ("w", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True))]),
    "c25_w12": lambda: FeatureUnion([
        ("c", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=600000)),
        ("w", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True))]),
    "cwonly": lambda: FeatureUnion([
        ("cw", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=400000))]),
}

if __name__ == "__main__":
    tr, te, Y = load("..")
    texts = tr["sentence"].map(norm).values
    tte = te["sentence"].map(norm).values
    cfgs = []
    for v in VECS:
        for kind, C in [("svc", 0.5), ("svc", 1.0), ("svc", 2.0)]:
            cfgs.append((v, kind, C))
    for kind, C in [("lr", 20.0), ("lr", 50.0), ("cnb", 0.3)]:
        cfgs.append(("cw25_w12", kind, C))
    res = {}
    for v, kind, C in cfgs:
        common.make_vec = VECS[v]
        import exp; exp.make_vec = VECS[v]
        t0 = time.time()
        P, Pte = oof(texts, Y, tte, kind, C)
        th = tune_thresholds(Y, P)
        s = macro_f1(Y, decide(P, th))
        s5 = macro_f1(Y, decide(P, np.full(NL, 0.5)))
        res[(v, kind, C)] = (s, s5, P, Pte, th)
        print(f"{v:10s} {kind} C={C}: @0.5={s5:.4f} tuned={s:.4f} ({time.time()-t0:.0f}s)", flush=True)
    with open("sweep.pkl", "wb") as f:
        pickle.dump({k: (v[0], v[1], v[2], v[3], v[4]) for k, v in res.items()}, f)
    print("best:", max(res.items(), key=lambda kv: kv[1][0])[0])
