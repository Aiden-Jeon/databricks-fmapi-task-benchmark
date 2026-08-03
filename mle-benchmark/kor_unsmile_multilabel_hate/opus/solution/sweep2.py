import time, pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from common import load, norm, macro_f1, tune_thresholds, decide, NL
import exp
from exp import oof


def V(cwr=(2, 5), cwdf=2, mf=600000, sub=True, char="char_wb", word=None, wdf=2, binary=False):
    parts = [("cw", TfidfVectorizer(analyzer=char, ngram_range=cwr, min_df=cwdf,
                                    sublinear_tf=sub, max_features=mf, binary=binary))]
    if word is not None:
        parts.append(("w", TfidfVectorizer(analyzer="word", ngram_range=word, min_df=wdf, sublinear_tf=sub)))
    return lambda: FeatureUnion(parts)


CFG = {
    "cw25df2":        (V(), "svc", 0.5),
    "cw25df1":        (V(cwdf=1), "svc", 0.5),
    "cw15df2":        (V(cwr=(1, 5)), "svc", 0.5),
    "cw26df2":        (V(cwr=(2, 6), mf=900000), "svc", 0.5),
    "cw24df2":        (V(cwr=(2, 4)), "svc", 0.5),
    "cw35df2":        (V(cwr=(3, 5)), "svc", 0.5),
    "cw25df3":        (V(cwdf=3), "svc", 0.5),
    "cw25df2_C025":   (V(), "svc", 0.25),
    "cw25df2_C1":     (V(), "svc", 1.0),
    "cw25df2_nosub":  (V(sub=False), "svc", 0.5),
    "cw25df2_lr":     (V(), "lr", 20.0),
    "cw25df2_lr100":  (V(), "lr", 100.0),
    "cw25df2_w11df3": (V(word=(1, 1), wdf=3), "svc", 0.5),
    "char25df2":      (V(char="char"), "svc", 0.5),
}

if __name__ == "__main__":
    tr, te, Y = load("..")
    texts = tr["sentence"].map(norm).values
    tte = te["sentence"].map(norm).values
    out = {}
    for name, (vf, kind, C) in CFG.items():
        exp.make_vec = vf
        t0 = time.time()
        Ps, Ptes = [], []
        for seed in (0, 1):
            P, Pte = oof(texts, Y, tte, kind, C, seed=seed)
            Ps.append(P); Ptes.append(Pte)
        P = np.mean(Ps, 0); Pte = np.mean(Ptes, 0)
        th = tune_thresholds(Y, P)
        s = macro_f1(Y, decide(P, th)); s5 = macro_f1(Y, decide(P, np.full(NL, 0.5)))
        out[name] = (s, s5, P, Pte, th)
        print(f"{name:18s}: @0.5={s5:.4f} tuned={s:.4f} ({time.time()-t0:.0f}s)", flush=True)
    with open("sweep2.pkl", "wb") as f:
        pickle.dump(out, f)
    for k, v in sorted(out.items(), key=lambda kv: -kv[1][0]):
        print(f"  {k:18s} {v[0]:.4f}")
