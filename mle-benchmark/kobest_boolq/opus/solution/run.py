"""KoBEST BoolQ — sklearn-only solution.

Pipeline
--------
1. Handcrafted features (lexical overlap, negation mismatch, LSA alignment,
   sentence-spread / proximity binding).
2. Sparse text views:
     - question-only char n-gram TF-IDF (captures question-side wording cues)
     - "presence-tagged" cross n-grams: every question n-gram is suffixed with
       @1/@0 depending on whether it occurs in the paragraph (resp. in the
       LSA-best-matching sentence) -> cheap textual-entailment features.
3. 10 diverse base models -> out-of-fold probabilities (repeated stratified CV).
4. Logistic-regression stacker on base-model logits.

Measured 5-fold CV accuracy (train.csv, 3 seeds averaged):
    question-char TF-IDF LR      0.598 - 0.609
    presence-tagged cross LR     0.593 - 0.599
    dense handcrafted LR / ET    0.606 - 0.613
    mix (sparse+cross+dense) LR  0.632 - 0.635
    final blend                  0.634   (nested-CV check: 0.626)
Majority-class baseline is 0.500 (labels are balanced).

Usage:  python run.py            (writes ../outputs/submission.csv)
        python run.py --oof      (also prints cross-validated scores)
        python run.py --nested   (honest nested-CV estimate of the blend)
"""
import argparse
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix, hstack

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)

from features import build_features, head_stems, sentences          # noqa: E402
from features2 import LsaSim, build_features2                        # noqa: E402
from features3 import build_features3                               # noqa: E402

SEED = 42


# --------------------------------------------------------------------------- #
# text views
# --------------------------------------------------------------------------- #
def tag_text(q, ref, n=3, mark="@"):
    rs = re.sub(r"\s+", "", str(ref))
    qs = re.sub(r"\s+", " ", str(q))
    toks = []
    for i in range(max(0, len(qs) - n + 1)):
        g = qs[i:i + n]
        toks.append(g.replace(" ", "_") + mark + ("1" if g.replace(" ", "") in rs else "0"))
    for w in head_stems(q):
        toks.append("W" + w + mark + ("1" if w in rs else "0"))
    return " ".join(toks)


def best_sentences(df, lsa):
    qv = lsa.emb([str(q) for q in df.question.values])
    out = []
    for k, p in enumerate(df.paragraph.values):
        s = sentences(p)
        sv = lsa.emb(s)
        out.append(s[int(np.argmax(sv @ qv[k]))])
    return out


def prepare(trdf, tedf):
    t0 = time.time()
    lsa = LsaSim(150).fit(list(trdf.paragraph) + list(tedf.paragraph))
    views, dense = {}, {}
    for name, df in (("tr", trdf), ("te", tedf)):
        bs = best_sentences(df, lsa)
        tagp = [tag_text(q, p) for p, q in zip(df.paragraph, df.question)]
        tagb = [tag_text(q, b, mark="#") for b, q in zip(bs, df.question)]
        views[name] = {
            "qtext": df.question.values,
            "tagpara": np.array(tagp),
            "tagboth": np.array([a + " " + b for a, b in zip(tagp, tagb)]),
        }
        F = pd.concat([build_features(df), build_features2(df, lsa), build_features3(df)], axis=1)
        dense[name] = F
    print(f"[prepare] {time.time() - t0:.1f}s  dense={dense['tr'].shape}", flush=True)
    return views, dense


# --------------------------------------------------------------------------- #
# base models
# --------------------------------------------------------------------------- #
def base_models():
    return {
        "q_c13_lr": ("qtext", make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), min_df=2, sublinear_tf=True),
            LogisticRegression(C=0.5, max_iter=3000))),
        "q_c15_lr": ("qtext", make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), min_df=2, sublinear_tf=True),
            LogisticRegression(C=0.5, max_iter=3000))),
        "q_c24_nb": ("qtext", make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True),
            MultinomialNB(alpha=0.3))),
        "tagboth_lr": ("tagboth", make_pipeline(
            TfidfVectorizer(min_df=3, sublinear_tf=True, token_pattern=r"\S+"),
            LogisticRegression(C=0.3, max_iter=3000))),
        "tagpara_lr": ("tagpara", make_pipeline(
            TfidfVectorizer(min_df=3, sublinear_tf=True, token_pattern=r"\S+"),
            LogisticRegression(C=0.5, max_iter=3000))),
        "tagpara_nb": ("tagpara", make_pipeline(
            TfidfVectorizer(min_df=3, sublinear_tf=True, token_pattern=r"\S+"),
            MultinomialNB(alpha=0.3))),
        "dense_lr": ("dense", make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000))),
        "dense_et": ("dense", ExtraTreesClassifier(800, min_samples_leaf=8, n_jobs=-1, random_state=0)),
        "dense_rf": ("dense", RandomForestClassifier(800, min_samples_leaf=6, n_jobs=-1, random_state=0)),
        "dense_hgb": ("dense", HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=40,
            l2_regularization=2.0, random_state=0)),
    }


def mix_models():
    """Strongest family: sparse(question) + sparse(cross-tag) + dense in one LR."""
    return {
        "mix_a": ("mixpara", MixModel(C=0.5, dense_w=0.15)),
        "mix_b": ("mixpara", MixModel(C=0.5, dense_w=0.30)),
        "mix_c": ("mixboth", MixModel(C=1.0, dense_w=0.30)),
    }


class MixModel:
    """question char TF-IDF + tagged cross n-grams + scaled dense features -> LR."""

    def __init__(self, C=1.0, dense_w=0.3):
        self.C, self.dense_w = C, dense_w

    def fit(self, X, y):
        q, tag, F = X
        self.vq = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), min_df=2, sublinear_tf=True)
        self.vt = TfidfVectorizer(min_df=3, sublinear_tf=True, token_pattern=r"\S+")
        self.sc = StandardScaler()
        Z = hstack([self.vq.fit_transform(q), self.vt.fit_transform(tag),
                    csr_matrix(self.sc.fit_transform(F)) * self.dense_w]).tocsr()
        self.m = LogisticRegression(C=self.C, max_iter=5000).fit(Z, y)
        return self

    def predict_proba(self, X):
        q, tag, F = X
        Z = hstack([self.vq.transform(q), self.vt.transform(tag),
                    csr_matrix(self.sc.transform(F)) * self.dense_w]).tocsr()
        return self.m.predict_proba(Z)


def get_view(view, V, F, idx=None):
    if view == "dense":
        return F.values if idx is None else F.values[idx]
    if view.startswith("mix"):
        tagview = "tagpara" if view == "mixpara" else "tagboth"
        q, t, d = V["qtext"], V[tagview], F.values
        if idx is not None:
            q, t, d = q[idx], t[idx], d[idx]
        return (q, t, d)
    x = V[view]
    return x if idx is None else x[idx]


def fit_predict(name, spec, Vtr, Ftr, ytr, tri, Vev, Fev, evi):
    view, mdl = spec
    m = clone(mdl) if hasattr(mdl, "get_params") else MixModel(mdl.C, mdl.dense_w)
    m.fit(get_view(view, Vtr, Ftr, tri), ytr[tri])
    Xe = get_view(view, Vev, Fev, evi)
    return m.predict_proba(Xe)[:, 1]


def all_models():
    m = dict(base_models())
    m.update(mix_models())
    return m


def compute_oof(V, F, y, seeds=(42, 7, 2024), n_splits=5, models=None, idx=None):
    """OOF probabilities on rows `idx` (default: all)."""
    models = models or all_models()
    idx = np.arange(len(y)) if idx is None else idx
    yv = y[idx]
    oof = {n: np.zeros(len(idx)) for n in models}
    for seed in seeds:
        cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
        for a, b in cv.split(idx, yv):
            tri, vai = idx[a], idx[b]
            for n, spec in models.items():
                oof[n][b] += fit_predict(n, spec, V, F, y, tri, V, F, vai) / len(seeds)
    return pd.DataFrame(oof, index=idx)


def fit_full_predict(V, F, y, Vte, Fte, models=None, idx=None):
    models = models or all_models()
    idx = np.arange(len(y)) if idx is None else idx
    n_te = len(Vte["qtext"]) if isinstance(Vte, dict) else len(Fte)
    out = {}
    for n, spec in models.items():
        out[n] = fit_predict(n, spec, V, F, y, idx, Vte, Fte, np.arange(n_te))
    return pd.DataFrame(out)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def make_stacker():
    return LogisticRegression(C=0.2, max_iter=5000)


def rank(x):
    return pd.Series(np.asarray(x)).rank(pct=True).values


def blend(P):
    """Final blend: 0.5 * rank-mean(mix family) + 0.5 * rank-mean(other base models)."""
    mix_cols = [c for c in P.columns if c.startswith("mix_")]
    base_cols = [c for c in P.columns if not c.startswith("mix_")]
    m = rank(P[mix_cols].rank(pct=True).mean(1).values)
    b = rank(P[base_cols].rank(pct=True).mean(1).values)
    return 0.5 * m + 0.5 * b


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nested", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--oof", action="store_true", help="report cross-validated scores")
    args = ap.parse_args()

    tr = pd.read_csv(os.path.join(ROOT, "train.csv"))
    te = pd.read_csv(os.path.join(ROOT, "test.csv"))
    y = tr.label.values
    views, dense = prepare(tr, te)
    Vtr, Vte = views["tr"], views["te"]
    Ftr, Fte = dense["tr"], dense["te"]
    seeds = (42, 7, 2024)[:args.seeds]

    if args.oof:
        t0 = time.time()
        O = compute_oof(Vtr, Ftr, y, seeds=seeds)
        print(f"[oof] {time.time() - t0:.1f}s", flush=True)
        for c in O.columns:
            print(f"  {c:12s} acc={accuracy_score(y, O[c] > .5):.4f} auc={roc_auc_score(y, O[c]):.4f}")
        print(f"  {'BLEND':12s} acc={accuracy_score(y, blend(O) > .5):.4f} "
              f"auc={roc_auc_score(y, blend(O)):.4f}")
        O.to_csv(os.path.join(HERE, "oof_base.csv"), index=False)

    if args.nested:
        outer = StratifiedKFold(5, shuffle=True, random_state=99)
        pn = np.zeros(len(y))
        for a, b in outer.split(np.arange(len(y)), y):
            Tb = pd.DataFrame({n: fit_predict(n, spec, Vtr, Ftr, y, a, Vtr, Ftr, b)
                               for n, spec in all_models().items()})
            pn[b] = blend(Tb)
        print(f"[nested] blend acc={accuracy_score(y, pn > .5):.4f} auc={roc_auc_score(y, pn):.4f}")

    T = fit_full_predict(Vtr, Ftr, y, Vte, Fte)
    final = blend(T)
    pred = (final > 0.5).astype(int)

    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    sub = pd.DataFrame({"id": te.id, "label": pred})
    sub.to_csv(os.path.join(ROOT, "outputs", "submission.csv"), index=False)
    T.to_csv(os.path.join(HERE, "test_base.csv"), index=False)
    print("wrote outputs/submission.csv", sub.shape, sub.label.value_counts().to_dict())


if __name__ == "__main__":
    main()
