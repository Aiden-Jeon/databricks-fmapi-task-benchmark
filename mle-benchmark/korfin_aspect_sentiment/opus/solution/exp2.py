"""Model / feature search round 2: extra encodings + model zoo + blending."""
import sys, time, pickle, os
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from scipy import sparse
from scipy.special import softmax

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import build_frame, numeric_feats
from exp_baseline import make_blocks, LABELS

SEED = 42
CACHE = "/tmp/opencode/korfin_blocks.pkl"


def get_data():
    tr = pd.read_csv("train.csv")
    te = pd.read_csv("test.csv")
    y = tr.label.map({l: i for i, l in enumerate(LABELS)}).values
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            Xtr, Xte = pickle.load(f)
    else:
        Ftr, Fte = build_frame(tr), build_frame(te)
        blocks = make_blocks(Ftr, Fte)
        Xtr = sparse.hstack([b[1] for b in blocks]).tocsr()
        Xte = sparse.hstack([b[2] for b in blocks]).tocsr()
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        with open(CACHE, "wb") as f:
            pickle.dump((Xtr, Xte), f)
    return tr, te, y, Xtr, Xte


PRIOR = None


def enc_features(fit_df, fit_y, apply_df, k_sent=1.0, k_asp=3.0):
    """Neighbour-based encodings computed from fit_df only.

    - same-sentence other-aspect label distribution
    - aspect-level label distribution (target encoding)
    """
    prior = np.bincount(fit_y, minlength=3) / len(fit_y)
    sent_cnt = defaultdict(lambda: np.zeros(3))
    sent_asp_cnt = defaultdict(lambda: np.zeros(3))
    asp_cnt = defaultdict(lambda: np.zeros(3))
    for s, a, lab in zip(fit_df.sentence.values, fit_df.aspect.values, fit_y):
        sent_cnt[s][lab] += 1
        sent_asp_cnt[(s, a)][lab] += 1
        asp_cnt[a][lab] += 1

    rows = []
    for s, a in zip(apply_df.sentence.values, apply_df.aspect.values):
        c = sent_cnt.get(s, None)
        c = (c.copy() if c is not None else np.zeros(3))
        sa = sent_asp_cnt.get((s, a), None)
        if sa is not None:
            c = c - sa  # exclude the same (sentence, aspect) pair -> no self leak
        n_s = c.sum()
        p_s = (c + k_sent * prior) / (n_s + k_sent)

        ca = asp_cnt.get(a, None)
        ca = (ca.copy() if ca is not None else np.zeros(3))
        if sa is not None:
            ca = ca - sa
        ca = np.maximum(ca, 0)
        n_a = ca.sum()
        p_a = (ca + k_asp * prior) / (n_a + k_asp)
        rows.append(np.concatenate([p_s, [min(n_s, 5) / 5.0], p_a, [min(n_a, 10) / 10.0]]))
    return np.asarray(rows)


MODELS = {
    "lr_c0.5": lambda: LogisticRegression(C=0.5, max_iter=3000),
    "lr_c1": lambda: LogisticRegression(C=1, max_iter=1000),
    "svc_c0.1": lambda: LinearSVC(C=0.1),
    "svc_c0.3": lambda: LinearSVC(C=0.3),
    "svc_c1": lambda: LinearSVC(C=1),
    "sgd_mh": lambda: SGDClassifier(loss="modified_huber", alpha=1e-5, max_iter=3000,
                                    random_state=SEED),
    "cnb": lambda: ComplementNB(alpha=0.3),
}


def proba(m, X):
    if hasattr(m, "predict_proba"):
        return m.predict_proba(X)
    d = m.decision_function(X)
    return softmax(d * 2.0, axis=1)


def main():
    tr, te, y, Xtr, Xte = get_data()
    num_tr = numeric_feats(tr)
    print("X", Xtr.shape, flush=True)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    folds = list(skf.split(Xtr, y))

    variants = {}
    # variant A: sparse only
    variants["A_sparse"] = None
    # variant B: sparse + enc features
    variants["B_enc"] = "enc"

    t0 = time.time()
    oofs = {}
    for vname, mode in variants.items():
        for mname, fn in MODELS.items():
            oof = np.zeros((len(y), 3))
            for tr_i, va_i in folds:
                Xa, Xb = Xtr[tr_i], Xtr[va_i]
                if mode == "enc":
                    Ea = enc_features(tr.iloc[tr_i], y[tr_i], tr.iloc[tr_i])
                    Eb = enc_features(tr.iloc[tr_i], y[tr_i], tr.iloc[va_i])
                    Xa = sparse.hstack([Xa, sparse.csr_matrix(Ea)]).tocsr()
                    Xb = sparse.hstack([Xb, sparse.csr_matrix(Eb)]).tocsr()
                m = fn()
                m.fit(Xa, y[tr_i])
                oof[va_i] = proba(m, Xb)
            key = f"{vname}|{mname}"
            oofs[key] = oof
            print(f"{key:22s} macroF1={f1_score(y, oof.argmax(1), average='macro'):.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    with open("/tmp/opencode/oofs.pkl", "wb") as f:
        pickle.dump((oofs, y), f)

    # greedy blend
    keys = list(oofs)
    scores = {k: f1_score(y, oofs[k].argmax(1), average="macro") for k in keys}
    order = sorted(keys, key=lambda k: -scores[k])
    cur = np.zeros_like(oofs[keys[0]])
    chosen = []
    best = -1
    for _ in range(8):
        cand_best, cand_key = best, None
        for k in order:
            trial = cur + oofs[k]
            s = f1_score(y, trial.argmax(1), average="macro")
            if s > cand_best + 1e-5:
                cand_best, cand_key = s, k
        if cand_key is None:
            break
        cur = cur + oofs[cand_key]
        chosen.append(cand_key)
        best = cand_best
        print(f"greedy add {cand_key:22s} -> {best:.4f}", flush=True)
    print("CHOSEN", chosen, best)


if __name__ == "__main__":
    main()
