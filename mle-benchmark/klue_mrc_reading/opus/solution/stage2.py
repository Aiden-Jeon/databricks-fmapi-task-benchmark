"""Stage 2: calibrate answer-vs-empty decision and write outputs/submission.csv."""
import os, pickle, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import KFold
import run as R

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
TASK = os.path.dirname(HERE)


def load():
    with open(os.path.join(CACHE, "stage2_rows.pkl"), "rb") as f:
        recs = pickle.load(f)
    with open(os.path.join(CACHE, "test_rows.pkl"), "rb") as f:
        trecs = pickle.load(f)
    X = np.stack([r["feat"] for r in recs])
    f1 = np.array([r["f1"] for r in recs])
    un = np.array([r["unans"] for r in recs])
    Xt = np.stack([r["feat"] for r in trecs])
    return recs, trecs, X, f1, un, Xt


def fit_models(X, f1, un, seed=0):
    reg = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06,
                                        max_leaf_nodes=31, min_samples_leaf=40,
                                        l2_regularization=1.0, early_stopping=False,
                                        random_state=seed)
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06,
                                         max_leaf_nodes=31, min_samples_leaf=40,
                                         l2_regularization=1.0, early_stopping=False,
                                         random_state=seed)
    reg.fit(X, f1)
    clf.fit(X, un)
    return reg, clf


def main():
    recs, trecs, X, f1, un, Xt = load()
    print("stage2 data", X.shape, "unans frac %.3f" % un.mean(),
          "mean f1(top1) %.4f" % f1.mean(), "answerable-only %.4f" % f1[un == 0].mean())

    # ---- cross-validated evaluation of the decision rule
    oof_r = np.zeros(len(X)); oof_c = np.zeros(len(X))
    for tri, vai in KFold(4, shuffle=True, random_state=0).split(X):
        reg, clf = fit_models(X[tri], f1[tri], un[tri])
        oof_r[vai] = reg.predict(X[vai])
        oof_c[vai] = clf.predict_proba(X[vai])[:, 1]
    from sklearn.metrics import roc_auc_score
    print("unans AUC %.4f | corr(reg,f1) %.3f" % (roc_auc_score(un, oof_c),
                                                  np.corrcoef(oof_r, f1)[0, 1]))
    print("baseline all-empty %.4f | all-answer %.4f" % (un.mean(), f1.mean()))
    rules = [("empty", lambda r, c: np.zeros(len(r), dtype=bool))]
    for k in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0]:
        rules.append(("ratio_%.2f" % k, (lambda kk: (lambda r, c: r > kk * c))(k)))
    for th in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]:
        rules.append(("reg_%.2f" % th, (lambda t: (lambda r, c: r > t))(th)))
    for d in [-0.1, -0.05, 0.0, 0.05, 0.1, 0.15]:
        rules.append(("diff_%.2f" % d, (lambda dd: (lambda r, c: r - c > dd))(d)))
    best = (-1, None, None)
    for name, fn in rules:
        ans = fn(oof_r, oof_c)
        sc = np.where(ans, f1, un).mean()
        print("  %-11s answer_frac %.3f score %.4f" % (name, ans.mean(), sc))
        if sc > best[0]:
            best = (sc, name, fn)
    print("BEST rule: %s score %.4f" % (best[1], best[0]))
    rule = best[2]

    # ---- final models on all data, apply to test
    reg, clf = fit_models(X, f1, un)
    pr = reg.predict(Xt)
    pc = clf.predict_proba(Xt)[:, 1]
    answer = rule(pr, pc)
    te = pd.read_csv(os.path.join(TASK, "test.csv"))
    texts = [r["text"] for r in trecs]
    out = [t if a else "" for t, a in zip(texts, answer)]
    sub = pd.DataFrame({"id": te.id, "answer": out})
    assert len(sub) == len(te) and sub.id.is_unique
    path = os.path.join(TASK, "outputs", "submission.csv")
    sub.to_csv(path, index=False)
    print("wrote", path, "| answered %.3f" % answer.mean())
    print(sub.head(15).to_string())


if __name__ == "__main__":
    main()
