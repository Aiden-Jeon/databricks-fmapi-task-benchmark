import copy
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score


DATA = "/tmp/kmle/M3_t23_korfin_asc_full_20260804_033756/task"
RANDOM_STATE = 42


def aspect_window(sentence, aspect, before=40, after=40):
    idx = sentence.find(aspect)
    if idx == -1:
        return sentence, "", ""
    s = max(0, idx - before)
    e = min(len(sentence), idx + len(aspect) + after)
    left = sentence[s:idx]
    right = sentence[idx + len(aspect):e]
    return sentence[s:e], left, right


def add_features(df):
    df = df.copy()
    df["sentence"] = df["sentence"].astype(str)
    df["aspect"] = df["aspect"].astype(str)
    ctxs, lefts, rights = [], [], []
    for _, r in df.iterrows():
        c, l, rr = aspect_window(r["sentence"], r["aspect"])
        ctxs.append(c); lefts.append(l); rights.append(rr)
    df["ctx"] = ctxs
    df["left"] = lefts
    df["right"] = rights
    df["combo"] = df["ctx"] + " [ASP] " + df["aspect"] + " [ASP] " + df["sentence"]
    df["left_asp"] = df["left"] + " [ASP] " + df["aspect"]
    df["asp_right"] = df["aspect"] + " [ASP] " + df["right"]
    df["left_only"] = df["left"]
    df["right_only"] = df["right"]
    return df


def build_mat(train_df, test_df, col, analyzer="char_wb", ngram=(1, 5), min_df=2):
    vec = TfidfVectorizer(
        analyzer=analyzer, ngram_range=ngram, min_df=min_df, max_df=0.95,
        sublinear_tf=True, lowercase=False,
    )
    Xtr = vec.fit_transform(train_df[col])
    Xte = vec.transform(test_df[col])
    return Xtr, Xte


def run_cv(Xtr, Xte, y, model, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y))
    te_prob = np.zeros((Xte.shape[0], len(np.unique(y))))
    for tr, va in skf.split(Xtr, y):
        m = copy.deepcopy(model)
        m.fit(Xtr[tr], y[tr])
        oof[va] = m.predict(Xtr[va])
        te_prob += m.predict_proba(Xte) / n_splits
    score = f1_score(y, oof, average="macro")
    return oof, te_prob, score


def main():
    train = pd.read_csv(f"{DATA}/train.csv")
    test = pd.read_csv(f"{DATA}/test.csv")
    train = add_features(train)
    test = add_features(test)

    le = LabelEncoder()
    y = le.fit_transform(train["label"])

    feats = [
        ("combo", "char_wb", (1, 5), 2),
        ("combo", "word", (1, 2), 2),
        ("left_asp", "char_wb", (1, 4), 2),
        ("asp_right", "char_wb", (1, 4), 2),
        ("left_only", "char_wb", (1, 4), 1),
        ("right_only", "char_wb", (1, 4), 1),
        ("ctx", "char_wb", (1, 4), 1),
    ]
    Xtr_parts, Xte_parts = [], []
    for col, an, ng, md in feats:
        a, b = build_mat(train, test, col, an, ng, md)
        Xtr_parts.append(a); Xte_parts.append(b)
    Xtr = hstack(Xtr_parts).tocsr()
    Xte = hstack(Xte_parts).tocsr()
    print("Xtr:", Xtr.shape, "Xte:", Xte.shape)

    models = {
        "lr_c1": LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced",
                                     n_jobs=-1, random_state=RANDOM_STATE),
        "lr_c07": LogisticRegression(C=0.7, max_iter=3000, class_weight="balanced",
                                     n_jobs=-1, random_state=RANDOM_STATE),
        "lr_c15": LogisticRegression(C=1.5, max_iter=3000, class_weight="balanced",
                                     n_jobs=-1, random_state=RANDOM_STATE),
    }
    results = {}
    te_probs = {}
    for name, model in models.items():
        oof, te_p, sc = run_cv(Xtr, Xte, y, model)
        results[name] = sc
        te_probs[name] = te_p
        print(f"{name}: {sc:.4f}")

    # weighted ensemble
    weights = np.array([results[n] for n in models])
    w = weights / weights.sum()
    te_avg = sum(te_probs[n] * ww for n, ww in zip(models, w))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_ens = np.zeros(len(y))
    for tr, va in skf.split(Xtr, y):
        fp = np.zeros((len(va), len(np.unique(y))))
        for i, (n, m) in enumerate(models.items()):
            mm = copy.deepcopy(m)
            mm.fit(Xtr[tr], y[tr])
            fp += mm.predict_proba(Xtr[va]) * w[i]
        oof_ens[va] = fp.argmax(1)
    ens_sc = f1_score(y, oof_ens, average="macro")
    print(f"ENSEMBLE: {ens_sc:.4f}")

    best_single = max(results, key=results.get)
    if ens_sc > results[best_single]:
        print("using ensemble", ens_sc)
        final = te_avg.argmax(1)
    else:
        print(f"using {best_single}", results[best_single])
        final = te_probs[best_single].argmax(1)

    labels = le.inverse_transform(final)
    out = pd.DataFrame({"id": test["id"], "label": labels})
    out.to_csv(f"{DATA}/outputs/submission.csv", index=False)
    print("Saved:", out.shape)
    print(out["label"].value_counts())


if __name__ == "__main__":
    main()
