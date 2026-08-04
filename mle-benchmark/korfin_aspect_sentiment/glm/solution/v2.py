import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score


DATA = "/tmp/kmle/M3_t23_korfin_asc_full_20260804_033756/task"
RANDOM_STATE = 42


def find_aspect_window(sentence, aspect, window=60):
    idx = sentence.find(aspect)
    if idx == -1:
        return sentence
    start = max(0, idx - window)
    end = min(len(sentence), idx + len(aspect) + window)
    return sentence[start:end]


def add_features(df):
    df = df.copy()
    df["sentence"] = df["sentence"].astype(str)
    df["aspect"] = df["aspect"].astype(str)
    df["ctx"] = df.apply(
        lambda r: find_aspect_window(r["sentence"], r["aspect"], 60), axis=1
    )
    df["left"] = df.apply(
        lambda r: r["sentence"][: r["sentence"].find(r["aspect"])]
        if r["aspect"] in r["sentence"]
        else "",
        axis=1,
    )
    df["right"] = df.apply(
        lambda r: r["sentence"][r["sentence"].find(r["aspect"]) + len(r["aspect"]) :]
        if r["aspect"] in r["sentence"]
        else "",
        axis=1,
    )
    df["combo"] = df["ctx"] + " [ASP] " + df["aspect"] + " [ASP] " + df["sentence"]
    df["combo_left"] = df["left"] + " [ASP] " + df["aspect"]
    df["combo_right"] = df["aspect"] + " [ASP] " + df["right"]
    df["aspect_in"] = df["aspect"].isin(df["sentence"]).astype(int).values
    return df


def build_matrix(train_df, test_df, col, analyzer="char_wb", ngram=(1, 5)):
    vec = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        lowercase=False,
    )
    Xtr = vec.fit_transform(train_df[col])
    Xte = vec.transform(test_df[col])
    return Xtr, Xte, vec


def cv_score(X, y, model, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        m = model.__class__.__new__(model.__class__)
        m.__dict__.update(model.__dict__)
        m.fit(X[tr], y[tr])
        oof[va] = m.predict(X[va])
    return f1_score(y, oof, average="macro"), oof


def main():
    train = pd.read_csv(f"{DATA}/train.csv")
    test = pd.read_csv(f"{DATA}/test.csv")

    train = add_features(train)
    test = add_features(test)

    le = LabelEncoder()
    y = le.fit_transform(train["label"])

    print("Building feature matrices...")
    Xtr_combo, Xte_combo, _ = build_matrix(train, test, "combo", "char_wb", (1, 5))
    Xtr_w, Xte_w, _ = build_matrix(train, test, "combo", "word", (1, 2))
    Xtr_l, Xte_l, _ = build_matrix(train, test, "combo_left", "char_wb", (1, 4))
    Xtr_r, Xte_r, _ = build_matrix(train, test, "combo_right", "char_wb", (1, 4))

    Xtr = hstack([Xtr_combo, Xtr_w, Xtr_l, Xtr_r]).tocsr()
    Xte = hstack([Xte_combo, Xte_w, Xte_l, Xte_r]).tocsr()
    print("Xtr shape:", Xtr.shape, "Xte shape:", Xte.shape)

    models = {
        "lr_c1": LogisticRegression(
            C=1.0, max_iter=2000, class_weight="balanced", n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "lr_c2": LogisticRegression(
            C=2.0, max_iter=2000, class_weight="balanced", n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "lr_c05": LogisticRegression(
            C=0.5, max_iter=2000, class_weight="balanced", n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "svm": CalibratedClassifierCV(
            LinearSVC(C=0.5, class_weight="balanced", random_state=RANDOM_STATE,
                      max_iter=5000),
            cv=3,
        ),
    }

    oof_dict = {}
    te_pred_dict = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for name, model in models.items():
        oof = np.zeros(len(y))
        te_pred = np.zeros((len(test), len(le.classes_)))
        for tr, va in skf.split(Xtr, y):
            import copy
            m = copy.deepcopy(model)
            m.fit(Xtr[tr], y[tr])
            oof[va] = m.predict(Xtr[va])
            te_pred += m.predict_proba(Xte) / 5
        score = f1_score(y, oof, average="macro")
        print(f"{name}: {score:.4f}")
        oof_dict[name] = oof
        te_pred_dict[name] = te_pred

    # simple average ensemble of probabilities
    te_avg = np.mean([te_pred_dict[n] for n in te_pred_dict], axis=0)
    oof_avg = np.zeros((len(y), len(le.classes_)))
    # build oof via stacking - but simpler: average predictions
    oof_labels = np.zeros((len(y), len(le.classes_)))
    for tr, va in skf.split(Xtr, y):
        import copy
        fold_probs = np.zeros((len(va), len(le.classes_)))
        for name, model in models.items():
            m = copy.deepcopy(model)
            m.fit(Xtr[tr], y[tr])
            fold_probs += m.predict_proba(Xtr[va]) / len(models)
        oof_avg[va] = fold_probs
    oof_pred = oof_avg.argmax(1)
    ens_score = f1_score(y, oof_pred, average="macro")
    print(f"ENSEMBLE avg proba: {ens_score:.4f}")

    final_te = te_avg.argmax(1)
    labels = le.inverse_transform(final_te)

    out = pd.DataFrame({"id": test["id"], "label": labels})
    out.to_csv(f"{DATA}/outputs/submission.csv", index=False)
    print("Saved submission:", out.shape)
    print(out["label"].value_counts())


if __name__ == "__main__":
    main()
