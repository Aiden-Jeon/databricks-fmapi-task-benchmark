"""Quick baseline: TF-IDF on [premise + question + alternative] rows, binary classifier scores each alternative, pick higher."""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold


def load():
    tr = pd.read_csv("train.csv")
    te = pd.read_csv("test.csv")
    tr["question"] = tr["question"].str.strip()
    te["question"] = te["question"].str.strip()
    return tr, te


def build_rows(df, label=None):
    """One row per (premise, q, alt) -> label is 1 if it's the correct alt, else 0.
       For test (label None), build both alternatives; we will score and compare."""
    rows = []
    labels = []
    ids = []
    which = []
    for _, r in df.iterrows():
        for k in (1, 2):
            text = r["premise"] + " " + r["question"] + " " + r[f"alternative_{k}"]
            rows.append(text)
            ids.append(r["id"])
            which.append(k)
            if label is not None:
                # correct alt is (label+1): label 0 -> alt1, label 1 -> alt2
                labels.append(1 if k == (r["label"] + 1) else 0)
    if label is not None:
        return rows, np.array(labels)
    return rows, ids, which


def main():
    tr, te = load()
    Xtr_text, ytr = build_rows(tr, label=True)
    Xte_text, ids_te, which_te = build_rows(te, label=None)

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
    Xtr = vec.fit_transform(Xtr_text)
    Xte = vec.transform(Xte_text)

    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=0)
    # CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(clf, Xtr, ytr, cv=cv, scoring="accuracy")
    print("CV pair-acc:", scores.mean(), scores.std())

    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]

    # pair back
    import numpy as np
    p = np.array(p)
    te_idx = te.set_index("id")
    pred = []
    for i in range(0, len(p), 2):
        a1, a2 = p[i], p[i + 1]
        pred.append(0 if a1 >= a2 else 1)
    out = pd.DataFrame({"id": te["id"], "label": pred})
    out.to_csv("outputs/submission.csv", index=False)
    print("Baseline submission written. predictions:", len(out))
    print(out["label"].value_counts().to_dict())


if __name__ == "__main__":
    main()
