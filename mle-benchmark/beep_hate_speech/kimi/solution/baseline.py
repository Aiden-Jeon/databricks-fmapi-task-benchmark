"""Baseline: TF-IDF (word + char n-grams) + LogisticRegression (OvR, class_weight balanced)."""
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

LABELS = ["none", "offensive", "hate"]


def main():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    Xtr_text = train["comment"].astype(str)
    Xte_text = test["comment"].astype(str)
    ytr = train["label"]

    word_vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True
    )
    char_vec = TfidfVectorizer(
        analyzer="char", ngram_range=(3, 5), min_df=2, sublinear_tf=True
    )

    Xtr = hstack([word_vec.fit_transform(Xtr_text), char_vec.fit_transform(Xtr_text)])
    Xte = hstack([word_vec.transform(Xte_text), char_vec.transform(Xte_text)])

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)

    sub = pd.DataFrame({"id": test["id"], "label": pred})
    sub.to_csv("outputs/submission.csv", index=False)
    print(sub["label"].value_counts())
    print("saved outputs/submission.csv")


if __name__ == "__main__":
    main()
