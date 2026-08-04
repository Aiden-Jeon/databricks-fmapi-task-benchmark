import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder


DATA = "/tmp/kmle/M3_t23_korfin_asc_full_20260804_033756/task"
RANDOM_STATE = 42


def find_aspect_window(sentence, aspect, window=30):
    idx = sentence.find(aspect)
    if idx == -1:
        return sentence
    start = max(0, idx - window)
    end = min(len(sentence), idx + len(aspect) + window)
    return sentence[start:end]


def build_features(df, vectorizer=None, fit=False):
    df = df.copy()
    df["sentence"] = df["sentence"].astype(str)
    df["aspect"] = df["aspect"].astype(str)
    df["text"] = df["sentence"] + " [SEP] " + df["aspect"]
    df["ctx"] = df.apply(
        lambda r: find_aspect_window(r["sentence"], r["aspect"], 60), axis=1
    )
    df["combo"] = df["ctx"] + " [ASP] " + df["aspect"] + " [ASP] " + df["sentence"]
    return df


def main():
    train = pd.read_csv(f"{DATA}/train.csv")
    test = pd.read_csv(f"{DATA}/test.csv")
    sub = pd.read_csv(f"{DATA}/sample_submission.csv")

    train = build_features(train)
    test = build_features(test)

    le = LabelEncoder()
    y = le.fit_transform(train["label"])

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(1, 5),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            lowercase=False,
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=2000,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )),
    ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        pipeline, train["combo"], y, cv=skf, scoring="f1_macro", n_jobs=-1
    )
    print("CV macro_f1:", scores, "mean:", scores.mean())

    pipeline.fit(train["combo"], y)
    preds = pipeline.predict(test["combo"])
    labels = le.inverse_transform(preds)

    out = pd.DataFrame({"id": test["id"], "label": labels})
    out.to_csv(f"{DATA}/outputs/submission.csv", index=False)
    print("Saved submission:", out.shape)
    print(out["label"].value_counts())


if __name__ == "__main__":
    main()
