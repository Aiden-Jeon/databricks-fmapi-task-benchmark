# KoBEST SentiNeg - 부정 표현 감성 분석
# TF-IDF (char + word n-gram) + 선형 분류기 앙상블
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load():
    train = pd.read_csv(os.path.join(BASE, "train.csv"))
    test = pd.read_csv(os.path.join(BASE, "test.csv"))
    return train, test

def make_union():
    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5),
        min_df=2, max_df=0.95, sublinear_tf=True,
    )
    word_vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2),
        min_df=2, max_df=0.95, sublinear_tf=True,
        token_pattern=r"(?u)\S+",
    )
    return FeatureUnion([("char", char_vec), ("word", word_vec)])

def build(model_name):
    vec = make_union()
    if model_name == "lr":
        clf = LogisticRegression(C=4.0, max_iter=3000, random_state=42)
    elif model_name == "svm":
        clf = LinearSVC(C=0.5, random_state=42)
    elif model_name == "nb":
        clf = ComplementNB(alpha=0.3)
    return Pipeline([("vec", vec), ("clf", clf)])

def main():
    train, test = load()
    X, y = train["sentence"].fillna(""), train["label"].values
    Xt = test["sentence"].fillna("")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = {}
    for name in ["lr", "svm", "nb"]:
        cv = cross_val_score(build(name), X, y, cv=skf, scoring="accuracy", n_jobs=3)
        scores[name] = cv
        print(f"{name}: {cv.mean():.4f} +/- {cv.std():.4f}")

    # 4개 모델 다수결 앙상블 (동점 시 1)
    preds = []
    for name in ["lr", "svm", "nb"]:
        m = build(name)
        m.fit(X, y)
        preds.append(m.predict(Xt))
    lr16 = Pipeline([("vec", make_union()), ("clf", LogisticRegression(C=16.0, max_iter=3000, random_state=42))])
    lr16.fit(X, y)
    preds.append(lr16.predict(Xt))
    preds = np.array(preds)
    final = (preds.sum(axis=0) >= 2).astype(int)

    sub = pd.DataFrame({"id": test["id"], "label": final})
    os.makedirs(os.path.join(BASE, "outputs"), exist_ok=True)
    out = os.path.join(BASE, "outputs", "submission.csv")
    sub.to_csv(out, index=False)
    print("saved:", out, len(sub), "rows; label dist:", sub["label"].value_counts().to_dict())
    assert sub["id"].is_unique and len(sub) == len(test)

if __name__ == "__main__":
    main()
