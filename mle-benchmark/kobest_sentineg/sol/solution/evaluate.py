"""Compare offline text-classification candidates on fixed folds."""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv")
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)

    char_params = {
        "analyzer": "char",
        "ngram_range": (2, 6),
        "min_df": 2,
        "sublinear_tf": True,
        "max_features": 150_000,
    }
    word_params = {
        "analyzer": "word",
        "ngram_range": (1, 2),
        "min_df": 2,
        "sublinear_tf": True,
        "max_features": 80_000,
    }
    candidates = {
        "char_C0.7": Pipeline(
            [("tfidf", TfidfVectorizer(**char_params)), ("clf", LinearSVC(C=0.7, dual=True))]
        ),
        "char_C1.2": Pipeline(
            [("tfidf", TfidfVectorizer(**char_params)), ("clf", LinearSVC(C=1.2, dual=True))]
        ),
        "word_C1.0": Pipeline(
            [("tfidf", TfidfVectorizer(**word_params)), ("clf", LinearSVC(C=1.0, dual=True))]
        ),
        "char_word_C0.8": Pipeline(
            [
                (
                    "features",
                    FeatureUnion(
                        [
                            ("char", TfidfVectorizer(**char_params)),
                            ("word", TfidfVectorizer(**word_params)),
                        ]
                    ),
                ),
                ("clf", LinearSVC(C=0.8, dual=True)),
            ]
        ),
    }

    for ngram_range, min_df in [((1, 5), 2), ((2, 5), 1), ((2, 5), 2), ((2, 7), 2), ((3, 6), 2)]:
        name = f"char_{ngram_range[0]}-{ngram_range[1]}_df{min_df}"
        candidates[name] = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="char",
                        ngram_range=ngram_range,
                        min_df=min_df,
                        sublinear_tf=True,
                        max_features=180_000,
                    ),
                ),
                ("clf", LinearSVC(C=0.7, dual=True)),
            ]
        )

    candidates["char_wb_2-6"] = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 6),
                    min_df=2,
                    sublinear_tf=True,
                    max_features=180_000,
                ),
            ),
            ("clf", LinearSVC(C=0.7, dual=True)),
        ]
    )

    for c in (0.3, 0.5, 1.0, 1.5):
        candidates[f"char_1-5_C{c}"] = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="char",
                        ngram_range=(1, 5),
                        min_df=2,
                        sublinear_tf=True,
                        max_features=180_000,
                    ),
                ),
                ("clf", LinearSVC(C=c, dual=True)),
            ]
        )

    for name, model in candidates.items():
        scores = cross_val_score(
            model, train["sentence"], train["label"], cv=folds, scoring="accuracy"
        )
        print(f"{name:18s} mean={scores.mean():.5f} std={scores.std():.5f} folds={scores}")


if __name__ == "__main__":
    main()
