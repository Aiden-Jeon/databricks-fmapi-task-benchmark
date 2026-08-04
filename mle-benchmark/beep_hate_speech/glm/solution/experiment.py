"""Experiment: sweep TF-IDF configs + classifiers with StratifiedKFold OOF macro-F1."""
import os
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TRAIN = os.path.join(ROOT, "train.csv")
TEST = os.path.join(ROOT, "test.csv")
RANDOM_STATE = 42
N_SPLITS = 5
CLASSES = ["none", "offensive", "hate"]
SORTED = sorted(CLASSES)  # ['hate','none','offensive']


def load():
    tr = pd.read_csv(TRAIN)
    te = pd.read_csv(TEST)
    tr["comment"] = tr["comment"].fillna("")
    te["comment"] = te["comment"].fillna("")
    return tr, te


def make_tfidf(tr_text, te_text, analyzer, ngram_range, min_df=2, max_df=0.95, max_features=200000, sublinear=True):
    vec = TfidfVectorizer(
        sublinear_tf=sublinear,
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        token_pattern=r"(?u)\b\w+\b",
    )
    vec.fit(list(tr_text) + list(te_text))
    Xtr = vec.transform(tr_text)
    Xte = vec.transform(te_text)
    return Xtr, Xte


def cv_oof_test(Xtr, Xte, y, model_factory, n_splits=N_SPLITS):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros((len(y), 3))
    test_proba = np.zeros((Xte.shape[0], 3))
    for tr_idx, va_idx in skf.split(Xtr, y):
        m = model_factory()
        m.fit(Xtr[tr_idx], y[tr_idx])
        oof[va_idx] = m.predict_proba(Xva_f := Xtr[va_idx])
        test_proba += m.predict_proba(Xte) / n_splits
    return oof, test_proba


def lr_factory(C=4.0):
    return LogisticRegression(
        C=C, max_iter=2000, solver="liblinear", class_weight="balanced", random_state=RANDOM_STATE
    )


def lgb_factory():
    # LightGBM needs dense-ish or sparse ok; use label encoding
    le = LabelEncoder().fit(SORTED)
    return LGBWrapper()


class LGBWrapper:
    def __init__(self):
        from sklearn.preprocessing import LabelEncoder
        self.le = LabelEncoder().fit(SORTED)
        self.model = None

    def fit(self, X, y):
        y_num = self.le.transform(y)
        dtr = lgb.Dataset(X, y_num)
        params = dict(
            objective="multiclass",
            num_class=3,
            metric="multi_logloss",
            learning_rate=0.05,
            num_leaves=63,
            min_data_in_leaf=20,
            feature_fraction=0.7,
            bagging_fraction=0.8,
            bagging_freq=5,
            verbosity=-1,
            seed=RANDOM_STATE,
        )
        self.model = lgb.train(params, dtr, num_boost_round=500)
        return self

    def predict_proba(self, X):
        p = self.model.predict(X)
        # p columns are in sorted-label order (0=hate,1=none,2=offensive) -> matches SORTED
        return p


from sklearn.preprocessing import LabelEncoder


def remap_to_classes(proba, src_order=SORTED, dst_order=CLASSES):
    out = np.zeros((proba.shape[0], len(dst_order)))
    for i, c in enumerate(src_order):
        j = dst_order.index(c)
        out[:, j] = proba[:, i]
    return out


def macro_f1_from_proba(y_true, proba, dst_order=CLASSES):
    pred = np.array([dst_order[i] for i in proba.argmax(1)])
    return f1_score(y_true, pred, average="macro")


def run():
    tr, te = load()
    y = tr["label"].values
    tr_text = tr["comment"].values
    te_text = te["comment"].values

    results = {}

    # Feature sets
    feats = {}
    # char 1-4
    feats["char14"] = make_tfidf(tr_text, te_text, "char_wb", (1, 4), max_features=200000)
    # char 2-5
    feats["char25"] = make_tfidf(tr_text, te_text, "char_wb", (2, 5), max_features=200000)
    # word 1-2
    feats["word12"] = make_tfidf(tr_text, te_text, "word", (1, 2), max_features=100000)
    # word 1-3
    feats["word13"] = make_tfidf(tr_text, te_text, "word", (1, 3), max_features=100000)

    # Combined
    Xtr = hstack([feats["char14"][0], feats["word12"][0]]).tocsr()
    Xte = hstack([feats["char14"][1], feats["word12"][1]]).tocsr()
    oof, tp = cv_oof_test(Xtr, Xte, y, lambda: lr_factory(C=4.0))
    oof_c = remap_to_classes(oof)
    tp_c = remap_to_classes(tp)
    f1 = macro_f1_from_proba(y, oof_c)
    results["lr_char14_word12_C4"] = f1
    print(f"lr_char14_word12_C4 OOF macroF1={f1:.4f}")

    for C in [1.0, 2.0, 4.0, 8.0, 16.0]:
        oof, tp = cv_oof_test(Xtr, Xte, y, lambda C=C: lr_factory(C=C))
        oof_c = remap_to_classes(oof)
        f1 = macro_f1_from_proba(y, oof_c)
        results[f"lr_C{C}"] = f1
        print(f"lr_C{C} OOF macroF1={f1:.4f}")

    # char25 + word12
    Xtr2 = hstack([feats["char25"][0], feats["word12"][0]]).tocsr()
    Xte2 = hstack([feats["char25"][1], feats["word12"][1]]).tocsr()
    oof, tp = cv_oof_test(Xtr2, Xte2, y, lambda: lr_factory(C=4.0))
    oof_c = remap_to_classes(oof)
    f1 = macro_f1_from_proba(y, oof_c)
    results["lr_char25_word12_C4"] = f1
    print(f"lr_char25_word12_C4 OOF macroF1={f1:.4f}")

    # char14 + word13
    Xtr3 = hstack([feats["char14"][0], feats["word13"][0]]).tocsr()
    Xte3 = hstack([feats["char14"][1], feats["word13"][1]]).tocsr()
    oof, tp = cv_oof_test(Xtr3, Xte3, y, lambda: lr_factory(C=4.0))
    oof_c = remap_to_classes(oof)
    f1 = macro_f1_from_proba(y, oof_c)
    results["lr_char14_word13_C4"] = f1
    print(f"lr_char14_word13_C4 OOF macroF1={f1:.4f}")

    # char14 only
    oof, tp = cv_oof_test(feats["char14"][0], feats["char14"][1], y, lambda: lr_factory(C=4.0))
    oof_c = remap_to_classes(oof)
    f1 = macro_f1_from_proba(y, oof_c)
    results["lr_char14_only_C4"] = f1
    print(f"lr_char14_only_C4 OOF macroF1={f1:.4f}")

    # LightGBM on char14+word12
    print("Running LightGBM (may be slow)...")
    try:
        oof, tp = cv_oof_test(Xtr, Xte, y, lambda: LGBWrapper())
        oof_c = remap_to_classes(oof)
        f1 = macro_f1_from_proba(y, oof_c)
        results["lgb_char14_word12"] = f1
        print(f"lgb_char14_word12 OOF macroF1={f1:.4f}")
    except Exception as e:
        print("LGB failed:", e)

    print("\n=== SUMMARY ===")
    for k, v in sorted(results.items(), key=lambda x: -x[1]):
        print(f"{v:.4f}  {k}")


if __name__ == "__main__":
    run()
