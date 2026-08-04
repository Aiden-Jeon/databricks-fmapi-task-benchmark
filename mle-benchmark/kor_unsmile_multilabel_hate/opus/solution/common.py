"""Shared utilities for the Korean UnSmile multi-label hate speech task."""
import re
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

N_LABELS = 10
CLASSES = [
    "여성/가족", "남성", "성소수자", "인종/국적", "연령",
    "지역", "종교", "기타 혐오", "악플/욕설", "clean",
]
CLEAN_IDX = 9


def load(data_dir="."):
    tr = pd.read_csv(f"{data_dir}/train.csv")
    te = pd.read_csv(f"{data_dir}/test.csv")
    return tr, te


def labels_to_matrix(s):
    s = s.astype(str).str.zfill(N_LABELS)
    return np.array([[int(c) for c in v] for v in s], dtype=np.int8)


def matrix_to_labels(Y):
    return ["".join(str(int(v)) for v in row) for row in Y]


_JAMO_LEAD = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_JAMO_VOWEL = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_JAMO_TAIL = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"


def decompose_hangul(text):
    """Break Korean syllables into jamo so that partial spelling variants
    (e.g. 'ㅅㅂ' vs '시발') share sub-token features."""
    out = []
    for ch in text:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            idx = code - 0xAC00
            out.append(_JAMO_LEAD[idx // 588])
            out.append(_JAMO_VOWEL[(idx % 588) // 28])
            t = _JAMO_TAIL[idx % 28]
            if t != " ":
                out.append(t)
        else:
            out.append(ch)
    return "".join(out)


_WS = re.compile(r"\s+")
_REPEAT = re.compile(r"(.)\1{2,}")


def normalize(text):
    text = str(text)
    text = _WS.sub(" ", text)
    text = _REPEAT.sub(r"\1\1", text)  # ㅋㅋㅋㅋㅋ -> ㅋㅋ
    return text.strip()


def build_features(train_texts, test_texts, verbose=False):
    """char_wb + word + jamo TF-IDF blocks, fitted on train only."""
    tr_n = [normalize(t) for t in train_texts]
    te_n = [normalize(t) for t in test_texts]
    tr_j = [decompose_hangul(t) for t in tr_n]
    te_j = [decompose_hangul(t) for t in te_n]

    blocks = [
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                 min_df=2, sublinear_tf=True,
                                 strip_accents=None, lowercase=True), tr_n, te_n),
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                 min_df=2, sublinear_tf=True,
                                 token_pattern=r"(?u)\S+"), tr_n, te_n),
        ("jamo", TfidfVectorizer(analyzer="char", ngram_range=(2, 5),
                                 min_df=3, sublinear_tf=True), tr_j, te_j),
    ]
    Xtr, Xte = [], []
    for name, vec, a, b in blocks:
        Xtr.append(vec.fit_transform(a))
        Xte.append(vec.transform(b))
        if verbose:
            print(f"  {name}: {Xtr[-1].shape[1]} features")
    return sparse.hstack(Xtr).tocsr(), sparse.hstack(Xte).tocsr()


def macro_f1(Y_true, Y_pred):
    """Macro F1 over labels that occur in the ground truth."""
    f1s = []
    for j in range(Y_true.shape[1]):
        t, p = Y_true[:, j], Y_pred[:, j]
        if t.sum() == 0:
            continue
        tp = float((t & p).sum())
        prec = tp / p.sum() if p.sum() else 0.0
        rec = tp / t.sum()
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s))


def apply_thresholds(P, thr, force_one=True):
    Y = (P >= thr[None, :]).astype(np.int8)
    if force_one:
        empty = Y.sum(1) == 0
        if empty.any():
            # every training row has >= 1 label -> assign the best-scoring one
            best = (P[empty] / np.maximum(thr[None, :], 1e-9)).argmax(1)
            Y[np.where(empty)[0], best] = 1
    # 'clean' is mutually exclusive with every hate category in the data
    hate_any = Y[:, :CLEAN_IDX].sum(1) > 0
    both = hate_any & (Y[:, CLEAN_IDX] == 1)
    if both.any():
        idx = np.where(both)[0]
        clean_rel = P[idx, CLEAN_IDX] / max(thr[CLEAN_IDX], 1e-9)
        hate_rel = (P[idx, :CLEAN_IDX] / np.maximum(thr[None, :CLEAN_IDX], 1e-9)).max(1)
        drop_hate = clean_rel > hate_rel
        for k, i in enumerate(idx):
            if drop_hate[k]:
                Y[i, :CLEAN_IDX] = 0
            else:
                Y[i, CLEAN_IDX] = 0
    return Y


def tune_thresholds(Y_true, P, grid=None, rounds=3, force_one=True):
    """Coordinate ascent on per-label thresholds to maximise macro F1."""
    if grid is None:
        grid = np.arange(0.05, 0.86, 0.01)
    thr = np.full(P.shape[1], 0.5)
    best = macro_f1(Y_true, apply_thresholds(P, thr, force_one))
    for _ in range(rounds):
        improved = False
        for j in range(P.shape[1]):
            cur = thr[j]
            for g in grid:
                thr[j] = g
                s = macro_f1(Y_true, apply_thresholds(P, thr, force_one))
                if s > best + 1e-9:
                    best, cur, improved = s, g, True
            thr[j] = cur
        if not improved:
            break
    return thr, best
