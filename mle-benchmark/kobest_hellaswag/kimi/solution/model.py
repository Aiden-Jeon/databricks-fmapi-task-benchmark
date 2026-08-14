# -*- coding: utf-8 -*-
"""
KoBEST HellaSwag — 랭킹 피처 + LogisticRegression 모델.

핵심 아이디어:
  - 가장 강한 신호는 "context 마지막 문장"과 ending의 TF-IDF(char_wb) 코사인 유사도.
  - pair별 절대 피처 대신, 문항 내 4개 후보 간 **상대 피처(문항 내 z-score, 차이)**를 사용해
    분류기가 후보 간 비교 구조를 학습하도록 한다.
  - 학습: (row, candidate) 단위 이진 분류(정답=1). 추론: 4개 점수 중 argmax.
  - 검증: 문항 단위 5-fold CV로 정확도 측정.

실행: python solution/model.py
"""
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

END = ["ending_1", "ending_2", "ending_3", "ending_4"]
SEED = 42

FINAL_ENDING = re.compile(
    r"(다|요|까|죠|네요|니다|습니다|었다|였다|한다|된다|있다|없다|같다|것이다|거다|란다)\.?$"
)
CONN_START = re.compile(
    r"^(그래서|그러나|그런데|그리고|그러자|그때|그제서야|그러고는|그 후|그다음|다음으로|이후|하지만|따라서|마침내|드디어|곧|이어서|계속해서|잠시 후|먼저|마지막으로|이제|다시)"
)


def last_sentences(text, n=1):
    sents = re.split(r"(?<=[.!?])\s+", str(text).strip())
    sents = [s for s in sents if s]
    return " ".join(sents[-n:]) if sents else str(text)


def tokset(s):
    return set(str(s).split())


class Featurizer:
    """train+test 텍스트로 벡터라이저를 fit하고, 문항별 (n,4,F) 피처 생성."""

    def __init__(self):
        self.vecs = {}

    def fit(self, dfs):
        corpus = []
        for df in dfs:
            corpus.append(df["context"].fillna(""))
            for c in END:
                corpus.append(df[c].fillna(""))
        corpus = pd.concat(corpus)
        specs = {
            "c13": ("char_wb", (1, 3)),
            "c12": ("char_wb", (1, 2)),
            "c24": ("char_wb", (2, 4)),
        }
        for name, (an, ng) in specs.items():
            v = TfidfVectorizer(analyzer=an, ngram_range=ng, min_df=1, sublinear_tf=True)
            v.fit(corpus)
            self.vecs[name] = v
        vw = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1,
                             sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
        vw.fit(corpus)
        self.vecs["word"] = vw
        return self

    def _sims(self, df, ctx_texts):
        out = {}
        for name, v in self.vecs.items():
            C = v.transform(ctx_texts)
            cols = []
            for c in END:
                E = v.transform(df[c].fillna(""))
                Cn = C.multiply(1.0 / (np.sqrt(C.multiply(C).sum(axis=1)) + 1e-9))
                En = E.multiply(1.0 / (np.sqrt(E.multiply(E).sum(axis=1)) + 1e-9))
                cols.append(np.asarray(Cn.multiply(En).sum(axis=1)).ravel())
            out[name] = np.vstack(cols).T  # (n,4)
        return out

    def transform(self, df):
        n = len(df)
        last1 = [last_sentences(c, 1) for c in df["context"]]
        last2 = [last_sentences(c, 2) for c in df["context"]]
        s_last1 = self._sims(df, last1)
        s_last2 = self._sims(df, last2)
        s_full = self._sims(df, df["context"].fillna("").tolist())

        # 원시 (n,4) 유사도
        raw = {}
        for name in self.vecs:
            raw[f"{name}_l1"] = s_last1[name]
            raw[f"{name}_l2"] = s_last2[name]
            raw[f"{name}_fu"] = s_full[name]

        # 토큰 jaccard (마지막 문장 기준)
        jac = np.zeros((n, 4))
        for i, c in enumerate(END):
            for r in range(n):
                a = tokset(last1[r]); b = tokset(df[c].iloc[r])
                jac[r, i] = len(a & b) / (len(a | b) or 1)
        raw["jac_l1"] = jac

        # 길이/구문 피처 (n,4)
        len_char = np.stack([df[c].fillna("").str.len().values for c in END], axis=1).astype(float)
        len_tok = np.stack([df[c].fillna("").str.split().str.len().values for c in END], axis=1).astype(float)
        final_end = np.stack([[1.0 if FINAL_ENDING.search(str(x).strip()) else 0.0 for x in df[c]] for c in END], axis=1)
        conn_start = np.stack([[1.0 if CONN_START.search(str(x).strip()) else 0.0 for x in df[c]] for c in END], axis=1)
        ends_dot = np.stack([[1.0 if str(x).strip().endswith(".") else 0.0 for x in df[c]] for c in END], axis=1)

        raw["len_char"] = len_char
        raw["len_tok"] = len_tok
        raw["final_end"] = final_end
        raw["conn_start"] = conn_start
        raw["ends_dot"] = ends_dot

        # 문항 내 정규화: z-score (4개 후보 기준)
        feat_blocks = []
        names = []
        for k, M in raw.items():
            mu = M.mean(axis=1, keepdims=True)
            sd = M.std(axis=1, keepdims=True) + 1e-9
            z = (M - mu) / sd
            feat_blocks.append(z)
            names += [f"{k}_z"] * 4
            # 원시값도 일부 포함 (유사도 계열만)
            if k.endswith(("_l1", "_fu", "_l2")):
                feat_blocks.append(M)
                names += [f"{k}_raw"] * 4
        X = np.concatenate(feat_blocks, axis=1)  # (n, 4*F)
        return X, names


def pair_flat(X):
    """(n,4*F) -> (n*4, F) : 후보별 행. 4개 연속이 한 문항."""
    n = X.shape[0]
    F = X.shape[1] // 4
    Xp = X.reshape(n, 4, F).reshape(n * 4, F)
    return Xp, F


def main():
    tr = pd.read_csv("train.csv")
    te = pd.read_csv("test.csv")

    fez = Featurizer().fit([tr, te])
    Xtr, names = fez.transform(tr)
    Xte, _ = fez.transform(te)
    y = tr["label"].values
    correct = (np.tile(np.arange(4), len(tr)) == np.repeat(y, 4)).astype(int)

    Xtr_p, F = pair_flat(Xtr)
    Xte_p, _ = pair_flat(Xte)
    groups = np.repeat(np.arange(len(tr)), 4)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(tr), dtype=int)
    for ti, vi in skf.split(np.zeros(len(tr)), y):
        mtr = np.isin(groups, ti)
        mva = np.isin(groups, vi)
        clf = LogisticRegression(C=1.0, max_iter=3000)
        clf.fit(Xtr_p[mtr], correct[mtr])
        p = clf.predict_proba(Xtr_p[mva])[:, 1].reshape(-1, 4)
        oof[vi] = p.argmax(axis=1)
    acc = accuracy_score(y, oof)
    print(f"[CV] 5-fold accuracy: {acc:.4f}")

    clf = LogisticRegression(C=1.0, max_iter=3000)
    clf.fit(Xtr_p, correct)
    pte = clf.predict_proba(Xte_p)[:, 1].reshape(-1, 4)
    pred = pte.argmax(axis=1)
    sub = pd.DataFrame({"id": te["id"], "label": pred})
    sub.to_csv("outputs/submission.csv", index=False)
    print("pred dist:", sub["label"].value_counts(normalize=True).round(3).to_dict())
    print("saved outputs/submission.csv", sub.shape)


if __name__ == "__main__":
    main()
