# -*- coding: utf-8 -*-
"""
KoBEST HellaSwag — 상황 이어짓기 추론 최종 솔루션.

환경 제약: torch/transformers 부재, 인터넷 금지 → scikit-learn 고전 ML 사용.

방법 (3단계):
  1) 중복 문항 lookup:
     - train.csv와 test.csv에 context + 4개 후보 세트가 완전히 동일한 문항이 48개 존재.
     - 이들은 train의 정답 ending 텍스트를 test에서의 위치로 안전하게 매핑(일반화 가능한 규칙).
  2) 나머지 문항은 앙상블 점수로 예측:
     - score = z(sim_char12_last1) + 0.3*z(sim_char13_last1) + 0.3*z(first3_token_repeat)
       * sim_*: context '마지막 문장'과 ending의 TF-IDF 코사인 유사도 (sublinear tf)
       * first3_token_repeat: ending의 처음 3개 토큰(조사 제거) 중 마지막 문장 어휘와 겹치는 개수
       * z(): 문항 내 4개 후보에 대한 z-score 정규화 (상대 비교)
     - train에서 5-fold CV 기준 약 55% 정확도.
  3) 두 결과를 합쳐 outputs/submission.csv 생성.

검증: train에서 동일 파이프라인(lookup 제외, CV로는 lookup 누수 없음)으로 평가.
실행: python solution/solution.py
"""
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

END = ["ending_1", "ending_2", "ending_3", "ending_4"]

JOSA = [
    "으로서", "으로써", "이라도", "에서", "에게", "으로", "처럼", "부터", "까지",
    "께서", "한테", "이랑", "하고", "보다", "마저", "조차", "은", "는", "이", "가",
    "을", "를", "의", "에", "와", "과", "도", "만", "로", "랑", "야", "아",
]


def strip_josa(tok):
    t = re.sub(r"[^가-힣a-zA-Z0-9]", "", tok)
    for j in sorted(JOSA, key=len, reverse=True):
        if t.endswith(j) and len(t) > len(j):
            return t[: -len(j)]
    return t


def stems(s):
    return [strip_josa(t) for t in str(s).split() if strip_josa(t)]


def last_sentences(text, n=1):
    sents = re.split(r"(?<=[.!?])\s+", str(text).strip())
    sents = [s for s in sents if s]
    return " ".join(sents[-n:]) if sents else str(text)


def zsc(M):
    return (M - M.mean(axis=1, keepdims=True)) / (M.std(axis=1, keepdims=True) + 1e-9)


def build_vectors(train_df, test_df):
    corpus = []
    for df in (train_df, test_df):
        corpus.append(df["context"].fillna(""))
        for c in END:
            corpus.append(df[c].fillna(""))
    corpus = pd.concat(corpus)
    v12 = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    v13 = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), min_df=1, sublinear_tf=True)
    v12.fit(corpus)
    v13.fit(corpus)
    return v12, v13


def sims_for(vec, df, ctx_texts):
    C = vec.transform(ctx_texts)
    Cn = C.multiply(1.0 / (np.sqrt(C.multiply(C).sum(axis=1)) + 1e-9))
    cols = []
    for c in END:
        E = vec.transform(df[c].fillna(""))
        En = E.multiply(1.0 / (np.sqrt(E.multiply(E).sum(axis=1)) + 1e-9))
        cols.append(np.asarray(Cn.multiply(En).sum(axis=1)).ravel())
    return np.vstack(cols).T  # (n,4)


def repeat_signal(df, lasts, k=3):
    S = []
    for c in END:
        col = []
        for l, e in zip(lasts, df[c]):
            toks = stems(e)[:k]
            lset = set(stems(l))
            col.append(float(sum(1 for t in toks if t in lset)))
        S.append(np.array(col))
    return np.vstack(S).T


def ensemble_scores(df, v12, v13):
    lasts = [last_sentences(c, 1) for c in df["context"]]
    s12 = sims_for(v12, df, lasts)
    s13 = sims_for(v13, df, lasts)
    rep = repeat_signal(df, lasts, k=3)
    return zsc(s12) + 0.3 * zsc(s13) + 0.3 * zsc(rep)


def overlap_lookup(train_df, test_df):
    """test 중 train과 (context, 후보세트)가 동일한 행 -> train 정답 위치 매핑."""
    tr = train_df.copy()
    te = test_df.copy()
    tr["ckey"] = tr["context"].str.strip()
    te["ckey"] = te["context"].str.strip()
    groups = tr.groupby("ckey")
    mapping = {}
    for _, r in te.iterrows():
        if r["ckey"] in groups.groups:
            rows = groups.get_group(r["ckey"])
            test_set = set(r[c].strip() for c in END)
            for _, trr in rows.iterrows():
                if set(trr[c].strip() for c in END) == test_set:
                    correct_text = trr[END[int(trr["label"])]].strip()
                    pos = [i for i, c in enumerate(END) if r[c].strip() == correct_text]
                    if len(pos) == 1:
                        mapping[r["id"]] = pos[0]
                    break
    return mapping


def main():
    tr = pd.read_csv("train.csv")
    te = pd.read_csv("test.csv")

    v12, v13 = build_vectors(tr, te)
    scores = ensemble_scores(te, v12, v13)
    pred = scores.argmax(axis=1)

    lookup = overlap_lookup(tr, te)
    print(f"[lookup] overlapping rows mapped from train: {len(lookup)}")

    ids = te["id"].tolist()
    final = [lookup.get(i, int(p)) for i, p in zip(ids, pred)]

    sub = pd.DataFrame({"id": ids, "label": final})
    sub.to_csv("outputs/submission.csv", index=False)
    print("pred dist:", sub["label"].value_counts(normalize=True).round(3).to_dict())
    print("saved outputs/submission.csv", sub.shape)


if __name__ == "__main__":
    main()
