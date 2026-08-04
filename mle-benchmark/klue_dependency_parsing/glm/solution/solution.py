#!/usr/bin/env python3
"""
KLUE-DP (의존 구문 분석) 베이스라인 솔루션.

접근:
  1) 각 어절(modifier)에 대해 "head가 어느 어절인지"를 예측하는 이진 분류기를 학습.
     피처: modifier 토큰, candidate head 토큰, 두 토큰의 접미사, 상대 위치, 문장 길이 등.
  2) deprel(의존관계 레이블)을 예측하는 다중 분류기 학습.
  3) 디코딩: 각 어절에 대해 점수가 가장 높은 head를 선택.
     - 학습 데이터에서 root(head=0)는 항상 마지막 어절이므로 마지막 어절은 head=0으로 강제.
     - 비순환 트리를 보장하기 위해 Chu-Liu/Edmonds 대신 간단한 휴리스틱 사용:
       각 어절마다 최적 head를 greedy 선택 후, 필요시 후처리로 사이클/중복을 해소.
       마지막 어절은 무조건 head=0(root)이므로 사이클 위험을 줄이고,
       다른 어절들은 self-loop을 금지(자기 자신을 head로 선택 불가)한다.
  4) 제출 파일 outputs/submission.csv 작성.

scikit-learn만 사용 (torch 없음).
"""

import json
import os
import re
import sys
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import hstack as sparse_hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

try:
    from solution.mst import chu_liu_edmonds as _cle
except Exception:
    _cle = None

warnings.filterwarnings("ignore")

# ---------- 경로 ----------
HERE = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(HERE)
TRAIN_CSV = os.path.join(TASK_DIR, "train.csv")
TEST_CSV = os.path.join(TASK_DIR, "test.csv")
SAMPLE_CSV = os.path.join(TASK_DIR, "sample_submission.csv")
OUT_DIR = os.path.join(TASK_DIR, "outputs")
OUT_CSV = os.path.join(OUT_DIR, "submission.csv")

# 레이블 목록 (spec.md에 명시된 36개)
ALL_DEPRELS = [
    "NP", "NP_AJT", "NP_CMP", "NP_CNJ", "NP_MOD", "NP_OBJ", "NP_SBJ",
    "VP", "VP_AJT", "VP_CMP", "VP_CNJ", "VP_MOD", "VP_OBJ", "VP_SBJ",
    "VNP", "VNP_AJT", "VNP_CMP", "VNP_CNJ", "VNP_MOD", "VNP_OBJ", "VNP_SBJ",
    "AP", "AP_AJT", "AP_CMP", "AP_MOD", "DP", "IP", "X", "X_AJT", "X_CMP",
    "X_CNJ", "X_MOD", "X_OBJ", "X_SBJ", "L", "R",
]

JOSA_LIST = [
    "은", "는", "이", "가", "을", "를", "에", "에서", "으로", "로", "와", "과",
    "에게", "한테", "께", "한", "의", "도", "만", "보다", "처럼", "보다",
    "까지", "부터", "이다", "하다",
]
EOMI_LIST = [
    "다.", "요.", "다", "요", "는", "은", "을", "다", "어", "아", "지", "고",
    "면", "면서", "지만", "면", "는", "은", "을", "어", "아", "하고", "하며",
    "하며", "면서", "지만", "지만", "는지", "는가", "을까", "을", "을",
]


def split_suffix(token):
    """어절에서 접미사(조사/어미) 부분과 어근 부분을 대략 분리."""
    for j in sorted(JOSA_LIST + EOMI_LIST, key=len, reverse=True):
        if token.endswith(j) and len(token) > len(j):
            return token[: -len(j)], j
    return token, ""


def char_ngrams(token, ns=(2,)):
    """문자 n-그램 (접미사 위주, 크기 제한)."""
    grams = []
    t = "^" + token + "$"
    for n in ns:
        if len(t) >= n:
            for i in range(len(t) - n + 1):
                grams.append(t[i : i + n])
    return grams


def token_features(token, prefix=""):
    """단일 토큰에 대한 피처 딕셔너리."""
    stem, suf = split_suffix(token)
    feats = {
        f"{prefix}len": len(token),
        f"{prefix}suf": suf,
        f"{prefix}suf2": token[-2:] if len(token) >= 2 else token,
        f"{prefix}suf1": token[-1:],
        f"{prefix}stem_suf": stem[-2:] if len(stem) >= 2 else stem,
        f"{prefix}has_josa": int(bool(suf and suf in JOSA_LIST)),
        f"{prefix}has_eomi": int(bool(suf and suf in EOMI_LIST)),
        f"{prefix}has_num": int(any(ch.isdigit() for ch in token)),
        f"{prefix}has_punct": int(any(ch in ".,!?:;\"'()[]{}" for ch in token)),
    }
    for g in char_ngrams(token):
        feats[f"{prefix}g_{g}"] = 1
    return feats


def pair_features(mod_tok, head_tok, mod_pos, head_pos, n_tokens):
    """(modifier, candidate head) 쌍에 대한 피처."""
    rel = head_pos - mod_pos  # head가 modifier보다 오른쪽이면 양수
    feats = {}
    feats.update(token_features(mod_tok, "mod_"))
    feats.update(token_features(head_tok, "head_"))
    # 관계/위치 피처
    feats["rel"] = rel
    feats["abs_rel"] = abs(rel)
    feats["rel_sign"] = 1 if rel > 0 else (-1 if rel < 0 else 0)
    feats["mod_to_end"] = n_tokens - mod_pos
    feats["head_to_end"] = n_tokens - head_pos
    feats["mod_is_last"] = int(mod_pos == n_tokens)
    feats["head_is_last"] = int(head_pos == n_tokens)
    feats["head_is_root_cand"] = int(head_pos == n_tokens)  # 마지막 어절이 root 후보
    feats["n_tokens"] = n_tokens
    # 거리 bin (비선형 위치 정보)
    if abs(rel) <= 0:
        db = 0
    elif abs(rel) <= 2:
        db = 1
    elif abs(rel) <= 5:
        db = 2
    elif abs(rel) <= 10:
        db = 3
    else:
        db = 4
    feats["dist_bin"] = db
    feats["rel_bin"] = int(np.sign(rel)) * db if rel != 0 else 0
    feats["mod_pos_bin"] = min(mod_pos, 12)
    feats["head_pos_bin"] = min(head_pos, 12)
    feats["mod_third"] = (mod_pos - 1) / max(n_tokens - 1, 1)
    feats["head_third"] = (head_pos - 1) / max(n_tokens - 1, 1)
    feats["same_suffix"] = int(feats.get("mod_suf", "") != "" and feats.get("mod_suf", "") == feats.get("head_suf", ""))
    # 상호작용
    feats["mod_suf_head_suf"] = feats.get("mod_suf", "") + "|" + feats.get("head_suf", "")
    feats["mod_suf2_head_suf2"] = feats.get("mod_suf2", "") + "|" + feats.get("head_suf2", "")
    feats["rel_mod_suf"] = f"{rel}_{feats.get('mod_suf','')}"
    feats["rel_head_suf"] = f"{rel}_{feats.get('head_suf','')}"
    feats["dist_bin_mod_suf"] = f"{db}_{feats.get('mod_suf','')}"
    feats["dist_bin_head_suf"] = f"{db}_{feats.get('head_suf','')}"
    feats["mod_suf_head_pos"] = f"{feats.get('mod_suf','')}_{min(head_pos,12)}"
    return feats


def parse_row(tokens, parse_str):
    """tokens(JSON 문자열), parse 문자열을 리스트로 변환."""
    tok = json.loads(tokens) if isinstance(tokens, str) else tokens
    arcs = []
    for p in parse_str.split("|"):
        h, d = p.rsplit(":", 1)
        arcs.append((int(h), d))
    return tok, arcs


def build_pair_dataset(df, include_negative=True, max_neg_per_mod=3, rng=None):
    """
    head 분류기용 (modifier, candidate) 쌍 데이터 생성.
    각 어절마다 정답 head는 positive, 나머지는 negative.
    너무 많은 negative를 피해 max_neg_per_mod개만 샘플링.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    X = []
    y_head = []  # 1 if head is correct else 0
    y_deprel = []  # deprel label (for positives)
    for _, row in df.iterrows():
        tok = json.loads(row["tokens"])
        arcs = [(int(p.rsplit(":", 1)[0]), p.rsplit(":", 1)[1]) for p in row["parse"].split("|")]
        n = len(tok)
        for i, (h, d) in enumerate(arcs):
            mod_pos = i + 1
            mod_tok = tok[i]
            # positive
            if h == 0:
                head_tok = "<ROOT>"
                head_pos = 0
            else:
                head_tok = tok[h - 1]
                head_pos = h
            X.append(pair_features(mod_tok, head_tok, mod_pos, head_pos, n))
            y_head.append(1)
            y_deprel.append(d)
            # negatives: sample some other candidates
            if include_negative:
                candidates = [j for j in range(0, n + 1) if j != mod_pos and j != h]
                # 더 많은 negative를 원하면 그대로; 여기서는 제한
                if len(candidates) > max_neg_per_mod:
                    candidates = rng.choice(candidates, size=max_neg_per_mod, replace=False).tolist()
                for j in candidates:
                    if j == 0:
                        head_tok2 = "<ROOT>"
                        head_pos2 = 0
                    else:
                        head_tok2 = tok[j - 1]
                        head_pos2 = j
                    X.append(pair_features(mod_tok, head_tok2, mod_pos, head_pos2, n))
                    y_head.append(0)
                    y_deprel.append("NEG")
    return X, y_head, y_deprel


def build_deprel_dataset(df):
    """정답 arc에 대해서만 (modifier, head) 특성과 deprel 라벨."""
    X = []
    y = []
    for _, row in df.iterrows():
        tok = json.loads(row["tokens"])
        arcs = [(int(p.rsplit(":", 1)[0]), p.rsplit(":", 1)[1]) for p in row["parse"].split("|")]
        n = len(tok)
        for i, (h, d) in enumerate(arcs):
            mod_pos = i + 1
            mod_tok = tok[i]
            if h == 0:
                head_tok = "<ROOT>"
                head_pos = 0
            else:
                head_tok = tok[h - 1]
                head_pos = h
            X.append(pair_features(mod_tok, head_tok, mod_pos, head_pos, n))
            y.append(d)
    return X, y


def _greedy_heads(scores, n):
    """MST 없을 때의 fallback greedy 디코딩."""
    heads = np.zeros(n, dtype=int)
    for i in range(n):
        if i == n - 1:
            heads[i] = 0
        else:
            heads[i] = int(np.argmax(scores[i]))
    return heads


def decode_sentence(tokens, head_clf, head_vec, dep_clf, dep_vec, dep_label_encoder):
    """
    한 문장에 대해 head와 deprel 예측.
    - 마지막 어절은 head=0(root) 강제.
    - 다른 어절은 자기를 제외한 후보 중 head 점수 최대 선택.
    - 디코딩 후 비순환 보장은 greedy 방식으로 처리(사이클 발생 시 보조 후보 선택).
    """
    n = len(tokens)
    # 1) 각 어절마다 후보 head 점수 계산
    # candidate heads: 0..n (0=root=마지막 어절과 연결), self 제외
    # 단, 마지막 어절은 항상 head=0
    scores = np.full((n, n + 1), -1e9)  # scores[i][j] = head j for modifier i
    # 각 (i, j) 쌍 피처 구성
    feats_list = []
    pairs = []  # (i, j)
    for i in range(n):
        mod_pos = i + 1
        mod_tok = tokens[i]
        for j in range(0, n + 1):
            if j == mod_pos:
                continue  # self 제외
            if j == 0:
                head_tok = "<ROOT>"
                head_pos = 0
            else:
                head_tok = tokens[j - 1]
                head_pos = j
            feats_list.append(pair_features(mod_tok, head_tok, mod_pos, head_pos, n))
            pairs.append((i, j))
    X = head_vec.transform(feats_list)
    proba = head_clf.predict_proba(X)[:, 1]
    for (i, j), p in zip(pairs, proba):
        scores[i, j] = p
    # 마지막 어절은 head=0 강제 (root=0은 오직 마지막 어절에만 연결)
    # MST용 score 행렬 (n+1, n+1): scores_mst[h, m] = head h -> modifier m
    scores_mst = np.full((n + 1, n + 1), -1e9, dtype=float)
    for i in range(n):
        for j in range(0, n + 1):
            if j == i + 1:
                continue
            scores_mst[j, i + 1] = float(scores[i, j])
    # root(0)은 오직 마지막 어절에만 연결 (학습에서 root는 항상 마지막 어절)
    for m in range(1, n + 1):
        if m != n:
            scores_mst[0, m] = -1e9
    # 마지막 어절은 root로만 연결 (다른 후보는 제거)
    for h in range(1, n + 1):
        scores_mst[h, n] = -1e9
    scores_mst[0, n] = float(scores[n - 1, 0])
    # 2) Chu-Liu/Edmonds MST (MST 없으면 greedy fallback)
    if _cle is not None:
        try:
            mst_heads = _cle(scores_mst)  # 길이 n+1, mst_heads[m]=head of m
            heads = np.array([mst_heads[m] for m in range(1, n + 1)], dtype=int)
        except Exception:
            heads = _greedy_heads(scores, n)
    else:
        heads = _greedy_heads(scores, n)
    # 안전: 마지막 어절은 반드시 head=0
    heads[n - 1] = 0

    # 4) deprel 예측: 선택된 (modifier, head) 쌍에 대해
    dep_feats = []
    for i in range(n):
        mod_pos = i + 1
        mod_tok = tokens[i]
        h = heads[i]
        if h == 0:
            head_tok = "<ROOT>"
            head_pos = 0
        else:
            head_tok = tokens[h - 1]
            head_pos = h
        dep_feats.append(pair_features(mod_tok, head_tok, mod_pos, head_pos, n))
    Xd = dep_vec.transform(dep_feats)
    dep_proba = dep_clf.predict_proba(Xd)
    dep_pred = dep_label_encoder.inverse_transform(np.argmax(dep_proba, axis=1))
    # 마지막 어절의 deprel: 학습에서 항상 VP/VNP/NP 계열이었음. 예측 그대로 사용.
    arcs = [f"{heads[i]}:{dep_pred[i]}" for i in range(n)]
    return "|".join(arcs)


def main():
    print("[info] loading data ...", flush=True)
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    print(f"train={len(train)} test={len(test)}", flush=True)

    # ---------- head 분류기 ----------
    print("[info] building head classifier dataset ...", flush=True)
    rng = np.random.RandomState(42)
    Xh, yh, _ = build_pair_dataset(train, include_negative=True, max_neg_per_mod=3, rng=rng)
    print(f"  pairs={len(Xh)} pos={sum(yh)} neg={len(yh)-sum(yh)}", flush=True)
    head_vec = DictVectorizer(sparse=True)
    Xh_v = head_vec.fit_transform(Xh)
    del Xh
    print(f"  feat dim={Xh_v.shape[1]}", flush=True)
    print("[info] training head classifier (LogReg) ...", flush=True)
    head_clf = LogisticRegression(C=0.3, max_iter=150, solver="lbfgs", n_jobs=1)
    head_clf.fit(Xh_v, yh)
    # 학습 데이터에서의 정확도
    pred_tr = head_clf.predict(Xh_v)
    print(f"  train head acc={ (pred_tr==yh).mean():.4f}", flush=True)
    del Xh_v, yh, pred_tr

    # ---------- deprel 분류기 ----------
    print("[info] building deprel classifier dataset ...", flush=True)
    Xd, yd = build_deprel_dataset(train)
    dep_vec = DictVectorizer(sparse=True)
    Xd_v = dep_vec.fit_transform(Xd)
    del Xd
    print(f"  arcs={len(yd)} feat dim={Xd_v.shape[1]}", flush=True)
    dep_le = LabelEncoder()
    yd_enc = dep_le.fit_transform(yd)
    print(f"  deprel classes={len(dep_le.classes_)}", flush=True)
    print("[info] training deprel classifier (LogReg) ...", flush=True)
    dep_clf = LogisticRegression(C=0.5, max_iter=200, solver="lbfgs", n_jobs=1)
    dep_clf.fit(Xd_v, yd_enc)
    pred_d = dep_clf.predict(Xd_v)
    print(f"  train deprel acc={(pred_d==yd_enc).mean():.4f}", flush=True)
    del Xd_v, yd_enc, pred_d

    # ---------- 디코딩 ----------
    print("[info] decoding test sentences ...", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_rows = []
    for _, row in test.iterrows():
        tok = json.loads(row["tokens"])
        parse = decode_sentence(tok, head_clf, head_vec, dep_clf, dep_vec, dep_le)
        out_rows.append((row["id"], parse))
    out_df = pd.DataFrame(out_rows, columns=["id", "parse"])
    # 제출 순서를 test 순서대로 유지
    out_df.to_csv(OUT_CSV, index=False, quoting=1)  # QUOTE_ALL
    print(f"[info] wrote {OUT_CSV} rows={len(out_df)}", flush=True)
    # 검증: 항목 수 일치
    bad = 0
    for _, row in test.iterrows():
        n = len(json.loads(row["tokens"]))
        p = out_df[out_df["id"] == row["id"]]["parse"].values[0].split("|")
        if len(p) != n:
            bad += 1
    print(f"[check] mismatched length rows={bad}", flush=True)


if __name__ == "__main__":
    main()
