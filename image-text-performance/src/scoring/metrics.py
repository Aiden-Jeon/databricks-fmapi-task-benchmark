"""
Quantitative metric functions for LLM benchmark scoring.

This module provides metrics for evaluating model outputs across tasks:
- Token-level string similarity (Token-F1, exact match)
- ANLS (DocVQA official metric — edit-distance based, typo tolerant)
- Classification metrics (accuracy, F1, confusion matrix)
- Multi-label set metrics (precision/recall/F1 over label sets)

Reference-based generation metrics (ROUGE, BERTScore) and structured
extraction (Cell-F1 for table parsing) live in their respective task modules.

See plan.md §3 for task-metric alignment and appendix "채점 방법론 상세"
for Korean tokenization and metric design choices.
"""

from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from .tokenizers import tokenize


def token_f1(pred: str, gold: str, lang: str) -> float:
    """
    Token-level F1 score (SQuAD-style).

    Tokenizes both strings using language-aware tokenization, then computes
    precision/recall/F1 at the token level. For Korean, uses Mecab morpheme
    tokenization; for English, whitespace splitting. Tokens are treated as
    a multiset (duplicates count).

    Formula: F1 = 2 * (overlap) / (len(pred_tokens) + len(gold_tokens))
    where overlap = min count of each token across both sequences (multiset intersection).

    Args:
        pred: Predicted text.
        gold: Gold reference text.
        lang: Language code ('ko' or 'en').

    Returns:
        F1 score in [0, 1]. Returns 1.0 if both are empty; 0.0 if one is empty
        and the other is not.
    """
    pred_tokens = tokenize(pred, lang)
    gold_tokens = tokenize(gold, lang)

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)

    overlap = sum((pred_counter & gold_counter).values())

    total_tokens = len(pred_tokens) + len(gold_tokens)
    f1 = 2.0 * overlap / total_tokens if total_tokens > 0 else 0.0

    return f1


def exact_match(pred: str, gold: str) -> float:
    """
    Exact match score (binary).

    Compares normalized strings (whitespace stripped, case-insensitive).

    Args:
        pred: Predicted text.
        gold: Gold reference text.

    Returns:
        1.0 if strings match (normalized), 0.0 otherwise.
    """
    pred_norm = pred.strip().lower()
    gold_norm = gold.strip().lower()
    return 1.0 if pred_norm == gold_norm else 0.0


# ANLS 임계값. DocVQA 공식 정의(Biten et al., ICCV 2019)의 τ=0.5:
# 정규화 편집거리가 0.5를 넘으면(= 유사도 0.5 미만) 0점으로 떨어뜨려, 우연한 부분일치에
# 점수를 주지 않는다. OCR/표기 오타(1~2글자)는 관용하되 다른 답은 통과시키지 않는 균형점.
ANLS_THRESHOLD = 0.5


def anls(pred: str, golds: list[str] | str, threshold: float = ANLS_THRESHOLD) -> float:
    """
    ANLS (Average Normalized Levenshtein Similarity) — DocVQA 공식 메트릭.

    문서 QA는 정답이 문서에 적힌 짧은 문자열("March 27, 1979", "$485")이라
    Token-F1은 표기 차이(쉼표·통화기호·대소문자)에 과하게 민감하고, exact_match는
    한 글자 오타도 0점으로 만든다. ANLS는 편집거리 기반이라 그 중간을 취한다.

    per-sample 계산:
        NL(pred, gold) = levenshtein(pred, gold) / max(len(pred), len(gold))
        s = 1 - NL  (문자열 유사도)
        ANLS = max over golds of (s if s >= threshold else 0.0)

    여러 정답(answers 리스트)이 주어지면 **최고 점수**를 취한다(공식 정의).
    문자열은 비교 전에 strip + lowercase로 정규화한다(공식 구현과 동일).

    Args:
        pred: 모델 예측 문자열.
        golds: 정답 문자열 리스트(또는 단일 문자열).
        threshold: τ. 이 값 미만의 유사도는 0으로 절단(기본 0.5).

    Returns:
        [0, 1] 범위 float. pred와 gold가 모두 비면 1.0, 한쪽만 비면 0.0.
    """
    import Levenshtein

    if isinstance(golds, str):
        golds = [golds]

    pred_norm = str(pred).strip().lower()

    best = 0.0
    for gold in golds:
        gold_norm = str(gold).strip().lower()

        if not pred_norm and not gold_norm:
            return 1.0
        if not pred_norm or not gold_norm:
            continue  # 한쪽만 비면 이 정답에 대해 0점

        dist = Levenshtein.distance(pred_norm, gold_norm)
        similarity = 1.0 - dist / max(len(pred_norm), len(gold_norm))
        if similarity >= threshold:
            best = max(best, similarity)

    return best


def binary_metrics(preds: list[int], golds: list[int]) -> dict:
    """
    Binary classification metrics (accuracy, F1, confusion matrix).

    Computes standard binary classification metrics assuming labels are 0/1.
    Also returns the confusion matrix as {tn, fp, fn, tp}.

    Note: AUROC is intentionally NOT computed here because chat LLMs do not
    provide probability/score outputs—only discrete class predictions. See
    plan.md (D12, appendix) for rationale.

    Args:
        preds: Predicted binary labels (0/1).
        golds: Gold binary labels (0/1).

    Returns:
        Dictionary with keys:
        - accuracy: float in [0, 1]
        - f1: float in [0, 1] (binary F1)
        - confusion_matrix: dict with keys {tn, fp, fn, tp}
    """
    acc = accuracy_score(golds, preds)
    f1 = f1_score(golds, preds, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(golds, preds, labels=[0, 1]).ravel()

    return {
        "accuracy": float(acc),
        "f1": float(f1),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def multilabel_prf(
    pred_sets: list[set], gold_sets: list[set]
) -> dict:
    """
    Multi-label precision/recall/F1 over label sets.

    Computes micro and macro averaged precision, recall, and F1 by comparing
    predicted label sets against gold label sets on a per-sample basis.

    Micro averaging: aggregate TP/FP/FN across all samples.
    Macro averaging: average per-sample P/R/F1.

    Args:
        pred_sets: List of predicted label sets.
        gold_sets: List of gold label sets.

    Returns:
        Dictionary with keys:
        - micro_precision, micro_recall, micro_f1
        - macro_precision, macro_recall, macro_f1
        - All values float in [0, 1].
    """
    if len(pred_sets) != len(gold_sets):
        raise ValueError("pred_sets and gold_sets must have same length")

    if not pred_sets:
        return {
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
        }

    # Micro: aggregate across samples
    total_tp = 0
    total_fp = 0
    total_fn = 0

    sample_precisions = []
    sample_recalls = []
    sample_f1s = []

    for pred, gold in zip(pred_sets, gold_sets):
        tp = len(pred & gold)
        fp = len(pred - gold)
        fn = len(gold - pred)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Per-sample metrics (macro)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * (p * r) / (p + r)
            if (p + r) > 0
            else 0.0
        )

        sample_precisions.append(p)
        sample_recalls.append(r)
        sample_f1s.append(f1)

    # Micro metrics
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (
        2 * (micro_p * micro_r) / (micro_p + micro_r)
        if (micro_p + micro_r) > 0
        else 0.0
    )

    # Macro metrics
    macro_p = np.mean(sample_precisions) if sample_precisions else 0.0
    macro_r = np.mean(sample_recalls) if sample_recalls else 0.0
    macro_f1 = np.mean(sample_f1s) if sample_f1s else 0.0

    return {
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
    }


def classification_metrics(preds: list[int], golds: list[int]) -> dict:
    """
    Multi-class classification metrics (accuracy, macro F1).

    Args:
        preds: Predicted class labels.
        golds: Gold class labels.

    Returns:
        Dictionary with keys:
        - accuracy: float in [0, 1]
        - macro_f1: float in [0, 1]
    """
    acc = accuracy_score(golds, preds)
    macro_f1 = f1_score(golds, preds, average="macro", zero_division=0)

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
    }


# BERTScore 설정. 한국어를 포함하므로 **다국어 모델**을 명시한다(기본 roberta-large는 영어
# 전용이라 한국어 점수가 무의미해진다 — plan 부록의 "BERTScore 다국어 모델 지정" 항목).
# num_layers는 이 모델의 권장값(bert_score 기본 매핑에 mBERT 항목이 있으나 명시해 고정).
BERTSCORE_MODEL = "bert-base-multilingual-cased"
BERTSCORE_LAYERS = 9

_bertscore_unavailable: str | None = None


def bertscore_f1(cands: list[str], refs: list[str]) -> dict:
    """BERTScore F1 평균. torch·bert_score가 없으면 이유를 담은 dict를 돌려준다.

    2026-08-05까지 태스크들이 `"deferred (torch 미설치)"`를 **하드코딩**해서, torch가
    설치된 환경에서도 계산하지 않았다. 여기서 실제로 시도하고, 불가할 때만 그 사실을
    (추측이 아닌 실제 예외 메시지로) 남긴다.

    첫 호출은 모델 다운로드·로드로 수십 초 걸리고 이후 캐시된다. 배치 메트릭이라
    스트리밍 누적이 불가해 태스크 끝에서 전체를 한 번에 계산한다.

    Returns:
        {"bertscore_f1": float, "bertscore_model": str} 또는
        {"bertscore": "unavailable (<이유>)"}
    """
    global _bertscore_unavailable

    pairs = [(c, r) for c, r in zip(cands, refs) if str(c or "").strip() and str(r or "").strip()]
    if not pairs:
        return {"bertscore": "unavailable (빈 입력)"}
    if _bertscore_unavailable:
        return {"bertscore": f"unavailable ({_bertscore_unavailable})"}

    try:
        import warnings

        from bert_score import score as _bs

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, _, f1 = _bs(
                [c for c, _ in pairs],
                [r for _, r in pairs],
                model_type=BERTSCORE_MODEL,
                num_layers=BERTSCORE_LAYERS,
                verbose=False,
                batch_size=16,
            )
        return {
            "bertscore_f1": round(float(f1.mean()), 4),
            "bertscore_model": BERTSCORE_MODEL,
            "bertscore_n": len(pairs),
        }
    except Exception as e:
        # 한 번 실패하면(미설치·다운로드 불가 등) 이후 호출은 재시도하지 않는다 —
        # 태스크마다 수십 초를 낭비하지 않기 위해.
        _bertscore_unavailable = f"{type(e).__name__}: {e}"[:120]
        return {"bertscore": f"unavailable ({_bertscore_unavailable})"}
