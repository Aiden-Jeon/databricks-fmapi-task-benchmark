"""
Quantitative metric functions for LLM benchmark scoring.

This module provides metrics for evaluating model outputs across tasks:
- Token-level string similarity (Token-F1, exact match)
- Classification metrics (accuracy, F1, confusion matrix)
- Multi-label set metrics (precision/recall/F1 over label sets)
- Reference-based generation metrics (ROUGE, BERTScore with language awareness)
- Structured extraction (Cell-F1 for table parsing)

Heavy dependencies (rouge_score, bert_score) are imported lazily inside stubs
to avoid installation requirements for modules that don't use them yet.

See plan.md §3 for task-metric alignment and appendix "채점 방법론 상세"
for Korean tokenization and metric design choices.
"""

from collections import Counter
from typing import Optional

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


def rouge_ko_en(pred: str, gold: str, lang: str) -> dict:
    """
    ROUGE score with Korean morpheme pre-tokenization.

    **Phase 1 stub**: This function will wrap rouge_score library with
    language-aware pre-tokenization:
    - For Korean (ko): tokenize via Mecab morphemes, re-join with spaces,
      then compute ROUGE (L, 1, 2).
    - For English (en): pass through to rouge_score directly.

    Implementation will follow plan.md appendix "채점 방법론 상세" (P0) on
    Korean tokenization necessity.

    Args:
        pred: Predicted text.
        gold: Gold reference text.
        lang: Language code ('ko' or 'en').

    Returns:
        Dictionary with keys:
        - rouge1: dict with {precision, recall, fmeasure}
        - rouge2: dict with {precision, recall, fmeasure}
        - rougeL: dict with {precision, recall, fmeasure}

    Raises:
        NotImplementedError: Phase 1 implementation pending.
    """
    raise NotImplementedError(
        "Phase 1: rouge-score + Mecab 전처리 구현 필요"
    )


def bertscore(
    preds: list[str], golds: list[str], lang: str
) -> dict:
    """
    BERTScore semantic similarity with multilingual model.

    **Phase 1 stub**: This function will compute BERTScore using a
    multilingual model:
    - For Korean (ko): `bert-base-multilingual-cased` or `klue/roberta-base`.
    - For English (en): same multilingual model for consistency.

    Model hash will be pinned for reproducibility. First run downloads
    the model for offline reproducibility.

    Args:
        preds: List of predicted texts.
        golds: List of gold reference texts.
        lang: Language code ('ko' or 'en').

    Returns:
        Dictionary with keys:
        - precision: list of float, one per sample
        - recall: list of float, one per sample
        - f1: list of float, one per sample

    Raises:
        NotImplementedError: Phase 1 implementation pending.
    """
    raise NotImplementedError(
        "Phase 1: BERTScore 다국어 모델 + 해시 고정 구현 필요"
    )


def cell_f1(pred_html: str, gold_html: str) -> float:
    """
    Cell-level F1 for table structure extraction (TXT-3).

    **Phase 1 stub**: This function will replace TEDS (which has no maintained
    package) with Cell-F1, which performs cell-by-cell fuzzy matching:
    1. Parse HTML tables (pred and gold) to extract cells.
    2. Build a bipartite matching between predicted and gold cells by
       position and text (fuzzy match threshold ~0.8).
    3. Return F1 over matched cells.

    See plan.md (D12, appendix) for rationale: TEDS package is unmaintained,
    so Cell-F1 is a practical substitute for v1.

    Args:
        pred_html: Predicted HTML table structure.
        gold_html: Gold HTML table structure.

    Returns:
        Cell-F1 score in [0, 1].

    Raises:
        NotImplementedError: Phase 1 implementation pending.
    """
    raise NotImplementedError(
        "Phase 1: Cell-F1 파싱 + 퍼지 매칭 구현 필요"
    )
