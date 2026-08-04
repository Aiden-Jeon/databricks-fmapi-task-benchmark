#!/usr/bin/env python3
"""Train a lightweight extractive Korean QA model and create a submission."""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit


TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]+")
SPACE_TOKEN_RE = re.compile(r"\S+")
TRIM_LEFT = " \t\n\r\"'“‘([{<《〈"
TRIM_RIGHT = " \t\n\r\"'”’)]}>》〉,.;:!?"
JOSA = (
    "으로부터", "에서부터", "이라고", "이라는", "에게서", "까지는", "에서는",
    "으로", "로부터", "에게", "에서", "부터", "까지", "보다", "처럼", "만큼",
    "이며", "이고", "였다", "이다", "에는", "에도", "라고", "라는", "와의",
    "은", "는", "이", "가", "을", "를", "에", "의", "와", "과", "도", "로",
)
GENERIC = {
    "무엇", "누구", "어디", "언제", "얼마", "몇", "어느", "것", "곳", "사람",
    "이름", "수", "해", "날", "날짜", "연도", "시기", "기관", "기업", "회사",
    "국가", "나라", "지역", "금액", "인가", "무엇인가", "누구인가", "어디인가",
}


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Return sentence-like character spans while preserving source offsets."""
    cuts = [0]
    boundary = r"(?:\n+|[!?][\"'”’)]*|(?<!\d)\.(?!\d)[\"'”’)]*)(?=\s*[^a-z])"
    for match in re.finditer(boundary, text):
        if match.end() - cuts[-1] >= 12:
            cuts.append(match.end())
    if cuts[-1] != len(text):
        cuts.append(len(text))
    spans = []
    for start, end in zip(cuts, cuts[1:]):
        while start < end and text[start].isspace():
            start += 1
        if end > start:
            spans.append((start, end))
    return spans or [(0, len(text))]


def normalize_token(token: str) -> str:
    token = token.lower()
    for suffix in JOSA:
        if token.endswith(suffix) and len(token) > len(suffix) + 1:
            return token[: -len(suffix)]
    return token


def content_terms(text: str) -> list[str]:
    terms = []
    for token in TOKEN_RE.findall(text.lower()):
        token = normalize_token(token)
        if len(token) > 1 and token not in GENERIC:
            terms.append(token)
    return terms


def char_ngrams(text: str) -> set[str]:
    compact = re.sub(r"\s+", "", text.lower())
    return {compact[i : i + 2] for i in range(max(0, len(compact) - 1))}


def sentence_features(question: str, sentence: str, index: int, count: int) -> dict[str, float]:
    q_terms = content_terms(question)
    s_lower = sentence.lower()
    exact = sum(term in s_lower for term in q_terms)
    qgrams = char_ngrams(question)
    sgrams = char_ngrams(sentence)
    gram_overlap = len(qgrams & sgrams) / math.sqrt(max(1, len(qgrams) * len(sgrams)))
    return {
        "term_exact": exact,
        "term_ratio": exact / max(1, len(set(q_terms))),
        "gram_overlap": gram_overlap,
        "position": index / max(1, count - 1),
        "length_log": math.log1p(len(sentence)),
    }


def rank_sentences(context: str, question: str) -> list[tuple[float, int, int, dict[str, float]]]:
    spans = split_sentences(context)
    ranked = []
    for i, (start, end) in enumerate(spans):
        feats = sentence_features(question, context[start:end], i, len(spans))
        score = 2.2 * feats["term_ratio"] + 1.3 * feats["gram_overlap"] + 0.12 * feats["term_exact"]
        ranked.append((score, start, end, feats))
    return sorted(ranked, reverse=True)


def trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start] in TRIM_LEFT:
        start += 1
    while end > start and text[end - 1] in TRIM_RIGHT:
        end -= 1
    return start, end


def candidate_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    sentence = text[start:end]
    words = [(start + m.start(), start + m.end()) for m in SPACE_TOKEN_RE.finditer(sentence)]
    candidates: set[tuple[int, int]] = set()
    for i in range(len(words)):
        for j in range(i, min(len(words), i + 8)):
            raw_left, raw_right = words[i][0], words[j][1]
            while raw_left < raw_right and text[raw_left].isspace():
                raw_left += 1
            while raw_right > raw_left and text[raw_right - 1].isspace():
                raw_right -= 1
            if 0 < raw_right - raw_left <= 90:
                candidates.add((raw_left, raw_right))
            left, right = trim_span(text, words[i][0], words[j][1])
            if 0 < right - left <= 90:
                candidates.add((left, right))
            raw_right = right
            value = text[left:raw_right]
            for suffix in JOSA:
                if value.endswith(suffix) and len(value) > len(suffix) + 1:
                    candidates.add((left, raw_right - len(suffix)))
    # Punctuation often joins adjacent Korean sentences without whitespace.
    for match in re.finditer(r"[가-힣A-Za-z0-9][가-힣A-Za-z0-9·&.()'’-]{0,40}", sentence):
        left, right = start + match.start(), start + match.end()
        candidates.add((left, right))
        value = text[left:right]
        for suffix in JOSA:
            if value.endswith(suffix) and len(value) > len(suffix) + 1:
                candidates.add((left, right - len(suffix)))
    for match in re.finditer(r"[‘“\"'《〈(<][^\n.!?]{1,70}?[’”\"'》〉)>]", sentence):
        candidates.add((start + match.start(), start + match.end()))
    return list(candidates)


def question_type(question: str) -> str:
    if re.search(r"몇|얼마|금액|가격|비율|수는|규모", question):
        return "number"
    if re.search(r"언제|연도|날짜|시기|몇 년|몇 월|며칠", question):
        return "date"
    if re.search(r"누구|사람|인물|이름", question):
        return "person"
    if re.search(r"어디|지역|장소|곳은|나라는|국가는", question):
        return "place"
    if re.search(r"회사|기업|기관|단체", question):
        return "org"
    return "other"


def expected_units(question: str) -> set[str]:
    units = set(re.findall(r"몇\s*(명|개|곳|건|대|회|차|년|월|일|시|분|초|원|배|퍼센트)", question))
    mappings = {
        "가격": "원", "금액": "원", "비용": "원", "연도": "년", "해는": "년",
        "날짜": "일", "인원": "명", "사람": "명", "비율": "%", "점유율": "%",
    }
    for clue, unit in mappings.items():
        if clue in question:
            units.add(unit)
    return units


def span_features(
    context: str,
    question: str,
    span: tuple[int, int],
    sent_info: tuple[float, int, int, dict[str, float]],
    sent_rank: int,
) -> dict[str, object]:
    start, end = span
    value = context[start:end]
    score, sent_start, sent_end, sent_feats = sent_info
    before = context[max(sent_start, start - 18) : start]
    after = context[end : min(sent_end, end + 18)]
    qtype = question_type(question)
    units = expected_units(question)
    q_tokens = TOKEN_RE.findall(question.lower())
    v_tokens = TOKEN_RE.findall(value.lower())
    features: dict[str, object] = {
        "sent_score": score,
        "sent_rank": sent_rank,
        "sent_term_ratio": sent_feats["term_ratio"],
        "sent_gram_overlap": sent_feats["gram_overlap"],
        "chars": min(len(value), 50),
        "chars_log": math.log1p(len(value)),
        "words": min(len(value.split()), 9),
        "has_digit": bool(re.search(r"\d", value)),
        "all_numeric": bool(re.fullmatch(r"[\d,.~～\-–年月일시분초%원명개곳건회차배㎝kKmM억만천조]+", value)),
        "has_latin": bool(re.search(r"[A-Za-z]", value)),
        "quoted": start > 0 and context[start - 1] in "\"'“‘《〈",
        "in_question": value.lower() in question.lower(),
        "qtype=" + qtype: True,
        "qtype_digit=" + qtype: bool(re.search(r"\d", value)),
        "expected_unit_match": bool(units and any(value.rstrip().endswith(unit) for unit in units)),
        "person_shape": qtype == "person" and bool(re.fullmatch(r"[가-힣]{2,4}", value)),
        "date_shape": qtype == "date" and bool(re.search(r"\d+(?:년|월|일)", value)),
        "place_shape": qtype == "place" and bool(re.search(r"(?:시|도|구|군|동|국|주)$", value)),
        "org_shape": qtype == "org" and bool(re.search(r"(?:사|청|부|원|회|행|대|소|관|단)$", value)),
        "at_sentence_start": start == sent_start,
        "at_sentence_end": end >= sent_end - 2,
        "before_last=" + (TOKEN_RE.findall(before.lower())[-1] if TOKEN_RE.findall(before.lower()) else "<B>"): True,
        "after_first=" + (TOKEN_RE.findall(after.lower())[0] if TOKEN_RE.findall(after.lower()) else "<E>"): True,
    }
    for n in (1, 2, 3):
        features[f"prefix{n}=" + value[:n].lower()] = True
        features[f"suffix{n}=" + value[-n:].lower()] = True
    if q_tokens:
        features["q_last=" + q_tokens[-1]] = True
        features["q_tail2=" + "_".join(q_tokens[-2:])] = True
    if v_tokens:
        features["v_first=" + v_tokens[0][:4]] = True
        features["v_last=" + v_tokens[-1][-4:]] = True
    return features


def kor_char_f1(truth: str, pred: str) -> float:
    truth = re.sub(r"\s+", "", truth)
    pred = re.sub(r"\s+", "", pred)
    if not truth or not pred:
        return float(truth == pred)
    common = sum((Counter(truth) & Counter(pred)).values())
    if not common:
        return 0.0
    precision, recall = common / len(pred), common / len(truth)
    return 2 * precision * recall / (precision + recall)


class ExtractiveQAModel:
    def __init__(self, random_state: int = 2026):
        self.random_state = random_state
        self.vectorizer = DictVectorizer()
        self.span_model = LogisticRegression(
            C=1.5, max_iter=250, class_weight={0: 1.0, 1: 8.0}, solver="liblinear", random_state=random_state
        )
        self.q_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=3, max_features=70000, sublinear_tf=True)
        self.answerable_model = LogisticRegression(C=1.2, max_iter=250, class_weight="balanced", solver="liblinear", random_state=random_state)
        self.threshold = 0.5

    def _row_candidates(self, context: str, question: str, top_sentences: int = 10):
        ranked = rank_sentences(context, question)
        result = []
        seen = set()
        for sent_rank, info in enumerate(ranked[:top_sentences]):
            for span in candidate_spans(context, info[1], info[2]):
                if span not in seen:
                    seen.add(span)
                    result.append((span, span_features(context, question, span, info, sent_rank)))
        return ranked, result

    @staticmethod
    def _answerability_numeric(ranked, candidate_prob: float) -> list[float]:
        scores = [item[0] for item in ranked]
        return [
            scores[0] if scores else 0.0,
            scores[1] if len(scores) > 1 else 0.0,
            (scores[0] - scores[1]) if len(scores) > 1 else 0.0,
            candidate_prob,
            len(ranked),
        ]

    def fit(self, frame: pd.DataFrame) -> "ExtractiveQAModel":
        feature_rows, labels = [], []
        rng = np.random.default_rng(self.random_state)
        for row in frame.itertuples(index=False):
            answer = "" if pd.isna(row.answer) else str(row.answer)
            ranked, candidates = self._row_candidates(row.context, row.question)
            positive_indices = []
            if answer:
                answer_start = row.context.find(answer)
                answer_span = (answer_start, answer_start + len(answer))
                for i, (span, _) in enumerate(candidates):
                    if span == answer_span:
                        positive_indices.append(i)
                if not positive_indices:
                    for sent_rank, info in enumerate(ranked):
                        if info[1] <= answer_start < info[2]:
                            candidates.append((answer_span, span_features(row.context, row.question, answer_span, info, sent_rank)))
                            positive_indices.append(len(candidates) - 1)
                            break
            pool = [i for i in range(len(candidates)) if i not in positive_indices]
            hard = sorted(pool, key=lambda i: (candidates[i][1]["sent_rank"], abs(candidates[i][1]["chars"] - max(1, len(answer)))))[:35]
            remaining = list(set(pool) - set(hard))
            random_neg = rng.choice(remaining, size=min(20, len(remaining)), replace=False).tolist() if remaining else []
            selected = positive_indices + hard + random_neg
            for i in selected:
                feature_rows.append(candidates[i][1])
                labels.append(int(i in positive_indices))
        matrix = self.vectorizer.fit_transform(feature_rows)
        self.span_model.fit(matrix, labels)

        numeric, best_probs = [], []
        for row in frame.itertuples(index=False):
            ranked, candidates = self._row_candidates(row.context, row.question)
            if candidates:
                probs = self.span_model.predict_proba(self.vectorizer.transform([x[1] for x in candidates]))[:, 1]
                best_prob = float(probs.max())
            else:
                best_prob = 0.0
            numeric.append(self._answerability_numeric(ranked, best_prob))
            best_probs.append(best_prob)
        q_matrix = self.q_vectorizer.fit_transform(frame.question)
        answerable_matrix = hstack([q_matrix, csr_matrix(np.asarray(numeric))], format="csr")
        self.answerable_model.fit(answerable_matrix, frame.answer.notna().astype(int))
        return self

    def predict_with_confidence(self, frame: pd.DataFrame) -> tuple[list[str], np.ndarray]:
        answers, numeric = [], []
        for row in frame.itertuples(index=False):
            ranked, candidates = self._row_candidates(row.context, row.question)
            if candidates:
                probs = self.span_model.predict_proba(self.vectorizer.transform([x[1] for x in candidates]))[:, 1]
                best = int(np.argmax(probs))
                span = candidates[best][0]
                answers.append(row.context[span[0] : span[1]])
                best_prob = float(probs[best])
            else:
                answers.append("")
                best_prob = 0.0
            numeric.append(self._answerability_numeric(ranked, best_prob))
        q_matrix = self.q_vectorizer.transform(frame.question)
        matrix = hstack([q_matrix, csr_matrix(np.asarray(numeric))], format="csr")
        confidence = self.answerable_model.predict_proba(matrix)[:, 1]
        return answers, confidence

    def predict(self, frame: pd.DataFrame) -> list[str]:
        answers, confidence = self.predict_with_confidence(frame)
        return [answer if prob >= self.threshold else "" for answer, prob in zip(answers, confidence)]


def validation_score(train: pd.DataFrame, seed: int) -> float:
    groups = pd.util.hash_pandas_object(train.context, index=False).to_numpy()
    train_idx, valid_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.18, random_state=seed).split(train, groups=groups))
    model = ExtractiveQAModel(seed).fit(train.iloc[train_idx].reset_index(drop=True))
    valid = train.iloc[valid_idx].reset_index(drop=True)
    raw_answers, confidence = model.predict_with_confidence(valid)
    truths = valid.answer.fillna("").astype(str).tolist()
    best = (-1.0, 0.5)
    for threshold in np.arange(0.25, 0.981, 0.025):
        predictions = [answer if prob >= threshold else "" for answer, prob in zip(raw_answers, confidence)]
        score = float(np.mean([kor_char_f1(t, p) for t, p in zip(truths, predictions)]))
        if score > best[0]:
            best = (score, float(threshold))
    answerable = [i for i, truth in enumerate(truths) if truth]
    oracle_gate = float(np.mean([kor_char_f1(truths[i], raw_answers[i]) for i in answerable]))
    print(f"validation_f1={best[0]:.5f} threshold={best[1]:.3f} answerable_extraction_f1={oracle_gate:.5f}")
    return best[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test.csv")
    parser.add_argument("--output", default="outputs/submission.csv")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    threshold = validation_score(train, args.seed) if args.validate else 0.5
    model = ExtractiveQAModel(args.seed).fit(train)
    model.threshold = threshold
    predictions = model.predict(test)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": test.id, "answer": predictions}).to_csv(output, index=False)
    print(f"wrote {len(test)} predictions to {output} (empty={sum(not x for x in predictions)})")


if __name__ == "__main__":
    main()
