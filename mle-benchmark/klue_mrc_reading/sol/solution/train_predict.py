#!/usr/bin/env python3
"""Offline extractive QA baseline for the K-MLE KLUE-MRC task.

The model uses only train.csv.  It combines a character TF-IDF sentence
retriever, a supervised span ranker, and an answerability classifier.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit


SEED = 20260731
SENT_RE = re.compile(r"[^.!?。！？\n]+(?:[.!?。！？]+|$)")
TOKEN_RE = re.compile(r"[^\s]+")
EDGE = " \t\r\n,.;:!?~·…()[]{}<>\"'“”‘’《》〈〉「」『』【】"
PARTICLES = (
    "으로부터", "에게서", "에서는", "으로는", "이라는", "이라고", "에서는",
    "까지도", "에서의", "으로의", "들에게", "보다도", "이라며", "이라고",
    "으로", "에서", "에게", "한테", "처럼", "보다", "부터", "까지", "와의",
    "과의", "에는", "에도", "만을", "만이", "들이", "라는", "라고", "이며",
    "의", "은", "는", "이", "가", "을", "를", "에", "와", "과", "도", "만",
)
UNITS = r"(?:조원|억원|만원|천원|달러|유로|원|명|개|건|곳|가구|채|대|회|차례|년|개월|월|주|일|시간|분|초|%|퍼센트|㎝|cm|mm|m|km|㎞|kg|㎏|t|톤|인치|위|점|배|세|살|권|종|개사|켤레)"
NUM_RE = re.compile(rf"\d[\d,.]*(?:\s*~\s*\d[\d,.]*)?\s*{UNITS}", re.I)


def normalize(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]", "", str(text)).lower()


def char_f1(pred: str, gold: str) -> float:
    from collections import Counter

    p, g = normalize(pred), normalize(gold)
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if not common:
        return 0.0
    return 2 * common / (len(p) + len(g))


def split_sentences(context: str) -> list[tuple[int, int, str]]:
    out = []
    for m in SENT_RE.finditer(context):
        text = m.group().strip()
        if text:
            start = m.start() + len(m.group()) - len(m.group().lstrip())
            out.append((start, start + len(text), text))
    return out or [(0, len(context), context)]


def grams(text: str, n: int = 2) -> set[str]:
    text = normalize(text)
    return {text[i : i + n] for i in range(max(0, len(text) - n + 1))}


def overlap(question: str, text: str) -> float:
    q = grams(question)
    if not q:
        return 0.0
    return len(q & grams(text)) / len(q)


def question_type(q: str) -> int:
    q = q.replace(" ", "")
    if re.search(r"몇|얼마|수는|비율|가격|금액|규모|높이|길이|거리|무게|속도|점유율", q):
        return 1
    if re.search(r"언제|연도|년도|날짜|시기|며칠|몇월", q):
        return 2
    if re.search(r"누구|인물|사람|이름|대표|회장|감독|선수", q):
        return 3
    if re.search(r"어디|지역|장소|도시|국가|나라|곳은", q):
        return 4
    if re.search(r"왜|이유|원인|목적", q):
        return 5
    return 0


def sentence_scores(question: str, sentences: list[tuple[int, int, str]]) -> np.ndarray:
    # Bigram overlap is robust to Korean particles and spacing variation.
    return np.array([overlap(question, s[2]) for s in sentences], dtype=np.float32)


def span_candidates(text: str) -> list[tuple[int, int, str]]:
    """Generate compact answer-like spans from one sentence."""
    found: dict[tuple[int, int], str] = {}

    def add(a: int, b: int) -> None:
        while a < b and text[a] in EDGE:
            a += 1
        while b > a and text[b - 1] in EDGE:
            b -= 1
        if 0 < b - a <= 55:
            found[(a, b)] = text[a:b]

    for m in NUM_RE.finditer(text):
        add(m.start(), m.end())
    for m in re.finditer(r"[‘'\"“](.{1,35}?)[’'\"”]", text):
        add(m.start(), m.end())
        add(m.start(1), m.end(1))

    toks = list(TOKEN_RE.finditer(text))
    for i, tok in enumerate(toks):
        for width in range(1, min(6, len(toks) - i) + 1):
            a, b = tok.start(), toks[i + width - 1].end()
            add(a, b)
            raw = text[a:b].rstrip(EDGE)
            if width <= 3:
                for particle in PARTICLES:
                    if raw.endswith(particle) and len(raw) > len(particle):
                        add(a, a + len(raw) - len(particle))
                        break
    return [(a, b, v) for (a, b), v in found.items()]


def span_features(q: str, context: str, sent: str, candidate: str, sent_score: float,
                  rel_start: int, rel_end: int, sent_rank: int) -> list[float]:
    qt = question_type(q)
    c = candidate.strip()
    norm_c = normalize(c)
    before = sent[max(0, rel_start - 20) : rel_start]
    after = sent[rel_end : rel_end + 20]
    is_num = bool(re.fullmatch(rf".*\d.*(?:{UNITS})?", c, re.I))
    is_date = bool(re.search(r"\d{2,4}년|\d{1,2}월|\d{1,2}일", c))
    quoted = (0 < rel_start <= len(sent) and sent[rel_start - 1] in "‘'\"“") or (c[:1] in "‘'\"“")
    person_context = bool(re.search(r"대표|회장|사장|장관|교수|감독|의원|씨|선수|기자|연구원", after[:12]))
    loc_context = bool(re.search(r"에서|으로|에 위치|지역|시|도|군|구|국", after[:10]))
    reason_context = bool(re.search(r"때문|위해|목적|이유", c + after[:8]))
    return [
        sent_score, overlap(q, sent), overlap(q, before + after),
        len(norm_c), len(c.split()), rel_start / max(1, len(sent)), sent_rank,
        context.count(c) if c else 0, float(norm_c in normalize(q)),
        float(is_num), float(is_date), float(quoted), float(person_context),
        float(loc_context), float(reason_context),
        float(qt == 1), float(qt == 2), float(qt == 3), float(qt == 4), float(qt == 5),
        float(qt == 1 and is_num), float(qt == 2 and is_date),
        float(qt == 3 and person_context), float(qt == 4 and loc_context),
        float(qt == 5 and reason_context),
        float(bool(re.search(r"(?:이다|였다|한다|했다|된다|말했다|밝혔다)", c))),
        float(c.endswith(tuple(PARTICLES))),
    ]


def example_candidates(q: str, context: str, top_sentences: int = 5):
    sents = split_sentences(context)
    scores = sentence_scores(q, sents)
    selected = np.argsort(-scores)[:top_sentences]
    rows = []
    for rank, si in enumerate(selected):
        base, _, sent = sents[int(si)]
        for a, b, c in span_candidates(sent):
            f = span_features(q, context, sent, c, float(scores[si]), a, b, rank)
            rows.append((c, f, base + a, int(si)))
    return rows, scores, sents


def train_span_ranker(train: pd.DataFrame, max_rows: int = 12000):
    rng = np.random.default_rng(SEED)
    X, y = [], []
    answerable = train[train.answer != ""]
    if len(answerable) > max_rows:
        answerable = answerable.sample(max_rows, random_state=SEED)
    covered = 0
    for row in answerable.itertuples(index=False):
        candidates, scores, sents = example_candidates(row.question, row.context, 5)
        positives = [x for x in candidates if x[0] == row.answer]
        if positives:
            covered += 1
            X.append(positives[0][1]); y.append(1)
        else:
            pos = row.context.find(row.answer)
            si = next((i for i, (a, b, _) in enumerate(sents)
                       if a <= pos and pos + len(row.answer) <= b), None)
            if si is None:
                si, a0, sent = 0, 0, row.context
            else:
                a0, _, sent = sents[si]
            rel = pos - a0
            X.append(span_features(row.question, row.context, sent, row.answer,
                                   float(scores[si]), rel, rel + len(row.answer), 5)); y.append(1)
        neg = [x for x in candidates if x[0] != row.answer]
        if neg:
            # Hard negatives by heuristic score plus random candidates.
            hard = sorted(neg, key=lambda z: z[1][0] + .2 * z[1][20], reverse=True)[:10]
            random_idx = rng.choice(len(neg), min(8, len(neg)), replace=False)
            for item in hard + [neg[i] for i in random_idx]:
                X.append(item[1]); y.append(0)
    print(f"span training examples={len(X):,}, exact candidate coverage={covered/len(answerable):.3f}")
    model = HistGradientBoostingClassifier(max_iter=180, learning_rate=.08, max_leaf_nodes=31,
                                           l2_regularization=2.0, random_state=SEED)
    model.fit(np.asarray(X, dtype=np.float32), y)
    return model


def answerability_features(q: str, context: str) -> list[float]:
    sents = split_sentences(context)
    scores = sentence_scores(q, sents)
    best = np.sort(scores)[::-1]
    qn = normalize(q)
    # Question content chunks absent from the article are useful for adversarial no-answer cases.
    chunks = [normalize(x) for x in re.findall(r"[A-Za-z가-힣0-9]{2,}", q)]
    chunks = [re.sub(r"(?:에서|에게|으로|의|은|는|이|가|을|를|에)$", "", x) for x in chunks]
    absent = sum(len(x) >= 2 and x not in normalize(context) for x in chunks)
    return [
        float(best[0]), float(best[1] if len(best) > 1 else 0),
        float(np.mean(scores)), float(absent), absent / max(1, len(chunks)),
        len(qn), len(context), len(sents), float(question_type(q)),
    ]


def train_answerability(train: pd.DataFrame):
    X = np.asarray([answerability_features(r.question, r.context) for r in train.itertuples()], dtype=np.float32)
    y = (train.answer != "").astype(int).to_numpy()
    model = ExtraTreesClassifier(n_estimators=350, min_samples_leaf=8, max_features=.9,
                                 class_weight="balanced", n_jobs=-1, random_state=SEED)
    model.fit(X, y)
    return model


def predict_one(q: str, context: str, span_model) -> tuple[str, float]:
    candidates, _, _ = example_candidates(q, context, 5)
    if not candidates:
        return "", 0.0
    X = np.asarray([x[1] for x in candidates], dtype=np.float32)
    probs = span_model.predict_proba(X)[:, 1]
    # The classifier is trained pointwise; a small prior favors concise candidates.
    adjusted = probs - .0015 * np.array([len(normalize(x[0])) for x in candidates])
    i = int(np.argmax(adjusted))
    answer = candidates[i][0].strip()
    if question_type(q) in (1, 2):
        numeric = NUM_RE.search(answer)
        if numeric:
            answer = numeric.group().strip()
    return answer, float(probs[i])


def answer_threshold(q: str) -> float:
    """Conservative thresholds selected once on the grouped holdout."""
    return {0: .825, 1: .575, 2: .400, 5: .725}.get(question_type(q), 1.01)


def validate(train: pd.DataFrame) -> None:
    groups = pd.util.hash_pandas_object(train.context, index=False).to_numpy()
    tr_idx, va_idx = next(GroupShuffleSplit(n_splits=1, test_size=.18, random_state=SEED).split(train, groups=groups))
    fit, valid = train.iloc[tr_idx], train.iloc[va_idx]
    span_model = train_span_ranker(fit)
    ans_model = train_answerability(fit)
    raw, span_p = zip(*(predict_one(r.question, r.context, span_model) for r in valid.itertuples()))
    ans_p = ans_model.predict_proba(np.asarray([answerability_features(r.question, r.context) for r in valid.itertuples()]))[:, 1]
    gold = valid.answer.tolist()
    print(f"raw answerable-only F1={np.mean([char_f1(p,g) for p,g in zip(raw,gold)]):.4f}")
    best = (-1.0, 0.5)
    for threshold in np.arange(.20, .81, .025):
        pred = [p if ap >= threshold else "" for p, ap in zip(raw, ans_p)]
        score = np.mean([char_f1(p, g) for p, g in zip(pred, gold)])
        if score > best[0]: best = (score, threshold)
    print(f"validation F1={best[0]:.4f}, answerability threshold={best[1]:.3f}")
    answerable = valid.answer != ""
    print(f"span F1 on answerable={np.mean([char_f1(p,g) for p,g in zip(np.array(raw)[answerable],valid.answer[answerable])]):.4f}")
    qtypes = np.array([question_type(q) for q in valid.question])
    for qt in range(6):
        mask = (qtypes == qt)
        amask = mask & answerable.to_numpy()
        if amask.any():
            sf1 = np.mean([char_f1(p, g) for p, g in zip(np.array(raw)[amask], valid.answer.to_numpy()[amask])])
            exact = np.mean(np.array(raw)[amask] == valid.answer.to_numpy()[amask])
            print(f"type={qt} n={mask.sum()} answerable={amask.sum()} span_f1={sf1:.3f} exact={exact:.3f} empty_rate={1-answerable.to_numpy()[mask].mean():.3f}")
    for threshold in [.5, .6, .7, .8, .9, .95]:
        take = ans_p >= threshold
        if take.any():
            quality = np.mean([char_f1(p, g) for p, g in zip(np.array(raw)[take], valid.answer.to_numpy()[take])])
            print(f"ans_threshold={threshold:.2f} selected={take.sum()} selected_f1={quality:.3f}")
    empty_scores = np.array([char_f1("", g) for g in gold])
    raw_scores = np.array([char_f1(p, g) for p, g in zip(raw, gold)])
    print(f"all-empty F1={empty_scores.mean():.4f}")
    for qt in range(6):
        best_policy = (empty_scores.mean(), 1.01, 0)
        for threshold in np.arange(.3, 1.001, .025):
            take = (qtypes == qt) & (ans_p >= threshold)
            score = np.where(take, raw_scores, empty_scores).mean()
            if score > best_policy[0]: best_policy = (score, threshold, int(take.sum()))
        print(f"policy type={qt}: F1={best_policy[0]:.4f} threshold={best_policy[1]:.3f} selected={best_policy[2]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("outputs/submission.csv"))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    train = pd.read_csv(args.data_dir / "train.csv", keep_default_na=False)
    if args.validate:
        validate(train)
        return
    test = pd.read_csv(args.data_dir / "test.csv", keep_default_na=False)
    span_model = train_span_ranker(train)
    ans_model = train_answerability(train)
    predictions = []
    for i, row in enumerate(test.itertuples(index=False), 1):
        answer, _ = predict_one(row.question, row.context, span_model)
        ap = ans_model.predict_proba(np.asarray([answerability_features(row.question, row.context)]))[0, 1]
        predictions.append(answer if ap >= answer_threshold(row.question) else "")
        if i % 500 == 0:
            print(f"predicted {i}/{len(test)}")
    submission = pd.DataFrame({"id": test.id, "answer": predictions})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({len(submission)} rows, {(submission.answer == '').sum()} empty)")


if __name__ == "__main__":
    main()
