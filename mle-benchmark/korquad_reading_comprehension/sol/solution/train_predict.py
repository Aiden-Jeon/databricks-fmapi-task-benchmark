#!/usr/bin/env python3
"""Train a from-scratch extractive QA span ranker and create submission.csv."""

import argparse
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GroupShuffleSplit


WORD_RE = re.compile(r"[가-힣A-Za-z0-9]+")
START_RE = re.compile(r"(?<![가-힣A-Za-z0-9])(?=[가-힣A-Za-z0-9])")
SENT_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")
GENERIC = {
    "은", "는", "이", "가", "을", "를", "에", "의", "로", "으로", "와", "과",
    "무엇", "뭐", "어떤", "것", "누구", "누가", "언제", "어디", "어떻게", "왜",
    "인가", "인가요", "일까", "하는가", "했는가", "한", "된", "되는", "했던",
}
PARTICLES = set("은는이가을를의에도와과로며라였인할한된될부터까지에서에게께만보다처럼")


def compact(text):
    return re.sub(r"\s+", "", str(text)).lower()


def char_f1(prediction, truth):
    pred, gold = compact(prediction), compact(truth)
    if not pred or not gold:
        return float(pred == gold)
    common = sum((Counter(pred) & Counter(gold)).values())
    if not common:
        return 0.0
    return 2.0 * common / (len(pred) + len(gold))


def question_kind(question):
    q = compact(question)
    if any(x in q for x in ("누구", "누가", "인물", "사람", "이름")):
        return "person"
    if any(x in q for x in ("언제", "몇년", "몇월", "몇일", "시기", "시대")):
        return "date"
    if any(x in q for x in ("어디", "장소", "지역", "출생지", "나라")):
        return "place"
    if any(x in q for x in ("얼마", "몇", "수는", "크기", "길이", "높이")):
        return "number"
    if any(x in q for x in ("왜", "이유", "원인")):
        return "reason"
    if any(x in q for x in ("어떻게", "방법")):
        return "method"
    return "thing"


def question_terms(question):
    words = WORD_RE.findall(question.lower())
    terms = []
    for word in words:
        stem = re.sub(r"(으로|에서|에게|에는|에는|이란|라는|이며|하고|했고|였다|이다|인가|은|는|이|가|을|를|의|에|로|와|과)$", "", word)
        if stem and stem not in GENERIC:
            terms.append(stem)
    return terms


def question_cue(question):
    words = WORD_RE.findall(question.lower())
    if not words:
        return "_"
    cue = re.sub(
        r"(입니까|인가요|인가|했나요|했는가|하는가|였는가|이었나|은|는|이|가|을|를|의|에|로|와|과)$",
        "", words[-1],
    )
    if cue in GENERIC and len(words) > 1:
        cue = words[-2]
    return cue[-8:] or "_"


def overlap_score(text, terms):
    text = text.lower()
    if not terms:
        return 0.0
    return sum(min(len(t), 6) for t in terms if t in text) / sum(min(len(t), 6) for t in terms)


def question_grams(question):
    text = re.sub(r"[^가-힣A-Za-z0-9]", "", question.lower())
    grams = {text[i:i + 3] for i in range(max(0, len(text) - 2))}
    # Interrogative boilerplate is not useful for finding supporting text.
    return [g for g in grams if not any(x in g for x in ("무엇", "어떤", "누구", "언제", "어디", "인가"))]


def gram_overlap(text, grams):
    if not grams:
        return 0.0
    normalized = re.sub(r"[^가-힣A-Za-z0-9]", "", text.lower())
    return sum(g in normalized for g in grams) / len(grams)


def row_info(context, question):
    terms = question_terms(question)
    grams = question_grams(question)
    sentences = []
    for match in SENT_RE.finditer(context):
        start, end = match.span()
        lexical = overlap_score(match.group(), terms)
        sentences.append((start, end, 0.55 * lexical + 0.45 * gram_overlap(match.group(), grams)))
    if not sentences:
        sentences = [(0, len(context), overlap_score(context, terms))]
    starts = [m.start() for m in START_RE.finditer(context)]
    if context and context[0].isalnum() and 0 not in starts:
        starts.insert(0, 0)
    return {
        "context": context,
        "question": question,
        "qcompact": compact(question),
        "terms": terms,
        "grams": grams,
        "kind": question_kind(question),
        "cue": question_cue(question),
        "sentences": sentences,
        "starts": starts,
    }


def sentence_at(info, position):
    for start, end, score in info["sentences"]:
        if start <= position < end:
            return start, end, score
    return 0, len(info["context"]), 0.0


def span_features(info, start, end):
    context, question = info["context"], info["qcompact"]
    answer = context[start:end]
    acompact = compact(answer)
    sent_start, sent_end, sent_score = sentence_at(info, start)
    before = context[max(0, start - 4):start]
    after = context[end:min(len(context), end + 4)]
    left = context[max(sent_start, start - 80):start].lower()
    right = context[end:min(sent_end, end + 80)].lower()
    term_left = sum(t in left for t in info["terms"])
    term_right = sum(t in right for t in info["terms"])
    nterms = max(1, len(info["terms"]))
    qtail = question[-8:]
    last = acompact[-1:] or "_"
    post1 = after[:1] or "$"
    post2 = after[:2] or "$"
    pre1 = before[-1:] or "^"
    length = len(acompact)
    words = WORD_RE.findall(answer)
    feats = {
        "bias": 1.0,
        "len": min(length, 30) / 10.0,
        "len2": min(length, 30) ** 2 / 100.0,
        "spaces": min(answer.count(" "), 5) / 5.0,
        "words": min(len(words), 6) / 6.0,
        "ctx_pos": start / max(1, len(context)),
        "sent_pos": (start - sent_start) / max(1, sent_end - sent_start),
        "sent_score": sent_score,
        "term_left": term_left / nterms,
        "term_right": term_right / nterms,
        "term_both": float(term_left > 0 and term_right > 0),
        "q_overlap": len(set(acompact) & set(question)) / max(1, len(set(acompact))),
        "in_question": float(length > 1 and acompact in question),
        "digits": sum(c.isdigit() for c in acompact) / max(1, length),
        "latin": sum("a" <= c <= "z" for c in acompact) / max(1, length),
        "post_particle": float(post1 in PARTICLES),
        "post_space": float(post1.isspace() or post1 == "$"),
        "kind=" + info["kind"]: 1.0,
        "kind_post=" + info["kind"] + ":" + post1: 1.0,
        "kind_post2=" + info["kind"] + ":" + post2: 1.0,
        "kind_last=" + info["kind"] + ":" + last: 1.0,
        "qtail_post=" + qtail[-4:] + ":" + post1: 1.0,
        "qtail_last=" + qtail[-4:] + ":" + last: 1.0,
        "boundary=" + pre1 + ":" + post1: 1.0,
        "post1=" + post1: 1.0,
        "post2=" + post2: 1.0,
        "pre1=" + pre1: 1.0,
        "last=" + last: 1.0,
    }
    if words:
        feats["first_shape=" + ("D" if words[0][0].isdigit() else "W")] = 1.0
    for term in info["terms"][-5:]:
        if term in left:
            feats["term_side=" + term[-4:] + ":L"] = 1.0
        if term in right:
            feats["term_side=" + term[-4:] + ":R"] = 1.0
    return feats


def start_features(info, start):
    context, question = info["context"], info["qcompact"]
    sent_start, sent_end, sent_score = sentence_at(info, start)
    before = context[max(sent_start, start - 80):start].lower()
    after = context[start:min(sent_end, start + 80)].lower()
    pre = context[max(0, start - 3):start]
    opening = context[start:min(len(context), start + 6)].lower()
    token_match = WORD_RE.match(context, start)
    token = token_match.group().lower() if token_match else context[start:start + 1].lower()
    nterms = max(1, len(info["terms"]))
    left_count = sum(t in before for t in info["terms"])
    right_count = sum(t in after for t in info["terms"])
    left_gram = gram_overlap(before, info["grams"])
    right_gram = gram_overlap(after, info["grams"])
    nearest = 999
    signed_nearest = 999
    feats = {
        "bias": 1.0,
        "ctx_pos": start / max(1, len(context)),
        "sent_pos": (start - sent_start) / max(1, sent_end - sent_start),
        "sent_score": sent_score,
        "term_left": left_count / nterms,
        "term_right": right_count / nterms,
        "term_both": float(left_count > 0 and right_count > 0),
        "gram_left": left_gram,
        "gram_right": right_gram,
        "gram_both": float(left_gram > 0 and right_gram > 0),
        "token_in_question": float(len(token) > 1 and token in question),
        "token_len": min(len(token), 15) / 10.0,
        "token_digit": float(any(c.isdigit() for c in token)),
        "kind=" + info["kind"]: 1.0,
        "kind_open=" + info["kind"] + ":" + opening[:2]: 1.0,
        "cue=" + info["cue"]: 1.0,
        "cue_open=" + info["cue"] + ":" + opening[:4]: 1.0,
        "cue_token=" + info["cue"] + ":" + token[:8]: 1.0,
        "cue_tokenend=" + info["cue"] + ":" + token[-4:]: 1.0,
        "cue_shape=" + info["cue"] + ":" + ("D" if any(c.isdigit() for c in token) else "W"): 1.0,
        "pre=" + (pre[-1:] or "^"): 1.0,
        "open1=" + opening[:1]: 1.0,
        "open2=" + opening[:2]: 1.0,
        "qtail_open=" + question[-4:] + ":" + opening[:1]: 1.0,
        "qtail_pre=" + question[-6:] + ":" + (pre[-1:] or "^"): 1.0,
        "qtail_tokenend=" + question[-6:] + ":" + token[-2:]: 1.0,
    }
    sentence = context[sent_start:sent_end].lower()
    for term in info["terms"][-6:]:
        positions = [m.start() + sent_start for m in re.finditer(re.escape(term), sentence)]
        if not positions:
            continue
        distance = min(positions, key=lambda p: abs(p - start)) - start
        if abs(distance) < nearest:
            nearest, signed_nearest = abs(distance), distance
        bucket = "Lfar" if distance < -35 else "L" if distance < 0 else "R" if distance < 35 else "Rfar"
        feats["qterm_side=" + term[-5:] + ":" + bucket] = 1.0
    feats["nearest"] = min(nearest, 100) / 100.0
    feats["nearest_left"] = float(signed_nearest < 0)
    return feats


def valid_span(context, start, end):
    text = context[start:end]
    if not text or len(compact(text)) == 0:
        return False
    if text[-1].isspace():
        return False
    return True


def training_examples(row, rng, negatives=65):
    context, question, answer = row.context, row.question, str(row.answer)
    gold_start = context.find(answer)
    gold_end = gold_start + len(answer)
    info = row_info(context, question)
    candidates = set()
    # Hard negatives share the correct start or end and teach exact particle boundaries.
    for delta in range(-8, 9):
        end = gold_end + delta
        if 0 < end <= len(context) and valid_span(context, gold_start, end):
            candidates.add((gold_start, end))
    nearby = [s for s in info["starts"] if abs(s - gold_start) < 140 and s != gold_start]
    rng.shuffle(nearby)
    for start in nearby[: negatives // 2]:
        length = max(1, len(answer) + int(rng.integers(-5, 6)))
        end = min(len(context), start + length)
        if valid_span(context, start, end):
            candidates.add((start, end))
    starts = info["starts"].copy()
    rng.shuffle(starts)
    for start in starts[:negatives]:
        length = int(rng.integers(1, 22))
        end = min(len(context), start + length)
        if valid_span(context, start, end):
            candidates.add((start, end))
    candidates.add((gold_start, gold_end))
    for start, end in candidates:
        yield span_features(info, start, end), int(start == gold_start and end == gold_end)


def boundary_examples(row, rng, negatives=70, max_length=28):
    context, answer = row.context, str(row.answer)
    gold_start = context.find(answer)
    gold_end = gold_start + len(answer)
    info = row_info(context, row.question)
    gold_sentence = sentence_at(info, gold_start)[:2]
    same_sentence = [s for s in info["starts"] if sentence_at(info, s)[:2] == gold_sentence and s != gold_start]
    other = [s for s in info["starts"] if s != gold_start and s not in same_sentence]
    rng.shuffle(same_sentence)
    rng.shuffle(other)
    starts = same_sentence[: negatives * 2 // 3] + other[: negatives // 3]
    starts.append(gold_start)
    for start in starts:
        yield "start", start_features(info, start), int(start == gold_start)
    # End labels are learned only with the correct start, independently of retrieval.
    limit = min(len(context), gold_start + max(max_length, len(answer) + 3))
    for end in range(gold_start + 1, limit + 1):
        if valid_span(context, gold_start, end):
            yield "end", span_features(info, gold_start, end), int(end == gold_end)


def fit_model(frame, epochs=2, seed=2026):
    hasher = FeatureHasher(n_features=2 ** 20, input_type="dict", alternate_sign=False)
    start_model = SGDClassifier(
        loss="log_loss", penalty="elasticnet", alpha=2e-6, l1_ratio=0.02,
        max_iter=1, tol=None, average=True, random_state=seed,
        class_weight={0: 1.0, 1: 24.0},
    )
    end_model = SGDClassifier(
        loss="log_loss", penalty="elasticnet", alpha=2e-6, l1_ratio=0.02,
        max_iter=1, tol=None, average=True, random_state=seed + 1,
        class_weight={0: 1.0, 1: 14.0},
    )
    rng = np.random.default_rng(seed)
    indices = np.arange(len(frame))
    for epoch in range(epochs):
        rng.shuffle(indices)
        start_features_batch, start_labels = [], []
        end_features_batch, end_labels = [], []
        seen_start = seen_end = 0
        for idx in indices:
            for kind, features, label in boundary_examples(frame.iloc[idx], rng):
                if kind == "start":
                    start_features_batch.append(features)
                    start_labels.append(label)
                else:
                    end_features_batch.append(features)
                    end_labels.append(label)
            if len(start_features_batch) >= 24000:
                start_model.partial_fit(hasher.transform(start_features_batch), np.asarray(start_labels), classes=np.array([0, 1]))
                seen_start += len(start_labels)
                start_features_batch, start_labels = [], []
            if len(end_features_batch) >= 24000:
                end_model.partial_fit(hasher.transform(end_features_batch), np.asarray(end_labels), classes=np.array([0, 1]))
                seen_end += len(end_labels)
                end_features_batch, end_labels = [], []
        if start_features_batch:
            start_model.partial_fit(hasher.transform(start_features_batch), np.asarray(start_labels), classes=np.array([0, 1]))
            seen_start += len(start_labels)
        if end_features_batch:
            end_model.partial_fit(hasher.transform(end_features_batch), np.asarray(end_labels), classes=np.array([0, 1]))
            seen_end += len(end_labels)
        print(f"epoch {epoch + 1}: {seen_start:,} start and {seen_end:,} end examples", flush=True)
    return hasher, start_model, end_model


def candidate_spans(info, max_length=24, top_sentences=3):
    context = info["context"]
    ranked = sorted(info["sentences"], key=lambda x: x[2], reverse=True)
    selected = {(s, e) for s, e, _ in ranked[:top_sentences]}
    # Include all tied best sentences and a fallback first sentence.
    if ranked:
        selected.update((s, e) for s, e, score in ranked if score == ranked[0][2])
    for start in info["starts"]:
        sent_start, sent_end, _ = sentence_at(info, start)
        if (sent_start, sent_end) not in selected:
            continue
        limit = min(len(context), sent_end, start + max_length)
        for end in range(start + 1, limit + 1):
            if valid_span(context, start, end):
                yield start, end


def predict_one(context, question, hasher, start_model, end_model):
    info = row_info(context, question)
    ranked_sentences = sorted(info["sentences"], key=lambda x: x[2], reverse=True)
    selected = {(s, e) for s, e, _ in ranked_sentences[:3]}
    if ranked_sentences:
        selected.update((s, e) for s, e, score in ranked_sentences if score == ranked_sentences[0][2])
    starts = [s for s in info["starts"] if sentence_at(info, s)[:2] in selected]
    if not starts:
        return context[:1]
    start_scores = start_model.decision_function(hasher.transform([start_features(info, s) for s in starts]))
    # End prediction for several starts lets a strong boundary score correct a close start decision.
    top = np.argsort(start_scores)[-5:]
    best_score, best_span = -math.inf, (starts[int(top[-1])], starts[int(top[-1])] + 1)
    for rank_idx in top:
        start = starts[int(rank_idx)]
        _, sent_end, _ = sentence_at(info, start)
        ends = [e for e in range(start + 1, min(len(context), sent_end, start + 28) + 1) if valid_span(context, start, e)]
        if not ends:
            continue
        end_scores = end_model.decision_function(hasher.transform([span_features(info, start, e) for e in ends]))
        end_idx = int(np.argmax(end_scores))
        # Start ranking dominates; the scaled end score resolves plausible alternatives.
        score = float(start_scores[int(rank_idx)] + 0.35 * end_scores[end_idx])
        if score > best_score:
            best_score, best_span = score, (start, ends[end_idx])
    return context[best_span[0]:best_span[1]].strip()


def predict_frame(frame, hasher, start_model, end_model):
    predictions = []
    for number, row in enumerate(frame.itertuples(index=False), 1):
        predictions.append(predict_one(row.context, row.question, hasher, start_model, end_model))
        if number % 1000 == 0:
            print(f"predicted {number:,}/{len(frame):,}", flush=True)
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test.csv")
    parser.add_argument("--output", default="outputs/submission.csv")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--validation-contexts", type=int, default=500)
    args = parser.parse_args()

    train = pd.read_csv(args.train, dtype={"id": str})
    if args.validate:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.12, random_state=2026)
        train_idx, valid_idx = next(splitter.split(train, groups=train["context"]))
        valid_contexts = set(train.iloc[valid_idx]["context"].drop_duplicates().iloc[:args.validation_contexts])
        valid = train[train["context"].isin(valid_contexts)].reset_index(drop=True)
        fit = train.iloc[train_idx].reset_index(drop=True)
        hasher, start_model, end_model = fit_model(fit, epochs=args.epochs)
        predictions = predict_frame(valid, hasher, start_model, end_model)
        score = np.mean([char_f1(p, g) for p, g in zip(predictions, valid["answer"])])
        exact = np.mean([compact(p) == compact(g) for p, g in zip(predictions, valid["answer"])])
        start_exact = np.mean([r.context.find(p) == r.context.find(str(r.answer)) for r, p in zip(valid.itertuples(), predictions)])
        print(f"validation rows={len(valid):,} char_f1={score:.6f} exact={exact:.6f} start={start_exact:.6f}")
        return

    test = pd.read_csv(args.test, dtype={"id": str})
    hasher, start_model, end_model = fit_model(train, epochs=args.epochs)
    predictions = predict_frame(test, hasher, start_model, end_model)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": test["id"], "answer": predictions}).to_csv(output, index=False)
    print(f"wrote {len(test):,} predictions to {output}")


if __name__ == "__main__":
    main()
