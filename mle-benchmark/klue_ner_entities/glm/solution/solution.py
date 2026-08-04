#!/usr/bin/env python3
"""Char-level CRF NER for KLUE-NER (t20_klue_ner).

Approach
--------
- KLUE-NER data is provided as sentence + list of "surface:type" pairs (not BIO).
  Entities are not space-token aligned (e.g. "은지원" inside "은지원은"), so we
  tag at the character level with a BIO scheme.
- We train a linear-chain CRF (sklearn_crfsuite / CRFsuite) with rich char-level
  features: char unigrams/bigrams, char-type, word (space-delimited) identity,
  word prefix/suffix, and gazetteer features built from the training entities.
- Predicted BIO tags are decoded back into "surface:type" pairs and written to
  outputs/submission.csv with the exact id set of test.csv.

Reproducibility
---------------
- No external data / internet. Only train.csv is used.
- Random state is fixed for any randomized step (none used here actually).
"""

import os
import re
import time
import pandas as pd
from sklearn_crfsuite import CRF


HERE = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(HERE)
TRAIN_CSV = os.path.join(TASK_DIR, "train.csv")
TEST_CSV = os.path.join(TASK_DIR, "test.csv")
SAMPLE_CSV = os.path.join(TASK_DIR, "sample_submission.csv")
OUT_CSV = os.path.join(TASK_DIR, "outputs", "submission.csv")

LABELS = ["PS", "LC", "OG", "DT", "TI", "QT"]


# ---------------------------------------------------------------------------
# Tag conversion
# ---------------------------------------------------------------------------
def build_tags(s, e):
    """Convert sentence + "surf:type|..." into char-level BIO tags.

    Entities are matched greedily left-to-right. We verified that for the
    provided train data this greedy alignment reproduces the original entity
    list exactly for every row (entities are listed in order of appearance
    and never overlap).
    """
    tags = ["O"] * len(s)
    if not e:
        return tags
    pos = 0
    for pair in e.split("|"):
        if not pair.strip():
            continue
        parts = pair.rsplit(":", 1)
        if len(parts) != 2:
            continue
        surf, typ = parts
        idx = s.find(surf, pos)
        if idx == -1:
            idx = s.find(surf)
        if idx >= 0:
            for k in range(len(surf)):
                tags[idx + k] = ("B" if k == 0 else "I") + "-" + typ
            pos = idx + len(surf)
    return tags


def reconstruct(s, tags):
    """Decode char-level BIO tags back into "surf:type|..." string."""
    entities = []
    i = 0
    n = len(tags)
    while i < n:
        t = tags[i]
        if t.startswith("B-"):
            typ = t[2:]
            start = i
            j = i + 1
            i_tag = "I-" + typ
            while j < n and tags[j] == i_tag:
                j += 1
            surf = s[start:j]
            entities.append(f"{surf}:{typ}")
            i = j
        else:
            i += 1
    return "|".join(entities)


# ---------------------------------------------------------------------------
# Gazetteer
# ---------------------------------------------------------------------------
def build_gazetteer(entities_series):
    """Build surface -> set(types) from training entities."""
    gaz = {}
    for e in entities_series.fillna(""):
        for pair in e.split("|"):
            if not pair.strip():
                continue
            parts = pair.rsplit(":", 1)
            if len(parts) == 2:
                gaz.setdefault(parts[0], set()).add(parts[1])
    # Freeze types as sorted tuple for stable feature names
    return {surf: tuple(sorted(ts)) for surf, ts in gaz.items()}


def compute_gaz_info(s, gaz):
    """For each char position, list of (types, pos_in_ent, ent_len, is_beg, is_end)."""
    n = len(s)
    info = [[] for _ in range(n)]
    for surf, types in gaz.items():
        L = len(surf)
        if L == 0 or L > 20 or L > n:
            continue
        start = 0
        while True:
            idx = s.find(surf, start)
            if idx == -1:
                break
            for k in range(L):
                info[idx + k].append((types, k, L, k == 0, k == L - 1))
            start = idx + 1
    return info


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def char_type(c):
    if c == " ":
        return "space"
    if c.isdigit():
        return "digit"
    if re.match("[A-Za-z]", c):
        return "alpha"
    if re.match("[가-힣]", c):
        return "hangul"
    return "punct"


def char_features(s, i, gaz_info_i):
    c = s[i]
    n = len(s)
    f = {
        "bias": 1.0,
        "c=" + c: 1.0,
        "ctype=" + char_type(c): 1.0,
    }
    if i > 0:
        f["c-1=" + s[i - 1]] = 1.0
        f["t-1=" + char_type(s[i - 1])] = 1.0
    else:
        f["c-1=<S>"] = 1.0
    if i < n - 1:
        f["c+1=" + s[i + 1]] = 1.0
        f["t+1=" + char_type(s[i + 1])] = 1.0
    else:
        f["c+1=<E>"] = 1.0
    if i > 1:
        f["c-2=" + s[i - 2]] = 1.0
    if i < n - 2:
        f["c+2=" + s[i + 2]] = 1.0
    if i > 0:
        f["c-1c=" + s[i - 1] + c] = 1.0
    if i < n - 1:
        f["cc+1=" + c + s[i + 1]] = 1.0
    # word (space-delimited) features
    ws = i
    while ws > 0 and s[ws - 1] != " ":
        ws -= 1
    we = i
    while we < n - 1 and s[we + 1] != " ":
        we += 1
    word = s[ws : we + 1]
    f["word=" + word] = 1.0
    f["wbeg"] = 1.0 if i == ws else 0.0
    f["wend"] = 1.0 if i == we else 0.0
    if len(word) >= 2:
        f["wpre2=" + word[:2]] = 1.0
        f["wsuf2=" + word[-2:]] = 1.0
    if len(word) >= 3:
        f["wpre3=" + word[:3]] = 1.0
        f["wsuf3=" + word[-3:]] = 1.0
    # gazetteer features
    for types, pos_in_ent, ent_len, is_beg, is_end in gaz_info_i:
        for t in types:
            if is_beg:
                f[f"gaz_b_{t}_{ent_len}"] = 1.0
            if is_end:
                f[f"gaz_e_{t}_{ent_len}"] = 1.0
            f[f"gaz_i_{t}_{pos_in_ent}"] = 1.0
    return f


def sent2features(s, gaz_info):
    return [char_features(s, i, gaz_info[i]) for i in range(len(s))]


def make_features(sentences, gaz):
    out = []
    for s in sentences:
        gi = compute_gaz_info(s, gaz)
        out.append(sent2features(s, gi))
    return out


# ---------------------------------------------------------------------------
# F1 metric (entity-level micro)
# ---------------------------------------------------------------------------
def entities_set(s, tags):
    return set(reconstruct(s, tags).split("|")) - {""}


def entity_f1(golds, preds, sentences):
    tp = fp = fn = 0
    for s, g, p in zip(sentences, golds, preds):
        gold = entities_set(s, g) if isinstance(g, list) else set(
            reconstruct(s, g).split("|")) - {""}
        # g may be tags list or entity string
        if isinstance(g, list):
            gold = entities_set(s, g)
        pred = entities_set(s, p) if isinstance(p, list) else set(p.split("|")) - {""}
        if isinstance(p, list):
            pred = entities_set(s, p)
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return f1, prec, rec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("Loading data...")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    print(f"train={len(train)} test={len(test)}")

    print("Building gazetteer...")
    gaz = build_gazetteer(train["entities"])
    print(f"gazetteer surfaces: {len(gaz)}")

    print("Extracting features (train)...")
    t = time.time()
    X_train = make_features(train["sentence"].tolist(), gaz)
    y_train = [build_tags(s, e) for s, e in zip(train["sentence"], train["entities"].fillna(""))]
    print(f"  train features: {time.time()-t:.1f}s")

    print("Training CRF...")
    t = time.time()
    crf = CRF(c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
    crf.fit(X_train, y_train)
    print(f"  train: {time.time()-t:.1f}s")

    print("Extracting features (test) + predicting...")
    t = time.time()
    X_test = make_features(test["sentence"].tolist(), gaz)
    y_pred = crf.predict(X_test)
    print(f"  test: {time.time()-t:.1f}s")

    print("Decoding entities...")
    rows = []
    for sid, s, tags in zip(test["id"], test["sentence"], y_pred):
        rows.append((sid, reconstruct(s, tags)))
    out = pd.DataFrame(rows, columns=["id", "entities"])

    # Ensure exact id match with sample (order independent)
    sample = pd.read_csv(SAMPLE_CSV)
    assert set(out["id"]) == set(sample["id"]), "id mismatch with sample"
    assert len(out) == len(sample), "row count mismatch"

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} ({len(out)} rows) in {time.time()-t0:.1f}s")

    # Quick sanity: how many non-empty predictions
    nonempty = (out["entities"].fillna("") != "").sum()
    print(f"Non-empty predictions: {nonempty}/{len(out)}")


if __name__ == "__main__":
    main()
