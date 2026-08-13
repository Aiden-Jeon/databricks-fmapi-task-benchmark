#!/usr/bin/env python
"""Grade one submission against a task's hidden answers.

Usage: grade.py --task t3_ynat --submission path/to/submission.csv
Prints one JSON line: {task, metric, valid, score, n, errors}.
Exit 0 = graded, 2 = invalid submission. Deterministic; safe to re-run.
Agents must NEVER see this file or kmle/private/.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss

ROOT = Path(__file__).resolve().parents[1]  # kmle/


def out(payload, code):
    print(json.dumps(payload, ensure_ascii=False))
    sys.exit(code)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--submission", required=True)
    args = ap.parse_args()

    meta = json.loads((ROOT / "packs" / args.task / "meta.json").read_text())
    id_col, pred_cols, metric = meta["id_col"], meta["pred_cols"], meta["metric"]
    res = {"task": args.task, "metric": metric, "valid": False, "score": None,
           "n": None, "errors": []}

    # Read id + prediction columns as strings so digit-only labels (e.g. the
    # multi-hot "0000000001") keep their leading zeros; numeric metrics coerce
    # back with to_numeric where needed.
    str_cols = {c: str for c in [id_col, *pred_cols]}
    answers = pd.read_csv(ROOT / "private" / args.task / "answers.csv",
                          dtype=str_cols)
    try:
        sub = pd.read_csv(args.submission, dtype=str_cols)
    except Exception as e:
        res["errors"].append(f"unreadable submission: {e}")
        out(res, 2)

    missing = [c for c in [id_col, *pred_cols] if c not in sub.columns]
    if missing:
        res["errors"].append(f"missing columns: {missing}")
        out(res, 2)
    if sub[id_col].duplicated().any():
        res["errors"].append("duplicate ids")
        out(res, 2)
    if set(sub[id_col]) != set(answers[id_col]):
        res["errors"].append(
            f"id mismatch: {len(set(answers[id_col]) - set(sub[id_col]))} missing, "
            f"{len(set(sub[id_col]) - set(answers[id_col]))} unexpected")
        out(res, 2)

    sub = sub.set_index(id_col).loc[answers[id_col]]
    y_true = answers[meta.get("truth_col", pred_cols[0])]

    if metric == "accuracy_str":
        y_pred = sub[pred_cols[0]].astype(str).str.strip()
        bad = ~y_pred.isin([str(c) for c in meta["classes"]])
        if bad.any():
            res["errors"].append(f"{int(bad.sum())} predictions outside allowed classes")
            out(res, 2)
        score = float((y_true.astype(str).to_numpy() == y_pred.to_numpy()).mean())
    elif metric == "pearson":
        y_pred = pd.to_numeric(sub[pred_cols[0]], errors="coerce")
        if y_pred.isna().any():
            res["errors"].append("non-numeric predictions")
            out(res, 2)
        yp = y_pred.to_numpy(dtype=float)
        yt = y_true.astype(float).to_numpy()
        score = 0.0 if yp.std() == 0 else float(np.corrcoef(yt, yp)[0, 1])
    elif metric == "korquad":
        def norm(s):
            return "".join(str(s).split()).lower()
        def char_f1(pred, truth):
            p, t = norm(pred), norm(truth)
            if not p or not t:
                return float(p == t)
            common = 0
            tc = list(t)
            for ch in p:
                if ch in tc:
                    tc.remove(ch)
                    common += 1
            if common == 0:
                return 0.0
            prec, rec = common / len(p), common / len(t)
            return 2 * prec * rec / (prec + rec)
        preds = sub[pred_cols[0]].fillna("").astype(str).to_numpy()
        truths = y_true.fillna("").astype(str).to_numpy()
        score = float(np.mean([char_f1(p, t) for p, t in zip(preds, truths)]))
    elif metric == "ner_f1":
        def parse_ents(s):
            s = "" if pd.isna(s) else str(s)
            out = set()
            for chunk in s.split("|"):
                chunk = chunk.strip()
                if ":" in chunk:
                    txt, lab = chunk.rsplit(":", 1)
                    if txt.strip():
                        out.add((txt.strip(), lab.strip()))
            return out
        preds = sub[pred_cols[0]]
        golds = y_true
        tp = fp = fn = 0
        for pg, gg in zip(preds, golds):
            pe, ge = parse_ents(pg), parse_ents(gg)
            tp += len(pe & ge)
            fp += len(pe - ge)
            fn += len(ge - pe)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        score = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    elif metric == "macro_f1":
        y_pred = sub[pred_cols[0]].astype(str).str.strip()
        bad = ~y_pred.isin([str(c) for c in meta["classes"]])
        if bad.any():
            res["errors"].append(f"{int(bad.sum())} predictions outside allowed classes")
            out(res, 2)
        score = f1_score(y_true.astype(str), y_pred, average="macro")
    elif metric == "accuracy":
        y_pred = pd.to_numeric(sub[pred_cols[0]], errors="coerce")
        if y_pred.isna().any():
            res["errors"].append("non-numeric predictions")
            out(res, 2)
        score = accuracy_score(y_true.astype(int), y_pred.round().astype(int))
    elif metric == "mae":
        y_pred = pd.to_numeric(sub[pred_cols[0]], errors="coerce")
        if y_pred.isna().any():
            res["errors"].append("non-numeric predictions")
            out(res, 2)
        score = float(np.mean(np.abs(y_true.astype(float).to_numpy()
                                     - y_pred.to_numpy())))
    elif metric == "rmse":
        y_pred = pd.to_numeric(sub[pred_cols[0]], errors="coerce")
        if y_pred.isna().any():
            res["errors"].append("non-numeric predictions")
            out(res, 2)
        score = float(np.sqrt(np.mean((y_true.astype(float).to_numpy()
                                       - y_pred.to_numpy()) ** 2)))
    elif metric == "multiclass_logloss":
        probs = sub[pred_cols].apply(pd.to_numeric, errors="coerce")
        if probs.isna().any().any() or (probs < 0).any().any():
            res["errors"].append("probabilities missing or negative")
            out(res, 2)
        p = probs.to_numpy(dtype=float)
        p = np.clip(p, 1e-15, None)
        p = p / p.sum(axis=1, keepdims=True)
        score = log_loss(y_true.astype(str), p, labels=meta["classes"])
    elif metric == "las":
        # Labeled Attachment Score for dependency parsing. Gold/pred serialize
        # one token per "|" chunk as "head:deprel" in token order. LAS = tokens
        # with correct head AND deprel; UAS = correct head only. Micro-averaged
        # over all gold tokens; a short/garbled prediction just loses tokens.
        def norm_head(h):
            try:
                return str(int(float(h)))
            except (ValueError, TypeError):
                return str(h).strip()

        def parse_dep(s):
            s = "" if pd.isna(s) else str(s)
            toks = []
            for chunk in s.split("|"):
                chunk = chunk.strip()
                if not chunk:
                    continue
                head, _, dep = chunk.partition(":")
                toks.append((norm_head(head), dep.strip()))
            return toks
        tot = las_c = uas_c = 0
        for pg, gg in zip(sub[pred_cols[0]], y_true):
            pe, ge = parse_dep(pg), parse_dep(gg)
            for i, (gh, gd) in enumerate(ge):
                tot += 1
                if i < len(pe) and pe[i][0] == gh:
                    uas_c += 1
                    if pe[i][1] == gd:
                        las_c += 1
        score = las_c / tot if tot else 0.0
        res["uas"] = round(uas_c / tot, 6) if tot else 0.0
    elif metric == "multilabel_f1":
        # kor_unsmile: fixed-order K-bit multi-hot string ("0100000000").
        # Macro-F1 across the K label columns; malformed pred → all-zeros row.
        K = len(meta["classes"])

        def bits(s):
            s = "" if pd.isna(s) else str(s)
            s = "".join(ch for ch in s if ch in "01")
            return [int(c) for c in s] if len(s) == K else None
        yt = [bits(v) for v in y_true]
        if any(v is None for v in yt):
            res["errors"].append("corrupt gold labels")
            out(res, 2)
        yp = [b if (b := bits(v)) is not None else [0] * K
              for v in sub[pred_cols[0]]]
        YT, YP = np.array(yt), np.array(yp)
        present = [j for j in range(K) if YT[:, j].sum() > 0]  # skip empty labels
        if not present:
            score = 0.0
        else:
            per = f1_score(YT[:, present], YP[:, present], average=None,
                           zero_division=0)
            score = float(np.mean(per))
    else:
        res["errors"].append(f"unknown metric {metric}")
        out(res, 2)

    res.update(valid=True, score=round(float(score), 6), n=len(answers))
    out(res, 0)


if __name__ == "__main__":
    main()
