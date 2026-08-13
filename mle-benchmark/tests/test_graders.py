#!/usr/bin/env python
"""Grader unit tests: perfect / baseline / malformed submissions per built task.

Run: .venv/bin/python kmle/tests/test_graders.py
Exit 0 = all pass. Part of the spec §11 verification plan.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # kmle/
GRADE = ROOT / "harness" / "grade.py"
PY = sys.executable
failures = []


def grade(task, df):
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        df.to_csv(f.name, index=False)
        path = f.name
    p = subprocess.run([PY, str(GRADE), "--task", task, "--submission", path],
                       capture_output=True, text=True)
    return json.loads(p.stdout.strip()), p.returncode


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"  {tag}  {name}  {detail}")
    if not cond:
        failures.append(name)


def perfect_sub(task, meta, answers):
    if meta["metric"] == "multiclass_logloss":
        sub = pd.DataFrame({"id": answers[meta["id_col"]]})
        for c in meta["classes"]:
            sub[c] = (answers[meta["truth_col"]] == c).astype(float)
        return sub
    return answers.rename(columns={meta.get("truth_col", meta["pred_cols"][0]):
                                   meta["pred_cols"][0]})


for pack in sorted((ROOT / "packs").iterdir()):
    task = pack.name
    meta = json.loads((pack / "meta.json").read_text())
    # read id + prediction columns as strings so digit-only labels (multi-hot
    # bitstrings) keep leading zeros — mirrors grade.py's reader.
    str_cols = {c: str for c in [meta["id_col"], *meta["pred_cols"]]}
    answers = pd.read_csv(ROOT / "private" / task / "answers.csv", dtype=str_cols)
    sample = pd.read_csv(pack / "sample_submission.csv", dtype=str_cols)
    hib = meta["higher_is_better"]
    print(f"[{task}] metric={meta['metric']}")

    res, rc = grade(task, perfect_sub(task, meta, answers))
    best_ok = (res["score"] >= 0.999999 if hib else res["score"] <= 0.01)
    check("perfect scores best", rc == 0 and res["valid"] and best_ok,
          f"score={res['score']}")
    perfect_score = res["score"]

    res, rc = grade(task, sample)
    worse = (res["score"] < perfect_score if hib else res["score"] > perfect_score)
    check("baseline scores worse than perfect", rc == 0 and res["valid"] and worse,
          f"score={res['score']}")

    res, rc = grade(task, sample.iloc[: len(sample) // 2])
    check("missing ids -> invalid", rc == 2 and not res["valid"], str(res["errors"]))

    res, rc = grade(task, sample.rename(columns={meta["pred_cols"][0]: "wrong"}))
    check("wrong columns -> invalid", rc == 2 and not res["valid"], str(res["errors"]))

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("ALL GRADER TESTS PASSED")
