#!/usr/bin/env python
"""Grade one benchmark run's submission from the results volume.

Usage: grade_run.py <RUN_ID_DIR>   (e.g. L1_t3_ynat_full_20260730_085352)
Downloads outputs/submission.csv from the artifacts volume, grades it locally
against kmle/private/<task>/answers.csv, appends to kmle/results/scores.csv.
Answers never leave this machine; the volume's private copy is untouched.
"""
import csv
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_CFG = json.loads((ROOT / "config.json").read_text())
PROFILE = _CFG["profile"]
VOL = (f"dbfs:/Volumes/{_CFG['catalog']}/{_CFG['schemas']['results']}"
       f"/{_CFG['volumes']['results']}")
SCORES = ROOT / "results" / "scores.csv"
PY = sys.executable

run_dir = sys.argv[1].rstrip("/")
# run_dir = <lane>_<task>_<mode>_<timestamp>; task may contain underscores
# (t6_klue_nli), so split on the mode marker rather than a fixed slice.
lane = run_dir.split("_")[0]
_m = re.search(r"^[^_]+_(.+?)_(full|smoke)_\d", run_dir)
task = _m.group(1) if _m else "_".join(run_dir.split("_")[1:3])

with tempfile.TemporaryDirectory() as td:
    sub = Path(td) / "submission.csv"
    meta = {}
    r = subprocess.run(["databricks", "fs", "cat", f"{VOL}/{run_dir}/result.json",
                        "-p", PROFILE], capture_output=True, text=True)
    if r.returncode == 0:
        meta = json.loads(r.stdout)
    dl = subprocess.run(["databricks", "fs", "cp",
                         f"{VOL}/{run_dir}/outputs/submission.csv", str(sub),
                         "-p", PROFILE], capture_output=True, text=True)
    if dl.returncode != 0:
        grade = {"task": task, "metric": None, "valid": False, "score": None,
                 "errors": ["submission.csv missing in artifacts"]}
    else:
        g = subprocess.run([PY, str(ROOT / "harness" / "grade.py"),
                            "--task", task, "--submission", str(sub)],
                           capture_output=True, text=True)
        grade = json.loads(g.stdout.strip())

row = {
    "graded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "run_dir": run_dir, "lane": lane, "task": task,
    "valid": grade["valid"], "metric": grade["metric"], "score": grade["score"],
    "wall_seconds": meta.get("wall_seconds"), "timed_out": meta.get("timed_out"),
    "agent_exit_code": meta.get("exit_code"),
    "errors": ";".join(grade.get("errors", [])),
}
SCORES.parent.mkdir(parents=True, exist_ok=True)
new = not SCORES.exists()
with open(SCORES, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row))
    if new:
        w.writeheader()
    w.writerow(row)
print(json.dumps(row, ensure_ascii=False, indent=2))
