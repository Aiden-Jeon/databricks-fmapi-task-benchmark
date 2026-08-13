#!/usr/bin/env python
"""Submit the K-MLE-Bench run matrix (automated lanes) as serverless one-time runs.

Usage:
  python submit_matrix.py                 # all 3 lanes x all 5 tasks, full mode
  python submit_matrix.py --lanes L1 --tasks t3_ynat            # subset
  python submit_matrix.py --mode smoke                          # smoke matrix
  python submit_matrix.py --lanes M7 --mode full                # one M-track model
                                                                # (M-track = pinned
                                                                # opencode, see runner.py)

Appends every submission to kmle/results/run_manifest.csv (the cost join key).
Requires: databricks CLI with profile newjeans-ontos.
"""
import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_CFG = json.loads((ROOT / "config.json").read_text())
PROFILE = _CFG["profile"]
CATALOG = _CFG["catalog"]
RUNNER = f"/Workspace{_CFG['workspace_runner_dir']}/runner.py"
MANIFEST = ROOT / "results" / "run_manifest.csv"
LANES = ["L1", "L2", "L3"]
TASKS = ["t1_pubg", "t2_spooky", "t3_ynat", "t4_nsmc", "t5_bike",
         "t6_klue_nli", "t7_klue_sts", "t8_beep", "t9_korquad",
         "t10_kornli", "t11_korsts", "t12_kobest_boolq", "t13_kobest_copa",
         "t14_kobest_wic", "t15_kobest_hellaswag", "t16_kobest_sentineg",
         "t17_pawsx_ko", "t18_klue_re", "t19_klue_mrc", "t20_klue_ner",
         "t21_kmmlu", "t23_korfin_asc", "t24_kor_unsmile", "t25_klue_dp"]


def submit(lane: str, task: str, mode: str) -> dict:
    payload = {
        "run_name": f"kmle-{mode}-{lane}-{task}",
        "environments": [{"environment_key": "default", "spec": {"client": "3"}}],
        "tasks": [{
            "task_key": "run",
            "environment_key": "default",
            "spark_python_task": {"python_file": RUNNER,
                                  "parameters": [lane, task, mode, CATALOG]},
            "timeout_seconds": 2400 if mode == "smoke" else 9000,
        }],
    }
    p = subprocess.run(
        ["databricks", "jobs", "submit", "--json", json.dumps(payload),
         "-p", PROFILE, "--no-wait", "-o", "json"],
        capture_output=True, text=True, check=True)
    info = json.loads(p.stdout)
    return {"submitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "lane": lane, "task": task, "mode": mode,
            "run_id": info["run_id"],
            "run_page_url": info.get("run_page_url", "")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lanes", nargs="*", default=LANES)
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--mode", default="full", choices=["full", "smoke"])
    args = ap.parse_args()

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    new_file = not MANIFEST.exists()
    with open(MANIFEST, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["submitted_utc", "lane", "task",
                                          "mode", "run_id", "run_page_url"])
        if new_file:
            w.writeheader()
        for lane in args.lanes:
            for task in args.tasks:
                row = submit(lane, task, args.mode)
                w.writerow(row)
                f.flush()
                print(f"submitted {lane} x {task} ({args.mode}) -> run {row['run_id']}")


if __name__ == "__main__":
    main()
