#!/usr/bin/env python
"""Poll all 'full' runs in the manifest until every one is terminal.

Prints one summary line per sweep; exits when done (or after --max-hours).
Network errors (VPN flaps) are treated as 'unknown, keep polling'.
"""
import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "config.json").read_text())
TERMINAL = {"TERMINATED", "INTERNAL_ERROR", "SKIPPED"}


def state(run_id: str) -> str:
    p = subprocess.run(["databricks", "jobs", "get-run", str(run_id),
                        "-p", CFG["profile"], "-o", "json"],
                       capture_output=True, text=True)
    if p.returncode != 0:
        return "?/?"
    try:
        s = json.loads(p.stdout)["state"]
        return f"{s.get('life_cycle_state', '?')}/{s.get('result_state', '-')}"
    except Exception:
        return "?/?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--max-hours", type=float, default=4)
    args = ap.parse_args()

    with open(ROOT / "results" / "run_manifest.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["mode"] == "full"]
    ids = {r["run_id"]: f"{r['lane']}x{r['task']}" for r in rows}

    deadline = time.time() + args.max_hours * 3600
    while True:
        states = {rid: state(rid) for rid in ids}
        done = [k for k, v in states.items() if v.split("/")[0] in TERMINAL]
        pend = {ids[k]: v for k, v in states.items()
                if v.split("/")[0] not in TERMINAL}
        fails = {ids[k]: v for k, v in states.items()
                 if v.endswith("/FAILED") or "INTERNAL_ERROR" in v}
        print(f"{time.strftime('%H:%M')} terminal={len(done)}/{len(ids)} "
              f"pending={list(pend)[:6]} fails={list(fails)}", flush=True)
        if len(done) == len(ids):
            print("MATRIX COMPLETE")
            print(json.dumps({ids[k]: v for k, v in states.items()}, indent=1))
            return
        if time.time() > deadline:
            print("POLLER DEADLINE REACHED")
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
