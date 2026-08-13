#!/usr/bin/env python
"""One-shot status snapshot of every 'full' run in the manifest.

Uses the paginated jobs/list-runs endpoint (bulk) instead of one get-run per
run — at 216+ runs the per-run version costs hundreds of API calls per sweep.
Prints a lane x state matrix plus per-seed-pass progress, and exits.
"""
import argparse
import collections
import csv
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "config.json").read_text())
TERMINAL = {"TERMINATED", "INTERNAL_ERROR", "SKIPPED"}


PAGE = 200


def list_runs(want: int):
    """Yield up to `want` run dicts, newest-first.

    The CLI flattens list-runs to a bare JSON array and drops next_page_token,
    so page with --offset rather than a cursor.
    """
    got = 0
    while got < want:
        p = subprocess.run(
            ["databricks", "jobs", "list-runs", "-p", CFG["profile"], "-o", "json",
             "--limit", str(min(PAGE, want - got)), "--offset", str(got)],
            capture_output=True, text=True)
        if p.returncode != 0:
            break
        try:
            d = json.loads(p.stdout)
        except json.JSONDecodeError:
            break
        runs = d if isinstance(d, list) else d.get("runs", [])
        if not runs:
            break
        yield from runs
        got += len(runs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=400,
                    help="how many recent runs to fetch (must cover the manifest)")
    args = ap.parse_args()

    with open(ROOT / "results" / "run_manifest.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["mode"] == "full"]
    # A (lane, task) pair appears once per seed pass; number them in submit order
    seen = collections.Counter()
    meta = {}
    for r in rows:
        key = (r["lane"], r["task"])
        seen[key] += 1
        meta[r["run_id"]] = (r["lane"], r["task"], seen[key])

    live = {str(x["run_id"]): x.get("state", {}) for x in list_runs(args.window)}

    by_state = collections.Counter()
    by_pass = collections.defaultdict(collections.Counter)
    fails = []
    for rid, (lane, task, npass) in meta.items():
        s = live.get(rid)
        if s is None:
            label = "not-in-window"  # older than the pages we walked
        else:
            lcs = s.get("life_cycle_state", "?")
            label = lcs if lcs not in TERMINAL else s.get("result_state", "TERMINAL")
        by_state[label] += 1
        by_pass[npass][label] += 1
        if label in {"FAILED", "INTERNAL_ERROR", "TIMEDOUT"}:
            fails.append(f"{lane}x{task}#{npass} {label}")

    print(f"manifest full-runs: {len(meta)}  |  in list-runs window: {len(live)}")
    for k, v in sorted(by_state.items(), key=lambda kv: -kv[1]):
        print(f"  {k:16s} {v}")
    print("\nby seed pass:")
    for npass in sorted(by_pass):
        c = by_pass[npass]
        tot = sum(c.values())
        succ = c.get("SUCCESS", 0)
        print(f"  pass {npass}: {tot:3d} runs | SUCCESS {succ:3d} | "
              + " ".join(f"{k}={v}" for k, v in sorted(c.items()) if k != "SUCCESS"))
    if fails:
        print("\nfailures:")
        for f_ in fails[:30]:
            print("  " + f_)


if __name__ == "__main__":
    main()
