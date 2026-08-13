#!/usr/bin/env python
"""Freeze the current campaign results into snapshots/<id>/ for longitudinal
tracking. A snapshot is the unit of comparison over time: re-running the same
model months later (or adding a new model) becomes a NEW snapshot, and
timeline.py renders the drift between snapshots.

Usage:
  python harness/snapshot.py                       # id = YYYY-MM auto
  python harness/snapshot.py --id 2026-08-v1 --notes "first 3-model campaign"

Writes:
  snapshots/<id>/board.csv     task,model,endpoint,metric,higher_is_better,
                               mean,std,se,n
  snapshots/<id>/snapshot.json id, frozen_utc, harness pin, endpoints, window,
                               notes — everything needed to interpret board.csv
                               without this working tree.
"""
import argparse
import csv
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from aggregate_repeats import LANE, NAMES  # noqa: E402  (same dir)
from build_repo import MODEL, TASKS       # noqa: E402


def harness_pin():
    m = re.search(r'OPENCODE_PIN = "([^"]+)"',
                  (ROOT / "harness" / "runner.py").read_text())
    return m.group(1) if m else "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default=time.strftime("%Y-%m"))
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    stats = list(csv.DictReader((ROOT / "results" / "repeats_stats.csv").open()))
    if not stats:
        raise SystemExit("results/repeats_stats.csv is empty — run "
                         "aggregate_repeats.py first")

    # campaign window from the manifest (submission timestamps)
    manifest = ROOT / "results" / "run_manifest.csv"
    window = {}
    if manifest.exists():
        ts = [r["submitted_utc"] for r in csv.DictReader(manifest.open())
              if r["mode"] == "full"]
        if ts:
            window = {"first_submitted_utc": min(ts),
                      "last_submitted_utc": max(ts)}

    endpoint = {v[0]: v[2] for v in MODEL.values()}
    out = ROOT / "snapshots" / args.id
    out.mkdir(parents=True, exist_ok=True)

    with (out / "board.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "model", "endpoint", "metric",
                                          "higher_is_better", "mean", "std",
                                          "se", "n"])
        w.writeheader()
        for r in stats:
            slug, metric, direction = TASKS.get(
                r["task"], (r["task"], "?", "higher"))
            w.writerow({"task": r["task"], "model": r["model"],
                        "endpoint": endpoint.get(r["model"], "?"),
                        "metric": metric,
                        "higher_is_better": direction == "higher",
                        "mean": r["mean"], "std": r["std"], "se": r["se"],
                        "n": r["n"]})

    models = sorted({r["model"] for r in stats})
    (out / "snapshot.json").write_text(json.dumps({
        "id": args.id,
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "harness": {"scaffold": "opencode", "pin": harness_pin(),
                    "lanes": {k: v for k, v in LANE.items() if v in models}},
        "models": {m: {"name": NAMES.get(m, m),
                       "endpoint": endpoint.get(m, "?")} for m in models},
        "campaign_window": window,
        "notes": args.notes,
    }, indent=2, ensure_ascii=False) + "\n")
    print(f"froze {len(stats)} cells for {models} -> snapshots/{args.id}/")


if __name__ == "__main__":
    main()
