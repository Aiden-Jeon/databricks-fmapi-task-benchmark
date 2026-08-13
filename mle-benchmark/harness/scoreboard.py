#!/usr/bin/env python
"""Grade every un-graded run artifact, then regenerate results/scoreboard.md.

Source of truth = the artifacts volume (each successful run wrote result.json +
outputs/submission.csv), NOT the run manifest (which may carry superseded rows).
Idempotent: re-running only grades new artifacts.

Usage: .venv/bin/python kmle/harness/scoreboard.py
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

PY = sys.executable

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "config.json").read_text())
PROFILE, CAT = CFG["profile"], CFG["catalog"]
VOL = f"dbfs:/Volumes/{CAT}/{CFG['schemas']['results']}/{CFG['volumes']['results']}"
SCORES = ROOT / "results" / "scores.csv"
BOARD = ROOT / "results" / "scoreboard.md"

# Task list + score direction are derived from the built packs' meta.json so the
# board always covers every task (ordered t1..t25 by numeric prefix).
def _task_num(name):
    p = name[1:].split("_", 1)[0]
    return int(p) if p.isdigit() else 999


TASKS, HIB = [], {}
for _p in sorted((ROOT / "packs").iterdir(), key=lambda d: _task_num(d.name)):
    _mj = _p / "meta.json"
    if not _mj.exists():
        continue
    TASKS.append(_p.name)
    HIB[_p.name] = json.loads(_mj.read_text())["higher_is_better"]
LANE_NAME = {"L1": "L1 ClaudeCode+Opus5", "L2": "L2 Codex+GPT5.6", "L3": "L3 opencode+GLM5.2",
             "M1": "Opus 5", "M2": "GPT-5.6-sol", "M3": "GLM 5.2",
             "M4": "Qwen3.5-122B", "M5": "Llama-4-Mav", "M6": "gpt-oss-120b"}


def grade_all():
    graded = set()
    if SCORES.exists():
        graded = {r["run_dir"] for r in csv.DictReader(open(SCORES))}
    listing = subprocess.run(["databricks", "fs", "ls", VOL, "-p", PROFILE],
                             capture_output=True, text=True).stdout.split()
    todo = [d for d in listing if "_full_" in d and d not in graded]
    for d in todo:
        subprocess.run([PY, str(ROOT / "harness" / "grade_run.py"), d],
                       capture_output=True, text=True)
    return len(todo)


def latest_by_cell():
    """Most recent valid (or latest) row per (lane, task)."""
    rows = list(csv.DictReader(open(SCORES)))
    by = {}
    for r in rows:
        lane = r["run_dir"].split("_")[0]
        key = (lane, r["task"])
        # prefer a valid row; otherwise keep latest by graded_utc
        cur = by.get(key)
        better = (cur is None or (r["valid"] == "True" and cur["valid"] != "True")
                  or (r["valid"] == cur["valid"] and r["graded_utc"] > cur["graded_utc"]))
        if better:
            by[key] = r
    return by


def fmt(cell, is_best):
    if cell is None:
        return "—"
    if cell["valid"] != "True":
        return "DNF"
    s = float(cell["score"])
    w = float(cell["wall_seconds"]) / 60 if cell["wall_seconds"] else 0
    txt = f"{s:.4g} ({w:.0f}m)"
    return f"**{txt}**" if is_best else txt


def section(by, lanes):
    lines = [f"| Task | {' | '.join(LANE_NAME[L] for L in lanes)} |",
             "|" + "---|" * (len(lanes) + 1)]
    for t in TASKS:
        cells = {L: by.get((L, t)) for L in lanes}
        valid = {L: float(c["score"]) for L, c in cells.items()
                 if c and c["valid"] == "True"}
        best = (max if HIB[t] else min)(valid.values()) if valid else None
        row = [t] + [fmt(cells[L], valid.get(L) == best and best is not None)
                     for L in lanes]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    n = grade_all()
    by = latest_by_cell()
    m_lanes = [L for L in ["M1", "M2", "M3", "M4"] if any((L, t) in by for t in TASKS)]
    s_lanes = [L for L in ["L1", "L2", "L3"] if any((L, t) in by for t in TASKS)]
    total = sum(1 for _ in by)
    valid = sum(1 for c in by.values() if c["valid"] == "True")
    with open(BOARD, "w") as f:
        f.write(f"# K-MLE-Bench Scoreboard\n\n")
        f.write(f"_{total} graded cells, {valid} valid. Newly graded this run: {n}._\n\n")
        f.write("> **Operational view — not the verdict.** Each cell is the *most "
                "recent* run, so bold marks the best single run, not a decided "
                "winner. Run-to-run spread reaches ±0.17 on structured-output "
                "tasks, so headline wins come from `aggregate_repeats.py` "
                "(mean ± std over all repeats + a tie test) — see "
                "`results/repeats_board.md`.\n\n")
        f.write("## PRIMARY — M-track (fixed opencode harness, model swapped)\n\n")
        f.write(section(by, m_lanes) + "\n\n")
        f.write("## REFERENCE — stack track (native harnesses)\n\n")
        f.write(section(by, s_lanes) + "\n\n")
        f.write("Excluded OSS (tool-calling failures, see report §6): "
                "Llama-4-Maverick (rejects tool schema), gpt-oss-120b (malformed tool calls).\n")
        f.write("Genie Code (L4) = UI-dependent reference lane, run separately.\n")
    print(open(BOARD).read())


if __name__ == "__main__":
    main()
