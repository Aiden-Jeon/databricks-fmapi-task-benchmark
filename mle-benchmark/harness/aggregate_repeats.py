#!/usr/bin/env python
"""Aggregate n≥1 repeat runs per (task, model) into a mean±std board and decide
each race. Reads results/scores.csv (all graded M-track runs); a cell's repeats
are simply its multiple valid runs (same fixed split, agent stochasticity = the
distribution). Emits results/repeats_board.md + results/repeats_stats.csv.

A race is called only when the leader's advantage clears the noise: we require
the gap between the top-two means to exceed the larger of their standard errors
(pooled). Otherwise it's reported as a statistical tie at the current n.
"""
import csv
import collections
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANE = {"M1": "opus", "M2": "sol", "M3": "glm", "M7": "kimi"}
NAMES = {"opus": "Opus 5", "sol": "GPT-5.6-sol", "glm": "GLM 5.2",
         "kimi": "Kimi K3"}
ORDER = ["opus", "sol", "glm", "kimi"]  # canonical column order


def load():
    runs = collections.defaultdict(lambda: collections.defaultdict(list))
    dnf = collections.defaultdict(lambda: collections.defaultdict(int))
    for r in csv.DictReader((ROOT / "results" / "scores.csv").open()):
        if r["lane"] not in LANE:
            continue
        m = LANE[r["lane"]]
        if r["valid"] == "True" and r["score"]:
            runs[r["task"]][m].append(float(r["score"]))
        else:
            dnf[r["task"]][m] += 1
    return runs, dnf


def cell_stats(vals):
    n = len(vals)
    mean = sum(vals) / n
    sd = st.stdev(vals) if n >= 2 else 0.0
    se = sd / math.sqrt(n) if n >= 2 else 0.0
    return mean, sd, se, n


def decide(cells, hib):
    """Call a race from {model: (mean, sd, se, n)}. Returns (kind, leader, rival).

    kind is "win" | "tie" | "undecided". The leader is compared against EVERY
    challenger, not just rank #2: a high-variance model can sit third by mean
    yet still have a band overlapping the leader, which must block the call.
    """
    if not cells:
        return None, None, None
    ranked = sorted(cells.items(), key=lambda kv: kv[1][0], reverse=hib)
    m1, s1 = ranked[0]
    if len(ranked) == 1:
        return "win", m1, None
    overlap = [(abs(s1[0] - s[0]), m) for m, s in ranked[1:]
               if abs(s1[0] - s[0]) <= max(s1[2], s[2]) and max(s1[2], s[2]) > 0]
    if overlap:
        return "tie", m1, min(overlap)[1]
    if s1[3] < 2:  # leader still n=1 — no variance estimate, so no call
        return "undecided", m1, None
    return "win", m1, None


def main():
    runs, dnf = load()
    packs = {p.name: json.loads((p / "meta.json").read_text())
             for p in (ROOT / "packs").iterdir() if (p / "meta.json").exists()}
    tasks = sorted(packs, key=lambda n: int(n[1:].split("_")[0])
                   if n[1:].split("_")[0].isdigit() else 999)

    # Board columns = canonical order, but only models with at least one run
    # (or DNF) anywhere — a model without any campaign data gets no column.
    seen = {m for t in runs for m in runs[t]} | {m for t in dnf for m in dnf[t]}
    models = [m for m in ORDER if m in seen]

    stats_rows = []
    lines = ["# n≥1 Repeat Board — mean ± std (M-track)",
             "",
             "Each cell = mean ± sample std over that model's valid runs (n). "
             "A race is **called** only when the top-two means differ by more than "
             "the larger standard error; else **tie**. DNF counts shown when a "
             "model failed to submit.",
             "",
             "| Task (metric) | " + " | ".join(NAMES[m] for m in models) +
             " | verdict |",
             "|---|" + "---|" * (len(models) + 1)]
    wins = collections.Counter()
    ties = 0
    for t in tasks:
        meta = packs[t]
        hib = meta["higher_is_better"]
        cells = {}
        for m in models:
            vals = runs[t].get(m, [])
            if vals:
                cells[m] = cell_stats(vals)
        if not cells:
            continue

        def fmt(m):
            if m not in cells:
                d = dnf[t].get(m, 0)
                return f"DNF×{d}" if d else "—"
            mean, sd, se, n = cells[m]
            star = "" if n >= 2 else "†"  # † = still n=1
            return f"{mean:.4f}±{sd:.4f} (n={n}){star}"

        kind, m1, rival = decide(cells, hib)
        if kind == "tie":
            verdict = f"tie ({NAMES[m1]}≈{NAMES[rival]})"
            ties += 1
        elif kind == "undecided":
            verdict = f"{NAMES[m1]}?"
        else:
            verdict = f"**{NAMES[m1]}**"
            wins[m1] += 1
        for m in models:
            if m in cells:
                mean, sd, se, n = cells[m]
                stats_rows.append({"task": t, "model": m, "mean": round(mean, 6),
                                   "std": round(sd, 6), "se": round(se, 6), "n": n})
        lines.append(f"| {t} ({meta['metric']}) | " +
                     " | ".join(fmt(m) for m in models) + f" | {verdict} |")

    lines += ["",
              "Decided wins — " +
              " · ".join(f"{NAMES[m]} {wins[m]}" for m in models) +
              f" · statistical ties {ties}.",
              "† = still n=1 (repeat pending); '?' = leader n=1, undecided."]
    (ROOT / "results" / "repeats_board.md").write_text("\n".join(lines))
    with (ROOT / "results" / "repeats_stats.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "model", "mean", "std", "se", "n"])
        w.writeheader()
        w.writerows(stats_rows)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
