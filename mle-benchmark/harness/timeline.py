#!/usr/bin/env python
"""Render TIMELINE.md from snapshots/*/board.csv — how each (task, model) cell
moves across campaign snapshots. This is the "did the model change under us /
did the new model beat the old board" view.

A move is flagged **significant** only when |Δmean| between two snapshots
exceeds twice the pooled standard error of the two cells — same philosophy as
the win/tie call in aggregate_repeats.py: never claim a change smaller than
the measured run-to-run noise.

Usage: python harness/timeline.py     # writes mle-benchmark/TIMELINE.md
"""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPS = ROOT / "snapshots"


def load():
    snaps = {}
    for d in sorted(p for p in SNAPS.iterdir() if (p / "board.csv").exists()):
        meta = json.loads((d / "snapshot.json").read_text())
        cells = {}
        for r in csv.DictReader((d / "board.csv").open()):
            cells[(r["task"], r["model"])] = r
        snaps[d.name] = {"meta": meta, "cells": cells}
    return snaps


def fmt_cell(r):
    if r is None:
        return "—"
    mean, std, n = float(r["mean"]), float(r["std"]), int(r["n"])
    return f"{mean:.4g}±{std:.2g} (n={n})" if n >= 2 else f"{mean:.4g} (n=1)"


def delta(prev, cur):
    """Δ vs previous snapshot with a noise gate. Returns markdown or ''."""
    if prev is None or cur is None:
        return ""
    p, c = float(prev["mean"]), float(cur["mean"])
    d = c - p
    se = math.hypot(float(prev["se"]), float(cur["se"]))
    hib = cur["higher_is_better"] == "True"
    arrow = "▲" if (d > 0) == hib else "▼"       # improvement vs regression
    sig = se > 0 and abs(d) > 2 * se
    s = f"{arrow}{abs(d):.4g}"
    return f"**{s}**" if sig else f"{s} (≈noise)"


def main():
    snaps = load()
    if not snaps:
        raise SystemExit("no snapshots/ found — run harness/snapshot.py first")
    ids = list(snaps)  # sorted by id = chronological (YYYY-MM… naming)

    lines = ["# TIMELINE — 시점별 성능 추적",
             "",
             "각 캠페인 스냅샷(`snapshots/<id>/`) 간에 같은 (task, model) 셀이 어떻게",
             "움직였는지 본다. 새 모델 추가·기존 모델의 시점별 변화 모두 새 스냅샷 한 줄로",
             "들어온다. **Δ가 굵게** = 두 스냅샷 pooled SE의 2배를 넘는, 노이즈 밖의 변화.",
             "`(≈noise)` = 실행간 편차 안이므로 변화를 주장하지 않는다.",
             "",
             "## 스냅샷",
             "",
             "| id | frozen (UTC) | harness pin | models | notes |",
             "|---|---|---|---|---|"]
    for sid in ids:
        m = snaps[sid]["meta"]
        models = " · ".join(f"{v['name']}" for v in m["models"].values())
        lines.append(f"| {sid} | {m['frozen_utc']} | `{m['harness']['pin']}` "
                     f"| {models} | {m.get('notes', '')} |")

    # union of tasks/models across snapshots, stable order
    tasks, models = [], []
    for sid in ids:
        for (t, mo) in snaps[sid]["cells"]:
            if t not in tasks:
                tasks.append(t)
            if mo not in models:
                models.append(mo)

    lines += ["", "## 셀별 추이", "",
              "마지막 열 Δ = 직전 스냅샷 대비 (그 모델이 두 스냅샷 모두에 있을 때만)."]
    for t in tasks:
        any_row = next((snaps[s]["cells"].get((t, m))
                        for s in ids for m in models
                        if snaps[s]["cells"].get((t, m))), None)
        metric = any_row["metric"] if any_row else "?"
        hdr = "| model | " + " | ".join(ids) + " | Δ (last vs prev) |"
        lines += ["", f"### {t} — {metric}", "", hdr,
                  "|---|" + "---|" * (len(ids) + 1)]
        for mo in models:
            cells = [snaps[s]["cells"].get((t, mo)) for s in ids]
            if not any(cells):
                continue
            present = [c for c in cells if c]
            d = delta(present[-2], present[-1]) if len(present) >= 2 else ""
            lines.append(f"| {mo} | " +
                         " | ".join(fmt_cell(c) for c in cells) +
                         f" | {d} |")

    (ROOT / "TIMELINE.md").write_text("\n".join(lines) + "\n")
    print(f"wrote TIMELINE.md ({len(ids)} snapshots, {len(tasks)} tasks, "
          f"{len(models)} models)")


if __name__ == "__main__":
    main()
