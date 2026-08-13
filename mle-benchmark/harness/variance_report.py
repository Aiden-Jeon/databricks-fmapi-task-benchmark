#!/usr/bin/env python
"""Characterise run-to-run variance across repeats: by model and by output shape.

Motivation: the n=1 board called races on gaps of 0.1-0.7% while identical
reruns move by several percent, so variance is a first-class result, not a
footnote. Emits results/variance.md + results/variance_cells.csv.

Relative std (std / |mean|, in %) is the comparison unit — the suite mixes
metrics whose absolute scales differ by orders of magnitude (RMSE ~300 vs
MAE ~0.02), so raw std is not comparable across tasks.
"""
import csv
import collections
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANE = {"M1": "opus", "M2": "sol", "M3": "glm"}
NAMES = {"opus": "Opus 5", "sol": "GPT-5.6-sol", "glm": "GLM 5.2"}


def output_shape(meta):
    """Objective output-shape class, read off the pack — not a hand grouping."""
    if meta.get("classes") is not None:
        return "closed-set label"
    m = meta["metric"]
    if m in ("las", "ner_f1"):
        return "structured parse"
    if m == "korquad":
        return "free-text span"
    if m == "multilabel_f1":
        return "multi-label set"
    if m in ("mae", "rmse", "multiclass_logloss", "pearson"):
        return "numeric regression"
    return "open string label"


def main():
    shapes, metrics = {}, {}
    for p in sorted((ROOT / "packs").iterdir()):
        mj = p / "meta.json"
        if mj.exists():
            meta = json.loads(mj.read_text())
            shapes[p.name] = output_shape(meta)
            metrics[p.name] = meta["metric"]

    vals = collections.defaultdict(list)
    for r in csv.DictReader((ROOT / "results" / "scores.csv").open()):
        if r["lane"] in LANE and r["valid"] == "True" and r["score"]:
            vals[(r["task"], LANE[r["lane"]])].append(float(r["score"]))

    cells = []
    for (task, model), v in vals.items():
        if len(v) < 2:
            continue
        mean = sum(v) / len(v)
        sd = st.stdev(v)
        cells.append({"task": task, "model": model, "shape": shapes.get(task, "?"),
                      "metric": metrics.get(task, "?"), "n": len(v),
                      "mean": round(mean, 6), "std": round(sd, 6),
                      "rel_std_pct": round(sd / abs(mean) * 100, 3) if mean else None,
                      "min": round(min(v), 6), "max": round(max(v), 6)})
    cells.sort(key=lambda c: -(c["rel_std_pct"] or 0))

    by_model = collections.defaultdict(list)
    by_shape = collections.defaultdict(list)
    for c in cells:
        if c["rel_std_pct"] is not None:
            by_model[c["model"]].append(c["rel_std_pct"])
            by_shape[c["shape"]].append(c["rel_std_pct"])

    L = ["# Run-to-run variance (M-track repeats)", "",
         "Unit = **relative std** (std / |mean|, %) so tasks with different metric "
         "scales are comparable. Only cells with n≥2 appear.", "",
         "## By model — consistency is a model property", "",
         "| Model | cells (n≥2) | median rel-std | max rel-std |", "|---|---|---|---|"]
    for m, v in sorted(by_model.items(), key=lambda kv: st.median(kv[1])):
        L.append(f"| {NAMES[m]} | {len(v)} | **{st.median(v):.2f}%** | {max(v):.2f}% |")
    L += ["", "## By output shape — a weak predictor, with big exceptions", "",
          "| Output shape | cells | median rel-std | max rel-std |", "|---|---|---|---|"]
    for s, v in sorted(by_shape.items(), key=lambda kv: -st.median(kv[1])):
        L.append(f"| {s} | {len(v)} | {st.median(v):.2f}% | {max(v):.2f}% |")
    L += ["", "Structured-output tasks carry the highest median, but the single most "
          "variable cell is a plain tabular regression — so output shape explains "
          "part of the spread, not all of it.", "",
          "## Most variable cells", "",
          "| rel-std | model | task | shape | runs |", "|---|---|---|---|---|"]
    for c in cells[:10]:
        L.append(f"| {c['rel_std_pct']:.2f}% | {NAMES[c['model']]} | {c['task']} | "
                 f"{c['shape']} | {c['min']:.4g} → {c['max']:.4g} (n={c['n']}) |")
    tight = [c for c in cells if (c["rel_std_pct"] or 0) < 0.5]
    L += ["", f"{len(tight)} of {len(cells)} cells are near-deterministic "
          f"(rel-std < 0.5%), so high variance is concentrated, not universal.", ""]

    (ROOT / "results" / "variance.md").write_text("\n".join(L))
    with (ROOT / "results" / "variance_cells.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cells[0].keys()))
        w.writeheader()
        w.writerows(cells)
    print("\n".join(L))


if __name__ == "__main__":
    main()
