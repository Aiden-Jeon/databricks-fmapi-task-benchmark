#!/usr/bin/env python
"""Run the non-harness (direct-inference) lane across every direct-eligible task
× the 3 core models, and assemble a scoreboard. Reuses direct_infer's per-task
run (Korean instruction + few-shot + test row → label, no agent, no code).

Emits results/direct_scoreboard.md and results/direct_scores.csv.
Usage: direct_sweep.py [n_test] [k_shot]
"""
import concurrent.futures as cf
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import direct_infer as di  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS = [("opus", "databricks-claude-opus-5"),
          ("sol", "databricks-gpt-5-6-sol"),
          ("glm", "databricks-glm-5-2")]


def eligible_tasks():
    out = []
    for d in sorted((ROOT / "packs").glob("t*/")):
        m = d / "meta.json"
        if not m.exists():
            continue
        meta = json.loads(m.read_text())
        if meta.get("classes"):
            out.append((d.name, meta))
    return out


def run_one(task, meta, model):
    """One (task, model) cell → accuracy over n sampled test rows."""
    classes = meta["classes"]
    is_str = meta["metric"] == "accuracy_str"
    idc = meta["id_col"]
    tr = pd.read_csv(ROOT / "packs" / task / "train.csv", dtype={idc: str})
    te = pd.read_csv(ROOT / "packs" / task / "test.csv", dtype={idc: str})
    ans = pd.read_csv(ROOT / "private" / task / "answers.csv", dtype={idc: str})
    input_cols = [c for c in te.columns if c != idc]
    per = max(1, K // max(1, len(classes)))
    shots = pd.concat([g.sample(min(per, len(g)), random_state=1)
                       for _, g in tr.groupby("label")]).head(K)
    te = te.sample(min(N, len(te)), random_state=1).reset_index(drop=True)

    def infer(row):
        p = di.build_prompt(meta.get("title_ko", task), classes, input_cols,
                            shots, row, is_str)
        try:
            return row[idc], di.parse_label(di.chat(model, p), classes, is_str)
        except Exception:
            return row[idc], (str(classes[0]) if is_str else -999)

    preds = {}
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for rid, lab in ex.map(infer, [r for _, r in te.iterrows()]):
            preds[rid] = lab
    amap = dict(zip(ans[idc], ans["label"].astype(str)))
    acc = sum(1 for rid, p in preds.items() if str(p) == amap.get(rid)) / len(preds)
    return acc, len(preds)


def main():
    global N, K
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    tasks = eligible_tasks()
    rows = []
    for task, meta in tasks:
        cells = {}
        for short, model in MODELS:
            acc, n = run_one(task, meta, model)
            cells[short] = acc
            print(f"{task:24s} {short:5s} acc={acc:.4f} (n={n})", flush=True)
        rows.append({"task": task, "metric": meta["metric"],
                     "title_ko": meta.get("title_ko", task), **cells})
    df = pd.DataFrame(rows)
    outd = ROOT / "results"
    df.to_csv(outd / "direct_scores.csv", index=False)

    # markdown scoreboard
    lines = ["# Non-Harness (Direct-Inference) Scoreboard",
             "",
             f"Direct chat-API inference — Korean instruction + {K}-shot + test row "
             f"→ label. No coding agent, no code execution. n={N} sampled test rows "
             "per cell. Accuracy (proxy for macro-F1/logloss tasks — hard labels only).",
             "",
             "| Task (metric) | Opus 5 | GPT-5.6-sol | GLM 5.2 |",
             "|---|---|---|---|"]
    for r in rows:
        best = max(r["opus"], r["sol"], r["glm"])
        def cell(v):
            s = f"{v:.3f}"
            return f"**{s}**" if abs(v - best) < 1e-9 else s
        lines.append(f"| {r['title_ko']} ({r['metric']}) | "
                     f"{cell(r['opus'])} | {cell(r['sol'])} | {cell(r['glm'])} |")
    # win counts
    w = {"opus": 0, "sol": 0, "glm": 0}
    for r in rows:
        best = max(r["opus"], r["sol"], r["glm"])
        for k in w:
            if abs(r[k] - best) < 1e-9:
                w[k] += 1
    lines += ["",
              f"Wins (incl. ties): Opus 5 {w['opus']} · GPT-5.6-sol {w['sol']} · "
              f"GLM 5.2 {w['glm']} (of {len(rows)} tasks).",
              "",
              "*Direct-eligible = fixed class set. The other 7 tasks "
              "(regression/correlation/extractive-QA/NER) require code and are "
              "M-track only.*"]
    (outd / "direct_scoreboard.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
