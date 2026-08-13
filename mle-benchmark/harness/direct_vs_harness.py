#!/usr/bin/env python
"""Overlay the non-harness (direct-inference) scoreboard against the fixed-harness
M-track, on the tasks where both lanes are scored by accuracy (10 of the 13
direct-eligible tasks; ynat/beep use macro-F1 and spooky log-loss, so those are
shown direct-only). Answers the real question: does the coding-agent harness help
or hurt on pure Korean NLU?

Reads results/direct_scores.csv + results/scores.csv. Emits
results/direct_vs_harness.md.
"""
import csv
import collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANE = {"M1": "opus", "M2": "sol", "M3": "glm"}
# tasks where the harness metric is also accuracy / accuracy_str
ACC_TASKS = {"t4_nsmc", "t6_klue_nli", "t10_kornli", "t12_kobest_boolq",
             "t13_kobest_copa", "t14_kobest_wic", "t15_kobest_hellaswag",
             "t16_kobest_sentineg", "t17_pawsx_ko", "t18_klue_re"}


def harness_scores():
    h = collections.defaultdict(dict)
    for r in csv.DictReader((ROOT / "results" / "scores.csv").open()):
        if r["lane"] in LANE and r["valid"] == "True" and r["score"]:
            h[r["task"]][LANE[r["lane"]]] = float(r["score"])
    return h


def main():
    direct = {r["task"]: r for r in
              csv.DictReader((ROOT / "results" / "direct_scores.csv").open())}
    har = harness_scores()
    lines = ["# Non-Harness vs Fixed-Harness — Korean NLU",
             "",
             "Same model, same task, same Korean data. **Direct** = chat-API "
             "inference (instruction + few-shot + test row → label, no agent, no "
             "code). **Harness** = the fixed opencode coding agent writing and "
             "running a classifier. Accuracy; direct on sampled test rows, harness "
             "on full test.",
             "",
             "| Task | Model | Harness | Direct | Δ (direct−harness) |",
             "|---|---|---|---|---|"]
    deltas = []
    for task in sorted(ACC_TASKS):
        if task not in direct:
            continue
        d = direct[task]
        for m in ("opus", "sol", "glm"):
            hv = har.get(task, {}).get(m)
            dv = float(d[m])
            if hv is None:
                lines.append(f"| {d['title_ko']} | {m} | — (DNF) | {dv:.3f} | — |")
                continue
            delta = dv - hv
            deltas.append(delta)
            mark = "🟢" if delta > 0.02 else ("🔴" if delta < -0.02 else "⚪")
            lines.append(f"| {d['title_ko']} | {m} | {hv:.3f} | {dv:.3f} | "
                         f"{mark} {delta:+.3f} |")
    mean = sum(deltas) / len(deltas) if deltas else 0.0
    wins = sum(1 for x in deltas if x > 0.02)
    losses = sum(1 for x in deltas if x < -0.02)
    lines += ["",
              f"**Mean Δ = {mean:+.3f}** across {len(deltas)} model×task cells "
              f"(direct better in {wins}, harness better in {losses}, "
              f"within ±0.02 in {len(deltas)-wins-losses}).",
              "",
              "Interpretation: on pure language-understanding tasks, forcing the "
              "model to *write a classifier* (harness) throws away the judgment it "
              "applies when it reads each example *directly*. The coding-agent "
              "harness earns its cost on tabular/regression tasks (pubg, bike) "
              "where code is essential — not on NLU.",
              "",
              "*ynat & beep (macro-F1) and spooky (log-loss) are direct-only in "
              "this overlay — different metric from the harness cell — see "
              "direct_scoreboard.md for their direct accuracy.*"]
    (ROOT / "results" / "direct_vs_harness.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
