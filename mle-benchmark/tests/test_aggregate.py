#!/usr/bin/env python
"""Tests for the repeat-aggregation race caller.

This logic produces the benchmark's headline number ("Opus N / GPT N / ties N"),
so its edge cases are worth pinning: a wrong call here mis-states the result.

Run: .venv/bin/python kmle/tests/test_aggregate.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
from aggregate_repeats import cell_stats, decide  # noqa: E402

FAILS = []


def check(name, got, want):
    if got != want:
        FAILS.append(f"{name}: got {got!r}, want {want!r}")


def cell(*vals):
    return cell_stats(list(vals))


# --- clean separation: leader's band clears every challenger -----------------
check("clear win, higher-is-better",
      decide({"opus": cell(0.90, 0.91), "sol": cell(0.50, 0.51)}, True),
      ("win", "opus", None))

check("clear win, lower-is-better (leader = smallest)",
      decide({"opus": cell(0.02, 0.021), "sol": cell(0.40, 0.41)}, False),
      ("win", "opus", None))

# --- overlapping bands must block the call ----------------------------------
# sol leads (mean .910) over opus (mean .900); gap .010 == sol's se .010 -> tie
check("overlap on rank #2 -> tie",
      decide({"sol": cell(0.900, 0.920), "opus": cell(0.895, 0.905)}, True),
      ("tie", "sol", "opus"))

# The regression that shipped a wrong headline: a high-variance model sitting
# THIRD by mean still overlaps the leader, so the race is not decided. Modeled
# on real t25_klue_dp numbers (Opus 0.5731/0.8199 vs GPT 0.799/0.7394).
third = decide({"opus": cell(0.5731, 0.8199),
                "sol": cell(0.7990, 0.7394),
                "glm": cell(0.7213, 0.7491)}, True)
check("high-variance THIRD place blocks the call", third[0], "tie")
check("...and names the overlapping challenger", third[2], "opus")

# --- n=1 has no variance estimate, so it cannot win -------------------------
check("leader at n=1 -> undecided",
      decide({"opus": cell(0.99), "sol": cell(0.50, 0.51)}, True),
      ("undecided", "opus", None))

check("sole model present still wins (nothing to compare)",
      decide({"opus": cell(0.42, 0.43)}, True),
      ("win", "opus", None))

check("empty cells -> no verdict", decide({}, True), (None, None, None))

# --- identical runs: zero variance must not create a spurious tie -----------
check("zero-variance leader beats zero-variance rival",
      decide({"opus": cell(0.80, 0.80), "sol": cell(0.70, 0.70)}, True),
      ("win", "opus", None))

# Exactly-equal means: gap 0 <= se, so tie (never an arbitrary winner).
check("identical means -> tie, not a coin flip",
      decide({"opus": cell(0.70, 0.80), "sol": cell(0.72, 0.78)}, True)[0],
      "tie")

# --- direction handling: lower-is-better must not invert the leader ---------
check("lower-is-better picks the smaller mean as leader",
      decide({"opus": cell(300.0, 320.0), "sol": cell(205.0, 218.0)}, False)[1],
      "sol")

# --- cell_stats basics ------------------------------------------------------
m, sd, se, n = cell(0.5, 0.7)
check("mean of two runs", round(m, 6), 0.6)
check("n counted", n, 2)
check("n=1 reports zero sd", cell(0.5)[1], 0.0)
check("n=1 reports zero se", cell(0.5)[2], 0.0)
check("se < sd for n=2", se < sd, True)

if FAILS:
    print(f"FAIL {len(FAILS)} check(s):")
    for f in FAILS:
        print("  " + f)
    sys.exit(1)
print("all aggregate-caller checks pass")
