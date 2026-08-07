#!/usr/bin/env python3
"""데이터셋별 변별력 비교 — 천장 효과를 정량화한다.

자체 제작 케이스와 외부 벤치마크가 모델을 얼마나 구분하는지 같은 기준으로 잰다.

    python3 -m src.discrimination

산출 지표:

- **셀 만점 비율**: (케이스 × 실험조건) 중 반복 전부 정답인 비율.
  높을수록 천장 효과가 크다 — 문제가 쉬워 모델이 구분되지 않는다.
- **1위·최하위 정확도 폭**: 클수록 변별력이 크다.
- **1위·2위 신뢰구간 중첩**: 겹치면 상위 두 모델의 차이를 확인할 수 없다.
- **순위**: 데이터셋 간 순위가 일치하는지 확인한다.
"""
from __future__ import annotations

import collections
import json
import math
import pathlib
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parent.parent


def wilson(c: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = c / n
    d = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (ctr - m, ctr + m)


def load(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def analyze(rows: list[dict[str, Any]], label: str, note: str = "") -> dict[str, Any] | None:
    per_arm: dict[str, list[int]] = collections.defaultdict(list)
    per_cell: dict[tuple[str, str], list[int]] = collections.defaultdict(list)

    for r in rows:
        if r.get("outcome") != "ok":
            continue
        s = r.get("score") or {}
        if s.get("correct") is None:
            continue
        v = 1 if s["correct"] else 0
        per_arm[r["arm"]].append(v)
        per_cell[(r["arm"], r["case_id"])].append(v)

    if not per_arm:
        return None

    cells = [sum(v) / len(v) for v in per_cell.values()]
    n_perfect = sum(1 for x in cells if x == 1.0)

    stats = {}
    for a, v in per_arm.items():
        c, n = sum(v), len(v)
        lo, hi = wilson(c, n)
        stats[a] = {"acc": c / n, "lo": lo, "hi": hi, "n": n}

    order = sorted(stats, key=lambda k: -stats[k]["acc"])
    top, second, last = order[0], order[1], order[-1]
    overlap = stats[top]["lo"] < stats[second]["hi"]
    spread = stats[top]["acc"] - stats[last]["acc"]

    print(f"\n{'─' * 78}\n{label}")
    if note:
        print(f"  {note}")
    print(f"  케이스 {len(set(c for _, c in per_cell))}  ·  호출 {sum(len(v) for v in per_arm.values())}")
    print(f"  셀 만점 비율   {n_perfect}/{len(cells)} = {n_perfect / len(cells):.1%}"
          f"   ← 높을수록 천장 효과")
    for a in order:
        s = stats[a]
        print(f"    {a:16} {s['acc']:.3f}  [{s['lo']:.3f}, {s['hi']:.3f}]  n={s['n']}")
    print(f"  순위          {' > '.join(order)}")
    print(f"  1위–최하위 폭  {spread:.3f}   ← 클수록 변별력")
    print(f"  1·2위 구간     {'겹침 → 차이 확인 불가' if overlap else '분리 → 차이 확인됨'}")

    return {"label": label, "order": order, "spread": spread,
            "ceiling": n_perfect / len(cells), "overlap": overlap, "stats": stats}


def main() -> int:
    R = ROOT / "results"
    out = []

    r = analyze(load(R / "all" / "scored.jsonl"),
                "[자체 제작] 본 실험 66 케이스",
                "직접 작성. 실험 조건 A/A′/B 전체")
    if r:
        out.append(r)

    core = load(R / "all" / "scored.jsonl")
    r = analyze([x for x in core if x["arm"] in ("opus-adaptive", "sol", "glm")],
                "[자체 제작] 본 실험 66 케이스 — 실험 조건 A만",
                "아래 외부 데이터셋과 같은 실험 조건")
    if r:
        out.append(r)

    pair = load(R / "parity-PAIR" / "scored.jsonl")
    r = analyze([x for x in pair if x.get("lang") == "KO"],
                "[자체 제작] PAIR 45 케이스 (한국어)",
                "본 실험 케이스 중 영어 대응이 성립하는 45개")
    if r:
        out.append(r)

    fcb = load(R / "fcb" / "scored.jsonl")
    r = analyze(fcb,
                "[외부] FunctionChat-Bench CallDecision 606 (전체)",
                "카카오 · Apache-2.0 · 한국어 function calling 전용")
    if r:
        out.append(r)
    if fcb:
        for cat in ("FCB-CALL", "FCB-REJECT", "FCB-SLOT-all", "FCB-SLOT-some"):
            sub = [x for x in fcb if x["category"] == cat]
            if not sub:
                continue
            sr = analyze(sub, f"  └ {cat}")
            # CALL 100건이 유일하게 변별력이 있는 부분집합이라 요약에도 싣는다.
            # 나머지 506건은 세 모델 모두 99% 이상이라 전체 평균을 끌어올린다.
            if sr and cat == "FCB-CALL":
                sr["label"] = "[외부] FunctionChat-Bench CALL 100 (변별 부분집합)"
                out.append(sr)

    ob = load(R / "parity-OB" / "scored.jsonl")
    r = analyze([x for x in ob if x.get("lang") == "KO"],
                "[외부] OrchestrationBench 한국어 222",
                "카카오 · Apache-2.0 · 다중 에이전트 계획 파생 태스크")
    if r:
        out.append(r)

    print(f"\n{'═' * 78}\n요약 — 변별력 비교\n{'═' * 78}")
    print(f"  {'데이터셋':52} {'셀만점':>7} {'폭':>7}  순위")
    for x in out:
        print(f"  {x['label']:52} {x['ceiling']:6.1%} {x['spread']:7.3f}  "
              f"{' > '.join(a.replace('opus-adaptive', 'opus') for a in x['order'])}")

    orders = {tuple(a.replace("opus-adaptive", "opus") for a in x["order"]) for x in out}
    print(f"\n  순위 일치 여부: "
          f"{'모든 데이터셋에서 동일' if len(orders) == 1 else f'{len(orders)}가지로 갈림'}")
    for o in sorted(orders):
        print(f"    {' > '.join(o)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
