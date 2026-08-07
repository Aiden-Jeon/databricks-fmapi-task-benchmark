"""언어 패리티 지표 — KP / P_q / P_a / P_c / KPS.

설계는 `../PARITY.md`. 교차언어 일관성 문헌
(RankC · xC · κp · Information Parity)에 맞췄다.

두 트랙을 **다르게** 다룬다:

- **PAIR** — 내가 만든 한/영 쌍. `pair_key`로 항목이 대응되므로
  **P_a(항목별 일치)까지 낼 수 있다.** 이게 진짜 패리티다.
- **OB** — OrchestrationBench. 한/영이 **번역이 아니라 서로 다른 시나리오**다
  (실측: gold 에이전트 집합 일치 27/219). 쌍이 없으므로 **P_q와 P_c만** 낸다.
  시나리오 난이도 차이가 교란으로 남는다 — 카카오 자체 리더보드도 이 방식이다.

사용:
    python -m src.parity --pair results/parity-PAIR --ob results/parity-OB
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections import defaultdict
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from score import PRICING, USD_PER_DBU, wilson  # noqa: E402


def usd(arm: str, u: dict[str, int]) -> float:
    p = PRICING[arm.split("-")[0]]
    fresh = max(0, u["prompt_tokens"] - u["cache_read_tokens"] - u["cache_write_tokens"])
    return USD_PER_DBU * (fresh / 1e6 * p["in"]
                          + u["cache_read_tokens"] / 1e6 * p["cr"]
                          + u["cache_write_tokens"] / 1e6 * p["cw"]
                          + u["completion_tokens"] / 1e6 * p["out"])


def load(run: str) -> list[dict[str, Any]]:
    return [json.loads(l) for l in
            (pathlib.Path(run) / "scored.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]


def summarize(rows: list[dict[str, Any]], lang: str) -> dict[str, Any]:
    r = [x for x in rows if x.get("lang") == lang and x["score"] is not None]
    if not r:
        return {"n": 0}
    k = sum(1 for x in r if x["score"].get("correct"))
    tot_usd = sum(x.get("usd", 0.0) for x in r)
    lat = sorted(x["latency_ms"] for x in r if x.get("latency_ms"))
    return {
        "n": len(r), "k": k, "acc": k / len(r),
        "usd": tot_usd,
        "usd_per_correct": tot_usd / k if k else float("nan"),
        "p50": lat[len(lat) // 2] if lat else 0,
        "prompt_tok": sum(x["usage"]["prompt_tokens"] for x in r),
        "completion_tok": sum(x["usage"]["completion_tokens"] for x in r),
    }


def parity(ko: dict[str, Any], en: dict[str, Any], agree: float | None) -> dict[str, Any]:
    """P_q / P_a / P_c / KP / KPS."""
    if not ko.get("n") or not en.get("n"):
        return {}
    s_ko, s_en = ko["acc"], en["acc"]
    p_q = min(1.0, s_ko / s_en) if s_en else float("nan")
    c_ko, c_en = ko["usd_per_correct"], en["usd_per_correct"]
    p_c = min(1.0, c_en / c_ko) if c_ko else float("nan")
    out = {"S_KO": s_ko, "S_EN": s_en, "P_q": p_q, "KP": 1 - p_q, "P_c": p_c,
           "C_KO": c_ko, "C_EN": c_en}
    if agree is not None:
        chance = s_ko * s_en + (1 - s_ko) * (1 - s_en)
        p_a = max(0.0, (agree - chance) / (1 - chance)) if chance < 1 else 1.0
        out.update({"agree": agree, "chance": chance, "P_a": p_a})
        eps = 1e-6
        vals = [max(p_q, eps), max(p_a, eps), max(p_c, eps)]
        out["KPS"] = 3 / sum(1 / v for v in vals)
    return out


def item_agreement(rows: list[dict[str, Any]]) -> tuple[float | None, dict[str, int]]:
    """같은 pair_key에서 한/영이 **같은 판정**을 낸 비율.

    반복이 있으므로 케이스별 다수결로 판정을 확정한 뒤 비교한다.
    """
    byk: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for r in rows:
        if r["score"] is None or not r.get("pair_key"):
            continue
        byk[(r["pair_key"], r["lang"])].append(bool(r["score"].get("correct")))
    keys = {k for k, _ in byk}
    cell = {"both_ok": 0, "both_bad": 0, "ko_only": 0, "en_only": 0}
    n = agree = 0
    for k in keys:
        ko, en = byk.get((k, "KO")), byk.get((k, "EN"))
        if not ko or not en:
            continue
        vk = sum(ko) * 2 >= len(ko)   # 다수결
        ve = sum(en) * 2 >= len(en)
        n += 1
        agree += int(vk == ve)
        cell["both_ok" if (vk and ve) else
             "both_bad" if (not vk and not ve) else
             "ko_only" if vk else "en_only"] += 1
    cell["n"] = n
    return (agree / n if n else None), cell


def show(title: str, rows: list[dict[str, Any]], paired: bool) -> dict[str, Any]:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)
    arms = sorted({r["arm"] for r in rows})
    out: dict[str, Any] = {}
    hdr = f"{'arm':16s} {'S_KO':>13s} {'S_EN':>13s} {'KP↓':>8s} {'P_q':>7s}"
    if paired:
        hdr += f" {'P_a':>7s}"
    hdr += f" {'P_c':>7s}"
    if paired:
        hdr += f" {'KPS↑':>7s} {'raw일치':>8s} {'불일치':>7s}"
    hdr += f"  {'$/정답 KO→EN':>18s}"
    print(hdr)
    for a in arms:
        r = [x for x in rows if x["arm"] == a]
        ko, en = summarize(r, "KO"), summarize(r, "EN")
        ag, cell = item_agreement(r) if paired else (None, {})
        p = parity(ko, en, ag)
        if not p:
            continue
        out[a] = {"KO": ko, "EN": en, **p, "agreement_cells": cell}
        pk, lk, hk = wilson(ko["k"], ko["n"])
        pe, le, he = wilson(en["k"], en["n"])
        line = (f"{a:16s} {pk:.3f}[{lk:.2f},{hk:.2f}]".ljust(16 + 14)
                + f" {pe:.3f}[{le:.2f},{he:.2f}]".ljust(14)
                + f" {p['KP']:8.3f} {p['P_q']:7.3f}")
        if paired:
            line += f" {p['P_a']:7.3f}"
        line += f" {p['P_c']:7.3f}"
        if paired:
            dis = cell["ko_only"] + cell["en_only"]
            line += f" {p['KPS']:7.3f} {p['agree']:8.3f} {dis:4d}/{cell['n']:<3d}"
        line += f"  {p['C_KO']:.5f}→{p['C_EN']:.5f}"
        print(line)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair")
    ap.add_argument("--ob")
    ap.add_argument("--out", default="results/parity_report.json")
    a = ap.parse_args()
    report: dict[str, Any] = {}

    if a.pair:
        rows = load(a.pair)
        report["PAIR"] = show(
            "PAIR — 진짜 쌍 비교 (내 45개 케이스 한/영, pair_key로 항목 대응)\n"
            "KP = 한국어 페널티(낮을수록 좋음) · P_a = 우연보정 항목일치 · KPS = 조화평균\n"
            "⚠️ 정확도가 천장(>0.93)이면 우연일치가 이미 ~0.91이라 P_a·KPS가 불안정하다(카파 역설).\n"
            "   불일치 1~3건이 P_a를 크게 흔든다 → **raw 일치율과 불일치 건수를 같이 봐야 한다.**",
            rows, paired=True)
        # 카테고리별 KP
        print("\n  카테고리별 KP (한국어에서 얼마나 손해 보는가)")
        cats = sorted({r["category"] for r in rows})
        arms = sorted({r["arm"] for r in rows})
        print("  " + f"{'category':22s} " + " ".join(f"{x:>16s}" for x in arms))
        for c in cats:
            cells = []
            for arm in arms:
                sub = [r for r in rows if r["category"] == c and r["arm"] == arm]
                ko, en = summarize(sub, "KO"), summarize(sub, "EN")
                if not ko.get("n") or not en.get("n") or not en["acc"]:
                    cells.append("n/a")
                    continue
                kp = 1 - min(1.0, ko["acc"] / en["acc"])
                cells.append(f"{ko['acc']:.2f}/{en['acc']:.2f} KP{kp:+.2f}")
            print("  " + f"{c:22s} " + " ".join(f"{x:>16s}" for x in cells))

    if a.ob:
        rows = load(a.ob)
        report["OB"] = show(
            "OB — OrchestrationBench 모집단 비교 (219 EN vs 222 KO, **쌍 아님**)\n"
            "한/영이 서로 다른 시나리오다 → P_a·KPS 계산 불가. 난이도 차이가 교란으로 남는다.",
            rows, paired=False)
        print("\n  보조 지표 — 에이전트 집합 F1 / 타입 정확도")
        arms = sorted({r["arm"] for r in rows})
        print("  " + f"{'arm':16s} {'agent_F1 KO':>12s} {'agent_F1 EN':>12s} "
              f"{'type_acc KO':>12s} {'type_acc EN':>12s}")
        for arm in arms:
            cells = []
            for lang in ("KO", "EN"):
                r = [x for x in rows if x["arm"] == arm and x.get("lang") == lang and x["score"]]
                f1 = sum(x["score"].get("agent_f1", 0) for x in r) / len(r) if r else 0
                ty = sum(1 for x in r if x["score"].get("type_correct")) / len(r) if r else 0
                cells += [f1, ty]
            print("  " + f"{arm:16s} {cells[0]:12.3f} {cells[2]:12.3f} "
                  f"{cells[1]:12.3f} {cells[3]:12.3f}")

    pathlib.Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    print(f"\n→ {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
