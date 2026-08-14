"""채점기 — 세션 단위 deterministic 채점 + arm×시나리오 집계.

원칙(function-calling-json/score.py 계승):
- 한 숫자로 합치지 않는다. 정확도·안정성·효율·비용을 분해해서 본다.
- 비율에는 Wilson 신뢰구간. pass^k = C(c,k)/C(n,k) (τ-bench 정의, "전부 성공").
- 인프라 실패 세션은 채점에서 제외한다.

사용:
    python -m src.score results/<run-id>
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import statistics
import sys
from collections import defaultdict
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parent.parent
CASES_DIR = ROOT / "cases"

USD_PER_DBU = 0.07
PRICING = {  # DBU per 1M tokens (image-text-performance/config/pricing.yaml 교차검증)
    "opus": {"in": 71.429, "out": 357.143, "cw": 89.286, "cr": 7.143},
    "sol": {"in": 71.429, "out": 428.571, "cw": 89.286, "cr": 7.143},
    "glm": {"in": 20.000, "out": 62.857, "cw": 0.0, "cr": 3.714},
}

ASK_PAT = re.compile(
    r"(알려주|말씀해|어느|어디|언제|몇\s*[명분일]|필요합니다|필요해|"
    r"확인\s*(부탁|해\s*주)|여쭤|알 수 없|모르|특정할 수 없|정해\s*주|선택해\s*주|\?)"
)

# 멀티턴 에이전트의 "정보 부족 → 되묻기/판정보류" 를 폭넓게 인식한다.
# ASK_PAT(단발 되물음 어휘)만으로는 "조건부", "판정 보류", "확정할 수 없습니다",
# "진단확정일 미기재 → 확인 필요" 같은 분석적 되묻기를 놓친다(더 잘한 모델을 감점).
HOLD_PAT = re.compile(
    r"(조건부|판정\s*(보류|불가|을?\s*내릴\s*수\s*없|이?\s*불가)|보류|"
    r"확정(할|하기|지)?\s*(수\s*없|불가|어렵)|산정\s*불가|판단\s*불가|"
    r"진단(확정)?일.{0,24}(없|미기재|기재되(지|어\s*있지)|미상|null|필요|확인|요청|제출)|"
    r"(추측|가정)(하지|없이).{0,12}(판정|계산|지급)|추가\s*(자료|정보|서류))"
)


def is_clarification(text: str) -> bool:
    return bool(ASK_PAT.search(text) or HOLD_PAT.search(text))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return (float("nan"), 0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def pass_hat_k(n: int, c: int, k: int) -> float | None:
    if n < k:
        return None
    return math.comb(c, k) / math.comb(n, k)


def usd(arm: str, u: dict[str, int]) -> float:
    p = PRICING[arm.split("-")[0]]
    fresh_in = max(0, u.get("prompt_tokens", 0) - u.get("cache_read_tokens", 0)
                   - u.get("cache_write_tokens", 0))
    # billable_output = completion_tokens (reasoning 포함, 더하지 않음)
    return USD_PER_DBU * (
        fresh_in / 1e6 * p["in"]
        + u.get("cache_read_tokens", 0) / 1e6 * p["cr"]
        + u.get("cache_write_tokens", 0) / 1e6 * p["cw"]
        + u.get("completion_tokens", 0) / 1e6 * p["out"]
    )


def load_case_index() -> dict[str, dict[str, Any]]:
    """case_id -> {expect, scoring} 로드."""
    idx: dict[str, dict[str, Any]] = {}
    for sdir in CASES_DIR.iterdir():
        if not sdir.is_dir():
            continue
        scoring = {}
        sp = sdir / "_scoring.json"
        if sp.exists():
            scoring = json.loads(sp.read_text(encoding="utf-8"))
        for p in sorted(sdir.glob("*.json")):
            if p.name.startswith("_"):
                continue
            case = json.loads(p.read_text(encoding="utf-8"))
            idx[case["id"]] = {"expect": case.get("expect", {}),
                               "scoring": scoring.get(case["id"], {})}
    return idx


def _all_match(patterns: list[str], text: str) -> bool:
    return all(re.search(p, text, re.IGNORECASE) for p in patterns) if patterns else False


def _any_match(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns) if patterns else False


def score_session(sess: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    expect = meta["expect"]
    scoring = meta["scoring"]
    text = sess.get("final_text") or ""
    should_ask = bool(expect.get("should_ask"))

    if sess["outcome"] != "ok":
        return {"scorable": False, "outcome": sess["outcome"]}
    # 생성 오류(finish_reason=error)로 최종 답이 잘린 세션은 채점 제외(인프라성 실패).
    turns = sess.get("turns") or []
    if turns and turns[-1].get("finish_reason") == "error":
        return {"scorable": False, "outcome": "generation_error"}

    # 정답성
    if should_ask:
        # 되묻기/판정보류가 정답. 정보 부족을 인정하고 확정 판정을 내리지 않아야 한다.
        correct = is_clarification(text)
    else:
        correct = _all_match(scoring.get("accept", []), text)

    # 함정(진단용)
    trap_fell = _any_match(scoring.get("reject", []), text)

    # 필수 도구 커버리지
    req = set(expect.get("required_tools", []))
    called = set(sess.get("tools_called", []))
    tool_cov = len(req & called) / len(req) if req else 1.0

    # 인자 검증(유의미한 케이스만)
    arg_ok = None
    for tool, want in (scoring.get("arg_check") or {}).items():
        calls = sess.get("tool_args_by_name", {}).get(tool, [])
        arg_ok = any(all(str(c.get(k)) == str(v) for k, v in want.items()) for c in calls)

    # 에러 복구(fault 케이스만)
    recovery = sess.get("recovered_after_fault") if sess.get("fault_triggered") else None

    return {
        "scorable": True, "correct": correct, "trap_fell": trap_fell,
        "tool_coverage": tool_cov, "arg_ok": arg_ok, "recovery": recovery,
        "steps": sess.get("steps"), "max_steps": sess.get("max_steps"),
        "latency_ms": sess.get("elapsed_ms"), "usd": usd(sess["arm"], sess.get("usage", {})),
        "usage": sess.get("usage", {}),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    args = ap.parse_args()
    run_dir = pathlib.Path(args.run_dir)

    idx = load_case_index()
    sessions = [json.loads(l) for l in (run_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    scored_fh = (run_dir / "scored.jsonl").open("w", encoding="utf-8")
    # (arm, scenario) -> list of session scores ; (arm, case) -> correct list
    by_as: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_ac: dict[tuple[str, str], list[bool]] = defaultdict(list)

    for sess in sessions:
        meta = idx.get(sess["case_id"])
        if not meta:
            continue
        sc = score_session(sess, meta)
        row = {"session_id": sess["session_id"], "arm": sess["arm"],
               "scenario": sess["scenario"], "case_id": sess["case_id"], **sc}
        scored_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        if sc["scorable"]:
            by_as[(sess["arm"], sess["scenario"])].append(sc)
            by_ac[(sess["arm"], sess["case_id"])].append(sc["correct"])
    scored_fh.close()

    # case_id -> scenario 매핑
    case_scen = {s["case_id"]: s["scenario"] for s in sessions}

    # 집계
    report: dict[str, Any] = {"by_arm_scenario": [], "by_arm_case": []}
    pass5_by_ac: dict[tuple[str, str], float | None] = {}
    for (arm, case_id), corrects in sorted(by_ac.items()):
        n, c = len(corrects), sum(corrects)
        p5 = pass_hat_k(n, c, 5)
        pass5_by_ac[(arm, case_id)] = p5
        report["by_arm_case"].append({
            "arm": arm, "case_id": case_id, "scenario": case_scen.get(case_id, ""),
            "n": n, "correct": c, "pass5": p5,
        })

    rows_md = ["| Model | Scenario | Accuracy [95% CI] | Steps | Median Latency (ms) | USD | mean pass^5 |",
               "|---|---|---|---:|---:|---:|---:|"]
    for (arm, scen), scs in sorted(by_as.items()):
        n = len(scs)
        c = sum(1 for s in scs if s["correct"])
        p, lo, hi = wilson(c, n)
        lat = statistics.median(s["latency_ms"] for s in scs) if scs else 0
        steps = statistics.mean(s["steps"] for s in scs) if scs else 0
        cost = statistics.mean(s["usd"] for s in scs) if scs else 0
        tool_cov = statistics.mean(s["tool_coverage"] for s in scs) if scs else 0
        # 시나리오 pass^5 = 이 arm·시나리오에 속한 케이스들의 pass5 평균
        p5s = [v for (a, cid), v in pass5_by_ac.items()
               if a == arm and case_scen.get(cid) == scen and v is not None]
        mean_p5 = statistics.mean(p5s) if p5s else float("nan")
        report["by_arm_scenario"].append({
            "arm": arm, "scenario": scen, "n": n, "correct": c,
            "accuracy": p, "ci_lo": lo, "ci_hi": hi,
            "median_latency_ms": lat, "mean_steps": steps, "mean_usd": cost,
            "mean_tool_coverage": tool_cov, "mean_pass5": mean_p5,
        })
        rows_md.append(f"| {arm} | {scen} | {p:.2f} [{lo:.2f},{hi:.2f}] | "
                       f"{steps:.1f} | {lat:.0f} | ${cost:.4f} | {mean_p5:.2f} |")

    (run_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "grade_results.json").write_text(
        json.dumps(report["by_arm_scenario"], ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "REPORT.md").write_text("\n".join(rows_md) + "\n", encoding="utf-8")

    print("\n".join(rows_md))
    print(f"\n→ {run_dir}/scored.jsonl, report.json, grade_results.json, REPORT.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
