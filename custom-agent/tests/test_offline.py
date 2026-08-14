"""네트워크 없이 하네스 로직 검증 — 도구 fixture 해석, fault 복구, 채점 분기.

실행: uv run python -m tests.test_offline
"""

from __future__ import annotations

import json
import pathlib

from src.score import load_case_index, score_session
from src.tools import ToolExecutor

ROOT = pathlib.Path(__file__).resolve().parent.parent
CASES = ROOT / "cases"


def load_case(scenario: str, cid: str) -> dict:
    return json.loads((CASES / scenario / f"{cid}.json").read_text(encoding="utf-8"))


def load_clauses(scenario: str) -> dict:
    p = CASES / scenario / "_clauses.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def check(name: str, cond: bool) -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def test_tools_azure() -> None:
    print("test_tools_azure")
    ex = ToolExecutor(load_case("azure_troubleshoot", "A-1"))
    r = json.loads(ex.execute("get_azure_resource", {"type": "role_assignments", "name": "sp-7f3a"}))
    check("A-1 role_assignments 조회", "Owner" in json.dumps(r, ensure_ascii=False))
    r2 = ex.execute("get_logs", {"cluster_id": "clu-a1"})
    check("A-1 로그에 AuthorizationPermissionMismatch", "AuthorizationPermissionMismatch" in r2)


def test_tools_insurance_clauses() -> None:
    print("test_tools_insurance_clauses")
    ex = ToolExecutor(load_case("insurance_policy", "B-2"), load_clauses("insurance_policy"))
    hits = json.loads(ex.execute("search_clause", {"keyword": "소액암"}))
    check("search_clause 소액암 히트", any("소액암" in h.get("clause_id", "") for h in hits))
    body = json.loads(ex.execute("get_clause", {"clause_id": "통합소액암진단-지급액"}))
    check("get_clause 200만 포함", "200만" in body["text"])
    dz = json.loads(ex.execute("lookup_disease_code", {"code": "D05"}))
    check("D05 통합소액암 분류", dz["policy_class"] == "통합소액암")


def test_fault_recovery() -> None:
    print("test_fault_recovery")
    ex = ToolExecutor(load_case("azure_troubleshoot", "A-4"))
    r1 = json.loads(ex.execute("get_job_run", {"run_id": "run-8842"}))  # on_call=1 → fault
    check("A-4 첫 호출 503 fault", "error" in r1 and ex.fault_triggered)
    r2 = json.loads(ex.execute("get_job_run", {"run_id": "run-8842"}))  # 재시도 → 실제
    check("A-4 재시도 성공", "Cluster validation error" in json.dumps(r2, ensure_ascii=False))
    check("A-4 recovered_after_fault", ex.recovered_after_fault)


def test_scoring() -> None:
    print("test_scoring")
    idx = load_case_index()

    # A-1 정답 세션
    good = {"outcome": "ok", "arm": "opus", "final_text": "원인은 data-plane RBAC 누락입니다. Storage Blob Data Contributor 를 부여하세요.",
            "tools_called": ["get_logs", "get_azure_resource"],
            "tool_args_by_name": {"get_azure_resource": [{"type": "role_assignments", "name": "sp-7f3a"}]},
            "steps": 3, "max_steps": 8, "elapsed_ms": 1000, "usage": {"prompt_tokens": 1000, "completion_tokens": 200}}
    s = score_session(good, idx["A-1"])
    check("A-1 정답 correct", s["correct"] is True)
    check("A-1 tool_coverage=1", s["tool_coverage"] == 1.0)
    check("A-1 arg_ok", s["arg_ok"] is True)
    check("A-1 usd>0", s["usd"] > 0)

    # A-1 함정 세션 (SAS 토큰으로 우회 결론)
    trap = dict(good, final_text="SAS 토큰으로 교체하면 됩니다.")
    st = score_session(trap, idx["A-1"])
    check("A-1 함정 correct=False", st["correct"] is False)
    check("A-1 함정 trap_fell=True", st["trap_fell"] is True)

    # B-2 정답 (100만원)
    b2 = {"outcome": "ok", "arm": "glm", "final_text": "제자리암은 통합소액암 200만원, 계약 1년 이내라 50% 감액하여 지급액 100만원입니다.",
          "tools_called": ["get_contract", "get_claim", "lookup_disease_code", "get_clause"],
          "tool_args_by_name": {}, "steps": 5, "max_steps": 12, "elapsed_ms": 2000,
          "usage": {"prompt_tokens": 2000, "completion_tokens": 300}}
    sb = score_session(b2, idx["B-2"])
    check("B-2 정답(100만) correct", sb["correct"] is True)

    # B-2 오답 (600만원 - 감액 누락)
    b2w = dict(b2, final_text="유사암 지급액은 600만원입니다.")
    sbw = score_session(b2w, idx["B-2"])
    check("B-2 오답(600만) correct=False", sbw["correct"] is False)

    # B-4 되묻기 정답
    b4 = {"outcome": "ok", "arm": "sol", "final_text": "진단확정일이 확인되지 않습니다. 진단일을 알려주시겠어요?",
          "tools_called": ["get_contract", "get_claim"], "tool_args_by_name": {},
          "steps": 2, "max_steps": 10, "elapsed_ms": 800, "usage": {"prompt_tokens": 500, "completion_tokens": 50}}
    s4 = score_session(b4, idx["B-4"])
    check("B-4 되묻기 correct", s4["correct"] is True)

    # B-4 추측 오답
    b4w = dict(b4, final_text="계약일 기준으로 지급 대상이며 1000만원 지급합니다.")
    s4w = score_session(b4w, idx["B-4"])
    check("B-4 추측 correct=False", s4w["correct"] is False)


if __name__ == "__main__":
    test_tools_azure()
    test_tools_insurance_clauses()
    test_fault_recovery()
    test_scoring()
    print("\n모든 오프라인 검증 통과")
