"""Mock 도구 실행기 — fixture 기반 + fault 주입 + (보험) 공유 조항 카탈로그.

케이스 JSON 의 fixtures 키 규칙: "toolname:arg1[:arg2]" 또는 "toolname".
보험의 search_clause/get_clause 는 케이스 fixture 가 아니라 시나리오 공유
_clauses.json (실제 상품 규칙)을 서빙한다.
"""

from __future__ import annotations

import json
from itertools import permutations
from typing import Any


class ToolExecutor:
    def __init__(self, case: dict[str, Any], clauses: dict[str, Any] | None = None) -> None:
        self.fixtures: dict[str, Any] = case.get("fixtures") or {}
        self.fault = case.get("fault")  # {"tool","on_call","error"} 또는 None
        self.clauses = clauses or {}
        self.call_counts: dict[str, int] = {}
        self.fault_triggered = False
        self.recovered_after_fault = False

    def _candidates(self, name: str, args: dict[str, Any]) -> list[str]:
        vals = [str(v) for v in args.values()]
        cands: list[str] = []
        # 인자 순서를 모르므로 순열을 시도한다(인자 ≤ 2라 비용 무시 가능).
        for perm in permutations(vals):
            cands.append(name + ":" + ":".join(perm))
        for v in vals:
            cands.append(f"{name}:{v}")
        cands.append(name)
        # 중복 제거(순서 유지)
        seen, out = set(), []
        for c in cands:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out

    def _serve_clause(self, name: str, args: dict[str, Any]) -> str | None:
        if not self.clauses:
            return None
        if name == "search_clause":
            kw = str(args.get("keyword", "")).strip()
            hits = []
            for cid, c in self.clauses.items():
                hay = cid + " " + c.get("title", "") + " " + " ".join(c.get("keywords", []))
                if kw and kw in hay:
                    hits.append({"clause_id": cid, "title": c.get("title", "")})
            return json.dumps(hits or [{"note": f"'{kw}' 에 해당하는 조항 없음"}],
                              ensure_ascii=False)
        if name == "get_clause":
            cid = str(args.get("clause_id", "")).strip()
            c = self.clauses.get(cid)
            if c:
                return json.dumps({"clause_id": cid, "title": c.get("title", ""),
                                   "text": c["text"]}, ensure_ascii=False)
            return json.dumps({"error": f"조항 '{cid}' 없음", "available": list(self.clauses)},
                              ensure_ascii=False)
        return None

    def execute(self, name: str, args: dict[str, Any]) -> str:
        self.call_counts[name] = self.call_counts.get(name, 0) + 1
        n = self.call_counts[name]

        # fault 주입: 지정 도구의 on_call 번째 호출을 실패시킨다.
        if self.fault and self.fault.get("tool") == name and self.fault.get("on_call") == n:
            self.fault_triggered = True
            return json.dumps({"error": self.fault["error"]}, ensure_ascii=False)
        # fault 이후 같은 도구를 다시 부르면 복구로 기록한다.
        if self.fault and self.fault.get("tool") == name and self.fault_triggered and n > self.fault["on_call"]:
            self.recovered_after_fault = True

        # 보험 조항 카탈로그
        served = self._serve_clause(name, args)
        if served is not None:
            return served

        # fixture 조회
        for key in self._candidates(name, args):
            if key in self.fixtures:
                v = self.fixtures[key]
                return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)

        return json.dumps(
            {"error": f"'{name}' 에 대한 데이터 없음 (args={args})"}, ensure_ascii=False)
