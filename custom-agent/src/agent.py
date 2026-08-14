"""멀티턴 도구 사용 에이전트 루프.

단일 루프: tools 를 주고, 모델이 tool_calls 를 내면 Mock 실행 결과를 대화에 붙여 반복,
텍스트만 내면 그것을 최종 답으로 종료. response_format 은 쓰지 않는다.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from .auth import Auth
from .fmapi import post_once
from .tools import ToolExecutor


def _assistant_toolcall_msg(tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": tc.get("id") or f"call_{i}", "type": "function",
             "function": {"name": tc["name"],
                          "arguments": tc.get("args_raw")
                          if isinstance(tc.get("args_raw"), str)
                          else json.dumps(tc.get("args") or {}, ensure_ascii=False)}}
            for i, tc in enumerate(tool_calls)
        ],
    }


def run_session(
    client: httpx.Client, auth: Auth, arm: str, case: dict[str, Any],
    tools: list[dict[str, Any]], system: str, clauses: dict[str, Any] | None,
    run_id: str, rep: int,
) -> dict[str, Any]:
    max_steps = int(case.get("expect", {}).get("max_steps", 10))
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": case["prompt"]},
    ]
    ex = ToolExecutor(case, clauses)

    turns: list[dict[str, Any]] = []
    tools_called: list[str] = []
    tool_args_by_name: dict[str, list[dict[str, Any]]] = {}
    total = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0,
             "cache_read_tokens": 0, "cache_write_tokens": 0}
    final_text = ""
    outcome = "ok"
    hit_max = False
    t_start = time.perf_counter()

    for step in range(max_steps):
        tags = {"benchmark_run": run_id, "arm": arm, "case_id": case["id"],
                "scenario": case.get("scenario", ""), "rep": str(rep), "step": str(step)}
        rec = post_once(client, auth, arm, messages, tools, tags)

        u = rec.get("usage") or {}
        for k in total:
            total[k] += u.get(k, 0)
        turns.append({
            "step": step, "outcome": rec["outcome"], "status": rec.get("status"),
            "latency_ms": rec.get("latency_ms"), "retries": rec.get("retries"),
            "finish_reason": rec.get("finish_reason"),
            "cumulative_prompt_tokens": total["prompt_tokens"],
            "tool_calls": [{"name": tc["name"], "args": tc["args"]}
                           for tc in rec.get("tool_calls", [])],
            "text": (rec.get("text") or "")[:2000],
            "error": rec.get("error"),
        })

        if rec["outcome"] != "ok":
            outcome = rec["outcome"]
            break

        calls = rec.get("tool_calls") or []
        if calls:
            messages.append(_assistant_toolcall_msg(calls))
            for i, tc in enumerate(calls):
                tools_called.append(tc["name"])
                tool_args_by_name.setdefault(tc["name"], []).append(tc["args"])
                result = ex.execute(tc["name"], tc["args"])
                messages.append({"role": "tool",
                                 "tool_call_id": tc.get("id") or f"call_{i}",
                                 "content": result})
        else:
            final_text = rec.get("text") or ""
            break
    else:
        hit_max = True
        final_text = turns[-1]["text"] if turns else ""

    return {
        "session_id": f"{run_id}:{arm}:{case['id']}:r{rep}",
        "run_id": run_id, "arm": arm, "case_id": case["id"],
        "scenario": case.get("scenario", ""), "rep": rep,
        "outcome": outcome, "hit_max_steps": hit_max,
        "steps": len(turns), "max_steps": max_steps,
        "final_text": final_text,
        "tools_called": tools_called,
        "tool_args_by_name": tool_args_by_name,
        "fault_triggered": ex.fault_triggered,
        "recovered_after_fault": ex.recovered_after_fault,
        "usage": total,
        "elapsed_ms": (time.perf_counter() - t_start) * 1000,
        "turns": turns,
    }
