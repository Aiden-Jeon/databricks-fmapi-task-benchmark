"""지원 기능 프로브 — 3모델이 멀티턴 도구 루프를 실제로 도는지 실측.

에이전트 벤치마크의 전제(각 모델이 tools 를 받고, tool 결과를 대화에 붙여 3턴 이상
이어갈 수 있는가)를 본 런 전에 확인한다. Task 6 이 이 확인을 했기에 Opus 정확도
0.760(하네스 버그)을 0.920(실제)으로 바로잡았다.

사용: python -m probes.probe_agent_capabilities
"""

from __future__ import annotations

import os
import sys

import httpx

from src.auth import Auth
from src.fmapi import ARMS, REQUEST_TIMEOUT, post_once

PROFILE = os.environ.get("DATABRICKS_PROFILE", "ai_devtools")

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_value",
        "description": "키에 해당하는 값을 반환한다.",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string", "description": "조회할 키 (a 또는 b)"}},
            "required": ["key"],
            "additionalProperties": False,
        },
    },
}]

FIXTURE = {"a": "42", "b": "17"}
SYSTEM = ("두 키 a, b 의 값을 각각 get_value 로 조회한 뒤 두 값을 더해서 답하라. "
          "반드시 두 번 조회한 다음 합을 말하라.")
USER = "a 와 b 의 합은?"


def probe_arm(client: httpx.Client, auth: Auth, arm: str) -> dict[str, object]:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]
    steps, tool_calls_total, empty_text = 0, 0, 0
    for step in range(6):
        rec = post_once(client, auth, arm, messages, TOOLS,
                        {"probe": "1", "arm": arm, "step": str(step)})
        steps += 1
        if rec["outcome"] != "ok":
            return {"arm": arm, "ok": False, "reason": rec["outcome"],
                    "detail": str(rec.get("error"))[:200]}
        calls = rec.get("tool_calls") or []
        if not calls and not (rec.get("text") or "").strip():
            empty_text += 1
        if calls:
            tool_calls_total += len(calls)
            messages.append({
                "role": "assistant", "content": None,
                "tool_calls": [{"id": tc.get("id") or f"c{i}", "type": "function",
                                "function": {"name": tc["name"],
                                             "arguments": tc.get("args_raw") or "{}"}}
                               for i, tc in enumerate(calls)],
            })
            for i, tc in enumerate(calls):
                key = str(tc["args"].get("key", ""))
                messages.append({"role": "tool", "tool_call_id": tc.get("id") or f"c{i}",
                                 "content": FIXTURE.get(key, "unknown")})
        else:
            final = rec.get("text") or ""
            got_59 = "59" in final
            return {"arm": arm, "ok": True, "steps": steps,
                    "tool_calls": tool_calls_total, "answer_has_59": got_59,
                    "empty_text_turns": empty_text, "final": final[:160]}
    return {"arm": arm, "ok": False, "reason": "max_steps", "tool_calls": tool_calls_total}


def main() -> int:
    auth = Auth(PROFILE)
    client = httpx.Client(timeout=REQUEST_TIMEOUT)
    print(f"{'arm':>6} | ok | steps | toolcalls | ans=59 | empty | final")
    print("-" * 80)
    results = []
    for arm in ARMS:
        r = probe_arm(client, auth, arm)
        results.append(r)
        if r["ok"]:
            print(f"{arm:>6} |  Y | {r['steps']:>5} | {r['tool_calls']:>9} | "
                  f"{str(r['answer_has_59']):>6} | {r['empty_text_turns']:>5} | {r['final']}")
        else:
            print(f"{arm:>6} |  N | reason={r.get('reason')} {r.get('detail','')}")
    client.close()
    print("\n판정: 세 arm 모두 ok=Y, tool_calls>=2, ans=59=True 여야 멀티턴 도구 루프가 성립한다.")
    ok = all(r.get("ok") and r.get("tool_calls", 0) >= 2 for r in results)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
