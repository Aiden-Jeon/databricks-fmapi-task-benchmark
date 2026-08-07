"""FMAPI capability probe — function calling & structured output.

세 모델(opus / sol / glm)이 Databricks FMAPI 위에서 실제로 무엇을 지원하는지
**직접 호출로** 확인한다. 벤치마크 설계는 이 결과 위에서만 성립한다:
지원하지 않는 모드로 측정하면 그건 모델 성능이 아니라 게이트웨이 400이다.

각 프로브는 독립적으로 실패해도 되고, 실패 사유(HTTP status + body 앞부분)를
그대로 기록한다. 판정은 사람이 결과 JSON을 보고 한다.

사용:
    python probe_capabilities.py --profile <your-profile>
    python probe_capabilities.py --profile <your-profile> --models opus
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

MODELS = {
    "opus": "databricks-claude-opus-5",
    "sol": "databricks-gpt-5-6-sol",
    "glm": "databricks-glm-5-2",
}

# 프로브에 공통으로 쓰는 도구 하나. 한국어 설명 + 한국어 enum을 일부러 섞어
# 한국어 스키마 처리까지 같은 호출로 관찰한다.
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "특정 도시의 현재 날씨를 조회한다.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "도시 이름 (예: 서울, 부산)"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_news",
        "description": "키워드로 뉴스를 검색한다.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

PERSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "city": {"type": "string"},
    },
    "required": ["name", "age", "city"],
    "additionalProperties": False,
}


def get_auth(profile: str) -> tuple[str, str]:
    def cli(*args: str) -> str:
        proc = subprocess.run(
            ["databricks", *args, "--profile", profile],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"databricks {' '.join(args)} 실패: {proc.stderr.strip()}")
        return proc.stdout

    host = json.loads(cli("auth", "env"))["env"]["DATABRICKS_HOST"].rstrip("/")
    token = json.loads(cli("auth", "token"))["access_token"]
    return host, token


@dataclass
class ProbeResult:
    probe: str
    model: str
    endpoint: str
    ok: bool
    status: int | None = None
    latency_ms: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class Prober:
    def __init__(self, host: str, token: str, timeout: float = 180.0) -> None:
        self.host = host
        self.token = token
        self.client = httpx.Client(timeout=timeout)

    def call(self, endpoint: str, payload: dict[str, Any]) -> tuple[int, Any, float]:
        url = f"{self.host}/serving-endpoints/{endpoint}/invocations"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        t0 = time.perf_counter()
        resp = self.client.post(url, json=payload, headers=headers)
        dt = (time.perf_counter() - t0) * 1000
        try:
            body = resp.json()
        except Exception:
            body = resp.text[:500]
        return resp.status_code, body, dt

    def close(self) -> None:
        self.client.close()


def _msg(text: str) -> list[dict[str, Any]]:
    return [{"role": "user", "content": text}]


def _tool_calls(body: Any) -> list[dict[str, Any]]:
    """응답에서 tool_calls를 꺼낸다. 없으면 빈 리스트."""
    try:
        return body["choices"][0]["message"].get("tool_calls") or []
    except Exception:
        return []


def _text(body: Any) -> str:
    try:
        content = body["choices"][0]["message"].get("content")
    except Exception:
        return ""
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


# ── 개별 프로브 ────────────────────────────────────────────────────────────────
# 각 함수는 (payload, judge) 를 돌려준다. judge(status, body) -> (ok, detail)


def probe_basic_tool_call(_model: str) -> tuple[dict, Any]:
    payload = {
        "messages": _msg("서울 날씨 알려줘."),
        "tools": [WEATHER_TOOL],
        "max_tokens": 1024,
    }

    def judge(status: int, body: Any) -> tuple[bool, dict]:
        calls = _tool_calls(body)
        if not calls:
            return False, {"reason": "tool_calls 없음", "text": _text(body)[:200]}
        fn = calls[0].get("function", {})
        args_raw = fn.get("arguments")
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        except Exception:
            return False, {"reason": "arguments JSON 파싱 실패", "raw": str(args_raw)[:200]}
        return fn.get("name") == "get_weather", {
            "name": fn.get("name"),
            "args": args,
            "n_calls": len(calls),
        }

    return payload, judge


def probe_parallel_tool_calls(_model: str) -> tuple[dict, Any]:
    payload = {
        "messages": _msg("서울과 부산 날씨를 각각 알려줘. 두 도시 모두 조회해."),
        "tools": [WEATHER_TOOL],
        "max_tokens": 1024,
    }

    def judge(status: int, body: Any) -> tuple[bool, dict]:
        calls = _tool_calls(body)
        cities = []
        for c in calls:
            try:
                a = c["function"]["arguments"]
                cities.append(json.loads(a).get("city") if isinstance(a, str) else a.get("city"))
            except Exception:
                pass
        return len(calls) >= 2, {"n_calls": len(calls), "cities": cities}

    return payload, judge


def probe_irrelevance(_model: str) -> tuple[dict, Any]:
    """도구가 무관할 때 호출을 참는가 (abstention)."""
    payload = {
        "messages": _msg("피보나치 수열의 10번째 항이 뭐야?"),
        "tools": [WEATHER_TOOL, SEARCH_TOOL],
        "max_tokens": 1024,
    }

    def judge(status: int, body: Any) -> tuple[bool, dict]:
        calls = _tool_calls(body)
        return len(calls) == 0, {"n_calls": len(calls), "text": _text(body)[:200]}

    return payload, judge


def probe_tool_choice_required(_model: str) -> tuple[dict, Any]:
    payload = {
        "messages": _msg("서울 날씨"),
        "tools": [WEATHER_TOOL],
        "tool_choice": "required",
        "max_tokens": 1024,
    }

    def judge(status: int, body: Any) -> tuple[bool, dict]:
        return len(_tool_calls(body)) > 0, {"n_calls": len(_tool_calls(body))}

    return payload, judge


def probe_tool_choice_named(_model: str) -> tuple[dict, Any]:
    payload = {
        "messages": _msg("아무거나 해줘"),
        "tools": [WEATHER_TOOL, SEARCH_TOOL],
        "tool_choice": {"type": "function", "function": {"name": "search_news"}},
        "max_tokens": 1024,
    }

    def judge(status: int, body: Any) -> tuple[bool, dict]:
        calls = _tool_calls(body)
        if not calls:
            return False, {"reason": "tool_calls 없음"}
        return calls[0].get("function", {}).get("name") == "search_news", {
            "name": calls[0].get("function", {}).get("name")
        }

    return payload, judge


def probe_json_object_mode(_model: str) -> tuple[dict, Any]:
    payload = {
        "messages": _msg(
            "다음 문장에서 이름/나이/도시를 뽑아 JSON으로만 답해. "
            "'김철수는 32세이고 서울에 산다.' 키는 name, age, city."
        ),
        "response_format": {"type": "json_object"},
        "max_tokens": 1024,
    }

    def judge(status: int, body: Any) -> tuple[bool, dict]:
        txt = _text(body)
        try:
            parsed = json.loads(txt)
        except Exception:
            return False, {"reason": "JSON 파싱 실패", "text": txt[:200]}
        return True, {"parsed": parsed}

    return payload, judge


def probe_json_schema_strict(_model: str) -> tuple[dict, Any]:
    payload = {
        "messages": _msg("'김철수는 32세이고 서울에 산다.' 에서 정보를 뽑아줘."),
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "person", "schema": PERSON_SCHEMA, "strict": True},
        },
        "max_tokens": 1024,
    }

    def judge(status: int, body: Any) -> tuple[bool, dict]:
        txt = _text(body)
        try:
            parsed = json.loads(txt)
        except Exception:
            return False, {"reason": "JSON 파싱 실패", "text": txt[:200]}
        missing = [k for k in PERSON_SCHEMA["required"] if k not in parsed]
        return not missing, {"parsed": parsed, "missing": missing}

    return payload, judge


def probe_multi_turn_tool_result(_model: str) -> tuple[dict, Any]:
    """tool 역할 메시지를 되먹였을 때 정상적으로 이어받는가."""
    payload = {
        "messages": [
            {"role": "user", "content": "서울 날씨 알려줘."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "서울"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"temp_c": 31, "condition": "맑음"}',
            },
        ],
        "tools": [WEATHER_TOOL],
        "max_tokens": 1024,
    }

    def judge(status: int, body: Any) -> tuple[bool, dict]:
        txt = _text(body)
        return ("31" in txt or "맑" in txt), {"text": txt[:300]}

    return payload, judge


def probe_reasoning_effort_none(_model: str) -> tuple[dict, Any]:
    payload = {"messages": _msg("1+1은?"), "reasoning_effort": "none", "max_tokens": 256}

    def judge(status: int, body: Any) -> tuple[bool, dict]:
        return status == 200, {"text": _text(body)[:100]}

    return payload, judge


def probe_reasoning_effort_low(_model: str) -> tuple[dict, Any]:
    payload = {"messages": _msg("1+1은?"), "reasoning_effort": "low", "max_tokens": 256}

    def judge(status: int, body: Any) -> tuple[bool, dict]:
        return status == 200, {"text": _text(body)[:100]}

    return payload, judge


def probe_thinking_disabled(_model: str) -> tuple[dict, Any]:
    payload = {
        "messages": _msg("1+1은?"),
        "thinking": {"type": "disabled"},
        "max_tokens": 256,
    }

    def judge(status: int, body: Any) -> tuple[bool, dict]:
        return status == 200, {"text": _text(body)[:100]}

    return payload, judge


def probe_temperature_zero(_model: str) -> tuple[dict, Any]:
    payload = {"messages": _msg("1+1은?"), "temperature": 0, "max_tokens": 256}

    def judge(status: int, body: Any) -> tuple[bool, dict]:
        return status == 200, {"text": _text(body)[:100]}

    return payload, judge


def probe_seed(_model: str) -> tuple[dict, Any]:
    payload = {"messages": _msg("1+1은?"), "seed": 42, "max_tokens": 256}

    def judge(status: int, body: Any) -> tuple[bool, dict]:
        return status == 200, {"text": _text(body)[:100]}

    return payload, judge


def probe_hangul_escaping(_model: str) -> tuple[dict, Any]:
    """JSON 안의 한글을 raw로 내는가, \\uXXXX로 이스케이프하는가."""
    payload = {
        "messages": _msg(
            '{"city": "서울", "district": "강남구"} 를 그대로 다시 출력해. JSON만.'
        ),
        "response_format": {"type": "json_object"},
        "max_tokens": 512,
    }

    def judge(status: int, body: Any) -> tuple[bool, dict]:
        txt = _text(body)
        return "서울" in txt or "\\uc11c" in txt.lower(), {
            "raw_hangul": "서울" in txt,
            "escaped": "\\u" in txt,
            "text": txt[:200],
        }

    return payload, judge


PROBES = {
    "basic_tool_call": probe_basic_tool_call,
    "parallel_tool_calls": probe_parallel_tool_calls,
    "irrelevance_abstention": probe_irrelevance,
    "tool_choice_required": probe_tool_choice_required,
    "tool_choice_named": probe_tool_choice_named,
    "json_object_mode": probe_json_object_mode,
    "json_schema_strict": probe_json_schema_strict,
    "multi_turn_tool_result": probe_multi_turn_tool_result,
    "reasoning_effort_none": probe_reasoning_effort_none,
    "reasoning_effort_low": probe_reasoning_effort_low,
    "thinking_disabled": probe_thinking_disabled,
    "temperature_zero": probe_temperature_zero,
    "seed": probe_seed,
    "hangul_escaping": probe_hangul_escaping,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--probes", nargs="*", default=list(PROBES))
    ap.add_argument("--out", default="capability_matrix.json")
    args = ap.parse_args()

    host, token = get_auth(args.profile)
    print(f"host={host}")
    prober = Prober(host, token)
    results: list[ProbeResult] = []

    for model in args.models:
        endpoint = MODELS[model]
        for probe_name in args.probes:
            builder = PROBES[probe_name]
            payload, judge = builder(model)
            try:
                status, body, dt = prober.call(endpoint, payload)
            except Exception as e:
                results.append(
                    ProbeResult(probe_name, model, endpoint, False, error=f"{type(e).__name__}: {e}")
                )
                print(f"  {model:5s} {probe_name:24s} EXC  {e}")
                continue

            if status != 200:
                snippet = json.dumps(body)[:300] if not isinstance(body, str) else body[:300]
                results.append(
                    ProbeResult(probe_name, model, endpoint, False, status, dt, error=snippet)
                )
                print(f"  {model:5s} {probe_name:24s} HTTP {status}  {snippet[:120]}")
                continue

            ok, detail = judge(status, body)
            usage = body.get("usage", {}) if isinstance(body, dict) else {}
            detail["usage"] = usage
            results.append(ProbeResult(probe_name, model, endpoint, ok, status, dt, detail))
            print(f"  {model:5s} {probe_name:24s} {'PASS' if ok else 'FAIL'}  {dt:7.0f}ms  {json.dumps(detail, ensure_ascii=False)[:140]}")

    prober.close()

    out = Path(args.out)
    out.write_text(
        json.dumps(
            [
                {
                    "probe": r.probe,
                    "model": r.model,
                    "endpoint": r.endpoint,
                    "ok": r.ok,
                    "status": r.status,
                    "latency_ms": r.latency_ms,
                    "detail": r.detail,
                    "error": r.error,
                }
                for r in results
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
