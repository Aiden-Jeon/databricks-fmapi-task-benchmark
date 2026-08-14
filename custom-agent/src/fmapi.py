"""FMAPI chat/completions 호출 코어 — 3개 모델 arm, 단일 요청, 응답 정규화.

function-calling-json/src/runner.py 에서 이식·축약했다. 이 벤치마크는 **chat 경로만** 쓴다.
최종 답이 자유 텍스트라 response_format(JSON 스키마 출력)을 요구하지 않으므로,
tools+response_format 비호환(GLM 도구 누락 / Opus HTTP 400)을 애초에 밟지 않는다.

실측으로 강제된 모델 설정 (METHODOLOGY §3):
- opus: `thinking:{disabled}` + tools 는 원시 XML 을 흘리고 tool_calls 를 비운다(8/12).
        **adaptive 필수** (0/135).
- sol : chat 경로에서 tools 를 쓰려면 `reasoning_effort:"none"` 이어야 한다(아니면 400).
- glm : QPH 7,200 = concurrency 2 가 전체 페이스를 정한다.
"""

from __future__ import annotations

import datetime as dt
import json
import random
import time
import uuid
from typing import Any

import httpx

from .auth import Auth, is_auth_expiry

GATEWAY_PATH = "/ai-gateway/mlflow/v1/chat/completions"

# 도구를 쓰는 3개 arm. 전부 chat 경로.
ARMS: dict[str, dict[str, Any]] = {
    "opus": {"model": "system.ai.claude-opus-5",
             "params": {"thinking": {"type": "adaptive"}},
             "concurrency": 6},
    "sol": {"model": "system.ai.gpt-5-6-sol",
            "params": {"reasoning_effort": "none"},
            "concurrency": 6},
    "glm": {"model": "system.ai.glm-5-2",
            "params": {"reasoning_effort": "none"},
            "concurrency": 2},
}

MAX_TOKENS = 2048
REQUEST_TIMEOUT = 120.0
MAX_INFRA_RETRIES = 4
BACKOFF_INITIAL = 2.0


def normalize_usage(u: dict[str, Any]) -> dict[str, int]:
    """모델별 usage 형태를 하나로. reasoning_tokens 는 completion 에 포함되므로 더하지 않는다."""
    ptd = u.get("prompt_tokens_details") or {}
    ctd = u.get("completion_tokens_details") or {}
    cache_read = (
        u.get("cache_read_input_tokens")
        if u.get("cache_read_input_tokens") is not None
        else ptd.get("cached_tokens", 0)
    ) or 0
    cache_write = (
        u.get("cache_creation_input_tokens")
        if u.get("cache_creation_input_tokens") is not None
        else ptd.get("cache_write_tokens", 0)
    ) or 0
    return {
        "prompt_tokens": u.get("prompt_tokens") or 0,
        "completion_tokens": u.get("completion_tokens") or 0,
        "total_tokens": u.get("total_tokens") or 0,
        "reasoning_tokens": ctd.get("reasoning_tokens", 0) or 0,
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
    }


def extract_text(msg: dict[str, Any]) -> str:
    """GLM 은 content 가 비고 reasoning_content 에 답이 오는 경우가 있어 fallback 한다."""
    c = msg.get("content")
    if isinstance(c, list):
        t = "".join(p.get("text", "") for p in c
                    if isinstance(p, dict) and p.get("type") == "text")
        if t.strip():
            return t.strip()
        return (msg.get("reasoning_content") or "").strip()
    if isinstance(c, str) and c.strip():
        return c.strip()
    return (msg.get("reasoning_content") or "").strip()


def extract_tool_calls(msg: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        raw = fn.get("arguments")
        try:
            args = json.loads(raw) if isinstance(raw, str) else (raw or {})
            parse_ok = True
        except Exception:
            args, parse_ok = {}, False
        out.append({"id": tc.get("id"), "name": fn.get("name"), "args": args,
                    "args_raw": raw, "args_parse_ok": parse_ok})
    return out


def post_once(
    client: httpx.Client, auth: Auth, arm: str,
    messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
    tags: dict[str, str],
) -> dict[str, Any]:
    """chat/completions 단일 요청 + 인증 갱신 + 인프라 재시도.

    반환: {outcome, status, latency_ms, retries, [text, tool_calls, usage, finish_reason]}
    outcome ∈ {ok, gateway_reject, malformed_response, infra_fail}
    """
    spec = ARMS[arm]
    body: dict[str, Any] = {"model": spec["model"], "messages": messages,
                            "max_tokens": MAX_TOKENS, **spec["params"]}
    if tools:
        body["tools"] = tools
    url = auth.host + GATEWAY_PATH

    rec: dict[str, Any] = {}
    attempt, budget, last_err = 0, MAX_INFRA_RETRIES, None
    while attempt < budget:
        headers = {
            "Authorization": f"Bearer {auth.token}",
            "Content-Type": "application/json",
            "Databricks-Ai-Gateway-Request-Tags": json.dumps(tags),
        }
        t0 = time.perf_counter()
        try:
            r = client.post(url, json=body, headers=headers, timeout=REQUEST_TIMEOUT)
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(BACKOFF_INITIAL * (2 ** attempt) + random.uniform(0, 1))
            attempt += 1
            continue
        dt_ms = (time.perf_counter() - t0) * 1000

        if is_auth_expiry(r.status_code, r.text) and budget == MAX_INFRA_RETRIES:
            auth.refresh()
            budget += 1
            continue

        if r.status_code >= 500 or r.status_code == 429:
            last_err = f"HTTP {r.status_code}: {r.text[:200]}"
            wait = BACKOFF_INITIAL * (2 ** attempt) + random.uniform(0, 1)
            try:
                ra = (r.json().get("retry_after")
                      or r.json().get("error", {}).get("retry_after"))
                if ra:
                    wait = float(ra)
            except Exception:
                pass
            time.sleep(wait)
            attempt += 1
            continue

        if r.status_code != 200:
            return {"outcome": "gateway_reject", "status": r.status_code,
                    "latency_ms": dt_ms, "error": r.text[:500], "retries": attempt}

        data = r.json()
        try:
            choice = data["choices"][0]
        except (KeyError, IndexError):
            return {"outcome": "malformed_response", "status": 200, "latency_ms": dt_ms,
                    "error": json.dumps(data, ensure_ascii=False)[:500], "retries": attempt}
        msg = choice.get("message", {})
        return {
            "outcome": "ok", "status": 200, "latency_ms": dt_ms, "retries": attempt,
            "finish_reason": choice.get("finish_reason"),
            "text": extract_text(msg),
            "tool_calls": extract_tool_calls(msg),
            "usage": normalize_usage(data.get("usage") or {}),
            "request_id": r.headers.get("x-request-id"),
        }

    return {"outcome": "infra_fail", "status": None, "error": last_err, "retries": attempt}
