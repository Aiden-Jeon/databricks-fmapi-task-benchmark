"""서버가 주입하는 스캐폴딩 토큰을 모델별로 잰다.

Databricks [FMAPI REST 레퍼런스](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/api-reference)는
이렇게 적는다:

    "prompt_tokens includes all text added by our server."

그리고 structured outputs 문서는 "Prompt injection and other techniques are used to
enhance the quality"라고만 하고 **얼마나** 주입하는지는 말하지 않는다.

그런데 그 주입분은 **청구된다.** 그리고 모델마다 다르면 비용 비교의 전제가 깨진다.
그래서 잰다: 사용자 메시지를 전 조건 **바이트 동일**로 두고 `tools` / `response_format`만
바꿔 `prompt_tokens` 증가분을 본다. 차이는 전부 서버 주입분이다.

사용:
    python probe_scaffolding_overhead.py
    python probe_scaffolding_overhead.py --route gateway   # /ai-gateway/mlflow/v1 경로로
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any

import httpx

PROFILE = os.environ.get("DATABRICKS_PROFILE", "DEFAULT")

SERVING_EP = {
    "opus": "databricks-claude-opus-5",
    "sol": "databricks-gpt-5-6-sol",
    "glm": "databricks-glm-5-2",
}
GATEWAY_MODEL = {
    "opus": "system.ai.claude-opus-5",
    "sol": "system.ai.gpt-5-6-sol",
    "glm": "system.ai.glm-5-2",
}
# 모델별 "reasoning 최소" — 실측 확정 (probe_capabilities)
MIN_REASONING: dict[str, dict[str, Any]] = {
    "opus": {"thinking": {"type": "disabled"}},
    "sol": {"reasoning_effort": "none"},
    "glm": {"reasoning_effort": "none"},
}

# 전 조건 공통. 바이트 단위로 동일해야 delta가 순수 주입분이 된다.
USER_MSG = "'김철수는 32세이고 서울에 산다.' 에서 이름/나이/도시를 알려줘."

BASE_TOOL = {
    "type": "function",
    "function": {
        "name": "save_person",
        "description": "사람 정보를 저장한다.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "city": {"type": "string"},
            },
            "required": ["name", "age", "city"],
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


def n_tools(n: int) -> list[dict[str, Any]]:
    """구조가 같은 툴 n개. 이름만 달리해 개수 효과만 분리한다."""
    out = []
    for i in range(n):
        t = json.loads(json.dumps(BASE_TOOL))
        t["function"]["name"] = f"save_person_{i}"
        out.append(t)
    return out


def auth() -> tuple[str, str]:
    def cli(*a: str) -> str:
        p = subprocess.run(
            ["databricks", *a, "--profile", PROFILE], capture_output=True, text=True
        )
        if p.returncode != 0:
            raise RuntimeError(p.stderr)
        return p.stdout

    host = json.loads(cli("auth", "env"))["env"]["DATABRICKS_HOST"].rstrip("/")
    return host, json.loads(cli("auth", "token"))["access_token"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", choices=["serving", "gateway"], default="serving")
    args = ap.parse_args()

    host, token = auth()
    client = httpx.Client(timeout=180.0)
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    def prompt_tokens(arm: str, extra: dict[str, Any]) -> int | None:
        body: dict[str, Any] = {
            "messages": [{"role": "user", "content": USER_MSG}],
            "max_tokens": 256,
            **MIN_REASONING[arm],
            **extra,
        }
        if args.route == "gateway":
            body["model"] = GATEWAY_MODEL[arm]
            url = f"{host}/ai-gateway/mlflow/v1/chat/completions"
        else:
            url = f"{host}/serving-endpoints/{SERVING_EP[arm]}/invocations"
        r = client.post(url, json=body, headers=headers)
        if r.status_code != 200:
            return None
        return r.json().get("usage", {}).get("prompt_tokens")

    conditions: list[tuple[str, dict[str, Any]]] = [
        ("baseline (도구·스키마 없음)", {}),
        ("+ tools ×1", {"tools": n_tools(1)}),
        ("+ tools ×4", {"tools": n_tools(4)}),
        ("+ tools ×16", {"tools": n_tools(16)}),
        (
            "+ response_format json_schema",
            {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "p", "schema": PERSON_SCHEMA, "strict": True},
                }
            },
        ),
    ]

    arms = ("opus", "sol", "glm")
    print(f"route={args.route}  host={host}")
    print("prompt_tokens — 사용자 메시지는 전 조건 동일. 차이는 전부 서버 주입분.\n")
    print(f"{'조건':34s} " + " ".join(f"{a:>14s}" for a in arms))

    baseline: dict[str, int] = {}
    for label, extra in conditions:
        cells = []
        for arm in arms:
            v = prompt_tokens(arm, extra)
            if v is None:
                cells.append("HTTP 400")
                continue
            if not baseline:
                pass
            if label.startswith("baseline"):
                baseline[arm] = v
                cells.append(str(v))
            else:
                cells.append(f"{v}  (+{v - baseline[arm]})")
        print(f"{label:34s} " + " ".join(f"{c:>14s}" for c in cells))

    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
