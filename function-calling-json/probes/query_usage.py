"""system.ai_gateway.usage 조회 헬퍼 — 비용·레이턴시 측정 검증용.

Databricks SQL Statement Execution API로 쿼리를 돌린다.
SQL warehouse가 필요하고, 워크스페이스에서 실제로 무엇이 조회 가능한지를
문서가 아니라 **실행 결과로** 확인하는 것이 목적이다.

사용:
    python query_usage.py --check          # 권한·스키마·신선도 점검
    python query_usage.py --run <run_id>   # 태그로 벤치마크 run 조회
    python query_usage.py --sql "SELECT 1"
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any

import httpx

PROFILE = os.environ.get("DATABRICKS_PROFILE", "DEFAULT")
WAREHOUSE_ID = os.environ.get("DATABRICKS_WAREHOUSE_ID", "")  # SQL warehouse id


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


HOST, TOKEN = auth()
CLIENT = httpx.Client(timeout=300.0)
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}


def run_sql(sql: str, warehouse_id: str = WAREHOUSE_ID) -> dict[str, Any]:
    """쿼리를 실행하고 결과 dict를 돌려준다. 실패해도 예외를 던지지 않고 error를 담는다."""
    r = CLIENT.post(
        f"{HOST}/api/2.0/sql/statements",
        json={
            "statement": sql,
            "warehouse_id": warehouse_id,
            "wait_timeout": "50s",
            "on_wait_timeout": "CONTINUE",
        },
        headers=HEADERS,
    )
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}: {r.text[:400]}"}
    data = r.json()

    # 50초 안에 안 끝나면 폴링
    sid = data.get("statement_id")
    while data.get("status", {}).get("state") in ("PENDING", "RUNNING"):
        time.sleep(3)
        g = CLIENT.get(f"{HOST}/api/2.0/sql/statements/{sid}", headers=HEADERS)
        data = g.json()

    state = data.get("status", {}).get("state")
    if state != "SUCCEEDED":
        err = data.get("status", {}).get("error", {})
        return {"error": f"{state}: {err.get('error_code')} {err.get('message','')[:400]}"}

    manifest = data.get("manifest", {})
    cols = [c["name"] for c in manifest.get("schema", {}).get("columns", [])]
    rows = (data.get("result") or {}).get("data_array") or []
    return {"columns": cols, "rows": rows, "row_count": len(rows)}


def show(label: str, sql: str) -> dict[str, Any]:
    print(f"\n{'─' * 78}\n{label}\n{'─' * 78}")
    res = run_sql(sql)
    if "error" in res:
        print(f"  ✗ {res['error']}")
        return res
    cols = res["columns"]
    print("  " + " | ".join(cols))
    for row in res["rows"][:40]:
        print("  " + " | ".join("NULL" if v is None else str(v) for v in row))
    if res["row_count"] == 0:
        print("  (행 없음)")
    return res


def check() -> None:
    print(f"host={HOST}  warehouse={WAREHOUSE_ID}")

    show("1. system 카탈로그에서 보이는 스키마", "SHOW SCHEMAS IN system")

    show(
        "2. 권한 프로브 — system.billing / system.serving",
        "SELECT count(*) AS n FROM system.billing.usage LIMIT 1",
    )
    show(
        "2b. 권한 프로브 — system.serving.endpoint_usage",
        "SELECT count(*) AS n FROM system.serving.endpoint_usage LIMIT 1",
    )

    show(
        "3. system.ai_gateway.usage 신선도",
        """
        SELECT max(event_time) AS latest_event,
               current_timestamp() AS now_utc,
               round((unix_timestamp(current_timestamp())
                      - unix_timestamp(max(event_time)))/60.0, 1) AS lag_min
        FROM system.ai_gateway.usage
        WHERE event_time >= current_timestamp() - INTERVAL 3 HOURS
        """,
    )

    show(
        "4. 내 호출 — 최근 6시간, 모델·경로별 (토큰이 실제로 기록되는가)",
        """
        SELECT destination_name,
               api_type,
               count(*)                                        AS requests,
               count_if(input_tokens IS NULL)                  AS null_input_tok,
               count_if(output_tokens IS NULL)                 AS null_output_tok,
               sum(coalesce(input_tokens,0))                   AS input_tokens,
               sum(coalesce(output_tokens,0))                  AS output_tokens,
               sum(coalesce(token_details.output_reasoning_tokens,0)) AS reasoning_tokens,
               count_if(latency_ms IS NULL)                    AS null_latency,
               percentile_approx(latency_ms, 0.5)              AS p50_ms
        FROM system.ai_gateway.usage
        WHERE event_time >= current_timestamp() - INTERVAL 6 HOURS
          AND requester = current_user()
        GROUP BY ALL
        ORDER BY requests DESC
        """,
    )


def by_run(run_id: str) -> None:
    show(
        f"태그 조회 — benchmark_run = {run_id}",
        f"""
        SELECT request_tags['model_arm']  AS model_arm,
               request_tags['call_id']    AS call_id,
               destination_name,
               api_type,
               status_code,
               input_tokens, output_tokens, total_tokens,
               token_details.cache_read_input_tokens     AS cache_read,
               token_details.cache_creation_input_tokens AS cache_write,
               token_details.output_reasoning_tokens     AS reasoning,
               latency_ms, time_to_first_byte_ms,
               event_time
        FROM system.ai_gateway.usage
        WHERE event_time >= current_timestamp() - INTERVAL 6 HOURS
          AND request_tags['benchmark_run'] = '{run_id}'
        ORDER BY event_time
        """,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--run")
    ap.add_argument("--sql")
    a = ap.parse_args()
    try:
        if a.check:
            check()
        if a.run:
            by_run(a.run)
        if a.sql:
            show("ad-hoc", a.sql)
        if not (a.check or a.run or a.sql):
            check()
    finally:
        CLIENT.close()
    sys.exit(0)
