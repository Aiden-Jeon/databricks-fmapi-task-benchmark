#!/usr/bin/env python3
"""
warehouse.py — Grader-side SQL access: run ground-truth queries against the SAME
warehouse the candidate apps use. Credentials resolve like everything else in this
repo (env DATABRICKS_HOST/TOKEN wins, else ucode) + DATABRICKS_WAREHOUSE_ID.

Ground truth is ALWAYS recomputed at grading time — never hardcoded — so the grader
is immune to dataset drift and workspace differences.
"""
import functools
import os

from benchmark import fmapi_auth


class WarehouseUnavailable(RuntimeError):
    """Raised when grader-side SQL cannot run (missing creds/connector). The grader
    downgrades affected checks to 'skipped' rather than failing candidates."""


def creds() -> tuple[str, str, str]:
    host, token, _ = fmapi_auth.resolve_host_token()
    wh = os.getenv("DATABRICKS_WAREHOUSE_ID")
    if not host or not token or not wh:
        raise WarehouseUnavailable(
            "need DATABRICKS_HOST/TOKEN (or ucode) + DATABRICKS_WAREHOUSE_ID")
    return host, token, wh


@functools.lru_cache(maxsize=1)
def _connect():
    try:
        from databricks import sql as dbsql
    except ImportError as e:
        raise WarehouseUnavailable(f"databricks-sql-connector not installed: {e}") from e
    host, token, wh = creds()
    return dbsql.connect(
        server_hostname=host.replace("https://", ""),
        http_path=f"/sql/1.0/warehouses/{wh}",
        access_token=token,
    )


GT_MARKER = "--bench-gt"  # keep in sync with query_audit.GT_MARKER


def run(query: str, params: dict | None = None) -> list[tuple]:
    """Execute one query with native named parameters (:name). Returns all rows.
    Every grader query is prefixed with GT_MARKER so query_audit can exclude it
    from candidate-app query counts."""
    conn = _connect()
    with conn.cursor() as cur:
        cur.execute(f"{GT_MARKER}\n{query}", params or {})
        return cur.fetchall()


def run_gt(queries: dict[str, str], name: str, params: dict | None = None) -> list[tuple]:
    if name not in queries:
        raise WarehouseUnavailable(f"ground-truth query {name!r} not found")
    return run(queries[name], params)


def scalar(queries: dict[str, str], name: str, params: dict | None = None):
    rows = run_gt(queries, name, params)
    return rows[0][0] if rows and rows[0] else None
