#!/usr/bin/env python3
"""
query_audit.py — Count the queries a candidate app actually issued, via the SQL
warehouse Query History API (GET /api/2.0/sql/history/queries).

Precondition (from README): grading uses a DEDICATED warehouse so the history in the
grading time-window belongs to the candidate app + the grader's own GT queries. GT
queries are excluded by statement fingerprint (they all start with the marker
comment below, which warehouse.py prepends — keeping candidate traffic untouched).

Used for tier1 T5b (KPI not re-queried), tier2 E1–E3 (query budget / caching /
boundedness).
"""
import time

import requests

from benchmark import fmapi_auth, warehouse

GT_MARKER = "--bench-gt"          # prepended to grader GT queries; excluded from counts


class AuditWindow:
    """Bracket an interaction: `with AuditWindow() as w: ... ; n = w.count()`."""

    def __init__(self):
        self.start_ms = None
        self.end_ms = None

    def __enter__(self):
        self.start_ms = int(time.time() * 1000)
        return self

    def __exit__(self, *exc):
        self.end_ms = int(time.time() * 1000)
        # history ingestion lags a little; small settle before counting
        time.sleep(3)
        return False

    def queries(self) -> list[dict]:
        host, token, wh = warehouse.creds()
        url = f"{host}/api/2.0/sql/history/queries"
        params = {
            "filter_by.warehouse_ids": wh,
            "filter_by.query_start_time_range.start_time_ms": self.start_ms,
            "filter_by.query_start_time_range.end_time_ms": self.end_ms or int(time.time() * 1000),
            "max_results": 100,
        }
        out, page_token = [], None
        for _ in range(10):  # pagination guard
            if page_token:
                params = {"page_token": page_token}
            r = requests.get(url, params=params,
                             headers={"Authorization": f"Bearer {token}"}, timeout=30)
            r.raise_for_status()
            data = r.json()
            out.extend(data.get("res", []))
            page_token = data.get("next_page_token")
            if not data.get("has_next_page"):
                break
        return [q for q in out
                if not (q.get("query_text") or "").lstrip().startswith(GT_MARKER)]

    def count(self) -> int:
        return len(self.queries())

    def unbounded_over(self, tables: tuple[str, ...] = ("lineitem", "orders")) -> int:
        """tier2 E3: queries touching big tables with neither aggregation nor LIMIT."""
        bad = 0
        for q in self.queries():
            text = (q.get("query_text") or "").lower()
            if any(t in text for t in tables):
                if "limit" not in text and not any(
                        f in text for f in ("count(", "sum(", "avg(", "group by")):
                    bad += 1
        return bad
