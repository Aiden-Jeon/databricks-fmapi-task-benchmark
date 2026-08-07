"""클라이언트 `usage` vs `system.ai_gateway.usage` 교차검증.

두 소스가 어긋나면 그 자체가 보고할 발견이다. 신뢰 우선순위는
METHODOLOGY §4에 정의돼 있다 — 지연은 클라이언트, 토큰은 시스템 테이블.

시스템 테이블 지연이 18~26분이라 **실행 30분 뒤**에 돌려야 한다.

사용:
    python -m src.xcheck --run results/combined
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "probes"))
from query_usage import run_sql  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    outdir = pathlib.Path(args.run)

    raw = [json.loads(l) for l in (outdir / "raw.jsonl").read_text(encoding="utf-8").splitlines()
           if l.strip()]
    run_ids = sorted({r["run_id"] for r in raw})
    print(f"benchmark_run: {run_ids}")

    # 클라이언트 집계
    client: dict[str, dict[str, int]] = {}
    for r in raw:
        if r["outcome"] != "ok":
            continue
        c = client.setdefault(r["arm"], {"n": 0, "in": 0, "out": 0})
        c["n"] += 1
        c["in"] += r["usage"]["prompt_tokens"]
        c["out"] += r["usage"]["completion_tokens"]

    # 서버는 **거부된 요청(HTTP 400)도 행으로 남긴다** — 토큰 컬럼은 NULL이다.
    # 클라이언트의 "ok" 집합과 맞추려면 status_code=200으로 걸러야 한다.
    # 안 거르면 행 수가 어긋나고 NULL이 통계를 오염시킨다.
    ids = ", ".join(f"'{i}'" for i in run_ids)
    res = run_sql(f"""
        SELECT request_tags['model_arm'] AS arm,
               count_if(status_code = 200) AS n,
               count_if(status_code = 200 AND input_tokens IS NULL) AS null_in,
               sum(CASE WHEN status_code = 200 THEN coalesce(input_tokens,0) ELSE 0 END)  AS in_tok,
               sum(CASE WHEN status_code = 200 THEN coalesce(output_tokens,0) ELSE 0 END) AS out_tok,
               percentile_approx(CASE WHEN status_code = 200 THEN latency_ms END, 0.5) AS p50,
               percentile_approx(CASE WHEN status_code = 200 THEN time_to_first_byte_ms END, 0.5)
                   AS ttfb_p50,
               count_if(status_code <> 200) AS rejected
        FROM system.ai_gateway.usage
        WHERE event_time >= current_timestamp() - INTERVAL 6 HOURS
          AND request_tags['benchmark_run'] IN ({ids})
          AND request_tags['case_id'] IS NOT NULL
        GROUP BY ALL ORDER BY arm
    """)
    if "error" in res:
        print(f"✗ {res['error']}")
        return 1

    server = {row[0]: {"n": int(row[1]), "null_in": int(row[2]), "in": int(row[3]),
                       "out": int(row[4]), "p50": row[5], "ttfb": row[6],
                       "rejected": int(row[7])}
              for row in res["rows"]}

    print(f"\n{'arm':16s} {'calls c/s':>14s} {'in_tok c/s':>20s} {'out_tok c/s':>18s} "
          f"{'srv p50':>9s} {'srv ttfb':>9s}  일치")
    all_match = True
    for arm in sorted(client):
        c = client[arm]
        s = server.get(arm)
        if s is None:
            print(f"{arm:16s} {c['n']:>6d}/{'—':>7s}   (시스템 테이블 미도달)")
            all_match = False
            continue
        ok_in = c["in"] == s["in"]
        ok_out = c["out"] == s["out"]
        ok_n = c["n"] == s["n"]
        all_match &= ok_in and ok_out and ok_n
        mark = "✅" if (ok_in and ok_out and ok_n) else "⚠️"
        print(f"{arm:16s} {c['n']:>6d}/{s['n']:<7d} {c['in']:>9d}/{s['in']:<10d} "
              f"{c['out']:>8d}/{s['out']:<9d} {s['p50']:>9} {s['ttfb']:>9}  {mark}")
        if s["null_in"]:
            print(f"{'':16s}   ⚠️ 200 응답인데 input_tokens NULL {s['null_in']}건")
        if s["rejected"]:
            print(f"{'':16s}   ℹ️  게이트웨이 거부 {s['rejected']}건도 서버에 기록됨 "
                  f"(토큰 컬럼 NULL, status_code<>200)")

    print("\n" + ("두 소스 완전 일치 — 클라이언트 usage를 신뢰할 수 있다."
                  if all_match else
                  "불일치 있음 — 위 표의 ⚠️ 행을 확인할 것."))
    return 0 if all_match else 2


if __name__ == "__main__":
    sys.exit(main())
