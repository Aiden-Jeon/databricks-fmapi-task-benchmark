"""벤치마크 러너 — 시나리오별 케이스 × arm × 반복.

사용:
    python -m src.runner --repeats 5 --scenario all --arms opus sol glm
    python -m src.runner --smoke --limit 1        # 스모크: 1케이스 × arm × 1회
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import httpx

from .agent import run_session
from .auth import Auth
from .fmapi import ARMS, REQUEST_TIMEOUT

ROOT = pathlib.Path(__file__).resolve().parent.parent
CASES_DIR = ROOT / "cases"
SCENARIOS = ["azure_troubleshoot", "insurance_policy"]
PROFILE = os.environ.get("DATABRICKS_PROFILE", "ai_devtools")


def load_scenario(name: str) -> dict[str, Any]:
    d = CASES_DIR / name
    tools = json.loads((d / "_tools.json").read_text(encoding="utf-8"))
    system = (d / "_system.txt").read_text(encoding="utf-8")
    clauses_path = d / "_clauses.json"
    clauses = json.loads(clauses_path.read_text(encoding="utf-8")) if clauses_path.exists() else {}
    cases = []
    for p in sorted(d.glob("*.json")):
        if p.name.startswith("_"):
            continue
        cases.append(json.loads(p.read_text(encoding="utf-8")))
    return {"name": name, "tools": tools, "system": system, "clauses": clauses, "cases": cases}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--scenario", default="all", choices=["all", *SCENARIOS])
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--limit", type=int, help="시나리오당 케이스 수 제한(스모크용)")
    ap.add_argument("--cases", nargs="*", help="특정 case id 만 실행 (예: A-5 A-6)")
    ap.add_argument("--smoke", action="store_true", help="repeats=1, limit=1")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.smoke:
        args.repeats = 1
        args.limit = args.limit or 1

    scen_names = SCENARIOS if args.scenario == "all" else [args.scenario]
    scenarios = {n: load_scenario(n) for n in scen_names}

    run_id = "custagent-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
    outdir = pathlib.Path(args.out) if args.out else ROOT / "results" / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    auth = Auth(PROFILE)

    # 작업 단위: (scenario, case, arm, rep). arm 을 섞어 시간대 편향을 없앤다.
    jobs: list[tuple[str, dict[str, Any], str, int]] = []
    for sname in scen_names:
        cases = scenarios[sname]["cases"]
        if args.cases:
            cases = [c for c in cases if c["id"] in set(args.cases)]
        if args.limit:
            cases = cases[: args.limit]
        for rep in range(1, args.repeats + 1):
            for case in cases:
                for arm in args.arms:
                    jobs.append((sname, case, arm, rep))
    random.Random(42).shuffle(jobs)

    manifest = {
        "run_id": run_id, "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "profile": PROFILE, "repeats": args.repeats, "arms": args.arms,
        "scenarios": {n: len(scenarios[n]["cases"]) for n in scen_names},
        "arm_specs": {a: ARMS[a] for a in args.arms}, "total_sessions": len(jobs),
    }
    (outdir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"run_id={run_id}\nsessions={len(jobs)}  arms={args.arms}  repeats={args.repeats}")
    print(f"out={outdir}\n")

    sems = {a: threading.Semaphore(ARMS[a]["concurrency"]) for a in args.arms}
    client = httpx.Client(timeout=REQUEST_TIMEOUT,
                          limits=httpx.Limits(max_connections=20, max_keepalive_connections=20))
    lock = threading.Lock()
    fh = (outdir / "raw.jsonl").open("w", encoding="utf-8")
    done = {"n": 0, "ok": 0, "fail": 0}
    t_start = time.time()

    def work(job: tuple[str, dict[str, Any], str, int]) -> None:
        sname, case, arm, rep = job
        sc = scenarios[sname]
        with sems[arm]:
            rec = run_session(client, auth, arm, case, sc["tools"], sc["system"],
                              sc["clauses"], run_id, rep)
        with lock:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            done["n"] += 1
            done["ok" if rec["outcome"] == "ok" else "fail"] += 1
            if done["n"] % 10 == 0 or done["n"] == len(jobs):
                el = time.time() - t_start
                rate = done["n"] / el if el else 0
                eta = (len(jobs) - done["n"]) / rate if rate else 0
                print(f"  {done['n']:4d}/{len(jobs)}  ok={done['ok']} fail={done['fail']}  "
                      f"{rate:.2f}/s  ETA {eta/60:.1f}m", flush=True)

    with ThreadPoolExecutor(max_workers=sum(ARMS[a]["concurrency"] for a in args.arms)) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for f in as_completed(futs):
            f.result()

    fh.close()
    client.close()
    manifest["ended_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest["elapsed_s"] = round(time.time() - t_start, 1)
    manifest["counts"] = dict(done)
    (outdir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n완료 {done} in {manifest['elapsed_s']}s → {outdir}/raw.jsonl")
    return 1 if done["fail"] > 0.10 * len(jobs) else 0


if __name__ == "__main__":
    sys.exit(main())
