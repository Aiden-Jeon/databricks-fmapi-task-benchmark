"""벤치마크 러너 — Arm A (조건 통일).

설계는 `../METHODOLOGY.md`. 실측으로 강제된 것들:

- **경로는 `/ai-gateway/mlflow/v1/chat/completions`.** `/serving-endpoints`는
  `system.ai_gateway.usage`에 기록되지 않아 서버측 텔레메트리도 request_tags 귀속도
  못 얻는다(METHODOLOGY §4 — 24시간 573만 행 중 0건).
- **reasoning은 모델별 최소로 통일.** sol은 `reasoning_effort:"none"`이 아니면 tools에서 HTTP 400이 난다. opus·glm도 최소로 맞춰야 조건이 같아진다.
- **temperature를 못 쓴다.** opus·sol이 거부한다 → 결정성 확보 불가 → 반복이 필수.
- **동시성 상한이 모델마다 다르다.** glm QPH 7,200(초당 2건)은 나머지의 1/50이다.
- **OAuth 토큰이 실행보다 먼저 만료된다**(~1시간). 만료를 만나면 갱신 후 재시도한다.
- **재시도는 인프라 오류에만.** 형식 실패를 재시도하면 측정 대상이 사라진다.

사용:
    python -m src.runner --repeats 5 --out results/<run-id>
    python -m src.runner --repeats 1 --limit 6 --smoke
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import random
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import httpx

ROOT = pathlib.Path(__file__).resolve().parent.parent
CASES = ROOT / "cases" / "cases.jsonl"
PARITY_CASES = ROOT / "cases" / "parity_cases.jsonl"
FCB_CASES = ROOT / "cases" / "fcb_cases.jsonl"

PROFILE = os.environ.get("DATABRICKS_PROFILE", "DEFAULT")
GATEWAY_PATH = "/ai-gateway/mlflow/v1/chat/completions"
RESPONSES_PATH = "/ai-gateway/codex/v1/responses"

# `protocol`은 요청/응답 형태를 결정한다.
#   "chat"      — OpenAI chat/completions 호환. tools는 {"type":"function","function":{...}}
#   "responses" — OpenAI Responses API. tools는 평평, input/output 구조가 다름
#
# Arm A = 경로·reasoning 통일 (엄밀한 모델 비교)
# Arm B = 모델별 최선     (고객이 실제로 배포하는 조건). 경로가 갈리는 것을 명시한다.
ARMS: dict[str, dict[str, Any]] = {
    # ── Arm A ────────────────────────────────────────────────────────────────
    "opus": {"model": "system.ai.claude-opus-5", "params": {"thinking": {"type": "disabled"}},
             "concurrency": 6, "protocol": "chat", "path": GATEWAY_PATH, "arm": "A"},
    "sol": {"model": "system.ai.gpt-5-6-sol", "params": {"reasoning_effort": "none"},
            "concurrency": 6, "protocol": "chat", "path": GATEWAY_PATH, "arm": "A"},
    # glm QPH 7,200 = 초당 2건. 나머지의 1/50이라 여기가 전체 페이스를 정한다.
    "glm": {"model": "system.ai.glm-5-2", "params": {"reasoning_effort": "none"},
            "concurrency": 2, "protocol": "chat", "path": GATEWAY_PATH, "arm": "A"},
    # opus는 `thinking:{type:disabled}` + tools 조합에서 Anthropic 원시 XML을
    # content로 흘리고 tool_calls를 비운다(실측 8/12, 경로 무관, sonnet-5는 정상).
    # 즉 Arm A의 "reasoning 최소 통일" 조건이 opus만 망가뜨린다. 버그 보정용 arm.
    "opus-adaptive": {"model": "system.ai.claude-opus-5",
                      "params": {"thinking": {"type": "adaptive"}}, "concurrency": 6,
                      "protocol": "chat", "path": GATEWAY_PATH, "arm": "A'"},

    # ── Arm B — 모델별 최선 ──────────────────────────────────────────────────
    # opus: `output_config.effort`가 진짜 노브다. 문서가 안내하는 thinking/budget_tokens는 400.
    #       공식 문서는 thinking/budget_tokens라고 안내하지만 그건 400이 난다.
    "opus-B": {"model": "system.ai.claude-opus-5", "params": {"output_config": {"effort": "high"}},
               "concurrency": 6, "protocol": "chat", "path": GATEWAY_PATH, "arm": "B"},
    # sol: chat/completions에서는 tools+reasoning이 배타다(게이트웨이 제약).
    #      Responses 경로만 둘을 동시에 허용한다 → **경로가 달라진다.**
    "sol-B": {"model": "system.ai.gpt-5-6-sol", "params": {"reasoning": {"effort": "high"}},
              "concurrency": 6, "protocol": "responses", "path": RESPONSES_PATH, "arm": "B"},
    "glm-B": {"model": "system.ai.glm-5-2", "params": {"reasoning_effort": "high"},
              "concurrency": 2, "protocol": "chat", "path": GATEWAY_PATH, "arm": "B"},
}

MAX_TOKENS = 1024
REQUEST_TIMEOUT = 120.0
MAX_INFRA_RETRIES = 4
BACKOFF_INITIAL = 2.0


# ── 인증 (토큰 만료 대응) ─────────────────────────────────────────────────────

class Auth:
    """host/token을 들고 있다가 만료 시 갱신한다.

    토큰 수명(~1시간)이 실행 시간보다 짧을 수 있다. 과거 같은 워크스페이스에서
    52분 지점에 세 모델이 동시에 403을 맞고 run이 폐기된 사고가 있었다
    (같은 repo의 image-text-performance가 겪은 사고).
    """

    def __init__(self, profile: str) -> None:
        self.profile = profile
        self._lock = threading.Lock()
        self.host, self.token = self._fetch()
        self._last_refresh = time.time()

    def _fetch(self) -> tuple[str, str]:
        def cli(*a: str) -> str:
            p = subprocess.run(
                ["databricks", *a, "--profile", self.profile], capture_output=True, text=True
            )
            if p.returncode != 0:
                raise RuntimeError(f"databricks {' '.join(a)}: {p.stderr.strip()}")
            return p.stdout

        host = json.loads(cli("auth", "env"))["env"]["DATABRICKS_HOST"].rstrip("/")
        return host, json.loads(cli("auth", "token"))["access_token"]

    def refresh(self) -> None:
        """스레드 안전. 방금 갱신했으면 건너뛴다(동시 만료 시 CLI 폭주 방지)."""
        with self._lock:
            if time.time() - self._last_refresh < 20:
                return
            self.host, self.token = self._fetch()
            self._last_refresh = time.time()
            print("    [토큰 갱신]", flush=True)


_PERMISSION_MARKERS = ("permission", "not authorized", "forbidden")
_EXPIRY_MARKERS = ("credential", "invalid token", "expired", "not authenticated")


def is_auth_expiry(status: int, body: str) -> bool:
    """갱신하면 풀릴 인증 실패인가(권한 거부와 구분)."""
    low = body.lower()
    if any(s in low for s in _PERMISSION_MARKERS):
        return False
    if status == 403:
        return "invalid token" in low
    if status == 401:
        return any(s in low for s in _EXPIRY_MARKERS)
    return False


# ── 응답 정규화 (모델마다 형태가 다르다) ─────────────────────────────────────

def normalize_usage(u: dict[str, Any]) -> dict[str, int]:
    """4종 usage 형태를 하나로.

    chat/completions:
      sol : prompt_tokens_details.cached_tokens / completion_tokens_details.reasoning_tokens
      opus: 최상위 cache_read_input_tokens, reasoning 필드 없음
      glm : 최상위 + 중첩 둘 다, reasoning 필드 없음
    responses:
      input_tokens / output_tokens + input_tokens_details / output_tokens_details

    reasoning_tokens는 **출력 토큰 안에 포함**된다(METHODOLOGY §3 실측).
    따라서 billable_output = completion_tokens 이고, 더하면 안 된다.
    """
    ptd = u.get("prompt_tokens_details") or u.get("input_tokens_details") or {}
    ctd = u.get("completion_tokens_details") or u.get("output_tokens_details") or {}
    # Responses API는 input_tokens/output_tokens를 쓴다.
    prompt = u.get("prompt_tokens")
    prompt = u.get("input_tokens", 0) if prompt is None else prompt
    completion = u.get("completion_tokens")
    completion = u.get("output_tokens", 0) if completion is None else completion

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
        "prompt_tokens": prompt or 0,
        "completion_tokens": completion or 0,
        "total_tokens": u.get("total_tokens") or 0,
        "reasoning_tokens": ctd.get("reasoning_tokens", 0) or 0,
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
    }


def to_responses_tool(t: dict[str, Any]) -> dict[str, Any]:
    """chat 형식 툴 정의를 Responses API 형식(평평)으로."""
    f = t["function"]
    return {"type": "function", "name": f["name"],
            "description": f.get("description", ""), "parameters": f["parameters"]}


def parse_responses(body: dict[str, Any]) -> tuple[str, list[dict[str, Any]], str | None]:
    """Responses API 응답 → (text, tool_calls, finish_reason).

    output 배열에 reasoning / function_call / message 항목이 섞여 온다.
    """
    text_parts: list[str] = []
    calls: list[dict[str, Any]] = []
    kinds: list[str] = []
    for o in body.get("output") or []:
        kind = o.get("type")
        kinds.append(kind)
        if kind == "function_call":
            raw = o.get("arguments")
            try:
                args = json.loads(raw) if isinstance(raw, str) else (raw or {})
                ok = True
            except Exception:
                args, ok = {}, False
            calls.append({"name": o.get("name"), "args": args,
                          "args_raw": raw, "args_parse_ok": ok})
        elif kind == "message":
            for p in o.get("content") or []:
                if isinstance(p, dict) and p.get("text"):
                    text_parts.append(p["text"])
    fr = "tool_calls" if calls else (body.get("status") or ("stop" if text_parts else None))
    if body.get("incomplete_details"):
        fr = "length"
    return "".join(text_parts).strip(), calls, fr


def extract_text(msg: dict[str, Any]) -> str:
    c = msg.get("content")
    if isinstance(c, list):
        t = "".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
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
        out.append({"name": fn.get("name"), "args": args, "args_raw": raw, "args_parse_ok": parse_ok})
    return out


# ── 호출 ─────────────────────────────────────────────────────────────────────

def call_once(
    client: httpx.Client, auth: Auth, arm: str, case: dict[str, Any], rep: int, run_id: str
) -> dict[str, Any]:
    spec = ARMS[arm]
    messages: list[dict[str, Any]] = []
    if case.get("system"):
        messages.append({"role": "system", "content": case["system"]})
    # FCB 트랙은 멀티턴이라 messages를 그대로 싣는다. 없으면 단일 user 턴.
    if case.get("messages"):
        messages.extend(case["messages"])
    else:
        messages.append({"role": "user", "content": case["prompt"]})

    if spec["protocol"] == "responses":
        # Responses API — input 배열, 평평한 tools, text.format으로 구조화 출력.
        body = {"model": spec["model"], "input": messages,
                "max_output_tokens": MAX_TOKENS, **spec["params"]}
        if case.get("tools"):
            body["tools"] = [to_responses_tool(t) for t in case["tools"]]
            if case.get("response_format"):
                rf = case["response_format"]["json_schema"]
                body["text"] = {"format": {"type": "json_schema", "name": rf["name"],
                                           "schema": rf["schema"], "strict": True}}
        else:
            body["text"] = {"format": {"type": "json_schema", "name": "out",
                                       "schema": case["schema"], "strict": True}}
    else:
        body = {"model": spec["model"], "messages": messages,
                "max_tokens": MAX_TOKENS, **spec["params"]}
        if case.get("tools"):
            body["tools"] = case["tools"]
            if case.get("response_format"):
                body["response_format"] = case["response_format"]
        else:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "out", "schema": case["schema"], "strict": True},
            }

    call_id = str(uuid.uuid4())
    tags = {
        "benchmark_run": run_id,
        "call_id": call_id,
        "model_arm": arm,
        "case_id": case["id"],
        "category": case["category"],
        "rep": str(rep),
    }
    url = auth.host + spec["path"]

    rec: dict[str, Any] = {
        "run_id": run_id, "call_id": call_id, "arm": arm, "case_id": case["id"],
        "category": case["category"], "track": case["track"], "rep": rep,
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
    }

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

        # 인증 만료는 갱신 후 재시도 — 재시도 예산을 쓰지 않는다.
        if is_auth_expiry(r.status_code, r.text) and budget == MAX_INFRA_RETRIES:
            auth.refresh()
            budget += 1
            continue

        # 인프라 실패만 재시도. 429는 retry_after를 지킨다.
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

        # 4xx는 즉시 확정. 게이트웨이 거부이므로 형식 실패가 아니라 별도 분류한다.
        if r.status_code != 200:
            rec.update({"outcome": "gateway_reject", "status": r.status_code,
                        "latency_ms": dt_ms, "error": r.text[:500], "retries": attempt})
            return rec

        data = r.json()
        if spec["protocol"] == "responses":
            if "output" not in data:
                rec.update({"outcome": "malformed_response", "status": 200, "latency_ms": dt_ms,
                            "error": json.dumps(data, ensure_ascii=False)[:500],
                            "retries": attempt})
                return rec
            text, calls, finish = parse_responses(data)
        else:
            try:
                choice = data["choices"][0]
            except (KeyError, IndexError):
                rec.update({"outcome": "malformed_response", "status": 200, "latency_ms": dt_ms,
                            "error": json.dumps(data, ensure_ascii=False)[:500],
                            "retries": attempt})
                return rec
            msg = choice.get("message", {})
            text, calls = extract_text(msg), extract_tool_calls(msg)
            finish = choice.get("finish_reason")
        rec.update({
            "outcome": "ok",
            "status": 200,
            "latency_ms": dt_ms,
            "retries": attempt,
            "finish_reason": finish,
            "text": text,
            "tool_calls": calls,
            "usage": normalize_usage(data.get("usage") or {}),
            "request_id": r.headers.get("x-request-id"),
        })
        return rec

    rec.update({"outcome": "infra_fail", "status": None, "error": last_err,
                "retries": attempt})
    return rec


# ── 실행 ─────────────────────────────────────────────────────────────────────

def load_cases(limit: int | None, src: pathlib.Path = CASES,
               tracks: list[str] | None = None) -> list[dict[str, Any]]:
    cases = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    if tracks:
        cases = [c for c in cases if c["track"] in tracks]
    if limit:
        # 카테고리별로 골고루 뽑는다 — 앞에서 자르면 카테고리가 통째로 빠진다.
        by_cat: dict[str, list[dict[str, Any]]] = {}
        for c in cases:
            by_cat.setdefault(c["category"], []).append(c)
        out, i = [], 0
        while len(out) < limit and any(len(v) > i for v in by_cat.values()):
            for cat in sorted(by_cat):
                if len(by_cat[cat]) > i and len(out) < limit:
                    out.append(by_cat[cat][i])
            i += 1
        return out
    return cases


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--out")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--parity", action="store_true", help="parity_cases.jsonl 사용")
    ap.add_argument("--fcb", action="store_true", help="fcb_cases.jsonl 사용 (FunctionChat-Bench)")
    ap.add_argument("--tracks", nargs="*", help="track 필터 (OB / PAIR)")
    args = ap.parse_args()

    run_id = "fcjson-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
    outdir = pathlib.Path(args.out) if args.out else ROOT / "results" / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    src = FCB_CASES if args.fcb else (PARITY_CASES if args.parity else CASES)
    cases = load_cases(args.limit, src, args.tracks)
    auth = Auth(PROFILE)

    # 워밍업 — TLS/DNS 비용을 첫 케이스에 싣지 않는다.
    warm = httpx.Client(timeout=REQUEST_TIMEOUT)
    for arm in args.arms:
        sp = ARMS[arm]
        b = ({"model": sp["model"], "input": "ping", "max_output_tokens": 16, **sp["params"]}
             if sp["protocol"] == "responses"
             else {"model": sp["model"], "messages": [{"role": "user", "content": "ping"}],
                   "max_tokens": 8, **sp["params"]})
        try:
            warm.post(auth.host + sp["path"], json=b,
                      headers={"Authorization": f"Bearer {auth.token}",
                               "Content-Type": "application/json"})
        except Exception:
            pass
    warm.close()

    # 작업 단위: (arm, case, rep). arm을 라운드로빈으로 섞어 시간대 편향을 없앤다.
    jobs: list[tuple[str, dict[str, Any], int]] = []
    for rep in range(1, args.repeats + 1):
        for case in cases:
            for arm in args.arms:
                jobs.append((arm, case, rep))
    random.Random(42).shuffle(jobs)

    manifest = {
        "run_id": run_id, "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "routes": {a: ARMS[a]["path"] for a in args.arms}, "profile": PROFILE,
        "repeats": args.repeats, "n_cases": len(cases), "arms": {a: ARMS[a] for a in args.arms},
        "max_tokens": MAX_TOKENS, "total_calls": len(jobs), "smoke": args.smoke,
    }
    (outdir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"run_id={run_id}")
    print(f"cases={len(cases)}  arms={args.arms}  repeats={args.repeats}  calls={len(jobs)}")
    print(f"out={outdir}\n")

    # arm별로 동시성 상한이 다르므로 arm별 세마포어로 제한한다.
    sems = {a: threading.Semaphore(ARMS[a]["concurrency"]) for a in args.arms}
    client = httpx.Client(timeout=REQUEST_TIMEOUT,
                          limits=httpx.Limits(max_connections=20, max_keepalive_connections=20))
    lock = threading.Lock()
    fh = (outdir / "raw.jsonl").open("w", encoding="utf-8")
    done = {"n": 0, "ok": 0, "rej": 0, "fail": 0}
    t_start = time.time()

    def work(job: tuple[str, dict[str, Any], int]) -> None:
        arm, case, rep = job
        with sems[arm]:
            rec = call_once(client, auth, arm, case, rep, run_id)
        with lock:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            done["n"] += 1
            o = rec.get("outcome")
            done["ok" if o == "ok" else ("rej" if o == "gateway_reject" else "fail")] += 1
            if done["n"] % 25 == 0 or done["n"] == len(jobs):
                el = time.time() - t_start
                rate = done["n"] / el if el else 0
                eta = (len(jobs) - done["n"]) / rate if rate else 0
                print(f"  {done['n']:5d}/{len(jobs)}  ok={done['ok']} reject={done['rej']} "
                      f"fail={done['fail']}  {rate:.1f}/s  ETA {eta/60:.1f}m", flush=True)

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
    # 인프라 실패율이 10%를 넘으면 수치를 신뢰할 수 없다는 뜻으로 exit 1.
    return 1 if done["fail"] > 0.10 * len(jobs) else 0


if __name__ == "__main__":
    sys.exit(main())
