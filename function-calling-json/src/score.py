"""채점기 — Function Calling & JSON Output.

설계는 `../METHODOLOGY.md` §5. 원칙 셋:

1. **한 숫자로 합치지 않는다.** 정확도·안정성·정책·비용을 분해해서 본다.
   같은 "툴 미호출"이 되물음(정상)일 수도 포기(실패)일 수도 있고,
   같은 "툴 호출"이 정답일 수도 날짜 환각(최악)일 수도 있다.
2. **실패를 종류별로 분류한다.** 인프라 실패는 채점 제외, 게이트웨이 거부는
   capability로 기록, **형식 실패는 0점**(이게 측정 대상이다).
3. **비율 지표에는 Wilson 신뢰구간.** 정규근사는 0/1 근처에서 못 쓴다.
   구간이 겹치면 무승부로 보고한다.

`pass^k`는 τ-bench 정의: C(c,k)/C(n,k) — k회 뽑아 **전부** 성공할 확률.
"적어도 한 번 성공"인 pass@k와 혼동하면 안 된다.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys
from collections import Counter, defaultdict
from typing import Any

try:
    import jsonschema
except ImportError:
    jsonschema = None

ROOT = pathlib.Path(__file__).resolve().parent.parent
CASES = ROOT / "cases" / "cases.jsonl"
PARITY_CASES = ROOT / "cases" / "parity_cases.jsonl"
FCB_CASES = ROOT / "cases" / "fcb_cases.jsonl"

# 같은 repo의 image-text-performance/config/pricing.yaml 교차검증 값.
USD_PER_DBU = 0.07
PRICING = {  # DBU per 1M tokens
    "opus": {"in": 71.429, "out": 357.143, "cw": 89.286, "cr": 7.143},
    "sol": {"in": 71.429, "out": 428.571, "cw": 89.286, "cr": 7.143},
    "glm": {"in": 20.000, "out": 62.857, "cw": 0.0, "cr": 3.714},
}

# 되물음으로 인정할 신호. "정보가 없어 확인이 필요하다"는 취지가 드러나야 한다.
ASK_PAT = re.compile(
    r"(알려주|말씀해|어느|어디|언제|몇\s*[명분일]|필요합니다|필요해|"
    r"확인\s*(부탁|해\s*주)|여쭤|알 수 없|모르|특정할 수 없|정해\s*주|선택해\s*주|\?)"
)


# ── 통계 ─────────────────────────────────────────────────────────────────────

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """비율의 Wilson score interval. (point, lo, hi)."""
    if n == 0:
        return (float("nan"), 0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def pass_hat_k(n: int, c: int, k: int) -> float | None:
    """pass^k = C(c,k)/C(n,k). n회 중 c회 성공했을 때 k회 연속 성공 확률의 불편추정량."""
    if n < k:
        return None
    return math.comb(c, k) / math.comb(n, k)


# ── 정규화 ───────────────────────────────────────────────────────────────────

def norm_scalar(v: Any) -> Any:
    """비교 전 정규화. 타입 충실도는 별도로 재므로 여기서는 값 동등성만 본다."""
    if isinstance(v, str):
        s = v.strip()
        # "35000000" vs 35000000 — 값은 같다고 본다(타입은 type_fidelity에서 별도 감점).
        if re.fullmatch(r"-?\d+", s):
            return int(s)
        return re.sub(r"\s+", " ", s)
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def flatten(o: Any, prefix: str = "") -> dict[str, Any]:
    """중첩 객체를 JSON Pointer 경로 → 리프값 dict로."""
    out: dict[str, Any] = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(flatten(v, f"{prefix}/{k}"))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            out.update(flatten(v, f"{prefix}/{i}"))
    else:
        out[prefix] = norm_scalar(o)
    return out


def field_prf(pred: dict[str, Any], gold: dict[str, Any]) -> tuple[float, float, float]:
    P, G = flatten(pred), flatten(gold)
    inter = sum(1 for k, v in P.items() if k in G and G[k] == v)
    p = inter / len(P) if P else 0.0
    r = inter / len(G) if G else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


# ── FC 채점 ──────────────────────────────────────────────────────────────────

def compare_args(
    pred: dict[str, Any], gold: dict[str, Any], tool: dict[str, Any]
) -> dict[str, Any]:
    """인자 비교. **선택적 파라미터를 추가한 것은 오답이 아니다.**

    스모크에서 세 모델 모두 `get_weather(city="서울")`에 `unit="celsius"`를 덧붙였다.
    스키마상 `unit`은 optional이고 enum에 있는 값이므로 **정답으로 봐야 한다.**
    이걸 오답으로 처리하면 AST 매칭의 고질적 취약성을 그대로 재현하는 것이다
    (arXiv:2504.00914 — 실패의 70~90%가 모델이 아니라 채점 방식 탓).

    규칙:
      - gold에 있는 키는 **전부 있어야 하고 값이 같아야** 한다.
      - gold에 없지만 스키마에 있는 optional 키는 **허용**한다. 단 enum 위반이면 오답.
      - 스키마에 아예 없는 키는 **환각 파라미터**로 오답.
    """
    params = tool["function"].get("parameters", {})
    props = params.get("properties", {})
    gold_keys = set(gold)
    pred_keys = set(pred)

    matched = sum(1 for k in gold_keys if k in pred_keys
                  and norm_scalar(pred[k]) == norm_scalar(gold[k]))
    missing = sorted(gold_keys - pred_keys)
    wrong = sorted(k for k in gold_keys & pred_keys
                   if norm_scalar(pred[k]) != norm_scalar(gold[k]))
    hallucinated = sorted(pred_keys - set(props))
    extra_optional = sorted(pred_keys - gold_keys - set(hallucinated))
    enum_bad = [k for k in extra_optional
                if "enum" in props.get(k, {}) and pred[k] not in props[k]["enum"]]

    exact = (not missing and not wrong and not hallucinated and not enum_bad)
    # F1은 gold 키 기준 recall과, 환각 파라미터를 벌하는 precision으로 낸다.
    denom_p = matched + len(wrong) + len(hallucinated)
    p = matched / denom_p if denom_p else 0.0
    r = matched / len(gold_keys) if gold_keys else 1.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"exact": exact, "f1": f1, "missing": missing, "wrong": wrong,
            "hallucinated_param": hallucinated, "extra_optional": extra_optional,
            "enum_bad": enum_bad}


def score_fcb(rec: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    """FunctionChat-Bench CallDecision 채점.

    **채점 범위가 원본 벤치마크보다 좁다.** 원본은 REJECT / SLOT 카테고리를
    LLM 채점자로 평가해 "되물음 문장이 올바른 슬롯을 묻는가"까지 본다.
    이 저장소는 LLM 채점자를 쓰지 않으므로 **도구 호출을 억제했는가만** 본다.
    따라서 REJECT / SLOT 점수는 원본 리더보드보다 관대하며 직접 비교할 수 없다.

    CALL 카테고리는 정답 도구·인자가 명시돼 있어 원본과 같은 기준으로 채점한다.
    `acceptable_arguments`가 있으면 정답의 대체 표기로 인정한다 — 고유명사를
    영문으로 정규화한 것을 오답으로 세지 않기 위한 원본 벤치마크의 장치다.
    """
    exp = case["expect"]
    action = exp["action"]
    calls = rec.get("tool_calls") or []
    tools_by_name = {t["function"]["name"]: t for t in case["tools"]}

    s: dict[str, Any] = {
        "called": len(calls) > 0,
        "n_calls": len(calls),
        "expected_action": action,
        "hallucinated_tool": any(c["name"] not in tools_by_name for c in calls),
        "args_unparseable": any(not c["args_parse_ok"] for c in calls),
    }

    if action == "no_call":
        # REJECT / SLOT-all / SLOT-some — 호출을 억제했으면 정답.
        s["correct"] = not calls
        s["suppressed"] = not calls
        return s

    # CALL — 도구 1개를 정확한 인자로 호출해야 한다.
    gold = exp["calls"][0]
    acc = exp.get("acceptable_arguments") or {}
    s["tool_correct"] = len(calls) == 1 and calls[0]["name"] == gold["name"]
    if not s["tool_correct"]:
        s["correct"] = False
        s["args_exact"] = False
        s["args_f1"] = 0.0
        return s

    pred = calls[0]["args"]
    tool = tools_by_name[gold["name"]]
    r = compare_args(pred, gold["args"], tool)

    # 정답과 다르더라도 acceptable_arguments에 있으면 인정한다.
    if not r["exact"] and r["wrong"] and not r["missing"] \
            and not r["hallucinated_param"] and not r["enum_bad"]:
        rescued = [k for k in r["wrong"]
                   if k in acc and norm_scalar(pred[k]) in
                   {norm_scalar(v) for v in acc[k]}]
        if set(rescued) == set(r["wrong"]):
            r = {**r, "exact": True, "f1": 1.0, "wrong": [],
                 "accepted_alternative": rescued}

    s.update({"args_exact": r["exact"], "args_f1": r["f1"],
              "missing": r["missing"], "wrong": r["wrong"],
              "hallucinated_param": r["hallucinated_param"],
              "extra_optional": r["extra_optional"], "enum_bad": r["enum_bad"]})
    if r.get("accepted_alternative"):
        s["accepted_alternative"] = r["accepted_alternative"]
    s["correct"] = bool(r["exact"]) and not s["hallucinated_tool"]
    return s


def score_fc(rec: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    exp = case["expect"]
    action = exp["action"]
    calls = rec.get("tool_calls") or []
    text = rec.get("text") or ""
    tools_by_name = {t["function"]["name"]: t for t in case["tools"]}
    valid_names = set(tools_by_name)

    s: dict[str, Any] = {
        "called": len(calls) > 0,
        "n_calls": len(calls),
        "expected_action": action,
        "hallucinated_tool": any(c["name"] not in valid_names for c in calls),
        "args_unparseable": any(not c["args_parse_ok"] for c in calls),
    }

    if action == "no_call":
        # 무관 케이스: 안 부르는 게 정답.
        s["correct"] = len(calls) == 0
        s["policy"] = "abstained" if not calls else "spurious_call"
        return s

    if action == "ask":
        # 정보 부족: 되물어야 한다. 부르면 지어낸 것.
        asked = (not calls) and bool(ASK_PAT.search(text))
        s["correct"] = asked
        s["policy"] = ("asked" if asked else ("fabricated" if calls else "silent_giveup"))
        return s

    # action == "call"
    gold = exp["calls"]
    if not calls:
        # 부르지 않았다 — 되물었는지(정직) 그냥 답했는지(실패) 구분해서 기록한다.
        s["correct"] = False
        s["policy"] = "asked_when_should_call" if ASK_PAT.search(text) else "no_call"
        s["name_correct"] = False
        s["arg_f1"] = 0.0
        return s

    s["policy"] = "called"
    # 순서 무관 1:1 매칭. 같은 이름끼리 묶고 인자로 매칭한다.
    gold_names = Counter(g["name"] for g in gold)
    pred_names = Counter(c["name"] for c in calls)
    s["name_correct"] = gold_names == pred_names
    s["call_count_correct"] = len(calls) == len(gold)

    # 순서 무관 1:1 매칭. gold 각각에 대해 아직 안 쓰인 pred 중 f1이 최대인 것을 붙인다.
    used: set[int] = set()
    f1s: list[float] = []
    exact = 0
    detail: list[dict[str, Any]] = []
    for g in gold:
        tool = tools_by_name.get(g["name"])
        best_i: int | None = None
        best: dict[str, Any] | None = None
        for i, c in enumerate(calls):
            if i in used or c["name"] != g["name"] or tool is None:
                continue
            cmp = compare_args(c["args"], g["args"], tool)
            if best is None or cmp["f1"] > best["f1"]:
                best_i, best = i, cmp
        if best is None:
            f1s.append(0.0)
            detail.append({"name": g["name"], "matched": False})
            continue
        used.add(best_i)
        f1s.append(best["f1"])
        exact += 1 if best["exact"] else 0
        detail.append({"name": g["name"], **best})

    s["arg_f1"] = sum(f1s) / len(f1s) if f1s else 0.0
    s["arg_exact"] = exact == len(gold) and len(calls) == len(gold)
    s["extra_optional_args"] = sum(len(d.get("extra_optional", [])) for d in detail)
    s["hallucinated_param"] = any(d.get("hallucinated_param") for d in detail)
    s["arg_detail"] = detail
    s["correct"] = bool(s["name_correct"] and s["arg_exact"])
    return s


# ── SO 채점 ──────────────────────────────────────────────────────────────────

def strip_fences(t: str) -> str:
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        return m.group(1).strip()
    i, j = t.find("{"), t.rfind("}")
    return t[i : j + 1] if i != -1 and j > i else t


def score_so(rec: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    text = rec.get("text") or ""
    schema = case["schema"]
    gold = case["expect"]
    s: dict[str, Any] = {}

    # 1차 시도(수리 없이)와 코드펜스 제거 후를 나눠서 본다 — IFEval의 strict/loose.
    try:
        pred = json.loads(text)
        s["parse_strict"] = True
    except Exception:
        s["parse_strict"] = False
        try:
            pred = json.loads(strip_fences(text))
        except Exception:
            return {**s, "parse_loose": False, "schema_ok": False, "correct": False,
                    "field_f1": 0.0, "extra_keys": 0, "missing_required": 0,
                    "enum_violation": False, "type_mismatch": 0}
    s["parse_loose"] = True

    if jsonschema is not None:
        try:
            jsonschema.validate(pred, schema)
            s["schema_ok"] = True
        except Exception:
            s["schema_ok"] = False
    else:
        s["schema_ok"] = None

    props = schema.get("properties", {})
    req = schema.get("required", [])
    pred_keys = set(pred) if isinstance(pred, dict) else set()
    s["extra_keys"] = len(pred_keys - set(props))
    s["missing_required"] = len([k for k in req if k not in pred_keys])

    # enum 위반
    viol = False
    for k, spec in props.items():
        if "enum" in spec and k in pred_keys and pred[k] not in spec["enum"]:
            viol = True
    s["enum_violation"] = viol

    # 타입 충실도 — "123" vs 123, null vs 누락 구분
    tm = 0
    for k, spec in props.items():
        if k not in pred_keys or k not in gold:
            continue
        want = spec.get("type")
        want = want if isinstance(want, list) else [want]
        got = pred[k]
        ok = (
            (got is None and "null" in want)
            or (isinstance(got, bool) and "boolean" in want)
            or (isinstance(got, int) and not isinstance(got, bool) and ("integer" in want or "number" in want))
            or (isinstance(got, float) and "number" in want)
            or (isinstance(got, str) and "string" in want)
            or (isinstance(got, dict) and "object" in want)
            or (isinstance(got, list) and "array" in want)
        )
        if not ok:
            tm += 1
    s["type_mismatch"] = tm

    # 값 정확도. ambiguous 케이스는 note에 명시된 필드만 본다.
    graded_gold = dict(gold)
    if case.get("ambiguous") and case["category"] == "SO-3":
        graded_gold = {"category": gold["category"]}
    graded_pred = {k: pred.get(k) for k in graded_gold} if isinstance(pred, dict) else {}

    _, _, f1 = field_prf(graded_pred, graded_gold)
    s["field_f1"] = f1
    s["correct"] = bool(s["schema_ok"] is not False and f1 == 1.0)

    # SO-9 전용: 마스킹이 자릿수를 유출했는가 (부분 정답 없음)
    if case["category"] == "SO-9" and "rrn_masked" in gold:
        got = pred.get("rrn_masked", "") if isinstance(pred, dict) else ""
        m = re.match(r"^(\d{6})-(.*)$", str(got))
        s["rrn_leaked_digits"] = (
            len(re.findall(r"\d", m.group(2))) if m else None
        )
    return s


def score_ob(rec: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    """OrchestrationBench 파생 계획 태스크.

    에이전트 집합 F1(카카오의 `fn_name_f1` 대응) + 워크플로 타입 정확도.
    집합으로 보는 이유: 같은 에이전트를 몇 번 부르는지는 시나리오마다 다르고
    순서도 계획마다 정당한 변형이 있어서, 순서·중복까지 강제하면 채점 노이즈가 커진다.
    """
    text = rec.get("text") or ""
    gold = case["expect"]
    s: dict[str, Any] = {}
    try:
        pred = json.loads(text)
        s["parse_strict"] = True
    except Exception:
        s["parse_strict"] = False
        try:
            pred = json.loads(strip_fences(text))
        except Exception:
            return {**s, "parse_loose": False, "correct": False,
                    "agent_f1": 0.0, "type_correct": False}
    s["parse_loose"] = True

    P = {a for a in (pred.get("agents") or []) if isinstance(a, str)}
    G = set(gold["agents"])
    inter = len(P & G)
    prec = inter / len(P) if P else 0.0
    rec_ = inter / len(G) if G else 0.0
    f1 = 2 * prec * rec_ / (prec + rec_) if (prec + rec_) else 0.0
    s["agent_f1"] = f1
    s["agent_exact"] = P == G
    s["type_correct"] = pred.get("workflow_type") == gold["workflow_type"]
    s["n_pred_agents"] = len(P)
    s["n_gold_agents"] = len(G)
    # correct = 에이전트 집합 완전 일치 AND 타입 일치
    s["correct"] = bool(s["agent_exact"] and s["type_correct"])
    return s


# ── 비용 ─────────────────────────────────────────────────────────────────────

def usd(arm: str, u: dict[str, int]) -> float:
    # opus-adaptive / opus-B / sol-B / glm-B 는 같은 엔드포인트라 단가가 같다.
    p = PRICING[arm.split("-")[0]]
    fresh_in = max(0, u["prompt_tokens"] - u["cache_read_tokens"] - u["cache_write_tokens"])
    # billable_output = completion_tokens. reasoning은 그 안에 이미 포함된다(METHODOLOGY §3).
    return USD_PER_DBU * (
        fresh_in / 1e6 * p["in"]
        + u["cache_read_tokens"] / 1e6 * p["cr"]
        + u["cache_write_tokens"] / 1e6 * p["cw"]
        + u["completion_tokens"] / 1e6 * p["out"]
    )


# ── 집계 ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="results/<run-id> 디렉토리")
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--fcb", action="store_true")
    args = ap.parse_args()
    outdir = pathlib.Path(args.run)

    src = FCB_CASES if args.fcb else (PARITY_CASES if args.parity else CASES)
    cases = {c["id"]: c for c in
             (json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip())}
    recs = [json.loads(l) for l in (outdir / "raw.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]

    scored: list[dict[str, Any]] = []
    for r in recs:
        case = cases[r["case_id"]]
        row = {k: r[k] for k in ("arm", "case_id", "category", "track", "rep", "outcome")}
        row["latency_ms"] = r.get("latency_ms")
        row["usage"] = r.get("usage")
        row["retries"] = r.get("retries", 0)
        if r["outcome"] != "ok":
            row["score"] = None
        else:
            if case["track"] == "OB":
                row["score"] = score_ob(r, case)
            elif case["track"] == "FCB":
                row["score"] = score_fcb(r, case)
            elif case.get("tools"):
                row["score"] = score_fc(r, case)
            else:
                row["score"] = score_so(r, case)
            row["lang"] = case.get("lang")
            row["pair_key"] = case.get("pair_key")
            row["usd"] = usd(r["arm"], r["usage"])
        scored.append(row)

    (outdir / "scored.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in scored), encoding="utf-8")

    arms = sorted({r["arm"] for r in scored})
    report: dict[str, Any] = {"arms": arms}

    def agg(rows: list[dict[str, Any]], key: str) -> tuple[int, int]:
        ok = [r for r in rows if r["score"] is not None]
        return sum(1 for r in ok if r["score"].get(key)), len(ok)

    # 전체 정확도 + Wilson
    print("\n" + "=" * 92)
    print("정확도 (correct) — Wilson 95% CI. 구간이 겹치면 무승부.")
    print("=" * 92)
    print(f"{'category':10s} " + " ".join(f"{a:>26s}" for a in arms))
    overall: dict[str, Any] = {}
    cats = sorted({r["category"] for r in scored})
    for cat in cats + ["ALL"]:
        cells = []
        for a in arms:
            rows = [r for r in scored if r["arm"] == a and (cat == "ALL" or r["category"] == cat)]
            k, n = agg(rows, "correct")
            if n == 0:
                cells.append("n/a")
                continue
            p, lo, hi = wilson(k, n)
            cells.append(f"{p:.3f} [{lo:.2f},{hi:.2f}] {k}/{n}")
            overall.setdefault(cat, {})[a] = {"p": p, "lo": lo, "hi": hi, "k": k, "n": n}
        print(f"{cat:10s} " + " ".join(f"{c:>26s}" for c in cells))
    report["accuracy"] = overall

    # pass^k — 케이스 단위로 n회 중 c회 성공
    print("\n" + "=" * 92)
    print("안정성 — 평균 정확도 vs pass^k (k=전체 반복). 격차가 곧 불안정성.")
    print("=" * 92)
    reps = max(r["rep"] for r in scored)
    print(f"{'arm':6s} {'mean acc':>10s} {'pass^' + str(reps):>10s} {'gap':>8s} "
          f"{'unstable cases':>16s}")
    stab: dict[str, Any] = {}
    for a in arms:
        by_case: dict[str, list[bool]] = defaultdict(list)
        for r in scored:
            if r["arm"] == a and r["score"] is not None:
                by_case[r["case_id"]].append(bool(r["score"].get("correct")))
        means, phk, unstable = [], [], 0
        for cid, res in by_case.items():
            n, c = len(res), sum(res)
            means.append(c / n)
            v = pass_hat_k(n, c, min(reps, n))
            if v is not None:
                phk.append(v)
            if 0 < c < n:
                unstable += 1
        m = sum(means) / len(means) if means else 0
        pk = sum(phk) / len(phk) if phk else 0
        print(f"{a:6s} {m:10.3f} {pk:10.3f} {m - pk:8.3f} {unstable:>10d}/{len(by_case)}")
        stab[a] = {"mean_acc": m, f"pass^{reps}": pk, "gap": m - pk,
                   "unstable_cases": unstable, "n_cases": len(by_case)}
    report["stability"] = stab

    # 정책 분해 (FC)
    print("\n" + "=" * 92)
    print("정책 분해 (FC) — 같은 '미호출'도 되물음/포기가 다르다")
    print("=" * 92)
    pol: dict[str, Any] = {}
    print(f"{'arm':6s} " + " ".join(f"{p:>20s}" for p in
          ["abstained(FC-5)", "asked(FC-6)", "fabricated(FC-6)", "silent_drop(FC-X)"]))
    for a in arms:
        f5 = [r for r in scored if r["arm"] == a and r["category"] == "FC-5" and r["score"]]
        f6 = [r for r in scored if r["arm"] == a and r["category"] == "FC-6"
              and r["score"] and r["score"]["expected_action"] == "ask"]
        fx = [r for r in scored if r["arm"] == a and r["category"] == "FC-X"]
        fx_ok = [r for r in fx if r["score"]]
        ab = sum(1 for r in f5 if r["score"].get("policy") == "abstained")
        ask = sum(1 for r in f6 if r["score"].get("policy") == "asked")
        fab = sum(1 for r in f6 if r["score"].get("policy") == "fabricated")
        drop = sum(1 for r in fx_ok if not r["score"].get("called"))
        rej = sum(1 for r in fx if r["outcome"] == "gateway_reject")
        cells = [f"{ab}/{len(f5)}", f"{ask}/{len(f6)}", f"{fab}/{len(f6)}",
                 f"{drop}/{len(fx_ok)}" + (f" (+{rej} 400)" if rej else "")]
        print(f"{a:6s} " + " ".join(f"{c:>20s}" for c in cells))
        pol[a] = {"abstained": [ab, len(f5)], "asked": [ask, len(f6)],
                  "fabricated": [fab, len(f6)], "silent_drop": [drop, len(fx_ok)],
                  "gateway_reject": rej}
    report["policy"] = pol

    # SO-9 규제 — 마스킹 자릿수 유출
    leak = {a: Counter() for a in arms}
    for r in scored:
        if r["category"] == "SO-9" and r["score"] and r["score"].get("rrn_leaked_digits") is not None:
            leak[r["arm"]][r["score"]["rrn_leaked_digits"]] += 1
    if any(leak.values()):
        print("\n" + "=" * 92)
        print("SO-9 주민등록번호 마스킹 — 뒤 7자리 중 남은 숫자 개수 (0이 정답)")
        print("=" * 92)
        for a in arms:
            tot = sum(leak[a].values())
            good = leak[a].get(0, 0)
            detail = ", ".join(f"{k}자리 유출 ×{v}" for k, v in sorted(leak[a].items()) if k)
            print(f"  {a:6s} 정상 {good}/{tot}" + (f"   ⚠️ {detail}" if detail else ""))
        report["rrn_leak"] = {a: dict(leak[a]) for a in arms}

    # 시간·비용
    print("\n" + "=" * 92)
    print("시간·비용 (p99는 표본 부족으로 생략 — 10/(1-q) 규칙)")
    print("=" * 92)
    print(f"{'arm':6s} {'p50 ms':>9s} {'p90 ms':>9s} {'calls':>7s} {'USD':>10s} "
          f"{'USD/정답':>11s} {'prompt tok':>11s} {'compl tok':>10s} {'reason tok':>11s}")
    cost: dict[str, Any] = {}
    for a in arms:
        rows = [r for r in scored if r["arm"] == a and r["outcome"] == "ok"]
        lat = sorted(r["latency_ms"] for r in rows if r["latency_ms"])
        tot_usd = sum(r.get("usd", 0) for r in rows)
        ncorrect = sum(1 for r in rows if r["score"] and r["score"].get("correct"))
        pt = sum(r["usage"]["prompt_tokens"] for r in rows)
        ct = sum(r["usage"]["completion_tokens"] for r in rows)
        rt = sum(r["usage"]["reasoning_tokens"] for r in rows)
        p50 = lat[len(lat) // 2] if lat else 0
        p90 = lat[int(len(lat) * 0.9)] if lat else 0
        upc = tot_usd / ncorrect if ncorrect else float("nan")
        print(f"{a:6s} {p50:9.0f} {p90:9.0f} {len(rows):7d} {tot_usd:10.4f} {upc:11.5f} "
              f"{pt:11d} {ct:10d} {rt:11d}")
        cost[a] = {"p50_ms": p50, "p90_ms": p90, "calls": len(rows), "usd": tot_usd,
                   "usd_per_correct": upc, "prompt_tokens": pt, "completion_tokens": ct,
                   "reasoning_tokens": rt, "n_correct": ncorrect}
    report["cost"] = cost

    # 실패 분류
    print("\n" + "=" * 92)
    print("실패 분류 — 인프라/게이트웨이/형식을 섞으면 안 된다")
    print("=" * 92)
    fails: dict[str, Any] = {}
    for a in arms:
        rows = [r for r in scored if r["arm"] == a]
        oc = Counter(r["outcome"] for r in rows)
        ok = [r for r in rows if r["score"]]
        fmt = sum(1 for r in ok if r["track"] == "SO" and not r["score"].get("parse_loose"))
        halluc = sum(1 for r in ok if r["score"].get("hallucinated_tool"))
        badargs = sum(1 for r in ok if r["score"].get("args_unparseable"))
        print(f"  {a:6s} {dict(oc)}   JSON파싱실패={fmt} 환각툴={halluc} 인자파싱실패={badargs} "
              f"재시도={sum(r['retries'] for r in rows)}")
        fails[a] = {"outcomes": dict(oc), "json_parse_fail": fmt,
                    "hallucinated_tool": halluc, "args_unparseable": badargs}
    report["failures"] = fails

    (outdir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {outdir}/scored.jsonl, {outdir}/report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
