#!/usr/bin/env python3
"""FunctionChat-Bench CallDecision(606건) → 이 저장소의 케이스 형식으로 변환.

원본: https://github.com/kakao/FunctionChat-Bench (Apache-2.0)
데이터는 재배포하지 않는다. 실행 시 저장소를 직접 받아 변환한다.

    git clone --depth 1 https://github.com/kakao/FunctionChat-Bench.git /tmp/FunctionChat-Bench
    python3 cases/build_fcb_cases.py --fcb-root /tmp/FunctionChat-Bench

원본 4개 카테고리를 그대로 유지한다.

| category   | 건수 | 기대 동작                        |
|------------|-----:|----------------------------------|
| CALL       |  100 | 정확한 도구를 정확한 인자로 호출 |
| REJECT     |  100 | 도구가 부적합 → 호출하지 않음    |
| SLOT-all   |  100 | 필수 인자 전부 없음 → 되물음     |
| SLOT-some  |  306 | 필수 인자 일부 없음 → 되물음     |

채점 범위에 관한 주의사항:

원본 벤치마크는 REJECT / SLOT 카테고리를 LLM 채점자(rubric_*.txt)로 평가한다.
"되물음 문장이 올바른 슬롯을 묻고 있는가"까지 보기 때문이다.
이 저장소는 LLM 채점자를 쓰지 않으므로 **"도구 호출을 억제했는가"만 기계 채점**한다.
따라서 이 실험의 REJECT / SLOT 점수는 원본 리더보드 수치보다 관대하며,
카카오 리더보드와 직접 비교할 수 없다. CALL 카테고리는 정답 도구·인자가 명시돼 있어
원본과 같은 기준으로 기계 채점한다.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import unicodedata
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "fcb_cases.jsonl"

SRC_NAME = "FunctionChat-CallDecision.jsonl"

# 원본 category → 기대 동작
EXPECT_ACTION = {
    "CALL": "call",
    "REJECT": "no_call",
    "SLOT-all": "no_call",
    "SLOT-some": "no_call",
}


def nfc(o: Any) -> Any:
    """NFD 한글은 토큰이 8.5배가 된다. 전 문자열을 NFC로 정규화한다."""
    if isinstance(o, str):
        return unicodedata.normalize("NFC", o)
    if isinstance(o, list):
        return [nfc(x) for x in o]
    if isinstance(o, dict):
        return {nfc(k): nfc(v) for k, v in o.items()}
    return o


def parse_acceptable(raw: Any) -> dict[str, list[Any]]:
    """acceptable_arguments — 인자별 허용 대체값 목록.

    원본에서 JSON 문자열로 들어오기도 하고 dict로 들어오기도 한다.
    `마이크로소프트` 정답에 대해 `Microsoft`, `MSFT(Microsoft Corporation)`를
    허용하는 식이다. 고유명사 정규화 방식 차이를 오답으로 세지 않기 위한 장치다.
    """
    if not raw:
        return {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return {}
    if not isinstance(raw, dict):
        return {}
    return {k: (v if isinstance(v, list) else [v]) for k, v in raw.items()}


def normalize_tools(tools: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """무인자 도구의 `parameters: {}`를 `{"type":"object","properties":{}}`로 채운다.

    원본 데이터의 `getCurrentKoreaTime` / `getCurrentUTCTime` 두 도구는
    `parameters`가 빈 객체다. OpenAI 형식으로는 유효하지만 게이트웨이 동작이 갈린다.

      - GPT-5.6-sol, GLM 5.2 → 그대로 수용
      - Opus 5 → HTTP 400 `tools.N.custom.input_schema.type: Field required`

    이 차이는 모델의 도구 호출 능력이 아니라 스키마 검증 엄격도의 차이다.
    빈 스키마와 `{"type":"object","properties":{}}`는 의미가 같으므로 정규화해
    측정에서 이 요인을 제거한다. 차이 자체는 문서에 발견 사항으로 남긴다.
    영향 범위는 606건 중 73건이다.
    """
    out, fixed = [], 0
    for t in tools:
        fn = dict(t["function"])
        p = fn.get("parameters")
        if not isinstance(p, dict) or "type" not in p:
            fn["parameters"] = {"type": "object",
                                "properties": (p or {}).get("properties", {})}
            fixed += 1
        out.append({**t, "function": fn})
    return out, fixed


def gold_calls(gt: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for tc in gt.get("tool_calls") or []:
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        out.append({"name": fn.get("name"), "args": args or {}})
    return out


def convert(row: dict[str, Any], idx: int) -> tuple[dict[str, Any], int]:
    cat = row["category"]
    action = EXPECT_ACTION[cat]
    msgs = nfc(row["input_messages"])

    system = None
    if msgs and msgs[0]["role"] == "system":
        system = msgs[0]["content"]
        msgs = msgs[1:]

    expect: dict[str, Any] = {"action": action}
    if action == "call":
        expect["calls"] = gold_calls(nfc(row["ground_truth"]))
        acc = parse_acceptable(nfc(row.get("acceptable_arguments")))
        if acc:
            expect["acceptable_arguments"] = acc
    else:
        # 되물음 정답 문장. 채점에는 쓰지 않고 오답 분석 시 참고용으로만 보관한다.
        expect["gold_text"] = (nfc(row["ground_truth"]).get("content") or "")

    tools, n_fixed = normalize_tools(nfc(row["input_tools"]))

    return {
        "id": f"FCB-{cat}-{idx:03d}",
        "track": "FCB",
        "category": f"FCB-{cat}",
        "system": system,
        # 멀티턴(최대 3 user 턴). runner가 messages를 우선 사용한다.
        "messages": msgs,
        "prompt": msgs[-1]["content"],
        "tools": tools,
        "expect": expect,
        "note": f"FunctionChat-Bench CallDecision serial={row['serial_num']}",
        "ambiguous": False,
        "source": "kakao/FunctionChat-Bench (Apache-2.0)",
    }, n_fixed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fcb-root", required=True,
                    help="git clone한 FunctionChat-Bench 경로")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    src = pathlib.Path(args.fcb_root) / "data" / SRC_NAME
    if not src.exists():
        raise SystemExit(
            f"원본을 찾을 수 없다: {src}\n"
            "  git clone --depth 1 https://github.com/kakao/FunctionChat-Bench.git /tmp/FunctionChat-Bench"
        )

    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]

    cases, counter, total_fixed = [], {}, 0
    for r in rows:
        cat = r["category"]
        counter[cat] = counter.get(cat, 0) + 1
        c, n_fixed = convert(r, counter[cat])
        total_fixed += n_fixed
        cases.append(c)

    out = pathlib.Path(args.out)
    with out.open("w", encoding="utf-8") as f:
        for c in cases:
            # ensure_ascii=False — True면 한글이 이스케이프되어 토큰이 3.15배가 된다.
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"{len(cases)}건 → {out}")
    if total_fixed:
        n_cases = sum(1 for c in cases if "getCurrentKoreaTime" in json.dumps(c["tools"])
                      or "getCurrentUTCTime" in json.dumps(c["tools"]))
        print(f"  무인자 도구 스키마 정규화: 도구 정의 {total_fixed}건 "
              f"(parameters:{{}} → {{type:object}}). Opus 5가 원형을 HTTP 400으로 거부한다.")
    for cat in sorted(counter):
        n_multi = sum(1 for c in cases
                      if c["category"] == f"FCB-{cat}" and len(c["messages"]) > 1)
        n_acc = sum(1 for c in cases
                    if c["category"] == f"FCB-{cat}"
                    and c["expect"].get("acceptable_arguments"))
        print(f"  {cat:10} {counter[cat]:4}건  "
              f"(멀티턴 {n_multi}, 대체정답 보유 {n_acc}, 기대={EXPECT_ACTION[cat]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
