"""언어 패리티 케이스 생성 — 두 갈래.

## 배경: 초기 가정이 또 틀렸다

사전조사가 OrchestrationBench를 *"222 scenarios in Korean and 222 **parallel**
scenarios in English … a ready-made **paired** corpus"*라고 적었다. **틀렸다.**

실제로 확인해 보니:

- EN은 **219개**다(222 아님). KO만 220~222가 있다.
- 그리고 **한/영이 서로 번역이 아니다.** 같은 번호가 전혀 다른 시나리오다
  (#1: KO 퀵배송 / EN 택시). gold 에이전트 집합이 일치하는 건 **27/219**로,
  단일 에이전트 시나리오에서 우연히 겹치는 수준이다.
- README도 *"bilingual"*이라고만 하고 parallel이라 하지 않는다. 자체 리더보드도
  KO와 EN을 **따로** 보고한다.

`bilingual ≠ parallel`. 이 구분을 놓치면 "쌍 비교"라고 부르면서 실제로는
서로 다른 문제를 푼 점수를 비교하게 된다.

## 그래서 두 갈래로 나눈다

**(A) OB — 모집단 수준 비교** (`track: "OB"`)
   OrchestrationBench 219 KO vs 219 EN. 쌍이 아니므로 **항목별 일치(P_a)를 못 낸다.**
   시나리오 난이도 차이가 교란이다. 대신 카카오 자체 리더보드와 같은 방식이고
   **실제 한국어 멀티에이전트 계획 태스크**라는 외부 타당성이 있다.

**(B) PAIR — 진짜 쌍 비교** (`track: "PAIR"`)
   내 66개 FC/SO 케이스를 영어로 옮긴다. 의미가 대응됨을 내가 보장하므로
   **KP·P_a·P_c를 전부 낼 수 있다.** 대가는 외부 검증이 없다는 것.

둘을 같이 보고해야 정직하다. (A)는 외부 corpus·약한 통제, (B)는 자작·강한 통제.

## OB 파생 태스크 정의

OrchestrationBench 전체 하네스(멀티턴 에이전트 루프 + DAG 평가)는 돌리지 않는다.
대신 **첫 워크플로 계획 스텝**만 단일턴으로 떼어낸다:

  입력: 시스템 컨텍스트 + 사용자 첫 발화 + 17개 에이전트 목록(id·설명)
  출력: 호출할 에이전트 목록 + 워크플로 타입(independent/dependent)
  정답: 시나리오 YAML의 실제 첫 `main` 계획 스텝

채점은 **에이전트 집합 F1**(카카오의 `fn_name_f1` 대응)과 **타입 정확도**.
→ 카카오 리더보드 수치와 **직접 비교 불가**다. 파생 서브태스크임을 명시한다.

사용:
    python cases/build_parity_cases.py --ob-root /tmp/OrchestrationBench
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import re
from typing import Any

import yaml

OUT = pathlib.Path(__file__).parent / "parity_cases.jsonl"

# ── (A) OrchestrationBench ────────────────────────────────────────────────────

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "agents": {
            "type": "array",
            "items": {"type": "string"},
            "description": "호출할 에이전트 id 목록. 순서대로.",
        },
        "workflow_type": {"type": "string", "enum": ["independent", "dependent"]},
    },
    "required": ["agents", "workflow_type"],
    "additionalProperties": False,
}

PROMPT_KO = """당신은 멀티에이전트 오케스트레이터입니다.

## 컨텍스트
{system}

## 사용 가능한 에이전트
{agents}

## 사용자 요청
{user}

이 요청을 처리하기 위해 호출해야 할 에이전트를 정하세요.
- `agents`: 호출할 에이전트 id 목록 (필요한 것만, 순서대로)
- `workflow_type`: 에이전트들을 독립적으로 병렬 실행할 수 있으면 `independent`,
  앞 에이전트의 결과가 뒤 에이전트에 필요하면 `dependent`"""

PROMPT_EN = """You are a multi-agent orchestrator.

## Context
{system}

## Available agents
{agents}

## User request
{user}

Decide which agents to invoke to handle this request.
- `agents`: list of agent ids to invoke (only what's needed, in order)
- `workflow_type`: `independent` if the agents can run in parallel,
  `dependent` if a later agent needs an earlier agent's result"""


def load_agent_catalog(root: str, lang: str) -> str:
    """에이전트 카드 → id + 설명 목록 문자열."""
    lines = []
    for f in sorted(glob.glob(f"{root}/data/{lang}/multiagent_cards/*.json")):
        card = json.load(open(f, encoding="utf-8"))["agent_card"]
        desc = (card.get("description") or "").strip().replace("\n", " ")
        lines.append(f"- {card['agent_id']}: {desc[:150]}")
    return "\n".join(lines)


def first_plan(path: str) -> tuple[str | None, str | None, dict | None]:
    """시나리오에서 (시스템 컨텍스트, 첫 사용자 발화, 첫 워크플로 계획)."""
    doc = yaml.safe_load(open(path, encoding="utf-8"))
    system = user = None
    for s in doc.get("steps", []):
        content = (s.get("data") or {}).get("content")
        if s.get("agent") == "system" and system is None:
            system = content
        if s.get("agent") == "user" and user is None:
            user = content
        if (s.get("agent") == "main" and isinstance(content, str)
                and content.strip().startswith("workflow_")):
            try:
                return system, user, yaml.safe_load(content)
            except Exception:
                return system, user, None
    return system, user, None


def build_ob(root: str) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for lang, prompt_tmpl in (("KO", PROMPT_KO), ("EN", PROMPT_EN)):
        catalog = load_agent_catalog(root, lang)
        files = sorted(glob.glob(f"{root}/data/{lang}/scenario_data/*.yaml"),
                       key=lambda p: int(re.findall(r"(\d+)", os.path.basename(p))[0]))
        for f in files:
            n = int(re.findall(r"(\d+)", os.path.basename(f))[0])
            system, user, plan = first_plan(f)
            if not (user and plan):
                continue
            agents, types = [], []
            for wf in plan.values():
                for st in wf.get("steps") or []:
                    if st.get("name"):
                        agents.append(st["name"])
                types.append(wf.get("type"))
            if not agents:
                continue
            # 워크플로가 여러 개면 하나라도 dependent이면 dependent로 본다.
            wtype = "dependent" if "dependent" in types else "independent"
            cases.append({
                "id": f"OB-{lang}-{n:03d}",
                "track": "OB",
                "category": f"OB-{lang}",
                "lang": lang,
                "pair_key": None,  # 쌍이 아니다 — 모집단 비교만 가능
                "system": None,
                "prompt": prompt_tmpl.format(
                    system=(system or "").strip(), agents=catalog, user=str(user).strip()),
                "schema": PLAN_SCHEMA,
                "expect": {"agents": agents, "workflow_type": wtype},
                "note": "OrchestrationBench 파생 단일턴 계획 서브태스크. "
                        "카카오 리더보드 수치와 직접 비교 불가.",
                "ambiguous": False,
            })
    return cases


# ── (B) 내 66 케이스의 한/영 쌍 ───────────────────────────────────────────────
# 의미가 대응되도록 직접 옮긴다. **정답은 그대로 두고 입력 언어만 바꾼다.**
# 한국어 고유 항목(만/억 단위, 조사, 주민번호)은 영어에 대응이 없으므로 제외한다 —
# 억지로 옮기면 "같은 문제"가 아니게 되어 패리티의 전제가 깨진다.

EN_PROMPTS: dict[str, str] = {
    # FC-1
    "FC1-01": "Tell me the weather in Seoul.",
    "FC1-02": "Look up the stock price of Samsung Electronics.",
    "FC1-03": "Tell me the weather in Busan in Fahrenheit.",
    "FC1-04": "Find flights from Seoul to Jeju on 2026-08-06 for 1 passenger.",
    "FC1-05": "Order 100 ballpoint pens at 500 KRW each.",
    # FC-2
    "FC2-01": "What's Naver's stock price right now?",
    "FC2-02": "Show me Kakao's stock price trend over the last 30 days.",
    "FC2-03": "Tell me about SK Hynix's dividend.",
    "FC2-04": "Find me some news articles about LG Chem.",
    "FC2-05": "What is Hyundai Motor stock trading at right now?",
    # FC-3
    "FC3-01": "Tell me the weather in Seoul and Busan, each of them.",
    "FC3-02": "Look up the stock prices of both Samsung Electronics and SK Hynix.",
    "FC3-03": "Tell me the stock prices of Naver and Kakao.",
    "FC3-04": "Tell me the weather in Daegu, Gwangju, and Incheon, all of them.",
    # FC-5
    "FC5-01": "What is the 10th term of the Fibonacci sequence?",
    "FC5-02": "Tell me how to make a delicious kimchi stew.",
    "FC5-03": "What's the difference between a list and a tuple in Python?",
    "FC5-04": "Explain why diversification matters when investing in stocks.",
    "FC5-05": "How much is a transfer fee usually?",
    # FC-6
    "FC6-01": "Find me some flights.",
    "FC6-02": "Find flights departing from Seoul.",
    "FC6-03": "Find flights from Seoul to Jeju next Friday for two people.",
    "FC6-04": "Request time off for me.",
    "FC6-05": "Request annual leave from next Monday through Wednesday.",
    # FC-7
    "FC7-01": "I'm not feeling well, I'll take one day of sick leave tomorrow.",
    "FC7-02": "I need to take August 20 through 22 off to care for my father.",
    "FC7-03": "I'll take unpaid leave from 9/1 to 9/5.",
    "FC7-04": "Jeju weather in Celsius.",
    "FC7-05": "Five boxes of A4 paper, twenty-three thousand won per box. Place the order.",
    # FC-X
    "FCX-01": "Tell me the weather in Seoul.",
    "FCX-02": "Look up the stock price of Samsung Electronics.",
    "FCX-03": "Tell me the weather in Busan.",
    # SO-1
    "SO1-01": "Extract the information from: 'Kim Chulsoo is 32 years old and lives in Seoul.'",
    "SO1-02": "Extract the information from: 'Lee Younghee (28), resides in Busan.'",
    "SO1-03": "Extract the information from: 'Mr. Park Minsoo is forty-five years old and lives in Daejeon.'",
    "SO1-04": "Extract the information from: 'Choi Jiwoo, age 51, residence Gwangju'",
    "SO1-05": "Extract the information from: 'Jung Hanbyul turned thirty this year and lives in Incheon.'",
    # SO-3
    "SO3-01": "Classify this customer inquiry: 'It's been a week since I ordered and it still hasn't arrived.'",
    "SO3-02": "Classify this customer inquiry: 'The product I received has a cracked screen. Please exchange it immediately.'",
    "SO3-03": "Classify this customer inquiry: 'I just changed my mind — can I get my money back?'",
    "SO3-04": "Classify this customer inquiry: 'What are your business hours?'",
    # SO-5
    "SO5-01": "Extract the contact from 'Hong Gildong, 010-1234-5678'. Use null for missing values.",
    "SO5-02": "Extract the contact from 'Kim Seoyeon, seoyeon@example.com'. Use null for missing values.",
    "SO5-03": "Extract the contact from 'Park Doyoon'. Use null for missing values.",
    "SO5-04": "Extract the contact from 'Lee Hajun, 010-9876-5432, hajun@corp.co.kr'. Use null for missing values.",
}

# 영어에서 정답이 달라지는 케이스는 여기서 덮어쓴다.
# (한국어 표기 사명 → 영어에서는 영문명이 자연스럽다)
EN_EXPECT_OVERRIDE: dict[str, Any] = {
    "FC1-02": {"action": "call", "calls": [{"name": "get_stock_price",
                                            "args": {"company": "Samsung Electronics"}}]},
    "FC2-01": {"action": "call", "calls": [{"name": "get_stock_price", "args": {"company": "Naver"}}]},
    "FC2-02": {"action": "call", "calls": [{"name": "get_stock_history",
                                            "args": {"company": "Kakao", "days": 30}}]},
    "FC2-03": {"action": "call", "calls": [{"name": "get_dividend_info",
                                            "args": {"company": "SK Hynix"}}]},
    "FC2-04": {"action": "call", "calls": [{"name": "get_stock_news", "args": {"company": "LG Chem"}}]},
    "FC2-05": {"action": "call", "calls": [{"name": "get_stock_price",
                                            "args": {"company": "Hyundai Motor"}}]},
    "FC3-02": {"action": "call", "calls": [
        {"name": "get_stock_price", "args": {"company": "Samsung Electronics"}},
        {"name": "get_stock_price", "args": {"company": "SK Hynix"}}]},
    "FC3-03": {"action": "call", "calls": [
        {"name": "get_stock_price", "args": {"company": "Naver"}},
        {"name": "get_stock_price", "args": {"company": "Kakao"}}]},
    "FC1-01": {"action": "call", "calls": [{"name": "get_weather", "args": {"city": "Seoul"}}]},
    "FC1-03": {"action": "call", "calls": [{"name": "get_weather",
                                            "args": {"city": "Busan", "unit": "fahrenheit"}}]},
    "FC1-04": {"action": "call", "calls": [{"name": "search_flights",
                                            "args": {"origin": "Seoul", "destination": "Jeju",
                                                     "date": "2026-08-06", "passengers": 1}}]},
    "FC1-05": {"action": "call", "calls": [{"name": "create_order",
                                            "args": {"item": "ballpoint pen", "quantity": 100,
                                                     "unit_price_krw": 500}}]},
    "FC3-01": {"action": "call", "calls": [{"name": "get_weather", "args": {"city": "Seoul"}},
                                           {"name": "get_weather", "args": {"city": "Busan"}}]},
    "FC3-04": {"action": "call", "calls": [{"name": "get_weather", "args": {"city": "Daegu"}},
                                           {"name": "get_weather", "args": {"city": "Gwangju"}},
                                           {"name": "get_weather", "args": {"city": "Incheon"}}]},
    "FC6-03": {"action": "call", "calls": [{"name": "search_flights",
                                            "args": {"origin": "Seoul", "destination": "Jeju",
                                                     "date": "2026-08-14", "passengers": 2}}]},
    "FC7-04": {"action": "call", "calls": [{"name": "get_weather",
                                            "args": {"city": "Jeju", "unit": "celsius"}}]},
    "FC7-05": {"action": "call", "calls": [{"name": "create_order",
                                            "args": {"item": "A4 paper", "quantity": 5,
                                                     "unit_price_krw": 23000}}]},
    "FCX-01": {"action": "call", "calls": [{"name": "get_weather", "args": {"city": "Seoul"}}]},
    "FCX-02": {"action": "call", "calls": [{"name": "get_stock_price",
                                            "args": {"company": "Samsung Electronics"}}]},
    "FCX-03": {"action": "call", "calls": [{"name": "get_weather", "args": {"city": "Busan"}}]},
    "SO1-01": {"name": "Kim Chulsoo", "age": 32, "city": "Seoul"},
    "SO1-02": {"name": "Lee Younghee", "age": 28, "city": "Busan"},
    "SO1-03": {"name": "Park Minsoo", "age": 45, "city": "Daejeon"},
    "SO1-04": {"name": "Choi Jiwoo", "age": 51, "city": "Gwangju"},
    "SO1-05": {"name": "Jung Hanbyul", "age": 30, "city": "Incheon"},
    "SO5-01": {"name": "Hong Gildong", "phone": "010-1234-5678", "email": None},
    "SO5-02": {"name": "Kim Seoyeon", "phone": None, "email": "seoyeon@example.com"},
    "SO5-03": {"name": "Park Doyoon", "phone": None, "email": None},
    "SO5-04": {"name": "Lee Hajun", "phone": "010-9876-5432", "email": "hajun@corp.co.kr"},
}

EN_SYSTEM = "Today is Thursday, 2026-08-06."


def build_pairs(base_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """한국어 원본 + 대응 영어. 정답이 언어에 의존하는 것은 override로 교정."""
    out: list[dict[str, Any]] = []
    by_id = {c["id"]: c for c in base_cases}
    for cid, en_prompt in EN_PROMPTS.items():
        ko = by_id.get(cid)
        if ko is None:
            raise SystemExit(f"알 수 없는 case id: {cid}")
        # 한국어 쪽
        k = dict(ko)
        k.update({"id": f"PAIR-KO-{cid}", "track": "PAIR", "lang": "KO",
                  "pair_key": cid, "category": f"PAIR-{ko['category']}"})
        out.append(k)
        # 영어 쪽
        e = dict(ko)
        e.update({"id": f"PAIR-EN-{cid}", "track": "PAIR", "lang": "EN",
                  "pair_key": cid, "category": f"PAIR-{ko['category']}",
                  "prompt": en_prompt,
                  "system": EN_SYSTEM if ko.get("system") else None})
        if cid in EN_EXPECT_OVERRIDE:
            e["expect"] = EN_EXPECT_OVERRIDE[cid]
        out.append(e)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ob-root", default="/tmp/OrchestrationBench")
    args = ap.parse_args()

    base = [json.loads(l) for l in
            (pathlib.Path(__file__).parent / "cases.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]

    ob = build_ob(args.ob_root)
    pairs = build_pairs(base)
    cases = ob + pairs

    with OUT.open("w", encoding="utf-8") as fh:
        for c in cases:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    from collections import Counter

    print(f"→ {OUT}  ({len(cases)} cases)")
    print(f"  OB   : {Counter(c['lang'] for c in ob)}")
    print(f"  PAIR : {Counter(c['lang'] for c in pairs)}  "
          f"({len(pairs)//2} 쌍, 원본 66개 중 한국어 고유 항목 제외)")


if __name__ == "__main__":
    main()
