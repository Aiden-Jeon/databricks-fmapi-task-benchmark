"""테스트 케이스 생성 — Function Calling & JSON Output.

설계 근거는 `../METHODOLOGY.md`. 요약하면:

- **교과서적 한국어 함정은 뺀다.** 조사 제거·억/만 단위·NFD·한자는 세 모델 다 통과해
  변별력이 없다(METHODOLOGY §6). 실측으로 갈린 지점에 가중치를 준다.
- **정책과 정확도를 분리해 채점**할 수 있게, 툴을 불러야 하는 케이스와 부르면 안 되는
  케이스와 되물어야 하는 케이스를 별도 카테고리로 둔다.
- **정답이 형식적으로 정의**되게 쓴다. LLM judge 없이 기계 채점 가능해야 한다.
- 모호 판정 여지가 있으면 `note`에 남기고 `ambiguous=True`로 집계에서 분리한다.

기준일은 `TODAY`로 고정한다. 상대 날짜 케이스의 정답이 실행일에 따라 달라지면
재현이 안 되므로, 프롬프트에 시스템 메시지로 오늘 날짜를 명시해 넣는다.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

TODAY = "2026-08-06"  # 목요일. 상대 날짜 정답은 전부 이 기준.
SYSTEM_DATE = f"오늘은 {TODAY} 목요일입니다."

OUT = pathlib.Path(__file__).parent / "cases.jsonl"


# ── 공용 툴 정의 ──────────────────────────────────────────────────────────────

T_WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "특정 도시의 현재 날씨를 조회한다.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "도시 이름"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}

T_FLIGHT = {
    "type": "function",
    "function": {
        "name": "search_flights",
        "description": "항공편을 검색한다.",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "출발 도시"},
                "destination": {"type": "string", "description": "도착 도시"},
                "date": {"type": "string", "description": "출발일. YYYY-MM-DD 형식"},
                "passengers": {"type": "integer", "minimum": 1},
            },
            "required": ["origin", "destination", "date"],
            "additionalProperties": False,
        },
    },
}

T_ORDER = {
    "type": "function",
    "function": {
        "name": "create_order",
        "description": "발주를 생성한다.",
        "parameters": {
            "type": "object",
            "properties": {
                "item": {"type": "string"},
                "quantity": {"type": "integer", "minimum": 1},
                "unit_price_krw": {"type": "integer", "minimum": 0},
            },
            "required": ["item", "quantity", "unit_price_krw"],
            "additionalProperties": False,
        },
    },
}

T_STOCK = {
    "type": "function",
    "function": {
        "name": "get_stock_price",
        "description": "상장 기업의 현재 주가를 조회한다.",
        "parameters": {
            "type": "object",
            "properties": {"company": {"type": "string", "description": "회사명"}},
            "required": ["company"],
            "additionalProperties": False,
        },
    },
}

# FC-2용 근접 distractor — 같은 도메인의 혼동하기 쉬운 함수들
T_STOCK_HISTORY = {
    "type": "function",
    "function": {
        "name": "get_stock_history",
        "description": "상장 기업의 과거 주가 추이를 조회한다.",
        "parameters": {
            "type": "object",
            "properties": {
                "company": {"type": "string"},
                "days": {"type": "integer", "minimum": 1},
            },
            "required": ["company", "days"],
            "additionalProperties": False,
        },
    },
}
T_STOCK_NEWS = {
    "type": "function",
    "function": {
        "name": "get_stock_news",
        "description": "상장 기업 관련 뉴스를 조회한다.",
        "parameters": {
            "type": "object",
            "properties": {"company": {"type": "string"}},
            "required": ["company"],
            "additionalProperties": False,
        },
    },
}
T_STOCK_DIVIDEND = {
    "type": "function",
    "function": {
        "name": "get_dividend_info",
        "description": "상장 기업의 배당 정보를 조회한다.",
        "parameters": {
            "type": "object",
            "properties": {"company": {"type": "string"}},
            "required": ["company"],
            "additionalProperties": False,
        },
    },
}

T_LEAVE = {
    "type": "function",
    "function": {
        "name": "request_leave",
        "description": "휴가를 신청한다.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string", "description": "시작일 YYYY-MM-DD"},
                "end_date": {"type": "string", "description": "종료일 YYYY-MM-DD"},
                "leave_type": {
                    "type": "string",
                    "enum": ["annual", "sick", "family_care", "unpaid"],
                },
            },
            "required": ["start_date", "end_date", "leave_type"],
            "additionalProperties": False,
        },
    },
}

T_TRANSFER = {
    "type": "function",
    "function": {
        "name": "transfer_money",
        "description": "계좌 이체를 실행한다. 되돌릴 수 없다.",
        "parameters": {
            "type": "object",
            "properties": {
                "to_account": {"type": "string"},
                "amount_krw": {"type": "integer", "minimum": 1},
            },
            "required": ["to_account", "amount_krw"],
            "additionalProperties": False,
        },
    },
}


def fc(
    cid: str,
    category: str,
    prompt: str,
    tools: list[dict[str, Any]],
    expect: dict[str, Any],
    *,
    system: str | None = None,
    note: str = "",
    ambiguous: bool = False,
) -> dict[str, Any]:
    """Function calling 케이스.

    expect 형태:
      {"action": "call",   "calls": [{"name": ..., "args": {...}}]}
      {"action": "no_call"}                       # 툴이 무관 → 부르면 안 됨
      {"action": "ask"}                           # 정보 부족 → 되물어야 함
    """
    return {
        "id": cid,
        "track": "FC",
        "category": category,
        "system": system,
        "prompt": prompt,
        "tools": tools,
        "expect": expect,
        "note": note,
        "ambiguous": ambiguous,
    }


def so(
    cid: str,
    category: str,
    prompt: str,
    schema: dict[str, Any],
    expect: dict[str, Any],
    *,
    system: str | None = None,
    note: str = "",
    ambiguous: bool = False,
) -> dict[str, Any]:
    """Structured output 케이스. expect는 기대 JSON 객체."""
    return {
        "id": cid,
        "track": "SO",
        "category": category,
        "system": system,
        "prompt": prompt,
        "schema": schema,
        "expect": expect,
        "note": note,
        "ambiguous": ambiguous,
    }


def obj(props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


S = {"type": "string"}
I = {"type": "integer"}


def build() -> list[dict[str, Any]]:
    C: list[dict[str, Any]] = []

    # ── FC-1 단일 툴 · 명확 ───────────────────────────────────────────────────
    C += [
        fc("FC1-01", "FC-1", "서울 날씨 알려줘.", [T_WEATHER],
           {"action": "call", "calls": [{"name": "get_weather", "args": {"city": "서울"}}]}),
        fc("FC1-02", "FC-1", "삼성전자 주가 조회해줘.", [T_STOCK],
           {"action": "call", "calls": [{"name": "get_stock_price", "args": {"company": "삼성전자"}}]}),
        fc("FC1-03", "FC-1", "부산 날씨를 화씨로 알려줘.", [T_WEATHER],
           {"action": "call", "calls": [{"name": "get_weather",
                                          "args": {"city": "부산", "unit": "fahrenheit"}}]}),
        fc("FC1-04", "FC-1", f"{TODAY}에 서울에서 제주 가는 항공편 1명 찾아줘.", [T_FLIGHT],
           {"action": "call", "calls": [{"name": "search_flights",
                                          "args": {"origin": "서울", "destination": "제주",
                                                   "date": TODAY, "passengers": 1}}]}),
        fc("FC1-05", "FC-1", "볼펜 100개를 개당 500원에 발주해줘.", [T_ORDER],
           {"action": "call", "calls": [{"name": "create_order",
                                          "args": {"item": "볼펜", "quantity": 100,
                                                   "unit_price_krw": 500}}]}),
    ]

    # ── FC-2 다중 툴 중 선택 (근접 distractor) ───────────────────────────────
    STOCK_SET = [T_STOCK, T_STOCK_HISTORY, T_STOCK_NEWS, T_STOCK_DIVIDEND]
    C += [
        fc("FC2-01", "FC-2", "네이버 지금 주가 얼마야?", STOCK_SET,
           {"action": "call", "calls": [{"name": "get_stock_price", "args": {"company": "네이버"}}]}),
        fc("FC2-02", "FC-2", "카카오 최근 30일 주가 흐름 보여줘.", STOCK_SET,
           {"action": "call", "calls": [{"name": "get_stock_history",
                                          "args": {"company": "카카오", "days": 30}}]}),
        fc("FC2-03", "FC-2", "SK하이닉스 배당 어떻게 되는지 알려줘.", STOCK_SET,
           {"action": "call", "calls": [{"name": "get_dividend_info",
                                          "args": {"company": "SK하이닉스"}}]}),
        fc("FC2-04", "FC-2", "LG화학 관련 기사 좀 찾아줘.", STOCK_SET,
           {"action": "call", "calls": [{"name": "get_stock_news", "args": {"company": "LG화학"}}]}),
        fc("FC2-05", "FC-2", "현대차 주식 지금 얼마에 거래되고 있어?", STOCK_SET,
           {"action": "call", "calls": [{"name": "get_stock_price", "args": {"company": "현대차"}}]}),
    ]

    # ── FC-3 병렬 호출 (한국어 접속조사 포함) ────────────────────────────────
    C += [
        fc("FC3-01", "FC-3", "서울과 부산 날씨를 각각 알려줘.", [T_WEATHER],
           {"action": "call", "calls": [{"name": "get_weather", "args": {"city": "서울"}},
                                         {"name": "get_weather", "args": {"city": "부산"}}]}),
        fc("FC3-02", "FC-3", "삼성전자랑 SK하이닉스 둘 다 주가 조회해줘.", [T_STOCK],
           {"action": "call", "calls": [{"name": "get_stock_price", "args": {"company": "삼성전자"}},
                                         {"name": "get_stock_price", "args": {"company": "SK하이닉스"}}]},
           note="한국어 접속조사 '랑'. 사전 프로브에서 3모델 공통 실패한 패턴."),
        fc("FC3-03", "FC-3", "네이버하고 카카오 주가 알려줘.", [T_STOCK],
           {"action": "call", "calls": [{"name": "get_stock_price", "args": {"company": "네이버"}},
                                         {"name": "get_stock_price", "args": {"company": "카카오"}}]},
           note="접속조사 '하고'."),
        fc("FC3-04", "FC-3", "대구와 광주, 인천 날씨 모두 알려줘.", [T_WEATHER],
           {"action": "call", "calls": [{"name": "get_weather", "args": {"city": "대구"}},
                                         {"name": "get_weather", "args": {"city": "광주"}},
                                         {"name": "get_weather", "args": {"city": "인천"}}]}),
    ]

    # ── FC-5 무관 (abstention) — 툴을 부르면 안 된다 ─────────────────────────
    C += [
        fc("FC5-01", "FC-5", "피보나치 수열의 10번째 항이 뭐야?", [T_WEATHER, T_STOCK],
           {"action": "no_call"}),
        fc("FC5-02", "FC-5", "김치찌개 맛있게 끓이는 법 알려줘.", [T_WEATHER, T_FLIGHT],
           {"action": "no_call"}),
        fc("FC5-03", "FC-5", "파이썬에서 리스트랑 튜플 차이가 뭐야?", [T_STOCK, T_ORDER],
           {"action": "no_call"}),
        fc("FC5-04", "FC-5", "주식 투자할 때 분산투자가 왜 중요한지 설명해줘.", STOCK_SET,
           {"action": "no_call"},
           note="주식 도메인이지만 조회가 아니라 개념 설명 요청. 도메인 유사성 함정."),
        fc("FC5-05", "FC-5", "이체 수수료는 보통 얼마나 나와?", [T_TRANSFER],
           {"action": "no_call"},
           note="이체 도구가 있지만 실행이 아니라 일반 질문. 되돌릴 수 없는 도구라 오호출 비용이 크다."),
    ]

    # ── FC-6 정보 부족 — 되물어야 한다 (지어내면 안 된다) ────────────────────
    C += [
        fc("FC6-01", "FC-6", "항공편 좀 찾아줘.", [T_FLIGHT], {"action": "ask"},
           note="출발지·도착지·날짜 전부 없음."),
        fc("FC6-02", "FC-6", "서울에서 출발하는 항공편 찾아줘.", [T_FLIGHT], {"action": "ask"},
           note="도착지·날짜 없음."),
        fc("FC6-03", "FC-6", "다음 주 금요일에 서울에서 제주 가는 항공편 두 명 찾아줘.",
           [T_FLIGHT],
           {"action": "call", "calls": [{"name": "search_flights",
                                          "args": {"origin": "서울", "destination": "제주",
                                                   "date": "2026-08-14", "passengers": 2}}]},
           system=SYSTEM_DATE,
           note="시스템 메시지로 오늘 날짜를 주므로 '다음 주 금요일'=2026-08-14로 확정 가능. "
                "날짜를 주지 않으면 되묻는 게 정답이지만, 여기서는 줬으므로 호출이 정답."),
        fc("FC6-04", "FC-6", "휴가 신청해줘.", [T_LEAVE], {"action": "ask"},
           note="시작일·종료일·종류 전부 없음."),
        fc("FC6-05", "FC-6", "다음 주 월요일부터 수요일까지 연차 신청해줘.", [T_LEAVE],
           {"action": "call", "calls": [{"name": "request_leave",
                                          "args": {"start_date": "2026-08-10",
                                                   "end_date": "2026-08-12",
                                                   "leave_type": "annual"}}]},
           system=SYSTEM_DATE,
           note="8/6(목) 기준 다음 주 월=8/10, 수=8/12. '연차'→annual enum."),
    ]

    # ── FC-7 제약 인자 (enum · 포맷 · 타입) ──────────────────────────────────
    C += [
        fc("FC7-01", "FC-7", "몸이 안 좋아서 내일 하루 병가 쓸게.", [T_LEAVE],
           {"action": "call", "calls": [{"name": "request_leave",
                                          "args": {"start_date": "2026-08-07",
                                                   "end_date": "2026-08-07",
                                                   "leave_type": "sick"}}]},
           system=SYSTEM_DATE, note="'병가'→sick. 하루면 start=end."),
        fc("FC7-02", "FC-7", "아버지 간병 때문에 8월 20일부터 22일까지 쉬어야 해.", [T_LEAVE],
           {"action": "call", "calls": [{"name": "request_leave",
                                          "args": {"start_date": "2026-08-20",
                                                   "end_date": "2026-08-22",
                                                   "leave_type": "family_care"}}]},
           system=SYSTEM_DATE, note="'간병'→family_care."),
        fc("FC7-03", "FC-7", "무급으로 9/1부터 9/5까지 쉴게.", [T_LEAVE],
           {"action": "call", "calls": [{"name": "request_leave",
                                          "args": {"start_date": "2026-09-01",
                                                   "end_date": "2026-09-05",
                                                   "leave_type": "unpaid"}}]},
           system=SYSTEM_DATE, note="'9/1' 축약 날짜 → YYYY-MM-DD."),
        fc("FC7-04", "FC-7", "제주 날씨 섭씨로.", [T_WEATHER],
           {"action": "call", "calls": [{"name": "get_weather",
                                          "args": {"city": "제주", "unit": "celsius"}}]}),
        fc("FC7-05", "FC-7", "A4용지 다섯 박스, 한 박스에 이만 삼천원이야. 발주해.", [T_ORDER],
           {"action": "call", "calls": [{"name": "create_order",
                                          "args": {"item": "A4용지", "quantity": 5,
                                                   "unit_price_krw": 23000}}]},
           note="한글 수사 '다섯'→5, '이만 삼천원'→23000."),
    ]

    # ── FC-8 한국어 고유 인자 (실측 변별력 있는 것만) ────────────────────────
    C += [
        fc("FC8-01", "FC-8", "3만 5천 개를 개당 1만 2천원에 주문할게요. 품목은 종이컵.",
           [T_ORDER],
           {"action": "call", "calls": [{"name": "create_order",
                                          "args": {"item": "종이컵", "quantity": 35000,
                                                   "unit_price_krw": 12000}}]},
           note="만이 수량 배수와 통화 배수로 동시에 쓰임. METHODOLOGY §6 참고."),
        fc("FC8-02", "FC-8", "마스크 2천개, 단가 3천5백원으로 발주.", [T_ORDER],
           {"action": "call", "calls": [{"name": "create_order",
                                          "args": {"item": "마스크", "quantity": 2000,
                                                   "unit_price_krw": 3500}}]}),
        fc("FC8-03", "FC-8", "삼성전자의 주가를 알려줘.", [T_STOCK],
           {"action": "call", "calls": [{"name": "get_stock_price", "args": {"company": "삼성전자"}}]},
           note="조사 '의' 제거. 토크나이저가 어간을 재분절하는 fused-boundary 케이스."),
        fc("FC8-04", "FC-8", "엘지화학이 지금 얼마인지 봐줘.", [T_STOCK],
           {"action": "call", "calls": [{"name": "get_stock_price", "args": {"company": "엘지화학"}}]},
           note="조사 '이' 제거 + 한글 표기 사명."),
        fc("FC8-05", "FC-8", "모레 서울에서 부산 가는 비행기 3명.", [T_FLIGHT],
           {"action": "call", "calls": [{"name": "search_flights",
                                          "args": {"origin": "서울", "destination": "부산",
                                                   "date": "2026-08-08", "passengers": 3}}]},
           system=SYSTEM_DATE, note="'모레' = 8/6 + 2 = 8/8."),
        fc("FC8-06", "FC-8", "이번 주 일요일에 인천에서 대구 가는 편 2명 찾아줘.", [T_FLIGHT],
           {"action": "call", "calls": [{"name": "search_flights",
                                          "args": {"origin": "인천", "destination": "대구",
                                                   "date": "2026-08-09", "passengers": 2}}]},
           system=SYSTEM_DATE, note="8/6(목) 기준 이번 주 일요일 = 8/9."),
    ]

    # ── FC-X tools + response_format 교차 (조용한 tool drop 탐지) ────────────
    ANS = obj({"answer": S}, ["answer"])
    for i, (cid, p, tools, exp) in enumerate([
        ("FCX-01", "서울 날씨 알려줘.", [T_WEATHER],
         {"action": "call", "calls": [{"name": "get_weather", "args": {"city": "서울"}}]}),
        ("FCX-02", "삼성전자 주가 조회해줘.", [T_STOCK],
         {"action": "call", "calls": [{"name": "get_stock_price", "args": {"company": "삼성전자"}}]}),
        ("FCX-03", "부산 날씨 알려줘.", [T_WEATHER],
         {"action": "call", "calls": [{"name": "get_weather", "args": {"city": "부산"}}]}),
    ]):
        c = fc(cid, "FC-X", p, tools, exp,
               note="tools와 response_format을 동시에 준다. 200 OK + 스키마 통과 + 툴 미호출 = "
                    "silent tool drop. opus는 400으로 거부, glm은 조용히 버림(실측).")
        c["response_format"] = {"type": "json_schema",
                                "json_schema": {"name": "ans", "schema": ANS, "strict": True}}
        C.append(c)

    # ── SO-1 평면 추출 ────────────────────────────────────────────────────────
    PERSON = obj({"name": S, "age": I, "city": S}, ["name", "age", "city"])
    C += [
        so("SO1-01", "SO-1", "'김철수는 32세이고 서울에 산다.' 에서 정보를 뽑아라.",
           PERSON, {"name": "김철수", "age": 32, "city": "서울"}),
        so("SO1-02", "SO-1", "'이영희(28)는 부산 거주.' 에서 정보를 뽑아라.",
           PERSON, {"name": "이영희", "age": 28, "city": "부산"}),
        so("SO1-03", "SO-1", "'박민수 씨는 마흔다섯 살이며 대전에 삽니다.' 에서 정보를 뽑아라.",
           PERSON, {"name": "박민수", "age": 45, "city": "대전"},
           note="한글 수사 '마흔다섯'→45. 호칭 '씨' 제거."),
        so("SO1-04", "SO-1", "'최지우, 나이 51, 거주지 광주' 에서 정보를 뽑아라.",
           PERSON, {"name": "최지우", "age": 51, "city": "광주"}),
        so("SO1-05", "SO-1", "'정한별은 올해 서른이 되었고 인천에 산다.' 에서 정보를 뽑아라.",
           PERSON, {"name": "정한별", "age": 30, "city": "인천"}),
    ]

    # ── SO-3 enum 준수 ────────────────────────────────────────────────────────
    TICKET = obj({"category": {"type": "string",
                               "enum": ["배송문의", "환불요청", "제품불량", "기타"]},
                  "urgency": {"type": "string", "enum": ["low", "medium", "high"]}},
                 ["category", "urgency"])
    C += [
        so("SO3-01", "SO-3", "고객 문의를 분류하라: '주문한 지 일주일 됐는데 아직도 안 왔어요.'",
           TICKET, {"category": "배송문의", "urgency": "medium"}, ambiguous=True,
           note="urgency는 주관적 → category만 채점."),
        so("SO3-02", "SO-3", "고객 문의를 분류하라: '받은 제품 화면에 금이 가 있습니다. 당장 교환해주세요.'",
           TICKET, {"category": "제품불량", "urgency": "high"}, ambiguous=True,
           note="category만 채점."),
        so("SO3-03", "SO-3", "고객 문의를 분류하라: '단순 변심인데 돈 돌려받을 수 있나요?'",
           TICKET, {"category": "환불요청", "urgency": "low"}, ambiguous=True,
           note="category만 채점."),
        so("SO3-04", "SO-3", "고객 문의를 분류하라: '영업시간이 어떻게 되나요?'",
           TICKET, {"category": "기타", "urgency": "low"}, ambiguous=True,
           note="category만 채점."),
    ]

    # ── SO-4 additionalProperties:false — 환각 키 ────────────────────────────
    NARROW = obj({"company": S, "amount_krw": I}, ["company", "amount_krw"])
    C += [
        so("SO4-01", "SO-4",
           "'삼성전자와 3,500만원 규모 계약을 체결했다. 담당자는 김민준 대리, 계약일은 2026년 3월 3일.' "
           "에서 스키마에 정의된 필드만 뽑아라.",
           NARROW, {"company": "삼성전자", "amount_krw": 35000000},
           note="담당자·계약일이 본문에 있지만 스키마에 없다 → 넣으면 환각 키."),
        so("SO4-02", "SO-4",
           "'현대차와 12억원 공급 계약. 기간 3년, 연장 옵션 포함.' 에서 스키마 필드만 뽑아라.",
           NARROW, {"company": "현대차", "amount_krw": 1200000000}),
        so("SO4-03", "SO-4",
           "'네이버, 7천8백만원 규모 발주. 결제조건 30일.' 에서 스키마 필드만 뽑아라.",
           NARROW, {"company": "네이버", "amount_krw": 78000000}),
        so("SO4-04", "SO-4",
           "'카카오와 5억 2천만원 계약 체결, 담당 이서연 팀장.' 에서 스키마 필드만 뽑아라.",
           NARROW, {"company": "카카오", "amount_krw": 520000000}),
    ]

    # ── SO-5 null vs 누락 vs 빈문자열 ────────────────────────────────────────
    CONTACT = {
        "type": "object",
        "properties": {"name": S, "phone": {"type": ["string", "null"]},
                       "email": {"type": ["string", "null"]}},
        "required": ["name", "phone", "email"],
        "additionalProperties": False,
    }
    C += [
        so("SO5-01", "SO-5",
           "'홍길동, 010-1234-5678' 에서 연락처를 뽑아라. 없는 값은 null로 둬라.",
           CONTACT, {"name": "홍길동", "phone": "010-1234-5678", "email": None}),
        so("SO5-02", "SO-5",
           "'김서연, seoyeon@example.com' 에서 연락처를 뽑아라. 없는 값은 null로 둬라.",
           CONTACT, {"name": "김서연", "phone": None, "email": "seoyeon@example.com"}),
        so("SO5-03", "SO-5",
           "'박도윤' 에서 연락처를 뽑아라. 없는 값은 null로 둬라.",
           CONTACT, {"name": "박도윤", "phone": None, "email": None}),
        so("SO5-04", "SO-5",
           "'이하준, 010-9876-5432, hajun@corp.co.kr' 에서 연락처를 뽑아라. 없는 값은 null로 둬라.",
           CONTACT, {"name": "이하준", "phone": "010-9876-5432", "email": "hajun@corp.co.kr"}),
    ]

    # ── SO-7 한국어 값 정규화 ────────────────────────────────────────────────
    DEAL = obj({"company": S, "date": S, "amount_krw": I},
               ["company", "date", "amount_krw"])
    C += [
        so("SO7-01", "SO-7",
           "'삼성전자와 2026년 3월 3일에 3,500만원 계약을 맺었다.' 에서 뽑아라. "
           "date는 YYYY-MM-DD, amount_krw는 정수(원 단위).",
           DEAL, {"company": "삼성전자", "date": "2026-03-03", "amount_krw": 35000000}),
        so("SO7-02", "SO-7",
           "'현대건설, 26년 12월 1일자 1조 2천억원 수주.' 에서 뽑아라. "
           "date는 YYYY-MM-DD, amount_krw는 정수(원 단위).",
           DEAL, {"company": "현대건설", "date": "2026-12-01", "amount_krw": 1200000000000},
           note="'26년' 2자리 연도 → 2026. 조 단위."),
        so("SO7-03", "SO-7",
           "'포스코 24.7.15 계약, 금액 8억 5천만원.' 에서 뽑아라. "
           "date는 YYYY-MM-DD, amount_krw는 정수(원 단위).",
           DEAL, {"company": "포스코", "date": "2024-07-15", "amount_krw": 850000000}),
        so("SO7-04", "SO-7",
           "'LG전자와 지난달 15일 계약. 규모는 백이십억원.' 에서 뽑아라. "
           "date는 YYYY-MM-DD, amount_krw는 정수(원 단위).",
           DEAL, {"company": "LG전자", "date": "2026-07-15", "amount_krw": 12000000000},
           system=SYSTEM_DATE, note="'지난달 15일' = 2026-07-15. 한글 수사 '백이십억'."),
        so("SO7-05", "SO-7",
           "'SK텔레콤 2026-01-31 체결, 3억원.' 에서 뽑아라. "
           "date는 YYYY-MM-DD, amount_krw는 정수(원 단위).",
           DEAL, {"company": "SK텔레콤", "date": "2026-01-31", "amount_krw": 300000000}),
        so("SO7-06", "SO-7",
           "'한화솔루션, 오늘 자로 4,750만원 규모 계약.' 에서 뽑아라. "
           "date는 YYYY-MM-DD, amount_krw는 정수(원 단위).",
           DEAL, {"company": "한화솔루션", "date": TODAY, "amount_krw": 47500000},
           system=SYSTEM_DATE),
    ]

    # ── SO-9 한국 규제 준수 (마스킹) — 실측 discriminator ────────────────────
    MASK_RRN = obj({"name": S, "rrn_masked": S}, ["name", "rrn_masked"])
    MASK_ACC = obj({"name": S, "account_masked": S}, ["name", "account_masked"])
    C += [
        so("SO9-01", "SO-9",
           "'김민준, 901010-1234567 로 등록해줘.' 에서 이름과 주민등록번호를 뽑되, "
           "rrn_masked는 생년월일 6자리만 남기고 뒤 7자리를 전부 '*'로 마스킹하라.",
           MASK_RRN, {"name": "김민준", "rrn_masked": "901010-*******"},
           note=""
                "PIPA 제24조의2 — 7번째 자리는 성별·출생세기를 인코딩한다."),
        so("SO9-02", "SO-9",
           "'이수아 850315-2345678' 에서 이름과 주민등록번호를 뽑되, "
           "rrn_masked는 생년월일 6자리만 남기고 뒤 7자리를 전부 '*'로 마스킹하라.",
           MASK_RRN, {"name": "이수아", "rrn_masked": "850315-*******"}),
        so("SO9-03", "SO-9",
           "'박지호 020728-3456789 확인 부탁해요.' 에서 이름과 주민등록번호를 뽑되, "
           "rrn_masked는 생년월일 6자리만 남기고 뒤 7자리를 전부 '*'로 마스킹하라.",
           MASK_RRN, {"name": "박지호", "rrn_masked": "020728-*******"}),
        so("SO9-04", "SO-9",
           "'최유진, 계좌 110-234-567890' 에서 이름과 계좌번호를 뽑되, "
           "account_masked는 마지막 4자리만 남기고 나머지 숫자를 전부 '*'로 마스킹하라. "
           "하이픈은 유지하라.",
           MASK_ACC, {"name": "최유진", "account_masked": "***-***-**7890"},
           ambiguous=True, note="마스킹 표기 관례가 갈릴 수 있어 ambiguous. 자릿수 유출 여부만 본다."),
        so("SO9-05", "SO-9",
           "'정하윤 주민번호 991225-4567890' 에서 이름과 주민등록번호를 뽑되, "
           "rrn_masked는 생년월일 6자리만 남기고 뒤 7자리를 전부 '*'로 마스킹하라.",
           MASK_RRN, {"name": "정하윤", "rrn_masked": "991225-*******"}),
    ]

    return C


def main() -> None:
    cases = build()
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "중복 case id"
    with OUT.open("w", encoding="utf-8") as fh:
        for c in cases:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    from collections import Counter

    by_cat = Counter(c["category"] for c in cases)
    by_track = Counter(c["track"] for c in cases)
    print(f"→ {OUT}  ({len(cases)} cases)")
    print(f"  track: {dict(by_track)}")
    for cat in sorted(by_cat):
        amb = sum(1 for c in cases if c["category"] == cat and c["ambiguous"])
        print(f"    {cat:6s} {by_cat[cat]:3d}" + (f"  (ambiguous {amb})" if amb else ""))


if __name__ == "__main__":
    main()
