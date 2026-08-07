# function-calling-json

Databricks Foundation Model API의 세 모델에게 한국어 업무 요청과 도구 정의를 주고,
**올바른 도구를 올바른 인자로 호출하는가**(Function Calling)와
**요구된 JSON 스키마를 준수하는가**(Structured Output)를 측정한다.

외부 공개 벤치마크 2종(카카오, Apache-2.0)과 자체 제작 케이스를 함께 쓴다.
전체 **9,942 호출**, 인프라 오류 0건.

| 문서 | 내용 |
|---|---|
| [BENCHMARKS.md](./BENCHMARKS.md) | 데이터셋 출처 · 변별력 · 순위 역전 분석 |
| [METHODOLOGY.md](./METHODOLOGY.md) | 실험 설계 · 플랫폼 제약 · 채점 규칙 |
| [FINDINGS.md](./FINDINGS.md) | 결과 해석 |
| [COST.md](./COST.md) | 비용 산출 |
| [PARITY.md](./PARITY.md) | 한국어·영어 정확도 비교 |
| [TASK_DESCRIPTION.md](./TASK_DESCRIPTION.md) | 태스크 정의 |

---

## 용어

| 용어 | 정의 |
|---|---|
| **실험 조건 A** | 세 모델의 추론 설정을 각 모델이 지원하는 최소값으로 통일. 모델 간 비교가 성립한다 |
| **실험 조건 A′** | 조건 A에서 Opus 5만 `thinking:adaptive`로 변경. 응답 형식 오류를 제거하기 위한 보정 조건 |
| **실험 조건 B** | 각 모델의 최대 추론 설정. 모델마다 호출 경로가 달라 엄밀한 비교는 아니다 |
| **정확도** | 도구 이름과 인자가 정답과 일치한 비율 |
| **Wilson 95% 신뢰구간** | 비율 지표의 신뢰구간. 두 모델의 구간이 겹치면 **차이를 확인할 수 없다**고 판정한다 |
| **`pass^5`** | 5회 시행이 **모두** 성공할 확률의 추정값. `C(성공, 5)/C(시행, 5)`. "최소 1회 성공"인 `pass@5`와 다르다 |
| **도구 미호출** | 도구를 호출하지 않은 상태. 요청과 무관할 때는 정답, 필요한데 안 하면 오답 |
| **서버 자동 추가 프롬프트** | Databricks가 도구 정의와 JSON 스키마 전달을 위해 요청에 덧붙이는 텍스트. `prompt_tokens`에 포함되어 과금된다 |

---

## 데이터셋

| 데이터셋 | 출처 | 건수 | 반복 | 호출 |
|---|---|---:|---:|---:|
| FunctionChat-Bench CallDecision | 카카오 · Apache-2.0 | 606 | 2 | **3,636** |
| OrchestrationBench | 카카오 · Apache-2.0 | 441 | 2 | **2,646** |
| 자체 제작 (FC 38 + SO 28) | 직접 작성 | 66 | 5 | 2,310 |
| 자체 제작 한/영 45쌍 | 직접 작성 | 90 | 5 | 1,350 |
| **합계** | | | | **9,942** |

외부 데이터는 **재배포하지 않고** 변환 스크립트만 포함한다.

---

## 결과

### FunctionChat-Bench (외부, 606 케이스)

한국어 function calling 전용 공개 벤치마크.

| 카테고리 | 건수 | 기대 동작 | Opus 5 | GPT-5.6-sol | GLM 5.2 |
|---|---:|---|---:|---:|---:|
| CALL | 100 | 정확한 도구를 정확한 인자로 호출 | **0.965** | 0.855 | 0.935 |
| REJECT | 100 | 도구가 부적합 → 호출 안 함 | 0.990 | **1.000** | 0.995 |
| SLOT-all | 100 | 필수 인자 전부 없음 → 되물음 | 1.000 | 1.000 | 1.000 |
| SLOT-some | 306 | 필수 인자 일부 없음 → 되물음 | 0.997 | **1.000** | 0.998 |
| **전체** | **606** | | **0.991** | 0.976 | 0.988 |
| 응답시간 중앙값 | | | 2,482 ms | 1,769 ms | **1,327 ms** |
| 정답 1건당 비용 | | | $0.01042 | $0.00425 | **$0.00097** |

### OrchestrationBench (외부, 한국어 222)

다중 에이전트 계획에서 첫 계획 단계만 분리한 파생 태스크.

| | Opus 5 | GPT-5.6-sol | GLM 5.2 |
|---|---:|---:|---:|
| 정확도 (Wilson 95%) | 0.651 [0.61, 0.69] | **0.680 [0.64, 0.72]** | 0.538 [0.49, 0.58] |
| 에이전트 선택 F1 | 0.924 | **0.935** | 0.873 |

### 자체 제작 66 케이스

| | Opus 5 (A′) | GPT-5.6-sol | GLM 5.2 |
|---|:---:|:---:|:---:|
| 정확도 (조건 A) | 0.959 [0.93, 0.98] | **0.970 [0.95, 0.98]** | 0.924 [0.89, 0.95] |
| 정확도 (조건 B) | 0.940 [0.91, 0.96] | **0.979 [0.96, 0.99]** | 0.936 [0.90, 0.96] |
| `pass^5` (조건 A) | 0.937 | **0.955** | 0.894 |
| 응답시간 중앙값 (조건 A) | 1,929 ms | 1,944 ms | **949 ms** |
| 정답 1건당 비용 (조건 A) | $0.00750 | $0.00233 | **$0.00058** |

Opus 5는 조건 A′ 수치다. 조건 A(`thinking:disabled`)에서는 플랫폼의 응답 형식 오류로
0.930이 측정되는데 모델 성능이 아니다([METHODOLOGY §3.1](./METHODOLOGY.md)).

---

## 데이터셋에 따라 순위가 뒤바뀐다

| 순위 | 데이터셋 |
|---|---|
| GPT-5.6-sol > Opus 5 > GLM 5.2 | 자체 제작 66, 한/영 45쌍, OrchestrationBench |
| **Opus 5 > GLM 5.2 > GPT-5.6-sol** | **FunctionChat-Bench CALL 100** |

원인은 능력 차이가 아니라 **도구 호출 판단 기준의 차이**다.
호출해야 하는 100건과 하면 안 되는 506건을 함께 집계하면 성향이 드러난다.

| 모델 | 호출 정밀도 | 호출 재현율 | 불필요한 호출 | 필요한데 미호출 |
|---|---:|---:|---|---|
| Opus 5 | 0.980 | **1.000** | 4/1,012 | **0/200** |
| GLM 5.2 | 0.990 | 0.990 | 2/1,012 | 2/200 |
| GPT-5.6-sol | **1.000** | 0.920 | **0/1,012** | 16/200 |

GPT-5.6-sol은 불필요한 호출이 1,012건 중 0건이다. 호출 기준이 가장 엄격해
REJECT·SLOT에서 유일하게 1.000이고, 같은 이유로 정보가 모호하면 호출 대신 확인을 요청한다
(CALL 오답 29건 중 16건). Opus 5는 반대로 필요한 호출을 하나도 놓치지 않았다.

**같은 성향의 양면이며 우열이 아니다.** 상세는 [BENCHMARKS.md](./BENCHMARKS.md).

---

## 모델 선택 기준

| 우선순위 | 선택 | 근거 |
|---|---|---|
| 잘못된 호출의 비용이 큼 (결제·예약·발송) | **GPT-5.6-sol** | 불필요한 호출 0/1,012 |
| 놓친 호출의 비용이 큼 (조회·검색·추천) | **Opus 5** | 미호출 0/200, FunctionChat-Bench 전체 0.991 |
| 비용이 가장 중요 | **GLM 5.2** | 정답 1건당 비용이 Opus 5의 1/10. 정확도는 0.988로 근접 |
| 응답 속도가 중요 | **GLM 5.2** | 응답시간 중앙값 1,327 ms |
| 도구와 JSON 스키마 동시 사용 | **GPT-5.6-sol** | 세 모델 중 유일하게 지원 ([METHODOLOGY §3.2](./METHODOLOGY.md)) |

**한국 금융·구매 도메인에서 GLM 5.2를 쓸 경우 추론 설정을 켜지 않는다.**
켜면 한국어 만·억·조 단위 변환에서 자릿수를 하나 더 붙이는 오류가 발생한다
([FINDINGS §2](./FINDINGS.md)).

---

## 알아야 할 플랫폼 제약

상세와 재현 절차는 [METHODOLOGY §2~4](./METHODOLOGY.md).

| 제약 | 영향 |
|---|---|
| **세 모델에 공통 적용 가능한 추론 설정이 없다** | GPT-5.6-sol은 추론을 켜면 도구 사용 시 HTTP 400, Opus 5는 끄면 응답 형식 오류 |
| **Opus 5가 도구 호출을 본문 텍스트로 반환** | `thinking:disabled`에서 135회 중 6회. `thinking:adaptive`에서 0회 |
| **GLM 5.2가 `tools`+`response_format`에서 도구 호출 누락** | HTTP 200이고 스키마도 통과해 오류 신호가 없다 (30/30) |
| **`temperature`를 지정할 수 없다** | Opus 5·GPT-5.6-sol이 거부. 반복 측정이 필요하다 |
| **서버 자동 추가 프롬프트가 모델마다 3~13배 다르다** | 도구 1개당 Opus 5 116토큰 vs GPT-5.6-sol 29토큰. 입력 단가가 같아 그대로 비용 차이 |
| **`/serving-endpoints`는 시스템 테이블에 미기록** | 서버 측 측정이 불가능하다. 전 호출을 `/ai-gateway`로 보낸다 |
| **Opus 5가 `parameters`에 `type`이 없는 도구 정의를 거부** | 다른 모델에서 동작하던 정의가 HTTP 400 ([BENCHMARKS §6](./BENCHMARKS.md)) |

---

## 자체 제작 66 케이스 — 카테고리별 (실험 조건 A)

**굵은 글씨는 신뢰구간이 겹치지 않아 차이가 확인된 값**이다.

| 카테고리 | 측정 대상 | Opus 5 (A′) | GPT-5.6-sol | GLM 5.2 |
|---|---|---|---|---|
| FC-1 단일 도구 | 기본 호출 | 0.920 | **1.000** | **1.000** |
| FC-2 유사 도구 중 선택 | 같은 도메인 도구 4개 중 선택 | **0.960** | 0.800 | **1.000** |
| FC-3 병렬 호출 | 한 응답에 2~3건 | 1.000 | 1.000 | 1.000 |
| FC-5 무관한 요청 | 호출하지 않는가 | 1.000 | 1.000 | 1.000 |
| FC-6 정보 부족 | 되묻는가 | 1.000 | 1.000 | 1.000 |
| FC-7 제약 있는 인자 | enum · 날짜 형식 · 정수형 | **1.000** | **1.000** | 0.960 |
| FC-8 한국어 고유 표현 | 만/억 단위 · 조사 · 상대 날짜 | 0.667 | 0.833 | **1.000** |
| FC-X 도구+스키마 동시 지정 | 동시 지정 시 동작 | HTTP 400 | **1.000** | **0.000** |
| SO-1 평면 구조 추출 | 기본 | 1.000 | 1.000 | 1.000 |
| SO-3 enum 값 준수 | 정의된 값 집합에서 선택 | **1.000** | **1.000** | 0.700 |
| SO-4 정의되지 않은 키 | `additionalProperties: false` | 1.000 | 1.000 | 1.000 |
| SO-5 null 처리 | null / 누락 / 빈 문자열 구분 | 1.000 | 1.000 | 1.000 |
| SO-7 한국어 수치 변환 | 3,500만원 → 35000000 | 1.000 | 1.000 | 1.000 |
| SO-9 개인정보 마스킹 | 주민등록번호·계좌번호 | **1.000** | **1.000** | 0.880 |
| **전체** | | 0.959 | **0.970** | 0.924 |

14개 중 8개에서 세 모델이 동일하다. 차이가 난 곳은 FC-2, FC-8, FC-X, SO-3 네 곳이다.

**이 데이터셋은 변별력이 낮다** — 셀의 92.8%가 만점이고 1위와 최하위 차이가 0.045다.
다만 FC-5·FC-6의 1.000은 케이스가 쉬워서가 아니라 세 모델이 실제로 대등하기 때문이며,
외부 데이터 506건이 이를 확인했다([BENCHMARKS §2~3](./BENCHMARKS.md)).

### 오답이 발생한 15개 케이스

66개 중 51개는 모든 조건·반복에서 정답이었다. 숫자는 5회 중 오답 횟수다.

| 케이스 | Opus 5 (A) | Opus 5 (A′) | GPT-5.6-sol | GLM 5.2 | 현상 |
|---|:--:|:--:|:--:|:--:|---|
| FC1-01 `서울 날씨 알려줘` | 4 | 0 | 0 | 0 | 도구 호출을 본문 텍스트로 반환 |
| FC1-03 `부산 날씨를 화씨로` | 2 | 2 | 0 | 0 | `city`를 `"Busan"`으로 영문 변환 |
| FC7-04 `제주 날씨 섭씨로` | 4 | 0 | 0 | 0 | 동일한 형식 오류 |
| FC2-02 `카카오 30일 주가 흐름` | 2 | 1 | 1 | 0 | 필요한 도구 외에 추가 도구도 호출 |
| FC2-05 `현대차 지금 얼마` | 0 | 0 | 4 | 0 | `현대차`를 `현대자동차`로 변환 |
| FC7-01 `내일 하루 병가` | 0 | 0 | 0 | 1 | 도구 호출 대신 문장으로 응답 |
| FC7-03 `무급 9/1~9/5` | 1 | 0 | 0 | 0 | 실행 전 확인 요청 |
| FC8-01 `3만 5천 개 개당 1만 2천원` | 4 | 5 | 0 | 0 | 단가가 비정상이라고 판단해 확인 요청 |
| FC8-04 `엘지화학 얼마` | 5 | 5 | 5 | 0 | `엘지화학`을 `LG화학`으로 변환 |
| FCX-01~03 도구+스키마 | 400 | 400 | 0 | 5 | GLM 5.2가 도구 호출 없이 JSON만 반환 |
| SO3-03 `단순 변심 환불` | 0 | 0 | 0 | 4 | `환불요청`을 `배송문의`로 분류 |
| SO3-04 `영업시간 문의` | 0 | 0 | 0 | 2 | `기타`를 `배송문의`로 분류 |
| SO9-04 계좌번호 마스킹 | 0 | 0 | 0 | 3 | 마스킹 자릿수 미보존 (판정 모호) |

**FC8-04와 FC2-05는 정답 정의의 문제일 수 있다.** 정식 상호로 정규화하는 것도
입력 표기를 유지하는 것도 타당하다. 성능이 아니라 **정규화 방식의 차이**다.

---

## 한국어와 영어

두 데이터셋 모두에서 **세 모델의 한국어 정확도가 영어와 같거나 높다.**

| 데이터셋 | Opus 5 (한/영) | GPT-5.6-sol (한/영) | GLM 5.2 (한/영) |
|---|---|---|---|
| 자체 제작 45쌍 | 0.971 / 0.933 | **0.978** / 0.933 | 0.889 / 0.871 |
| OrchestrationBench | 0.651 / 0.605 | **0.680** / 0.591 | 0.538 / 0.459 |

한국어에서 발생하는 추가 부담은 정확도가 아니라 **비용**이다.
긴 프롬프트에서 정답 1건당 비용이 22~34% 높다. 상세는 [PARITY.md](./PARITY.md).

---

## 측정 지표

| | 지표 | 산출 방식 |
|---|---|---|
| 1 | 정확도 | 기계 채점 (`jsonschema` 검증 + 인자 대조). LLM 채점자 미사용 |
| 2 | 안정성 | `pass^5` |
| 3 | 응답시간 | 클라이언트 측정 중앙값·90분위 |
| 4 | 비용 | 응답 `usage` × 공시 DBU 단가 |

산출물이 JSON과 도구 호출이라 프로그램으로 완전히 검증할 수 있고,
API를 직접 호출하므로 모든 응답에 `usage`가 포함되어 호출 단위 비용이 나온다.

---

## 디렉토리 구조

```
function-calling-json/
├── README.md                    결과 요약
├── BENCHMARKS.md                데이터셋 출처 · 변별력 · 순위 역전
├── METHODOLOGY.md               실험 설계 · 플랫폼 제약 · 채점 규칙
├── FINDINGS.md                  결과 해석
├── COST.md                      비용 산출
├── PARITY.md                    한국어·영어 비교
├── TASK_DESCRIPTION.md          태스크 정의
├── cases/
│   ├── build_cases.py           자체 제작 66개 생성
│   ├── build_fcb_cases.py       FunctionChat-Bench 606개 변환
│   ├── build_parity_cases.py    한/영 45쌍 + OrchestrationBench 441개
│   └── cases.jsonl              자체 제작 케이스와 정답
├── src/
│   ├── runner.py                호출 실행 (프로토콜 분기 · 요청 태깅 · 동시성 제한)
│   ├── score.py                 채점
│   ├── parity.py                한국어·영어 비교 지표
│   ├── discrimination.py        데이터셋별 변별력 비교
│   └── xcheck.py                클라이언트 측정값과 시스템 테이블 대조
├── probes/
│   ├── probe_capabilities.py    실행 전 지원 기능 확인
│   ├── probe_scaffolding_overhead.py
│   └── query_usage.py
└── results/<실행ID>/            scored.jsonl · report.json · manifest.json
```

케이스·프롬프트·채점 규칙은 실험 조건 간 **바이트 단위로 동일**하다.

---

## 재현

```bash
cd function-calling-json
pip install httpx jsonschema pyyaml
export DATABRICKS_PROFILE=<프로파일명>

# 1. 지원 기능 확인 — 게이트웨이 제약은 릴리스마다 바뀐다
python3 probes/probe_capabilities.py

# 2. 자체 제작 66 케이스
python3 cases/build_cases.py
python3 -m src.runner --repeats 5                                # 조건 A
python3 -m src.runner --repeats 5 --arms opus-adaptive           # 조건 A′
python3 -m src.runner --repeats 5 --arms opus-B sol-B glm-B      # 조건 B
python3 -m src.score --run results/<실행ID>

# 3. FunctionChat-Bench (카카오, Apache-2.0)
git clone --depth 1 https://github.com/kakao/FunctionChat-Bench.git /tmp/FunctionChat-Bench
python3 cases/build_fcb_cases.py --fcb-root /tmp/FunctionChat-Bench
python3 -m src.runner --fcb --repeats 2 --arms opus-adaptive sol glm --out results/fcb
python3 -m src.score --run results/fcb --fcb

# 4. OrchestrationBench + 한국어·영어 비교
git clone --depth 1 https://github.com/kakao/OrchestrationBench.git /tmp/OrchestrationBench
python3 cases/build_parity_cases.py --ob-root /tmp/OrchestrationBench
python3 -m src.runner --parity --tracks PAIR --repeats 5 \
    --arms opus-adaptive sol glm --out results/parity-PAIR
python3 -m src.runner --parity --tracks OB --repeats 2 \
    --arms opus-adaptive sol glm --out results/parity-OB
python3 -m src.score  --run results/parity-PAIR --parity
python3 -m src.score  --run results/parity-OB   --parity
python3 -m src.parity --pair results/parity-PAIR --ob results/parity-OB

# 5. 데이터셋별 변별력 비교
python3 -m src.discrimination
```

모든 호출은 `/ai-gateway/mlflow/v1/chat/completions`를 사용한다.

---

## 유효성 점검

| 항목 | 결과 |
|---|---|
| 인프라 오류 | **0건** / 9,942 호출 (재시도 총 2회) |
| JSON 파싱 실패 (1차 시도) | **0건** |
| 정의되지 않은 도구 호출 · 인자 파싱 실패 | **0건** |
| 응답 길이 초과로 잘림 | 16건. 전부 FC-5의 설명 문장이며 채점에 영향 없음 (16/16 정답) |
| 클라이언트 측정값과 시스템 테이블 대조 | **1,290 호출 전수 일치** ([COST.md](./COST.md)) |

2026-08-06 ~ 08-07 실행.
