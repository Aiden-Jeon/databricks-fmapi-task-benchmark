# Function Calling & JSON Output (태스크 6)

## 배경

에이전트가 동작하려면 두 가지가 규격대로 반환되어야 한다.
**올바른 도구를 올바른 인자로 호출하는 것**과 **요구된 JSON 스키마를 준수하는 것**이다.

같은 저장소의 `mle-benchmark`가 관측한 바에 따르면, Databricks 환경의 에이전트
태스크에서 점수 차이를 만든 것은 모델의 문제 해결 능력이 아니라 **도구 사용 적합성**이었다
(72개 셀 중 7개 미완료, *"every one a harness/tool-conformance fault"*).

이 태스크는 그 변수를 **에이전트 프레임워크 없이 직접 측정**한다.

## 데이터셋

| 데이터셋 | 출처 | 라이선스 | 건수 |
|---|---|---|---:|
| FunctionChat-Bench CallDecision | 카카오 | Apache-2.0 | 606 |
| OrchestrationBench | 카카오 | Apache-2.0 | 441 |
| 자체 제작 (아래 트랙) | 직접 작성 | — | 66 |
| 자체 제작 한/영 45쌍 | 직접 작성 | — | 90 |

외부 데이터는 재배포하지 않고 변환 스크립트만 포함한다.
선정 근거는 [BENCHMARKS.md](./BENCHMARKS.md).

## 측정 대상 — 자체 제작 트랙

### Function Calling 트랙 (38 케이스)

| ID | 카테고리 | 내용 |
|---|---|---|
| FC-1 | 단일 도구 (요청 명확) | 기본 호출 능력 |
| FC-2 | 유사 도구 중 선택 | 같은 도메인의 혼동하기 쉬운 도구 4개 중 선택 |
| FC-3 | 병렬 호출 | 한 응답에 2~3건. 한국어 접속조사 `랑` / `하고` / `와` 포함 |
| FC-5 | 무관한 요청 | 도구가 적합하지 않을 때 **호출하지 않는가** |
| FC-6 | 정보 부족 | 필수 인자를 채울 수 없을 때 **되묻는가, 임의로 값을 만드는가** |
| FC-7 | 제약 있는 인자 | enum · 날짜 형식(YYYY-MM-DD) · 정수형 |
| FC-8 | 한국어 고유 표현 | 만·억 단위 · 조사 제거 · 상대 날짜 |
| FC-X | `tools`와 `response_format` 동시 지정 | 오류 신호 없는 도구 호출 누락 탐지 |

### Structured Output 트랙 (28 케이스)

| ID | 카테고리 | 내용 |
|---|---|---|
| SO-1 | 평면 구조 추출 | 기본 |
| SO-3 | enum 값 준수 | 정의된 라벨 집합 안에서 선택 |
| SO-4 | `additionalProperties: false` | 본문에는 있지만 스키마에 없는 값을 키로 추가하는가 |
| SO-5 | null 처리 | null / 키 누락 / 빈 문자열 구분 |
| SO-7 | 한국어 수치·날짜 변환 | 3,500만원 → 35000000, 26년 → 2026, 지난달 15일 |
| SO-9 | 개인정보 마스킹 | 주민등록번호·계좌번호 (개인정보보호법 제24조의2) |

## 고정 조건 (모델 간 바이트 단위로 동일)

| 항목 | 값 |
|---|---|
| 호출 경로 | `POST /ai-gateway/mlflow/v1/chat/completions` — `/serving-endpoints`는 `system.ai_gateway.usage`에 기록되지 않아 서버 측 측정이 불가능하다 |
| `max_tokens` | 1024 |
| `temperature` | **지정 불가** — Opus 5와 GPT-5.6-sol이 거부한다 |
| 반복 | 자체 제작 5회 · 외부 벤치마크 2회 |
| 요청 태깅 | 전 호출에 `Databricks-Ai-Gateway-Request-Tags` 헤더 부착 |
| 재시도 | 인프라 오류(5xx · 429 · 타임아웃)에만 적용. **형식 오류는 재시도하지 않는다** |
| 동시 요청 수 | Opus 5·GPT-5.6-sol 6, **GLM 5.2 2** (시간당 요청 한도 7,200건 = 초당 2건) |

추론 설정만 실험 조건별로 다르다. 세 모델에 공통 적용 가능한 설정이 없기 때문이며
([METHODOLOGY §2](./METHODOLOGY.md)), 조건 정의는 [README.md](./README.md)의 용어 표에 있다.

| 실험 조건 | Opus 5 | GPT-5.6-sol | GLM 5.2 |
|---|---|---|---|
| A (통일) | `thinking:{type:disabled}` | `reasoning_effort:"none"` | `reasoning_effort:"none"` |
| A′ (보정) | `thinking:{type:adaptive}` | — | — |
| B (최대) | `output_config:{effort:"high"}` | Responses 경로 + `reasoning.effort:"high"` | `reasoning_effort:"high"` |

## 채점

**LLM 채점자를 사용하지 않는다.** 산출물이 JSON과 도구 호출이라 프로그램으로
완전히 검증할 수 있다. 채점 규칙 전문은 [METHODOLOGY §6](./METHODOLOGY.md).

| 항목 | 규칙 |
|---|---|
| Function Calling | 도구 이름 일치 + 인자 대조. **스키마에 정의된 선택적 파라미터 추가는 오답이 아니다** |
| Structured Output | `jsonschema`(Draft 2020-12) 검증 + 필드 값 대조 |
| 비율 지표 | Wilson 95% 신뢰구간을 함께 보고. 구간이 겹치면 **차이를 확인할 수 없음**으로 판정 |
| 안정성 | `pass^k = C(성공, k)/C(시행, k)` — k회가 모두 성공할 확률 |
| 호출 판단 | 도구 호출률 / 무관한 요청 시 미호출 / 되물음 / 임의 값 생성 / 호출 누락을 분리 집계 |

실패는 **인프라 오류 · 게이트웨이 거부 · 형식 오류 · 호출 판단 오류**로 분류하며
섞지 않는다. 형식 오류만 0점 처리하고 나머지는 채점에서 제외하거나 별도 지표로 둔다.

## 실행 규칙

- 케이스와 정답은 `cases/build_cases.py`로 생성한다.
  상대 날짜의 정답이 실행일에 따라 달라지지 않도록 기준일(`2026-08-06`)을
  시스템 메시지로 고정한다.
- 정답 판정이 갈릴 수 있는 케이스는 `ambiguous=True`로 표시하고 집계에서 분리한다.
- 실행 전에 지원 기능을 확인한다(`probes/probe_capabilities.py`).
  게이트웨이 제약은 릴리스마다 변경된다.
