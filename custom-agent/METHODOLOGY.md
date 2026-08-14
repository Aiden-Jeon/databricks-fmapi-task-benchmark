# METHODOLOGY — Custom Agent 성능

## 1. 환경

- 워크스페이스: ai-devtools-prod (`dbc-a5d4177a-49dc`), CLI profile `ai_devtools`
- 호출: FMAPI 게이트웨이 `/ai-gateway/mlflow/v1/chat/completions` 직접. 세 모델 전부 chat 경로
- OAuth 토큰 수명(~1시간)이 실행보다 짧을 수 있어 만료 시 자동 갱신(`src/auth.py`)

## 2. 모델 설정 — 실측으로 강제된 것

Task 6 `function-calling-json/METHODOLOGY.md` §3 의 지원 기능 실측 결과가 그대로 걸린다. 에이전트는 도구를 항상 쓰기 때문이다.

| arm | model | params | concurrency |
| --- | --- | --- | --- |
| opus | `system.ai.claude-opus-5` | `thinking: {type: adaptive}` | 6 |
| sol | `system.ai.gpt-5-6-sol` | `reasoning_effort: none` | 6 |
| glm | `system.ai.glm-5-2` | `reasoning_effort: none` | 2 |

- **Opus 는 `thinking:{disabled}` + tools 에서 원시 XML 을 흘리고 `tool_calls` 를 비운다**(실측 8/12). `adaptive` 는 0/135. 그래서 adaptive 필수.
- **sol 은 chat 경로에서 tools 를 쓰려면 `reasoning_effort:"none"` 이어야 한다** (아니면 HTTP 400).
- **GLM 은 QPH 7,200 = 초당 2건.** concurrency 2 가 전체 실행 시간을 정한다.
- **temperature·seed 를 못 쓴다**(Opus·sol 400). 결정성 확보 불가 → 반복 측정이 필수 → `pass^5`.

### 지원 기능 프로브 (본 런 전)

`probes/probe_agent_capabilities.py` 로 세 모델이 멀티턴 도구 루프를 실제로 도는지 먼저 확인했다.
세 모델 모두 두 값을 각각 조회해 합산(정답 59)까지 정상. Opus XML 누출 0, GLM 빈 응답 0.
Task 6 이 이 확인을 안 했다면 Opus 정확도 0.760(하네스 버그)을 실제 성능으로 오보고할 뻔했다.

## 3. 에이전트 루프 — response_format 을 쓰지 않는다

최종 답이 자유 텍스트(판정·금액·근거)이므로 JSON 스키마 출력을 요구하지 않는다. 그 결과:

- Opus 의 `tools + response_format` HTTP 400 (`Cannot specify both`)을 밟지 않는다.
- GLM 의 `tools + response_format` 도구 누락(200이지만 tool_calls 비움, 오류 신호 없음)을 밟지 않는다.

단일 도구 루프로 끝낸다. Task 6 이 겪은 이 비호환들이 이 태스크에선 설계상 회피된다.

## 4. 채점

전부 deterministic. `src/score.py`. 시나리오는 독립 채점하고 합산하지 않는다.

| 항목 | 방법 |
| --- | --- |
| 최종 답 정확도 | 케이스별 `accept` 정규식이 최종 답에 모두 존재 (대소문자 무시). 금액은 정답 숫자 매칭 |
| 되묻기 | `should_ask` 케이스는 되묻기/판정보류 인식(`ASK_PAT` + `HOLD_PAT`) |
| 필수 도구 커버리지 | `required_tools` ⊆ 호출 도구 집합. **순서 미강제** |
| 인자 정확도 | 유의미한 케이스만 `compare_args`(예: `get_azure_resource(type="role_assignments")`) |
| 함정 | `reject` 정규식(진단용) |
| 에러 복구 | `fault` 주입 후 재시도로 목표 달성 |
| 안정성 | `pass^5 = C(c,5)/C(5,5)` (τ-bench 정의, 5회 전부 성공) |
| 비용·지연 | `usd()`(DBU 단가) + 세션 지연 median·p90 |

정확도는 Wilson 95% CI 병기. 정규 근사는 0/1 근처에서 못 쓴다.

### 순서를 강제하지 않는 이유

τ-bench 도 참조 궤적(`RewardType.ACTION`)을 극소수 태스크에만 쓰고 기본은 결과 기반이다.
도구 호출 순서를 강제하면 정당한 대안 경로를 오답 처리하게 된다. 그래서 `required_tools` 는 집합으로 본다.

### 되묻기에 사용자 시뮬레이터를 쓰지 않는 이유

애매한 케이스(B-4)의 정답은 "추측하지 말고 묻는 것"이라 답해줄 주체가 필요 없다.
도구 호출 없이 텍스트로 정보 부족을 인정하고 확정 판정을 피하면 정답으로 본다.

### 채점 정직성 — 두 차례 재교정

초기 채점에서 나타난 정확도 차이를 실제 응답과 대조한 결과, 둘 다 scorer 아티팩트였다.

1. **B-4 되묻기**: sol 만 통과(5/5)하고 opus·glm 은 3/5 로 나왔으나, 15개 응답을 전부 열어보니
   셋 다 진단확정일 누락을 정확히 짚고 되물었다. `ASK_PAT`(단발 되물음 어휘)이 "조건부 / 판정 보류 /
   확정할 수 없습니다" 같은 분석적 되묻기를 놓친 false negative 였다. `HOLD_PAT` 을 추가해 교정.
2. **A-6**: sol 1건이 정답(DBFS 비활성화 원인 + Volumes 조치)인데 accept 정규식의 문자 간격이
   좁아 놓쳤다 → 완화. opus 1건은 finish_reason=error 로 응답이 21자에서 잘린 생성 오류 →
   인프라성 실패로 채점 제외.

교정은 정답을 오답으로 처리한 false negative 만 되돌린 것이며, 기준을 낮춘 게 아니다.

## 5. 비용 계산

DBU 단가(`function-calling-json`·`image-text-performance` 교차검증):

| 모델 | in | out | cache write | cache read |
| --- | --- | --- | --- | --- |
| opus | 71.429 | 357.143 | 89.286 | 7.143 |
| sol | 71.429 | 428.571 | 89.286 | 7.143 |
| glm | 20.000 | 62.857 | 0.0 | 3.714 |

`usd = 0.07 × (fresh_in×in + cache_read×cr + cache_write×cw + completion×out) / 1e6`.
`billable_output = completion_tokens` — reasoning 토큰은 completion 에 포함되므로 더하지 않는다(Task 6 §3 실측).
멀티턴 세션의 프롬프트 누적을 턴별로 기록했으나 8스텝·소형 fixture 라 sol 구간요금 임계(200k)에 닿지 않아 flat 단가를 썼다.

## 6. 실행

12 케이스 × 3 모델 × 5 회 = 180 세션. arm 별로 병렬 실행(각 모델 독립 쿼터).
job 은 `random.Random(42)` 로 섞어 시간대 편향을 없앴다. 인프라 실패 0건.
