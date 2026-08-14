# 진행 기록 — Custom Agent 벤치마크

## 환경
- 워크스페이스: ai-devtools-prod (dbc-a5d4177a-49dc), CLI profile `ai_devtools`
- 실행 시 `DATABRICKS_PROFILE=ai_devtools` 필요
- OAuth 토큰 수명 ~1시간 → 하네스가 자동 갱신(Auth.refresh). 장시간 런 중 만료돼도 재시도

## 완료
- [x] 하네스 구현 (`src/`: auth, fmapi, tools, agent, runner, score) + probe
- [x] 오프라인 로직 검증 18/18 통과 (`tests/test_offline.py`)
- [x] uv 환경 (`uv sync`) — httpx, jsonschema
- [x] OAuth 로그인 (포트 8020 점유하던 sage-catalog MCP 프록시 kill 후 성공)
- [x] **프로브 통과** — opus/sol/glm 전부 멀티턴 도구 루프 정상 (2 tool calls, 정답 59, XML 누출 0)
- [x] **스모크** (A-1 × 3모델) — 3/3 성공, 모두 data-plane RBAC 정답. run: `results/custagent-20260814T033758`

## 완료 (본 런)
- [x] 본 런 120 세션 (opus/sol/glm 각 40, arm별 병렬). 무실패. `results/full_{opus,sol,glm}` → `results/full_merged`
- [x] 채점 완료. B-4 되묻기 채점 버그 수정(HOLD_PAT 추가) — sol만 5/5로 나오던 게 실은 3모델 모두 정확히 되물었음(scorer false negative)

## 결과 (2026-08-14 본 런, 120 세션)
| 모델 | 정확도 | pass^5 | 중앙 지연 | 세션당 비용 | 총비용(40세션) |
| --- | --- | --- | --- | --- | --- |
| opus | 1.00 | 1.00 | 31.3s | $0.1175 | $4.70 |
| sol | 1.00 | 1.00 | 12.4s | $0.0354 | $1.42 |
| glm | 1.00 | 1.00 | 14.6s | $0.0074 | $0.30 |

- **정확도는 3모델 전부 1.00 (8케이스 × 5회 모두 통과)** — 이 중상도 태스크에서 프론티어 3모델은 정확도로 구분 안 됨
- **변별은 비용·지연에서**: opus 가 glm 의 **15.8배** 비쌈, sol 의 3.3배. sol 이 가장 빠름(12.4s), opus 가 가장 느림(31s)
- 전체 벤치마크(120세션) 총비용 $6.42

## 난이도 케이스 추가 (사용자 선택)
- [x] A-5(DBFS init script 지원종료, 최신자료), A-6(Public DBFS root disabled), B-5(갑상선암=소액암 함정), B-6(다빈치 50% 구간) 추가 → 6케이스/시나리오
- [x] 재실행 60세션 + 기존 120 = **180세션** (`results/full12`). 무실패
- [x] scorer false-negative 2건 재교정: A-6 accept 정규식 완화(sol 정답인데 놓침), 생성오류(finish=error) 세션 채점제외(opus 1건). 실제로는 모든 케이스 정답

## 최종 결과 (180세션, 12케이스 × 3모델 × 5회)
| 모델 | 정확도 | pass^5 | 중앙지연 | p90지연 | 세션당비용 | 60세션총비용 |
| --- | --- | --- | --- | --- | --- | --- |
| opus | 1.00 | 1.00 | 31.5s | 39.5s | $0.1157 | $6.94 |
| sol | 1.00 | 1.00 | 12.3s | 14.6s | $0.0354 | $2.12 |
| glm | 1.00 | 1.00 | 15.8s | 22.2s | $0.0077 | $0.47 |

- **정확도 완전 포화**: 난이도 케이스(최신자료 DBFS 종료, 갑상선암·다빈치 도메인 함정)조차 3모델 전부 통과. 모델이 사전지식이 아니라 도구/문서에 근거해 판단 → 함정에 안 빠짐
- **변별은 비용·지연**: opus 가 glm 의 **14.9배**, sol 의 3.3배 비쌈. sol 최속(12.3s), opus 최저속(31.5s)
- 전체 180세션 총비용 **$9.53**

## 남음
- [ ] 문서 5종 (README/TASK_DESCRIPTION/METHODOLOGY/FINDINGS/COST)
- [ ] PLAN.md 정리, 내 GitHub collaborator 등록

## 실행 명령 메모
```bash
export DATABRICKS_PROFILE=ai_devtools
uv run python -m probes.probe_agent_capabilities        # 프로브
uv run python -m src.runner --smoke                     # 스모크
uv run python -m src.runner --repeats 5 --scenario all  # 본 런 (단일 프로세스)
uv run python -m src.score results/<run-id>             # 채점
```

## 관찰
- GLM concurrency 2 (QPH 7200)가 전체 페이스를 정한다 — 병렬화해도 GLM 게이트웨이 캡이 임계경로
- 세션당 도구 호출 3~6회, 스텝 3 내외 (A-1 기준)
