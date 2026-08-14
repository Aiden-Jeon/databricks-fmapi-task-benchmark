# Custom Agent 성능 벤치마크 (Task 5)

동일한 멀티턴 도구 사용 에이전트에 LLM 만 바꿔 끼워, 실제 업무형 태스크에서
**Opus 5 / GPT-5.6-sol / GLM-5.2** 의 런타임 성능을 비교한다.

Task 6(`function-calling-json`)이 단발 도구 호출 정확도를 다루므로, 여기서는 그와 겹치지 않는 축만 본다 —
**의존관계 있는 다단계 오케스트레이션, 도구 실패 후 복구, 되묻기 판단, 컨텍스트 누적 비용.**

## 시나리오 (각각 독립 검증)

| # | 시나리오 | 내용 | 케이스 |
| --- | --- | --- | --- |
| 1 | Azure Databricks 트러블슈팅 | 증상 → 로그·설정·문서 조회 → 근본 원인·조치. Azure 인프라 + Databricks 제품 레이어 혼합 | A-1~A-6 |
| 2 | 보험 약관 심사 | 청구 → 조항·계약·진단 조회 → 지급 판정·금액. 실제 「삼성 다모은 건강보험」 규칙 기반 | B-1~B-6 |

12 케이스 × 3 모델 × 5 회 = **180 세션**.

## 결과

| Model | Scenario | Accuracy [95% CI] | Steps | Median Latency | USD/session | pass^5 |
|---|---|---|---:|---:|---:|---:|
| opus | azure_troubleshoot | 1.00 [0.88,1.00] | 3.8 | 36.0s | $0.1253 | 1.00 |
| opus | insurance_policy | 1.00 [0.89,1.00] | 4.7 | 28.0s | $0.1068 | 1.00 |
| sol | azure_troubleshoot | 1.00 [0.89,1.00] | 3.1 | 11.5s | $0.0293 | 1.00 |
| sol | insurance_policy | 1.00 [0.89,1.00] | 4.4 | 13.2s | $0.0414 | 1.00 |
| glm | azure_troubleshoot | 1.00 [0.89,1.00] | 3.4 | 16.8s | $0.0073 | 1.00 |
| glm | insurance_policy | 1.00 [0.89,1.00] | 5.0 | 15.3s | $0.0081 | 1.00 |

### 요약

- **정확도로는 세 모델을 구분할 수 없다.** 12 케이스 × 5 회를 셋 다 전부 통과(pass^5=1.00). 난이도를 높인 케이스 — 학습 시점 이후 바뀐 규칙(DBFS init script 지원 종료), 강한 도메인 함정(갑상선암은 일반암이 아니라 소액암, 다빈치 수술 단계별 감액) — 조차 전부 맞혔다. 모델이 사전지식이 아니라 **도구·문서에 근거해 판단**하기 때문에 함정에 빠지지 않았다.
- **변별은 비용과 지연에서 극명하다.** opus 가 glm 의 **14.9배**, sol 의 3.3배 비싸다. 지연은 sol 이 가장 빠르고(중앙 12.3초), opus 가 가장 느리다(31.5초). **정확도가 동률이면 glm 이 경제성에서 압도한다.**

상세: [FINDINGS.md](./FINDINGS.md), [COST.md](./COST.md), 방법론: [METHODOLOGY.md](./METHODOLOGY.md), 태스크 명세: [TASK_DESCRIPTION.md](./TASK_DESCRIPTION.md)

## 재현

```bash
export DATABRICKS_PROFILE=ai_devtools          # ai-devtools-prod
uv sync
uv run python -m probes.probe_agent_capabilities        # 지원 기능 사전 확인
uv run python -m src.runner --repeats 5 --scenario all  # 본 런
uv run python -m src.score results/<run-id>             # 채점
```

- 모델 설정, 플랫폼 제약, 채점 규칙은 [METHODOLOGY.md](./METHODOLOGY.md).
- 정답 근거: `cases/*/ANSWERS.md`, 보험 약관 발췌: `policy/samsung_damoeun_health.md`.
