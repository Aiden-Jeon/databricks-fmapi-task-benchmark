# 평가 기준

> Task 5 — Custom Agent 벤치마크의 채점 기준. 세부 설계 배경은 `PLAN.md`.

## 시나리오

| ID | 시나리오 | 다루는 것 | 케이스 | 되묻기 | fault |
| --- | --- | --- | --- | --- | --- |
| 1 | Azure Databricks 트러블슈팅 | 증상 → 로그·설정·문서 조회 → 근본 원인·조치. Azure 인프라 2 + Databricks 제품 2 | A-1~A-4 | 없음 | A-4 |
| 2 | 보험 약관 심사 | 청구 → 조항·계약·진단 조회 → 지급 판정·금액 | B-1~B-4 | B-4 | 없음 |

각 케이스는 3모델(opus / sol / glm) × 5회 반복. 시나리오는 독립 채점하며 합산하지 않는다.

## 채점 항목 (단일 deterministic 스코어러)

| 항목 | 방법 | 배점 |
| --- | --- | --- |
| 최종 답 정확도 | `answer_key` 일치 (root cause / 지급 판정·금액) | Pass/Fail |
| 필수 도구 커버리지 | `required_tools` 를 모두 호출. **순서 미강제, 집합 판정** | 비율 |
| 인자 정확도 | `compare_args()` 로 도구 인자 검증 | 비율 |
| 함정 회피 | `trap_answers` 를 최종 결론으로 내지 않았는가 | Pass/Fail |
| 되묻기 | `should_ask` 케이스에서 `ASK_PAT` 매칭 (B-4) | Pass/Fail |
| 에러 복구 | `fault` 주입 후 재시도로 목표 달성 (A-4) | Pass/Fail |
| 스텝 효율 | 실제 턴 수 / `max_steps` | 비율 |
| 안정성 | `pass^5` — 5회 모두 통과 | 지표 |
| Latency | 세션 전체 median · p90 | 지표 |
| 비용 | `compute_usd()` + 턴별 누적 토큰 | 지표 |

- 최종 답 정확도는 Wilson 95% CI 병기.
- 정성 평가는 judge(`databricks-gemini-3-1-pro`) 1항목 — 최종 보고서를 실무에 바로 쓸 수 있는가 (1~5).
- **정답 정확도 판정 방식**: `answer_key` 는 정규화 후 문자열/수치 매칭. 금액은 정확히 일치해야 한다. root cause 는 핵심 키(예: `missing_data_plane_rbac`)를 최종 답이 담고 있는지로 본다. 경계 케이스는 judge 보조 판정.

## 케이스별 정답 요약

정답의 상세 근거는 각 시나리오 폴더의 `ANSWERS.md`.

| 케이스 | 정답 | 핵심 함정 |
| --- | --- | --- |
| A-1 | data-plane RBAC 누락 (Storage Blob Data Contributor 부여) | Owner 보이니 권한 맞다고 판단 |
| A-2 | Private DNS Zone 이 spoke VNet 미링크 (spoke 에 링크) | jumpbox private IP 나오니 DNS 정상 |
| A-3 | UC READ FILES grant 누락 (그룹에 grant 부여) | Azure RBAC 를 의심 (실제론 정상) |
| A-4 | cluster policy node_type allowlist 위반 (node_type 변경/policy 수정) | 코드·DBR·용량을 의심 |
| B-1 | 부지급 0원 (보장개시 전) | 감액 적용 500만원 |
| B-2 | 100만원 (통합소액암 200만 × 감액 50%) | 감액 미적용 200만원, 통합암 오분류 |
| B-3 | 125만원 (다빈치 500만 × 180일내 25%) | 50% 적용 250만원, 감액 미적용 500만원 |
| B-4 | 되묻기 (진단일 누락) | 계약일로 추측해 계산 |

## 표준화 프롬프트

각 케이스의 프롬프트는 `cases/<scenario>/<id>.json` 의 `prompt` 필드에 고정돼 있다. 시스템 프롬프트는 `_system.txt`, 도구 스키마는 `_tools.json` 으로 시나리오 내 전 케이스 공통이다. 3모델에 동일 입력을 준다.

## 결과 표 (README)

```
| Model | Scenario | Accuracy [95% CI] | Steps | Median Latency (ms) | USD | pass^5 |
```
