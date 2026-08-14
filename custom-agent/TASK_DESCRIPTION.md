# TASK_DESCRIPTION — Custom Agent 성능

## 무엇을 재는가

동일한 멀티턴 도구 사용 에이전트에 LLM 만 교체해, 실제 업무 태스크를 도구 루프로 풀게 하고
그 과정과 결과를 비교한다. 측정 축:

- 의존관계 있는 다단계 오케스트레이션 (뒤 단계가 앞 조회 결과에 의존)
- 도구 실패 후 대안 경로로의 복구
- 정보 부족 시 추측하지 않고 되묻는 판단
- 함정(그럴듯한 오답)과 최신 정보(학습 시점 이후 변경)에 대한 견고성
- 스텝 효율, 지연, 컨텍스트 누적 비용

Task 6(`function-calling-json`)은 단발 호출 정확도를 다룬다. 여기서는 멀티턴 오케스트레이션만 본다.

## 에이전트 루프

시스템 프롬프트 + 사용자 프롬프트로 시작한다. 모델이 도구를 호출하면 Mock 실행 결과를
대화에 붙여 반복하고, 텍스트만 내면 그것을 최종 답으로 종료한다. 최대 스텝은 케이스별로 정한다.
최종 답은 자유 텍스트(판정·금액·근거)이므로 구조화 출력(response_format)을 요구하지 않는다 —
그래서 tools + response_format 비호환 문제(Opus HTTP 400, GLM 도구 누락)를 밟지 않는다.

## 시나리오 1 — Azure Databricks 트러블슈팅

사용자가 보고한 장애를 로그·설정·리소스·공식 문서를 조회해 진단한다.
Azure 인프라(스토리지 권한·네트워킹)와 Databricks 제품(Unity Catalog·cluster policy·DBFS)을 섞었다.

| 케이스 | 근본 원인 | 함정 |
| --- | --- | --- |
| A-1 | management-plane 역할(Owner)만, data-plane RBAC 없음 | Owner 보이니 권한 맞다고 판단 |
| A-2 | Private DNS Zone 이 spoke VNet 에 미링크 | jumpbox 는 private IP 나오니 DNS 정상 |
| A-3 | UC READ FILES grant 누락 (Azure RBAC 는 정상) | Azure RBAC 를 의심 |
| A-4 | cluster policy node_type allowlist 위반 (fault 주입) | 코드·DBR·용량 의심 |
| A-5 | DBFS init script 지원 종료 — 이관 필요 (최신 자료) | DBFS 권한·경로 수정 시도 |
| A-6 | 관리자가 DBFS 루트/마운트 비활성화 (최신 자료) | 마운트 자격 증명 갱신 시도 |

도구: `get_cluster_events`, `get_logs`, `get_cluster_config`, `get_azure_resource(type,name)`,
`nslookup(host,source)`, `describe_uc_securable`, `get_uc_grants`, `get_cluster_policy`,
`get_job_run`, `get_doc(topic)`. 올바른 도구·인자 선택이 도구 선택 능력을 가른다.

## 시나리오 2 — 보험 약관 심사

실제 「삼성 다모은 건강보험(2403)」 규칙(`policy/samsung_damoeun_health.md`)을 근거로,
청구 건의 지급 여부와 금액을 판정한다. 조항 하나만 읽으면 틀리는 참조 구조라 멀티턴이 필수다.

| 케이스 | 정답 | 함정 |
| --- | --- | --- |
| B-1 | 부지급 0원 (보장개시 전 진단) | 감액 적용 500만원 |
| B-2 | 100만원 (통합소액암 200만 × 감액 50%) | 한 축만 적용 |
| B-3 | 125만원 (다빈치 500만 × 180일 이내 25%) | 50% 적용 250만원 |
| B-4 | 되묻기 (진단확정일 누락) | 계약일로 추측 |
| B-5 | 200만원 (갑상선암은 소액암, 1년 초과라 감액 없음) | 통합암 취급 1,000만원 |
| B-6 | 250만원 (다빈치 500만 × 180일 초과~1년 50%) | 25% 적용 125만원 |

도구: `search_clause`, `get_clause`, `get_contract`, `get_claim`, `lookup_disease_code`.
`search_clause` 없이 조항을 바로 찍는 경로를 막아, 조항 번호를 모르는 상태에서 탐색하게 한다.

## 데이터 재현성

- 모든 도구는 Mock — 케이스별 고정 fixture(`cases/*/*.json`) 와 시나리오 공유 자료
  (보험 조항 `cases/insurance_policy/_clauses.json`)를 반환한다. 외부 의존성이 없어 5회 반복이 가능하다.
- `fault` 필드로 특정 도구의 N번째 호출을 실패시켜 에러 복구를 측정한다(A-4).
- 표준 프롬프트는 각 케이스 JSON 의 `prompt`, 시스템 프롬프트는 `_system.txt`, 도구 스키마는 `_tools.json`.
  3 모델에 동일 입력을 준다.

## 정답의 신뢰성

정답은 Opus MAX 서브에이전트가 독립 재계산·원문 대조로 검증했다. 보험은 원문 PDF 수치까지
대조했고, Azure 는 공식 문서·에러 문자열을 교차 확인했다. `cases/*/ANSWERS.md` 에 근거를 남겼다.
