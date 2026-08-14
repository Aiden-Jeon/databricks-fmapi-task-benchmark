# Task 5 — Custom Agent 벤치마크

> 작업용 문서. 최종 산출물(README / TASK_DESCRIPTION / METHODOLOGY / FINDINGS / COST) 완성 후 삭제한다.

## Context

한국 AI SME 그룹이 Opus 5 / GPT-Sol / GLM 세 모델을 실제 업무 태스크에 붙여 비교하는 벤치마크를 진행 중이다. 태스크 6개를 나눠 맡았고 내 몫은 **Task 5 — Custom Agent 성능**이다.

동일한 멀티턴 도구 사용 에이전트에 LLM 만 스왑해 성능을 비교한다. 측정 축은 의존관계 있는 다단계 오케스트레이션, 도구 실패 후 복구, 되묻기 판단, 컨텍스트 누적 비용이다. Task 6(`function-calling-json`)이 단발 도구 호출 정확도를 케이스 1,113건으로 이미 다루므로 그쪽과 겹치지 않는 축만 본다.

레포에는 이미 5개 태스크가 구현돼 있다. 백지에서 시작하는 게 아니라 **기존 하네스·문서 규격에 내 태스크를 끼워 넣는 작업**이다.

| # | Task | 담당 | 폴더 |
| --- | --- | --- | --- |
| 1 | Databricks App 개발 | Jiyu Kim | `databricks-app-generation/` |
| 2 | MLE Benchmark | HYUN SUNG KIM | `mle-benchmark/` |
| 3 | HTML 슬라이드 | Jongseob Jeon | `html-slide-generation/` |
| 4 | 이미지 + 텍스트 이해 | Stefano (Woohyuk) Jang | `image-text-performance/` |
| **5** | **Custom agent 성능** | **Jaewoo Park** | `custom-agent/` |
| 6 | Function Calling & JSON | SeungWon Ji | `function-calling-json/` |

환경 — 워크스페이스 `ai-devtools-prod` (https://dbc-a5d4177a-49dc.cloud.databricks.com/), CLI profile `ai_devtools`. FMAPI 게이트웨이 직접 호출.

| 별칭 | 모델 ID |
| --- | --- |
| Opus 5 | `system.ai.claude-opus-5` |
| GPT-Sol | `system.ai.gpt-5-6-sol` |
| GLM | `system.ai.glm-5-2` |
| Judge | `databricks-gemini-3-1-pro` |

---

## 1. 설계

| 항목 | 결정 |
| --- | --- |
| 시나리오 | 2개 — `azure_troubleshoot`, `insurance_policy`. **각각 독립 검증** (합산 점수 없음) |
| 케이스 | 시나리오당 4건, 각 5회 반복 |
| 모델 설정 | 모델별로 **도구 사용이 실제 동작하는 단일 설정** |
| 에이전트 루프 | 2단계 — Phase 1 도구 루프(`tools` only) → Phase 2 최종 정리(`response_format` only) |
| 채점 | deterministic 단일 스코어러 + judge 1항목 |
| 규모 | 2 × 4 × 3모델 × 5회 = **120 세션**, 약 900 호출 |

---

## 2. 플랫폼 제약 — 설계의 근거

`function-calling-json/METHODOLOGY.md` §3 의 실측 결과를 확인했다. **에이전트는 도구를 항상 쓰기 때문에 이 제약이 정면으로 걸린다.**

| 제약 | 내용 | 대응 |
| --- | --- | --- |
| `tools` + `response_format` | Opus **HTTP 400** (`Cannot specify both`), GLM 200이지만 **도구 호출 누락**(30/30, 오류 신호 없음), sol 정상 | **2단계 루프 분리가 필수** |
| sol + tools (chat 경로) | **전부 400.** `reasoning_effort` 생략해도 발생 | sol 은 `/ai-gateway/codex/v1/responses` 또는 `reasoning_effort:'none'` |
| Opus `thinking:{disabled}` + tools | 원시 XML 흘리고 `tool_calls` 비움 (6/135). `adaptive` 는 0/135 | Opus 는 `thinking:{type:adaptive}` |
| `temperature` / `seed` | Opus·sol 400 | 온도 고정 불가 → 반복 측정 필수, `pass^5` |
| GLM 응답 파싱 | `content` 비고 `reasoning_content` 에 답이 오는 경우 | `_normalize_content()` 이식 |
| GLM 처리량 | QPH 7,200 = 초당 2건, 나머지의 1/50 | concurrency 2. 전체 페이스를 GLM 이 정한다 |
| 비용 공식 | Task 6 은 `billable_output = completion_tokens` (reasoning 이 이미 포함), Task 4 는 더한다 | **Task 6 따름** (측정 근거 있음). 상충은 팀에 보고 |
| sol 구간 요금 | 200k 초과 시 in 2배, out 1.5배 | 멀티턴은 실제로 임계에 닿는다. 턴별 누적 토큰 기록 |

Task 6 은 사전 확인을 했기 때문에 Opus 정확도 0.760(하네스 버그)을 0.920(실제)으로 바로잡았다. 확인 없이 돌리면 하네스 버그를 모델 성능으로 보고한다. **그래서 프로브가 0단계다.**

경로 상수 (`function-calling-json/src/runner.py:43`)

```python
GATEWAY_PATH   = "/ai-gateway/mlflow/v1/chat/completions"
RESPONSES_PATH = "/ai-gateway/codex/v1/responses"
```

---

## 3. 재사용 자산

새로 짜기 전에 가져온다. 중복 구현하면 결과 포맷이 어긋난다.

**`function-calling-json/src/runner.py`**
- `Auth` — `databricks` CLI subprocess, thread-safe refresh, 20초 쿨다운
- `is_auth_expiry()` — 401/403 을 권한 오류와 토큰 만료로 구분
- `normalize_usage()` — chat/responses 토큰 필드 통일
- `ARMS` 구조
- job shuffle + per-arm semaphore

**`function-calling-json/src/score.py`**
- `wilson(k, n)` — 95% CI
- `pass_hat_k(n, c, k)` = `C(c,k)/C(n,k)` — τ-bench 정의와 일치
- `compare_args()` — 인자 일치 + 스키마 밖 파라미터를 hallucination 판정
- `score_so()` — JSON schema 검증
- `usd()`, `PRICING`(40)
- `ASK_PAT`(47) — 되묻기 판정 정규식

**`image-text-performance/`**
- `config/pricing.yaml` — 구간 요금 포함 단가표, `usd_per_dbu: 0.07`
- `src/cost/pricing.py` — `compute_usd()`. reasoning 합산 부분만 2절대로 수정
- `src/cost/usage.py` — `build_usage_query()`, `ai_gateway.usage` 조인
- `src/adapters/fmapi.py` — `_normalize_content()`
- `src/results.py` — `SampleResult`, `RunManifest`

**`function-calling-json/probes/probe_capabilities.py`** — 프로브 패턴

---

## 4. 산출물 구조

```text
custom-agent/
├── README.md  TASK_DESCRIPTION.md  METHODOLOGY.md  FINDINGS.md  COST.md
├── pyproject.toml / uv.lock
├── config/{models.yaml, pricing.yaml}
├── probes/probe_agent_capabilities.py
├── policy/synthetic_policy.md
├── src/{auth.py, agent.py, runner.py, score.py, report.py, tools/}
├── cases/{azure_troubleshoot/, insurance_policy/}
├── results/<run_id>/{manifest.json, raw.jsonl, scored.jsonl, report.json}
└── grade_results.json
```

### 케이스 스키마

```json
{
  "id": "A-1",
  "scenario": "azure_troubleshoot",
  "prompt": "SP 에 Owner 를 줬는데 /data/dept-a/ 만 403 이 납니다",
  "tools": [ /* OpenAI function schema */ ],
  "fixtures": { "get_logs": "...", "get_azure_resource:role_assignments": "..." },
  "fault": null,
  "expect": {
    "answer_key": "missing_data_plane_rbac",
    "required_tools": ["get_logs", "get_azure_resource"],
    "trap_answers": ["subscription_owner_scope", "sas_token"],
    "max_steps": 8,
    "should_ask": false
  }
}
```

### 턴 레코드

Task 6 `raw.jsonl` 스키마에 더한다.

```
session_id, turn_index, phase("tools"|"final"),
cumulative_prompt_tokens,   # sol 200k 임계 추적
tool_name, tool_args, fault_triggered, recovered_after_fault
```

---

## 5. 시나리오 1 — Azure Databricks 트러블슈팅

Azure 에서 Databricks 를 쓰다 생긴 문제를 공식 문서를 참조해 진단한다. 권한 / DNS / 방화벽 / 쿼터로 레이어를 흩어 한 레이어에 강한 모델이 전체를 쓸어가지 않게 한다.

| # | 증상 | 근본 원인 | 함정 |
| --- | --- | --- | --- |
| **A-1** | Owner 권한인데 하위 경로만 403 `AuthorizationPermissionMismatch` | management-plane 역할만 있고 data-plane RBAC(`Storage Blob Data Contributor`) 없음 | IAM 에 Owner 가 보이니 권한은 맞다고 확신 |
| **A-2** | jumpbox 는 private IP 나오는데 클러스터는 부트스트랩 타임아웃 | Private DNS Zone 이 spoke VNet 에 미링크 | 내 PC 에서 private IP 나오니 DNS 정상 |
| **A-3** | 방화벽 trusted services 켰는데 여전히 403 | SP OAuth 트래픽은 trusted services 예외 대상이 아님 | ADF 는 같은 설정으로 동작 중 |
| **A-4** | vCPU 여유 있는데 `AZURE_QUOTA_EXCEEDED_EXCEPTION` | 총량은 여유, VM 계열별 쿼터 소진 | Portal 총 vCPU 화면만 보면 여유 |

### Mock 도구 7개

| 도구 | 반환 |
| --- | --- |
| `get_cluster_events(cluster_id)` | 이벤트 로그. `BOOTSTRAP_TIMEOUT_DUE_TO_MISCONFIG`, `AZURE_QUOTA_EXCEEDED_EXCEPTION` 등 |
| `get_logs(cluster_id)` | 드라이버 로그 발췌. `AuthorizationPermissionMismatch` 등 |
| `get_cluster_config(cluster_id)` | `node_type_id`, spark conf (ADLS 접근 방식) |
| `get_azure_resource(type, name)` | `type` = `nsg` / `dns_zone_links` / `storage_account` / `role_assignments`. 올바른 type 선택이 도구 선택 축 |
| `nslookup(host, source)` | `source` = `jumpbox` / `cluster`. A-2 의 핵심 |
| `get_vm_quota(region, family=None)` | 생략 시 총량, 지정 시 계열별. A-4 의 핵심 |
| `get_doc(topic)` | 공식 문서 발췌 |

`get_doc` 은 발췌 fixture 로 고정한다. 실제 웹 조회는 5회 반복 시 결과가 흔들려 재현성이 깨진다.

**에러 복구** — A-2 또는 A-4 에 `fault` 를 주입해 첫 조회 실패 후 대안 경로를 찾는지 본다.

### 근거 문서

- ADLS Gen2 접근 제어 (data-plane vs management-plane) — https://learn.microsoft.com/en-us/azure/storage/blobs/data-lake-storage-access-control-model
- Storage 403 트러블슈팅 — https://learn.microsoft.com/en-us/troubleshoot/azure/azure-storage/blobs/authentication/storage-troubleshoot-403-errors
- Private Endpoint DNS 트러블슈팅 — https://learn.microsoft.com/en-us/troubleshoot/azure/private-link/troubleshoot-private-endpoint-dns-resolution
- Storage 방화벽 trusted services — https://learn.microsoft.com/en-us/azure/storage/common/storage-network-security-trusted-azure-services
- Azure core limit (쿼터) — https://learn.microsoft.com/en-us/azure/databricks/kb/clusters/azure-core-limit
- 클러스터 에러 코드 — https://learn.microsoft.com/en-us/azure/databricks/compute/troubleshooting/cluster-error-codes

---

## 6. 시나리오 2 — 보험 약관

긴 문맥 이해, 조건 중첩, 참조 체인 추적을 본다. 조항 하나만 읽으면 답이 틀리는 구조라 멀티턴이 필수다.

### 합성 약관을 쓴다

표준약관(`보험업감독업무시행세칙` 별표15)의 조문 번호 체계·문장 투·용어를 따르되 **조항과 사실관계는 직접 작성한다.** 정답이 유일하게 도출되도록 설계로 보장한다.

- 저작권 문제가 없다 (개별 보험사 약관은 재배포 불가)
- 정답이 하나로 결정된다 (실제 분쟁 사례는 해석이 갈린 것들이라 벤치마크 정답으로 부적합)
- 참조 체인 깊이와 기간 중첩을 원하는 만큼 심을 수 있다

발표에서 "실제 약관과 같은 구조의 합성 약관"이라고 밝힌다.

| # | 상황 | 정답 | 함정 |
| --- | --- | --- | --- |
| **B-1** | 계약 2026-01-10, 진단 2026-03-20(70일). 책임개시일은 계약 90일 후, 이후 1년 50% 감액. 가입금액 5,000만원 | **부지급 0원** | 감액 적용해 2,500만원. 90일 대기가 감액보다 먼저 걸린다 |
| **B-2** | 계약 2025-06-01, 진단 2026-02-15, 제자리암. 유사암은 일반암의 20%, 계약 후 1년 50% 감액. 가입금액 3,000만원 | **300만원** (3,000 × 20% × 50%) | 한 축만 적용해 600만원 또는 1,500만원 |
| **B-3** | 보통약관은 계약일 개시, 암특약은 특약 체결일 +90일 개시. 계약 2025-11-01, 특약 추가 2026-01-15, 진단 2026-03-01 | 보통약관 **지급** / 특약 **부지급**, 분리 답변 | 한쪽만 보고 단일 결론 |
| **B-4** | 진단일이 명시되지 않은 청구 (`should_ask: true`) | **되묻기** | 계약일 기준으로 추측해 계산 |

### Mock 도구 5개

| 도구 | 반환 |
| --- | --- |
| `search_clause(keyword)` | 키워드가 든 조항 번호 목록 |
| `get_clause(article_no)` | 조항 원문 |
| `get_contract(contract_id)` | 계약일, 생년월일, 가입금액, 특약 목록 |
| `get_claim(claim_id)` | 진단일, 진단코드, 청구액 |
| `lookup_disease_code(code)` | 약관상 분류 (일반암 / 유사암 / 해당없음) |

조항 번호를 모르는 상태에서 시작하게 만든다. `search_clause` 없이 `get_clause` 로 정답 조항을 찍는 경로를 막는다.

### 근거

- 보험업감독업무시행세칙 별표15 표준약관 — https://law.go.kr/LSW/admRulLsInfoP.do?admRulSeq=2200000080415

---

## 7. 채점

단일 스코어러. 전부 deterministic.

| 항목 | 방법 |
| --- | --- |
| 최종 답 | `answer_key` 일치. 시나리오 1은 root cause, 2는 지급 판정 + 금액 |
| 필수 도구 커버리지 | `required_tools` 를 다 불렀는가. **순서 미강제, 집합 판정** |
| 인자 정확도 | `compare_args()` — 예: `get_azure_resource(type="role_assignments")` |
| 되묻기 | `should_ask` 케이스에서 `ASK_PAT` 매칭 |
| 함정 회피 | `trap_answers` 를 최종 결론으로 냈는가 |
| 에러 복구 | `fault` 주입 후 대안 경로로 목표 달성 |
| 안정성 | `pass^5` |
| 효율 / 비용 | 스텝 수 / `max_steps`, latency median·p90, USD |

정확도는 Wilson 95% CI 병기. 정성은 judge(`databricks-gemini-3-1-pro`) **1항목** — 최종 보고서를 실무에 바로 쓸 수 있는가 (1~5).

도구 호출 순서를 강제하지 않는 이유 — τ-bench 도 `RewardType.ACTION` 을 극소수 태스크에만 쓰고 기본은 결과 기반이다. 정당한 대안 경로를 오답 처리하지 않기 위해서다.

되묻기에 사용자 시뮬레이터를 두지 않는 이유 — 애매한 케이스의 정답은 "추측하지 말고 묻는 것"이므로 답해줄 주체가 필요 없다. 도구 호출 없이 텍스트가 나오면 `ASK_PAT` 으로 되묻기인지 판정하면 끝이다.

### README 결과 표

```
| Model | Scenario | Accuracy [95% CI] | Steps | Median Latency (ms) | USD | pass^5 |
```

### 선례

- τ-bench (pass^k 정의) — https://arxiv.org/abs/2406.12045
- τ²-bench 채점 (DB 해시, RewardType) — https://github.com/sierra-research/tau2-bench/blob/main/docs/evaluation.md
- BFCL v3 multi-turn (상태 기반 채점의 읽기 전용 한계) — https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html

---

## 8. 작업 순서

### 0단계 — 지원 기능 프로브

`probes/probe_agent_capabilities.py` 로 에이전트 특유 조합을 3모델에 실측한다.

- 도구 결과를 대화에 넣고 3턴 이상 이어갈 때 각 모델 거동
- sol Responses 경로에서 멀티턴 도구 루프가 되는지
- Opus `adaptive` + tools 를 멀티턴 반복해도 XML 누출이 0인지
- GLM `reasoning_content` fallback 이 멀티턴에서도 필요한지
- 2단계 분리가 3모델 전부 성립하는지

**완료 기준** — 결과 표가 `METHODOLOGY.md` 에 기록되고, 3모델 모두 멀티턴 도구 루프가 동작하는 설정이 확정된다.

### 1단계 — 합성 약관

`policy/synthetic_policy.md`. 15~25개 조항. 용어 정의 → 지급사유 → 세부규정(대기·감액) → 면책 → 특약. 참조 체인 2~3단을 의도적으로 심는다.

**완료 기준** — B-1~B-4 정답을 약관만 보고 손으로 도출했을 때 답이 하나로 나온다.

### 2단계 — 케이스 8건 + fixture

시나리오 1은 에러 메시지 전문·설정 덤프·문서 발췌, 시나리오 2는 계약·청구 메타데이터. `answer_key`, `required_tools`, `trap_answers` 정의.

**완료 기준** — fixture 만 읽고 사람이 정답에 도달한다. 도달 못 하면 단서가 부족한 것이다.

### 3단계 — 하네스

- `src/auth.py` — `Auth` 이식
- `src/agent.py` — 2단계 루프, 프로토콜 분기(chat / responses)
- `src/tools/` — Mock 도구 registry + fixture 로더 + fault 주입
- `src/runner.py` — `normalize_usage`, shuffle+semaphore 이식 후 세션 루프
- `src/score.py` — `compare_args`, `score_so`, `wilson`, `pass_hat_k`, `ASK_PAT` 이식 후 세션 단위 집계
- 구간 요금 반영한 `usd()` (reasoning 합산 안 함)

**완료 기준** — 케이스 1건 × 1모델 × 1회가 끝까지 돌고 `raw.jsonl` 에 턴 레코드가 남는다.

### 4단계 — 파일럿

케이스 2건 × 3모델 × 1회.

**완료 기준** — 실측치로 `BUDGET`·`REFERENCE_USD` 를 정하고, sol 의 `cumulative_prompt_tokens` 가 200k 에 접근하는지 확인한다.

### 5단계 — 본 런

120 세션. GLM concurrency 2 가 페이스를 정한다.

### 6단계 — 문서

`README` / `TASK_DESCRIPTION` / `METHODOLOGY` / `FINDINGS` / `COST` 5종 작성 후 이 파일 삭제.

---

## 9. 검증 방법

| 대상 | 방법 |
| --- | --- |
| 프로브 | `uv run python probes/probe_agent_capabilities.py` — 3모델 × 조합별 성공/실패 표 |
| 합성 약관 | 4개 케이스 정답을 약관만 보고 수동 도출. 답이 갈리면 조항 수정 |
| fixture | fixture 만 보고 정답 도달 가능한지 수동 확인 |
| 하네스 | 1케이스 × 1모델 × 1회 스모크. `raw.jsonl` 턴 레코드 검사 |
| 스코어러 | 정답 궤적과 오답 궤적을 손으로 만들어 넣고 채점이 갈리는지 확인 |
| 비용 | `build_usage_query()` 로 `ai_gateway.usage` 조인해 자체 계산값과 대조 |
| 본 런 | `pass^5` 가 0 또는 1 로만 나오면 케이스가 너무 쉽거나 어려운 것 — 난이도 재조정 |

---

## 10. 팀에 확인할 것

- [ ] **비용 공식 상충** — Task 4 와 Task 6 이 reasoning 토큰 처리에서 정반대. Task 4 가 이중 계상 중일 여지
- [ ] **모델 ID 표기** — Task 6 은 `system.ai.*`, Task 4 는 `databricks-*`
- [ ] **sol 구간 경계 200k** — 원 작성자가 `CONFIRM` 상태로 남김. 내 멀티턴 런으로 실측 확정 가능
- [ ] 내 GitHub 계정 전달 (collaborator 미등록)
- [ ] 표준약관 공공누리 유형 확인 (제1유형이면 출처표시만으로 가능)
