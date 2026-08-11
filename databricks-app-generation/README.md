# databricks-app-generation

여러 파운데이션 모델(FMAPI) / 에이전트 하네스를 **동일 프롬프트로** 실행하고 **동일 채점 + 사람 리뷰**로 비교하는 벤치마크 태스크: "**Databricks App(데이터 대시보드) 빌드**". "산출물을 만드는 단계"만 후보마다 교체하고, 프롬프트·지시문·예산·채점은 100% 동일하게 둔다.

> **상태: 실 워크스페이스 드라이런 완료 — `direct-fmapi` baseline 결과 확보 (n=1).** 워크스페이스 `ai-devtools-prod`, 채점 전용 serverless SQL warehouse에서 opus/sol/glm 3후보를 전 tier(1·2·3) 실행·채점했다. 결과는 아래 [테스트 결과](#테스트-결과-direct-fmapi-baseline-n1) 참조.
>
> - ✅ 전 tier GUI 채점 + tier2 E1–E4 엔지니어링 감사(Query History 연동) + tier3 robustness(R1 장애 부팅·R3 퍼저) 실 워크스페이스 검증 완료.
> - ⏳ **`pricing.json` 단가 미설정** — 비용 축은 N/A (final = 품질 0.7 + 시간 0.3으로 재정규화). 조직 FMAPI/DBU 요율표 확정 시 채우면 소급 계산됨. 토큰 수는 이미 기록.
> - **잔여 OPEN ITEM:** ② R2 중도 타임아웃 주입 ④ 배포 앱 OAuth 통과 스모크 ⑤ 차트 값 검증(Vega spec 파싱) + tier3 X1/X2 확장 케이스 GT 배선.
> - **다음 라운드 권장:** n=1은 실행 간 노이즈가 승부를 뒤집을 수 있으므로, 본 라운드는 **n≥3 반복(평균±표준편차)**과 **에이전트 하네스**로 확장할 것.

## TL;DR — 결과 요약

`direct-fmapi`(단발 completion, raw baseline)로 3후보를 전 tier 실행·채점한 **1회(n=1)** 결과. 비용 축은 단가 미설정으로 N/A.

| 후보 | 모델 (FMAPI) | tier1 | tier2 | tier3 | **final** | 총 소요 |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **opus** | `databricks-claude-opus-4-8` | 1.00 | 0.88 | 0.56 | **0.82** | 96초 |
| **sol** | `databricks-gpt-5-6-sol` | 1.00 | 0.13* | 0.65 | **0.53** | 352초 |
| **glm** | `databricks-glm-5-2` | 1.00 | 0.00† | 0.00† | **0.24** | 1,328초 |

- **tier1(게이트)은 세 모델 모두 만점** — 기본 App 빌드는 프론티어 모델에 천장 효과. 변별은 tier2·tier3에서 난다.
- **opus가 종합 1위** — 품질·속도 모두 우세(같은 tier1 결과에도 glm보다 20배 빠름).
- **tier3에서 프로파일이 갈림** — opus는 레거시 수리, sol은 장애 강건성(R1+R3 만점)에서 앞선다.
- \* sol tier2는 채점 중 사내 pip 프록시 장애로 부팅 실패(앱 결함 아님) → 저평가, 재채점 필요.
- † glm은 tier2·tier3에서 `max_tokens` 폭주로 산출물 미완성(정적 탈락). tier1은 완성.
- **n=1 주의** — 확정 순위는 n≥3 반복이 필요하다. 상세는 [테스트 결과](#테스트-결과-direct-fmapi-baseline-n1).

## 왜 이 태스크인가 (리서치 요약)

**Databricks Apps란:** Databricks 서버리스 위에서 컨테이너로 도는 웹 앱. Streamlit/Dash/Gradio(Python), Node.js 프레임워크 지원. 앱은 `app.yaml` 매니페스트로 실행 커맨드·환경변수·리소스를 선언하고, SQL warehouse / model serving endpoint / Genie / secret / volume 등 워크스페이스 리소스에 서비스 프린시펄(앱 인증) 또는 OBO(사용자 인증)로 접근한다. 리소스 ID 하드코딩 금지(선언→배포 시 바인딩)가 공식 베스트 프랙티스다.

**기존 오픈소스 벤치마크에서 가져온 것:**

| 벤치마크 | 무엇을 재나 | 우리가 차용한 것 |
|---|---|---|
| WebGen-Bench (arXiv:2505.03733) | 스펙 → 웹앱 from-scratch 생성, 101태스크 | "operation → expected result" 형식의 GUI 테스트 케이스, 웹 내비게이션(Playwright)으로 자동 실행, 기능+외관 이원 채점 |
| FullStack-Bench (arXiv:2602.03798) | 프론트+백엔드+DB 테스트 | 레이어별 검증 (UI만 아니라 데이터 정합성까지) |
| AppForge (arXiv:2510.07740) | 스펙 → Android 앱, 컴파일→에뮬레이터 배포→기능 테스트 | **compile→deploy→test 파이프라인**과 4대 메트릭(빌드 성공률·테스트 통과율·크래시율·종합 점수) |

**차별점:** 범용 웹앱 생성이 아니라 **Databricks Apps 규약 준수**(app.yaml, 리소스 선언, SDK 사용, 하드코딩 금지)와 **실 데이터 정합성**(SQL warehouse로 UC 테이블 조회)을 잰다. Field Engineering 관점에서 고객이 실제로 시키는 일과 같다.

## 태스크 정의 — 3-Tier 스위트

단일 태스크는 프론티어 모델 간 천장 효과가 우려된다는 리뷰(AppForge 18.8%·WebGen-Bench 27.8%가 보여주듯 태스크가 어려워야 격차가 벌어짐)를 수용해, 난이도 3단 구성으로 재편했다. 각 tier는 자체 `TASK_DESCRIPTION.md`(canonical, `instructions.txt`로 byte-copy)와 `test_cases.json`을 가진다. 스위트 구성은 `suite.json`.

| Tier | 태스크 | 가중치 | 예산 | 무엇을 가르나 |
|---|---|---|---|---|
| **1 — gate** | `samples.nyctaxi.trips` Streamlit 대시보드 from-scratch (KPI·필터·차트·테이블) | 0.15 | 15분 | 파이프라인 검증 + "App을 만들 수 있는가". **auto_score < 0.5면 스위트 전체 0점** |
| **2 — core** | `samples.tpch` 매출 분석 앱 — 3-테이블 조인 KPI, MoM 윈도우 함수, 상호작용 필터 2개, 페이지네이션, CSV, **파라미터 바인딩·쿼리 수 예산·캐싱 계약** | 0.50 | 30분 | 설계·SQL·엔지니어링 감각 (모델 변별의 중심) |
| **3 — differentiator** | 결함 8개가 심어진 레거시 앱(`tier3-differentiator/legacy_app/`) **수리 + 기능 추가 + 장애 강건성**(장애 주입·GUI 퍼저) | 0.35 | 25분 | 코드 이해·디버깅 축 — 생성력과 별개로 격차가 가장 크게 벌어지는 영역 |

프레임워크는 **Streamlit으로 고정** — 포맷 계약의 일부다. 이 벤치마크는 모델 비교가 목적이므로 산출물 형태를 byte-identical 지시문으로 고정해야 채점기 하나로 공정 비교가 된다.

**결정성 규칙 (전 tier 공통, 그라운드트루스와 앱이 같은 해석을 하도록 계약에 명문화):** 날짜 경계는 반개방 `[start 00:00, end+1d 00:00)`, 테이블 정렬 고정(ORDER BY 명시), top-N 동률은 보조 키 ASC로 tie-break, NULL 차원 값은 제외, 필터 변경 시 재조회 범위 명시(KPI 캐시 유지 등).

**채점 원칙 — 계약 준수 vs 기능 결함 분리:** GUI 테스트 셀렉터는 `key=` 우선 + 시맨틱 폴백(레이블·role·차트 존재) 2단으로 동작해, 키 오탈자 같은 순수 지시-준수 실수는 기능 점수를 깎지 않고 별도 `contract_compliance` 메트릭(정적 검증 내 소항목)으로만 보고한다. 반대로 LIMIT 후 집계(숫자 오답), app.yaml 오류(배포 불가), connector 누락(부팅 불가) 같은 **사용자에게 보이는 결함**은 기능 점수에서 그대로 감점한다. 구분 기준: "실 사용자에게 보이는가".

**후속 태스크 로드맵** (같은 뼈대 재사용, 별도 디렉토리):

- `databricks-app-chatbot` — serving endpoint 리소스로 챗봇 앱 (LLM 출력 비결정성 → LLM-judge 채점 필요, 그래서 2순위)
- `databricks-app-genie` — Genie Agents API 연동 대화형 분석 앱
- ~~`databricks-app-repair`~~ — **tier3-differentiator로 스위트에 편입됨**
- OBO(사용자 인증)·async/streaming 태스크 — 방향은 유효하나 사용자 신원 시뮬레이션·비결정성 채점 문제로 자동 채점 설계가 선행돼야 함 (리뷰 2차에서 제안됐으나 보류)

## 구조 (계획)

```
databricks-app-generation/       # 스위트 = 디렉토리 하나, tier = 하위 디렉토리
├── pyproject.toml               # 패키지 + console_scripts (run-task / grade-task)
├── suite.json                   # tier 구성·가중치·게이트 규칙·효율 축 설정
├── pricing.json                 # 모델별 토큰 단가 (비용 축)
├── src/benchmark/               # 러너·채점기 패키지
│   ├── task_spec.py                 # COMMON_PROMPT + 태스크 로더 (재사용)
│   ├── run_task.py                  # 후보 1개 × tier 1개 실행 → <tier>/<candidate>/app/
│   ├── grade_tasks.py               # Phase A 자동 채점 + 갤러리 생성
│   ├── app_runner.py                # [신규] 로컬 부팅·헬스체크·장애 주입 프록시
│   ├── gui_tests.py                 # [신규] Playwright: test_cases 실행 + 퍼저
│   ├── query_audit.py               # [신규] warehouse query history로 쿼리 수·바운드 검증
│   └── deploy.py                    # [신규] databricks apps deploy + 상태 폴링 (tier1만)
├── tier1-gate/                  # TASK_DESCRIPTION.md, test_cases.json, ground_truth.sql
├── tier2-core/                  # TASK_DESCRIPTION.md, test_cases.json, ground_truth.sql
├── tier3-differentiator/        # TASK_DESCRIPTION.md, test_cases.json, legacy_app/ (결함 8개)
│   └── legacy_app/                  # 후보에게 지시문과 함께 복사되는 레거시 코드
├── <tier>/opus/ sol/ glm/ …     # 후보 산출물: app/, run_meta.json, screenshots/
└── gallery/                     # 채점 후 사람 리뷰 갤러리 (tier × 후보 매트릭스)
```

후보 이름 → 기본 모델 매핑은 공용 뼈대 그대로: `opus` → `databricks-claude-opus-4-8`, `sol` → `databricks-gpt-5-6-sol`, `glm` → `databricks-glm-5-2`. `--model`로 덮어쓰기 가능.

## 산출물 계약 (핵심 설계 결정)

슬라이드(파일 1개)와 달리 앱은 **멀티파일**이다. 후보 디렉토리 산출물:

```
<candidate>/app/
├── app.yaml            # command + env (warehouse id는 valueFrom으로 주입)
├── app.py              # Streamlit 엔트리포인트
└── requirements.txt    # 의존성
```

- **에이전트 하네스** (claude-code / codex / pi / omnigent): `app/` 아래 파일을 직접 작성. 자연스러움.
- **direct-fmapi** (raw baseline): 단발 completion이 멀티파일을 낼 수 있도록 응답 포맷을 `=== FILE: <path> ===` 마커로 계약하고, 러너의 `extract_files()`가 파일로 분해한다.

**Databricks 특화 규칙 (채점기가 기계적으로 검사):**

- warehouse ID·호스트·토큰 **하드코딩 금지** — `app.yaml`의 `env`/`valueFrom`과 `DATABRICKS_WAREHOUSE_ID` 환경변수로만 접근 (공식 베스트 프랙티스이자 배포 이식성 조건)
- `databricks-sdk` 또는 `databricks-sql-connector`로 쿼리 (requirements.txt에 명시)
- 인증 코드 직접 작성 금지 — Apps 런타임이 주입하는 앱 인증(서비스 프린시펄) 사용

## 0) 준비 (한 번)

```bash
uv venv
uv pip install -e databricks-app-generation
uv run playwright install chromium
# 채점: grade-task [--tier ...] [--candidates ...] [--no-gui] [--no-deploy] [--blind]
#       grade-task --merge-human databricks-app-generation/gallery/human_scores.json

# 채점용 워크스페이스 자격 (배포 Phase + GUI 테스트의 실 쿼리에 필요)
# ucode 설치 시 자동, 또는 DATABRICKS_HOST/DATABRICKS_TOKEN + DATABRICKS_WAREHOUSE_ID
```

채점자(그레이더)가 쓰는 워크스페이스 사전 조건: serverless SQL warehouse 1개(2X-Small이면 충분), `samples` 카탈로그 활성화, Apps 활성화된 워크스페이스 + 앱 생성 권한. 후보 모델에게는 자격이 필요 없다 — 모델은 코드만 쓰고, 실행·배포는 채점기가 한다.

## 1) 후보별 실행

```bash
# tier 지정 실행 (--tier all 이면 1→2→3 순차, 게이트 실패 시 중단)
# raw baseline — 단발 completion + extract_files()
uv run run-task --tier tier1-gate --candidate opus --harness direct-fmapi
uv run run-task --tier all        --candidate glm  --harness direct-fmapi

# 에이전트 하네스 — 멀티턴으로 app/ 작성 (로컬 실행·자가수정 가능)
uv run run-task --tier all --candidate opus --harness claude-code
uv run run-task --tier all --candidate sol  --harness codex
uv run run-task --tier all --candidate pi-opus --harness pi \
    --pi-provider databricks-claude --model system.ai.claude-opus-5
```

공정성 규칙은 공용 뼈대와 동일: byte-identical `instructions.txt` + `prompt.txt`(tier3는 `legacy_app/` 사본 포함), tier별 고정 wall-clock 예산(15/30/25분, **timeout = 해당 tier 실패**), `run_meta.json`에 `effective_model`·소요시간·토큰 기록.

## 2) 채점 + 사람 리뷰

### Phase A (자동) — AppForge식 3단 관문 + 배포

| 단계 | 내용 | 가중치 |
|---|---|---|
| **A1 정적 검증** | 필수 파일 존재, `app.yaml` 유효(YAML 파싱 + `command` 키), `python -m py_compile`, requirements 파싱, **하드코딩 스캔**(호스트/토큰/warehouse ID 정규식), 리소스 접근이 env 경유인지 | 0.15 |
| **A2 로컬 부팅** | 격리 venv에 requirements 설치 → `app.yaml`의 command로 기동(채점자 자격을 env로 주입) → 60초 내 HTTP 200, 프로세스 생존 | 0.15 |
| **A3 GUI 기능 테스트** | Playwright가 `test_cases.json`의 케이스 실행 (operation → expected result). KPI 수치는 채점기가 `ground_truth.sql`을 warehouse에 직접 날려 얻은 값과 대조 → **화면이 그럴듯한지가 아니라 숫자가 맞는지**를 잰다 | 0.40 |
| **A4 배포 검증** | `databricks apps deploy` → status `RUNNING` 폴링 → 앱 URL 200 + 배포본 스모크 테스트 1건 → 채점 후 `stop`(과금 방지) | 0.30 |

tier별 `auto_score`는 각 `test_cases.json`의 가중치를 따른다 (tier1: 정적/부팅/GUI/배포, tier2: +엔지니어링 검증(쿼리 수·캐싱·바운드·지연), tier3: 수리(결함 8개 pass/fail)/확장/강건성(장애 주입 + 퍼저)). **A4 배포 검증은 tier1에만 적용**(비용 통제 — App 빌드 파이프라인 검증 목적은 tier1로 충분). 앞 관문 실패 시 뒤 관문 0점. 스위트 품질 점수:

`suite_quality = 0.15·tier1 + 0.50·tier2 + 0.35·tier3`, 단 **tier1 < 0.5면 suite 전체 0** (게이트).

보고 메트릭: tier별 auto_score + 부팅/배포 성공, 테스트 통과율, 크래시 수, contract_compliance, 쿼리 수 감사 결과. 결과는 `grade_results.json` + `gallery/index.html`(tier × 후보 매트릭스, 스크린샷 나란히). **리더보드에는 tier별 점수를 항상 병기** — "어디서부터 무너지는가"가 곧 모델 스토리다.

### 소요 시간 · 비용 (효율 축)

품질과 효율은 **분리 측정 후 합성**한다 — 시간·비용을 품질 가중치에 섞으면 "빠르지만 안 도는 앱"이 이기는 왜곡이 생기기 때문.

**측정 (러너가 `run_meta.json`에 기록):**

- `elapsed_minutes` — 러너 시작→산출물 확정까지 wall-clock (모든 하네스 공통으로 러너가 직접 측정)
- `input_tokens` / `output_tokens` — direct-fmapi는 API 응답의 usage 필드, claude-code/codex/pi는 하네스 세션 로그에서 추출, 추출 불가 시 `tokens_source: "unavailable"`로 명시하고 비용 축은 N/A 처리
- `estimated_cost_usd` — 토큰 × 모델별 단가. 단가는 `pricing.json`(모델명→$/1M input·output 토큰)에 한 곳에서만 관리. FMAPI는 DBU 과금이므로 DBU 단가 기준으로 환산해 기입

**점수화 (`test_cases.json`의 `efficiency` 블록):**

- `time_score = max(0, 1 − Σelapsed/70분)` — tier별 예산 합(15+30+25) 기준 고정 정규화. timeout된 tier는 0점 + elapsed를 예산으로 캡. 후보 간 상대 정규화는 후보가 추가될 때마다 과거 점수가 흔들려 배제
- `cost_score = max(0, 1 − Σcost/reference_cost)` — 고정 기준값(초기 $5.00, 첫 라운드 후 재조정) 대비
- `final_score = 0.70·auto_score + 0.15·time_score + 0.15·cost_score`, 단 **auto_score가 0이면 final도 0** (안 도는 앱이 효율로 점수 얻는 것 방지)

리더보드에는 final_score 하나만 내지 말고 **품질/시간/비용 3열을 항상 병기**한다 — Field Engineering 고객 대화에서는 "Opus가 품질 +10%지만 비용 3배" 같은 트레이드오프가 곧 메시지다.

### Phase B (사람)

`gallery/index.html`에서 후보별 스크린샷·코드 diff를 나란히 보고 **UX 품질**과 **코드 품질(SE 관점: 구조, 에러 핸들링, Databricks 관례 준수)** 각 1-5점 + 메모 → `human_scores.json` 다운로드 → `grade-task --merge-human`으로 병합. 공용 뼈대와 동일한 재현 가능 방식.

## 테스트 결과 (direct-fmapi baseline, n=1)

워크스페이스 `ai-devtools-prod`, 채점 전용 serverless SQL warehouse에서 세 후보를 **`direct-fmapi`**(단발 completion — raw baseline) 하네스로 전 tier 실행·채점한 결과다. **비용 축은 `pricing.json` 미설정으로 N/A**이므로 `final = (0.7·quality + 0.15·time) / 0.85`로 재정규화된다.

### tier별 auto_score

| Tier (가중치) | opus | sol | glm |
|---|:---:|:---:|:---:|
| **tier1 — gate** (0.15) | **1.00** | **1.00** | **1.00** |
| **tier2 — core** (0.50) | **0.88** | 0.13 ¹ | 0.00 ² |
| **tier3 — differentiator** (0.35) | 0.56 | **0.65** | 0.00 ² |

### 스위트 종합 (게이트 통과 후)

| 후보 | quality | time_score | **final** | 모델 (FMAPI) |
|---|:---:|:---:|:---:|---|
| **opus** | 0.785 | 0.98 | **0.82** | `databricks-claude-opus-4-8` |
| **sol** | 0.443 ¹ | 0.92 | **0.53** | `databricks-gpt-5-6-sol` |
| **glm** | 0.150 | 0.68 | **0.24** | `databricks-glm-5-2` |

**효율 축 (러너 측정):**

| 후보 | 총 소요 (tier1+2+3) | 총 출력 토큰 |
|---|:---:|:---:|
| opus | **96초** | 10,457 |
| sol | 352초 | 18,776 |
| glm | 1,328초 | 95,509 ³ |

¹ **sol tier2는 실제 실력이 아니다** — 채점 시점에 사내 pip 프록시가 일시 다운되어 격리 venv 부팅이 실패했다(앱 결함 아님, 인프라 장애). 프록시 정상 시 재채점 필요. sol의 quality/final은 이 때문에 저평가된 값이다.
² **glm tier2·tier3는 산출물 미완성** — completion이 `max_tokens`(32K)를 소진하고도 필수 파일을 다 못 만들어 정적 검증에서 탈락(게이트 내 `auto=0`). tier1은 완성했다.
³ glm 출력 토큰이 압도적으로 큰 것은 위 미완성(폭주)의 결과다.

### 핵심 관찰

- **tier1은 천장 효과** — 세 프론티어 모델 모두 GUI 8케이스 만점(1.00). "App을 만들 수 있는가"의 게이트로서는 통과했으나, 여기서는 모델이 갈리지 않는다. 변별은 tier2·tier3의 몫이라는 설계 의도가 실측으로 확인됐다.
- **속도 격차가 크다** — 동일 품질 tier1에서도 opus(21.6초) ≪ sol(82.9초) ≪ glm(445.4초). 결과물이 같아도 opus가 glm보다 20배 빠르다.
- **tier3에서 프로파일이 갈린다** — opus는 수리(gui 0.56) 우세, sol은 강건성(R1 장애 부팅 + R3 퍼저 1.00) 우세. tier3가 "코드 이해·디버깅·강건성"에서 격차를 벌린다는 목적을 달성.
- **glm은 복잡한 태스크에서 반복적으로 폭주** — tier2·tier3 모두 `max_tokens`까지 생성하고 미완성. tier1(단순)은 완성. 태스크 난이도와 완성률이 뚜렷이 반비례.
- **n=1 주의** — 위 숫자는 단일 실행이다. LLM 산출물과 GUI 상호작용의 실행 간 분산이 있으므로, 확정 순위는 n≥3 평균±표준편차로 내야 한다.

결과 원본: `grade_results.json` · 사람 리뷰 갤러리: `gallery/index.html`.

## 실전 유의점

- **모델명 footgun** — FMAPI 모델명은 `databricks-` 프리픽스 필수 (공용 뼈대와 동일).
- **토큰 집계 footgun** — 하네스마다 usage 노출 방식이 다르다(캐시된 토큰, 병렬 서브에이전트 호출 포함 여부). `run_meta.json`에 `tokens_source`(api-usage / session-log / unavailable)를 남겨 비교 가능성을 표시하고, 캐시 토큰은 캐시 단가로 별도 합산한다. 채점(그레이더) 비용은 전 후보 동일하므로 집계에서 제외.
- **그라운드트루스 드리프트** — `samples.nyctaxi.trips`는 정적 데이터셋이지만, 채점기가 기대값을 하드코딩하지 않고 **매 채점마다 `ground_truth.sql`로 재계산**해 워크스페이스 간 차이·데이터 갱신에 면역으로 만든다.
- **Streamlit 렌더링 타이밍** — Streamlit은 위젯 상호작용 시 rerun하므로 Playwright 테스트에 명시적 대기(`data-testid` 셀렉터 + networkidle)를 넣는다. 태스크 계약에서 주요 요소에 `key=`를 강제해 셀렉터를 안정화한다.
- **배포 비용/시간** — 배포 검증은 후보당 수 분 + 앱 컴퓨트 과금. tier1에만 적용하고, 채점기는 반드시 `stop`까지 수행, `--no-deploy` 플래그 지원.
- **쿼리 수 감사** — tier2/3의 쿼리 예산 검증은 warehouse query history에서 채점 세션의 쿼리를 시간창+세션 태그로 필터링한다. 후보 앱 쿼리에 채점기가 주입한 `QUERY_TAG` env를 쓰도록 강제하진 않으므로(계약 오염 방지), 채점 전용 warehouse를 써서 다른 트래픽과 격리하는 것이 전제.
- **tier3 결함 세트 비공개** — `seeded_defects`는 채점기 전용. 후보 지시문에는 사용자 증상 리포트만 준다. 결함 목록이 유출되면(예: 태스크 공개 시) 새 결함 세트로 로테이션.
- **후보 앱 이름 충돌** — 앱 URL은 이름 기반·변경 불가이므로 배포 시 `bench-<task>-<candidate>-<timestamp>`로 생성하고 채점 후 삭제한다.
- **비결정성** — LLM 산출물 자체의 분산이 크므로 후보당 **3회 반복 실행**(seed별 디렉토리 `opus-r1/ -r2/ -r3/`)을 기본으로 하고 평균±표준편차를 보고한다. 예산이 없으면 1회로 시작하되 표에 명시.

## 공정성 체크리스트

- 같은 태스크 · 같은 프롬프트/지시문(byte-identical) · 같은 예산(wall-clock) · 같은 채점기 · 같은 채점용 워크스페이스/warehouse.
- 시간·비용 정규화 기준(`budget_minutes`, `reference_cost_usd`, `pricing.json`)은 라운드 시작 전에 고정하고 라운드 중 변경 금지 — 변경 시 전 후보 재채점.
- **`direct-fmapi`는 단발 완성 1회** — 멀티파일 앱을 한 번에 뽑는 건 에이전트(로컬 실행→자가수정 가능)와 apples-to-apples가 아님 → "raw baseline"으로 명시.
- 채점기는 후보가 누구인지 모른 채 동일 파이프라인 실행. 사람 리뷰 갤러리도 후보명을 가린 블라인드 모드 지원(`--blind`).
- 배포 검증은 동일 워크스페이스·동일 warehouse에 순차 실행해 인프라 편차 제거.

## 참고 자료

- [Databricks Apps 개요](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/) · [Key concepts (app.yaml, 리소스, 인증)](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/key-concepts) · [공식 앱 템플릿](https://github.com/databricks/app-templates)
- [WebGen-Bench (arXiv:2505.03733)](https://arxiv.org/abs/2505.03733) — GUI 테스트 케이스 설계·자동 실행 방법론
- [AppForge (arXiv:2510.07740)](https://arxiv.org/abs/2510.07740) — compile→deploy→test 파이프라인·메트릭 설계
- [FullStack-Bench / FullStack-Agent (arXiv:2602.03798)](https://arxiv.org/pdf/2602.03798) — 프론트/백엔드/DB 레이어별 테스트
