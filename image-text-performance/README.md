# image-text-performance

Databricks Foundation Model API(FMAPI)로 서빙되는 LLM들의 **이미지·텍스트 처리 성능**을
표준 데이터셋 위에서 재현 가능하게 측정하고, 수치·그래프·정성 샘플·시간·비용을
**자동 리포트**로 생성하는 벤치마크.

> 설계·의사결정의 근거와 상세는 [`plan.md`](./plan.md) 참고. 이 README는 개요·사용법·핵심 주의사항 요약.

---

## 무엇을 측정하나

**이미지 (5)** · **텍스트 (8)**, 총 13개 태스크. 생성 태스크는 참조기반 메트릭 + LLM-judge,
분류 태스크는 정답 라벨 대비 정량 메트릭. 모든 태스크에 각 모델의 실제 출력을 나란히 보는
**정성 갤러리**가 붙는다.

| 그룹 | 태스크 |
|------|--------|
| 이미지 | 캡션(IMG-1) · 태그추출(IMG-2) · 무기/위협 판별(IMG-3) · NSFW 판별(IMG-4) · 사람 포함(IMG-5) |
| 텍스트 | 문서QA(TXT-1) · 표QA(TXT-2) · 표구조추출(TXT-3) · 한국어QA(TXT-4) · 요약(TXT-5) · 감정(TXT-6) · 키워드(TXT-7) · 비속어(TXT-8) |

- **언어**: 한국어 + 영어 병행(TXT-7은 한국어 표준셋 부재로 영어 한정, 이미지 태스크는 영어/시각 기반).
- **채점**: 하이브리드(정량 + LLM-judge) + 정성 갤러리. 한국어는 형태소(Mecab) 토큰화로 채점.

---

## 대상 모델

`config/models.yaml`에서 언제든 추가·교체할 수 있다. 1차 대상(각 계열 최신):

| 별칭 | FMAPI 엔드포인트 | Vision | 비고 |
|------|-----------------|:------:|------|
| opus | `databricks-claude-opus-5` | ✅ | |
| sol | `databricks-gpt-5-6-sol` | ✅ | |
| glm | `databricks-glm-5-2` | ❌ | 이미지 입력 미지원 → 이미지 태스크는 **N/A**로 자동 스킵 |
| judge | `databricks-gemini-3-1-pro` | ✅ | LLM-as-judge (평가 대상과 다른 계열로 bias 최소화, 텍스트·이미지 모두 채점) |

> vision 지원 여부는 `ai_devtools` 워크스페이스에서 **실제 이미지 호출로 검증**됨.

---

## Reasoning ON/OFF — 반드시 이해해야 하는 부분

**대상 4개 모델은 전부 reasoning(사고) 모델이다.** reasoning을 켜고 끄는 것은
정확도뿐 아니라 **수행 시간과 비용을 크게 좌우**한다. 예를 들어 `glm-5-2`는 reasoning이
켜지면 응답을 내부 사고(`reasoning_content`)에 먼저 쏟아내며 다른 모델보다 3~5배 느려지고
토큰(=비용)이 급증한다. 따라서 reasoning을 통제하지 않으면 모델 비교가 공정하지 않다.

**이 벤치마크는 reasoning을 두 모드로 측정하고 리포트에서 나란히 비교한다:**

- **`full`** — reasoning ON. 각 모델의 최대 사고 능력.
- **`minimal`** — reasoning을 끄거나(또는 최소화) 한 상태. 빠르고 저렴한 기본 응답.

실행 매트릭스 = **모델 × 태스크 × 샘플 × reasoning{minimal, full}**.

### 모델별 reasoning 제어 파라미터 (실측 확정)

제어 방식이 계열마다 다르다. config의 `minimal`/`full` 키가 아래 실제 파라미터로 매핑된다:

| 모델 | 제어 파라미터 | `minimal` | `full` |
|------|--------------|-----------|--------|
| sol (GPT식) | `reasoning_effort` | `none` | `high` |
| opus-5 (Claude식) | `thinking` + `output_config.effort` | `thinking:{type: disabled}` | `thinking:{type: adaptive}` |
| glm (GLM식) | `reasoning_effort` | `none` | 기본 |
| gemini (judge) | `generation_config.thinking_config.thinking_level` | 최소 레벨만 | 기본 |

### 중요한 한계

- **opus-5와 gemini는 reasoning을 완전히 끌 수 없다.** opus의 `effort`는 최소가 `low`(none 없음),
  gemini는 `none`/`minimal` 레벨을 거부한다. 그래서 `minimal` 모드는 "reasoning 완전 OFF"가 아니라
  **"각 모델이 지원하는 최소 reasoning"**으로 정의한다. 리포트도 이 정의를 명시한다.
- config 키를 `on`/`off`가 아니라 `minimal`/`full`로 쓴다 — YAML에서 `on`/`off`는 boolean으로
  파싱되는 예약어라 키로 쓰면 버그가 나고, 위 한계 때문에 "off"라는 말도 부정확하기 때문이다.

---

## 리포트

`reports/<run-id>/`에 벤치마크 **시점(run)별로 분리 생성·보존**된다. 구성:

0. **Executive Summary** — 모델별 성능을 사람이 한눈에 이해할 자연어 요약(어느 모델이 무엇을
   잘하고/못하고, 시간·비용은 어떤지). 수치에서 핵심 사실을 규칙으로 추출한 뒤 judge가 문단화해,
   실제 수치에 고정(hallucination 방지)된다.
1. **정량 요약** — 태스크 × 모델 점수 매트릭스 + 그래프(막대·레이더·히트맵). reasoning 모드별 분리, 통계 유의성 병기.
2. **정성 갤러리** — 고정 샘플에 대한 각 모델의 실제 출력·판정·정답·judge 근거를 나란히.
3. **성능 리포트** — 모델별 **수행 시간**(median/p95)과 **USD 비용**(입력·출력·reasoning·캐시 분해), reasoning ON/OFF delta.

### 시점별 리포트 (모델 추가 시)

모델을 추가하거나 데이터셋·프롬프트를 바꿔 다시 실행하면 **새 `run-id`로 리포트가 생성**되고,
과거 리포트는 덮어쓰지 않고 보존된다. 각 run의 `manifest.json`에 그 시점의 모델·데이터셋 버전·
reasoning 모드·pricing·코드 커밋 SHA가 기록되어 재현 가능하다. `reports/index.md`가 전체 run을
시간순으로 링크한다.

---

## 사용법

> ⚠️ 아직 스캐폴딩 단계다. 아래는 목표 인터페이스이며 구현이 진행되면 갱신된다.

### 사전 요구

- Python 3.10+
- Databricks CLI 프로파일(기본 `ai_devtools`) — FMAPI 호출과 `system.ai_gateway.usage` 조회에 사용
- 데이터셋은 HuggingFace에서 **streaming**으로 필요한 샘플만 받는다(전체 다운로드 X). `.cache/`에 캐시(gitignore).

### 설치

```bash
pip install -e .          # pyproject.toml의 의존성 설치
```

**선택 의존성 (없어도 fallback으로 동작):**
- **한국어 형태소 분석** (`konlpy` + `mecab`, `mecab-ko-dic`): 설치 시 ROUGE·Token-F1의 한국어 채점이 형태소 단위로 정밀해진다. **미설치 시 음절(syllable) 단위로 자동 fallback** (공백 분리보다 정확). 리포트에 사용된 backend가 표시된다.
  - macOS: `brew install mecab` + `mecab-ko-dic` 별도 설치
  - Linux: `apt-get install mecab libmecab-dev mecab-ko-dic`
- **BERTScore** (`bert-score` + `torch`, 수 GB): 설치 시 IMG-1·TXT-5 의미유사도 채점 활성화. 미설치 시 해당 지표는 `deferred`로 표시되고 ROUGE·token-F1·judge로 대체된다.

### 실행

```bash
# 전체 벤치마크 (3모델 × 13태스크 × reasoning{minimal,full} × 50샘플)
python -m src.runner

# 빠른 확인: 특정 모델·모드·샘플 수로 제한
python -m src.runner --models sol --reasoning-modes minimal --samples 3

# 실행 매트릭스만 미리보기(FMAPI 호출 없음)
python -m src.runner --dry-run
```

결과는 `results/<run-id>/`(원시 JSONL·scores·manifest), 리포트는 `reports/<run-id>/report.md`,
전체 run 목록은 `reports/index.md`에 생성된다.

**주요 옵션:** `--models opus,sol,glm` (모델 필터) · `--reasoning-modes minimal,full` · `--samples N` (샘플 수) · `--dry-run` · `--config` / `--tasks` (설정 경로).

---

## 설정 파일

| 파일 | 역할 |
|------|------|
| `config/models.yaml` | 대상 모델·엔드포인트·capability·reasoning 모드·judge |
| `config/tasks.yaml` | 실행할 태스크·샘플 수(기본 50, seed 고정)·언어·프롬프트 |
| `config/pricing.yaml` | 모델별 DBU 단가 + `usd_per_dbu` (비용 계산) |
| `config/judge_rubrics.yaml` | 태스크별 judge 채점 루브릭(1–5 앵커) |
| `datasets/registry.yaml` | 데이터셋 출처·버전·해시·라이선스 |

---

## 비용·시간 측정

각 FMAPI 호출의 `request_id`를 기록해 두고, `system.ai_gateway.usage` 테이블과 조인하여
`latency_ms`·토큰 사용량을 실측한 뒤 `config/pricing.yaml`의 DBU 단가로 USD를 환산한다.
(`system.billing`은 현재 권한이 없어 `ai_gateway.usage`를 쓴다.)

- **비용 공식**: `usd = usd_per_dbu × (billable_input×dbu_in + billable_output×dbu_out + cache…) / 1e6`
- **reasoning 토큰 주의**: `billable_output = completion_tokens + reasoning_tokens`. reasoning 토큰이
  `completion_tokens`에 포함되지 않으므로, 이를 빼먹으면 비용이 크게 과소 계상된다.
- **단가 확정 상태**: DBU 단가는 Databricks 공식 pricing 페이지에서 2개 소스로 교차검증(GLM-5.2는 재확인 완료). `usd_per_dbu=0.07`은 Premium Model Serving 표준단가로 확정 채택 — `system.billing`은 워크스페이스 권한이 없어 계약별 실단가 대조는 불가(권한 확보 시 사후 대조 가능). 계약 단가가 다르면 `pricing.yaml`의 이 한 줄만 바꾸면 전체 비용이 재계산된다.

---

## 데이터 취급 주의 (민감 태스크)

- **IMG-4(NSFW)**: 이미지를 **repo에 절대 커밋하지 않는다.** 런타임에만 로컬 캐시로 내려받아 처리하고,
  정성 갤러리에도 썸네일 없이 **판정값만** 표시한다. (공개 GitHub repo이므로.)
- **IMG-3**: `Subh775/WeaponDetection`(CC-BY-4.0)을 **무기/위협 존재 여부(binary)**로 측정한다.
  '폭력 정도(severity)'가 아니라 **'무기'** 측정임을 리포트에 명시한다(서열 severity 데이터가 존재하지 않아서).
- 데이터셋 라이선스는 `datasets/registry.yaml`에 기록한다. 일부는 mirror의 라이선스 태그가 원본과
  다르므로 원본 라이선스를 별도 기록한다(예: KorQuAD는 CC-BY-ND).

---

## 데이터셋 로딩 (streaming)

대상 데이터셋 일부는 매우 크다(COCO ~19GB, PubTabNet ~16GB, DocVQA ~11GB). 전체를 받으면
디스크가 폭발하므로, `src/datasets_loader.py`는 HuggingFace **streaming**으로 seed 고정 subset만
받는다(50샘플에 수십 GB를 받지 않음). script-based로 로드 불가한 원본은 parquet mirror로 대체하고
(`datasets/registry.yaml`의 `mirror_of` 기록), 실제 로드 실패 시 **합성 데이터로 우회하지 않고**
명확히 실패한다(표준 데이터셋 원칙, requirement #5).

## 상태

**구현 완료 — 13개 태스크 전체가 end-to-end 동작** (데이터 로딩 → FMAPI 실행 → 채점 → 시간·비용·Executive Summary 리포트). 로드맵 Phase 0~4 완료:
- Phase 0 스캐폴딩(FMAPI 어댑터·설정·채점/비용 골격)
- Phase 1 텍스트 안전 태스크(TXT-4/5/6/8)
- Phase 2 이미지 태스크(IMG-1~5)
- Phase 3 문서·표 태스크(TXT-1/2/3/7)
- Phase 4 시점별 리포트 축적·재현성 메타·인덱스

세부 설계·의사결정은 [`plan.md`](./plan.md), 결정 이력은 §2(D1–D14) 참고.
