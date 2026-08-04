# image-text-performance

Databricks Foundation Model API(FMAPI)로 서빙되는 LLM들의 **이미지·텍스트 처리 성능**을
표준 데이터셋 위에서 재현 가능하게 측정하고, 수치·그래프·정성 샘플·시간·비용을
**자동 리포트**로 생성하는 벤치마크.

> 설계·의사결정의 근거와 상세는 [`plan.md`](./plan.md) 참고. 이 README는 개요·사용법·핵심 주의사항 요약.

## 📊 리포트 바로가기

- **[▶ 전체 리포트 인덱스 (reports/index.md)](./reports/index.md)** — 모든 run을 시간순으로 링크. **최신 결과는 여기 맨 위 줄.**
- **[최신 리포트 (2026-08-03T23-44)](./reports/2026-08-03T23-44/report.md)** — 30샘플, opus·sol·glm. 리포트 내 "고객 설명용 프레젠테이션" 배너로 슬라이드(HTML)도 바로 볼 수 있다.

> 새 run을 돌리면 `reports/<run-id>/report.md`가 생기고 `reports/index.md` 맨 위에 자동 추가된다. 위 "최신 리포트" 링크는 새 run 후 갱신하면 된다.

---

## 무엇을 측정하나

**이미지 (6)** · **텍스트 (8)**, 총 14개 태스크. 생성 태스크는 참조기반 메트릭 + LLM-judge,
분류 태스크는 정답 라벨 대비 정량 메트릭. 모든 태스크에 각 모델의 실제 출력을 나란히 보는
**정성 갤러리**가 붙는다.

| 그룹 | 태스크 |
|------|--------|
| 이미지 | 캡션(IMG-1) · 태그추출(IMG-2) · 무기/위협 판별(IMG-3) · NSFW 판별(IMG-4) · 사람 포함(IMG-5) · 표이미지 구조추출(IMG-6) |
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

## Reasoning — OFF로 고정

**대상 모델은 전부 reasoning(사고) 모델이지만, 이 벤치마크는 reasoning을 OFF(minimal)로 고정해 측정한다.**

- **이유**: (1) reasoning의 효과가 **특정 태스크의 성능 개선에만 한정**되고, (2) full reasoning은
  **실험 시간을 크게 늘린다**. 특히 `databricks-glm-5-2`는 full reasoning 시 응답을 내부 사고
  (`reasoning_content`)에 먼저 쏟아내며 3~5배 느려지고, 15초 타임아웃을 자주 초과한다(실측: 160호출 중 19 오류).
- 그래서 실행 매트릭스를 **모델 × 태스크 × 샘플** (reasoning 축 제거)로 두어 실용성을 확보한다.

### 모델별 reasoning 제어 파라미터 (실측 확정)

config의 `minimal` 키가 아래 실제 파라미터로 매핑된다:

| 모델 | 제어 파라미터 | `minimal` (사용) |
|------|--------------|-----------|
| sol (GPT식) | `reasoning_effort` | `none` |
| opus (Claude식) | `thinking` | `thinking:{type: disabled}` |
| glm (GLM식) | `reasoning_effort` | `none` |
| gemini (judge) | `generation_config.thinking_config.thinking_level` | 최소 레벨만 |

- **주의**: `databricks-claude-opus-5`와 judge(`gemini`)는 reasoning을 완전히 끌 수 없다(opus effort 최소가 `low`,
  gemini는 `none`/`minimal` 레벨 거부). 그래서 "OFF"는 **각 모델이 지원하는 최소 reasoning**으로 정의한다.
  리포트의 "Reasoning 정책" 섹션에 이 사실이 표기된다.
- **full reasoning으로 다시 비교하고 싶다면**: `config/models.yaml`의 `reasoning_modes:`를
  `[minimal, full]`로 되돌리면 된다(각 모델의 `full` 파라미터 정의는 그대로 유지되어 있음). 단 실행 시간이 배로 늘고 glm 타임아웃이 발생한다.
- config 키를 `on`/`off`가 아니라 `minimal`/`full`로 쓴다 — YAML에서 `on`/`off`는 boolean으로 파싱되는 예약어라 키로 쓰면 버그가 난다.

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

## 사용법 — Vibe agent로 실행

이 벤치마크는 **Vibe agent(Claude Code 기반 SA 에이전트)에게 자연어로 지시**해 운영한다.
설치·모델 추가·실험 실행·리포트 생성·결과 해석까지 아래 명령들을 그대로 붙여넣으면 된다.
(에이전트가 내부적으로 의존성 설치, `config/*.yaml` 편집, 벤치마크 실행, 리포트·프레젠테이션 생성을 수행한다.)

**전제**: Databricks CLI 프로파일(기본 `ai_devtools`)이 설정돼 있어야 한다(FMAPI 호출·`system.ai_gateway.usage` 조회용).

### 1) 최초 설치·환경 준비

> "image-text-performance 벤치마크를 처음 실행할 수 있게 필요한 라이브러리를 설치하고 환경을 준비해줘.
> 한국어 형태소 분석(mecab)이나 BERTScore(torch)는 없으면 자동 fallback되니 무거우면 건너뛰어도 돼."

- 에이전트가 `pyproject.toml`의 의존성을 설치한다.
- 데이터셋은 HuggingFace **streaming**으로 필요한 샘플만 받는다(전체 다운로드 X, `.cache/`에 캐시).
- 선택 의존성: **mecab**(한국어 형태소 — 없으면 음절 fallback), **bert-score+torch**(의미유사도 — 없으면 `deferred`, ROUGE·judge로 대체). 리포트에 어떤 backend가 쓰였는지 표시된다.

### 2) 실험 실행 → 리포트 생성

> "image-text-performance 벤치마크를 10샘플로 돌려서 리포트를 새로 뽑아줘."

- 3모델(opus·sol·glm) × 13태스크 × reasoning OFF로 실행하고, `reports/<run-id>/`에 리포트·그래프·정성 갤러리·고객용 프레젠테이션(HTML)을 생성한다.
- 빠른 확인만 원하면: > "opus와 sol만 텍스트 태스크로 3샘플만 빠르게 돌려줘."
- 대규모(정확도 우선): > "HF 토큰 설정하고 glm 타임아웃을 60초로 올린 뒤 전체 50샘플로 백그라운드 실행해줘." (전체는 HF rate limit·glm 지연으로 수 시간+ 걸린다.)

### 3) 모델 추가 후 재실험

> "이 벤치마크에 `databricks-claude-sonnet-5`를 sonnet이라는 별칭으로 추가하고(vision 지원, reasoning은 off),
> pricing 단가도 채운 다음 10샘플로 재실행해서 리포트를 새로 뽑아줘."

- 에이전트가 `config/models.yaml`(모델·capability·reasoning)과 `config/pricing.yaml`(DBU 단가)을 편집하고 재실행한다.
- 새 `run-id` 리포트가 생성되고 기존 run은 보존된다 → `reports/index.md`에서 시점 비교 가능.
- 모델 제거·교체도 동일하게 지시하면 된다.

### 4) 결과 해석

> "가장 최근 리포트에서 한국어 태스크(TXT-4) 성능과 모델별 비용·속도를 요약해줘."
> "정성 비교에서 모델 판정이 갈린 이미지 샘플을 보여주고 어느 모델이 맞았는지 알려줘."

### 5) reasoning 비교가 필요하면

> "reasoning을 켠 경우와 끈 경우를 비교하고 싶어. reasoning ON/OFF 둘 다로 돌려서 리포트 뽑아줘."

- 기본은 reasoning OFF 고정(실험 시간·비용 절감). 위처럼 지시하면 ON/OFF 둘 다 측정한다(실행 시간 배증).

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

## 데이터 소스 (출처)

모든 태스크는 **공개 표준 데이터셋**을 HuggingFace Hub에서 로드한다(합성·더미 데이터 없음, requirement #5).
태스크별 출처·언어·라이선스는 아래와 같고, 버전·split·mirror 여부·해시는 [`datasets/registry.yaml`](./datasets/registry.yaml)에 고정 기록된다.

| 태스크 | 데이터셋 | HuggingFace ID | 언어 | 라이선스 |
|--------|----------|----------------|:----:|----------|
| IMG-1 캡션 | COCO Captions (Karpathy split) | [yerevann/coco-karpathy](https://huggingface.co/datasets/yerevann/coco-karpathy) | en | CC-BY-4.0(주석) / 이미지는 Flickr 약관 |
| IMG-2 태그추출 | COCO | [detection-datasets/coco](https://huggingface.co/datasets/detection-datasets/coco) | en | CC-BY-4.0(주석) |
| IMG-3 무기/위협 | WeaponDetection | [Subh775/WeaponDetection](https://huggingface.co/datasets/Subh775/WeaponDetection) | en | CC-BY-4.0 |
| IMG-4 NSFW | NSFW Image Classification | [DarkyMan/nsfw-image-classification](https://huggingface.co/datasets/DarkyMan/nsfw-image-classification) | en | CC(variant 미상) · **민감** |
| IMG-5 사람 포함 | COCO (person 파생 binary) | [detection-datasets/coco](https://huggingface.co/datasets/detection-datasets/coco) | en | CC-BY-4.0(주석) |
| IMG-6 표이미지 구조추출 | PubTabNet (image+HTML) | [apoidea/pubtabnet-html](https://huggingface.co/datasets/apoidea/pubtabnet-html) | en | CDLA-Permissive-1.0 |
| TXT-1 문서QA | DocumentVQA | [HuggingFaceM4/DocumentVQA](https://huggingface.co/datasets/HuggingFaceM4/DocumentVQA) | en | mirror apache-2.0 / 원본 research-use |
| TXT-2 표QA | WikiTableQuestions | [lighteval/wikitablequestions](https://huggingface.co/datasets/lighteval/wikitablequestions) | en | CC-BY-4.0 (mirror) |
| TXT-3 표구조추출 | PubTabNet (HTML) | [apoidea/pubtabnet-html](https://huggingface.co/datasets/apoidea/pubtabnet-html) | en | CDLA-Permissive-1.0 |
| TXT-4 한국어QA | KorQuAD v1 | [KorQuAD/squad_kor_v1](https://huggingface.co/datasets/KorQuAD/squad_kor_v1) | ko | CC-BY-ND-4.0 |
| TXT-5 요약 | CNN/DailyMail(en) · Naver News 요약(ko) | [abisee/cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) · [daekeun-ml/naver-news-summarization-ko](https://huggingface.co/datasets/daekeun-ml/naver-news-summarization-ko) | en·ko | apache-2.0 |
| TXT-6 감정 | SST-2(en) · NSMC(ko) | [stanfordnlp/sst2](https://huggingface.co/datasets/stanfordnlp/sst2) · [Blpeng/nsmc](https://huggingface.co/datasets/Blpeng/nsmc) | en·ko | GLUE · CC-BY-2.0 |
| TXT-7 키워드 | INSPEC | [memray/inspec](https://huggingface.co/datasets/memray/inspec) | en | research-use |
| TXT-8 비속어 | Jigsaw Toxic Comment(en) · APEACH(ko) | [thesofakillers/jigsaw-toxic-comment-classification-challenge](https://huggingface.co/datasets/thesofakillers/jigsaw-toxic-comment-classification-challenge) · [jason9693/APEACH](https://huggingface.co/datasets/jason9693/APEACH) | en·ko | CC-BY-SA / CC0 · CC-BY-SA-4.0 |

- **mirror 주의**: script-based 원본이 로드 불가하면 parquet mirror를 쓴다. mirror의 라이선스 태그가 원본과 다를 수 있어 registry에 **원본 라이선스를 함께 기록**한다(WikiTableQuestions·NSMC·Jigsaw·DocVQA 등).
- **비상업 라이선스 주의**: KorQuAD(CC-BY-ND) · DocVQA 원본(research-use) · INSPEC(research-use) 등은 **연구·평가 용도 한정**이다. 상업 활용 전 각 데이터셋 원 라이선스를 확인한다.
- **시간·비용 데이터 소스**는 데이터셋이 아니라 `system.ai_gateway.usage`(FMAPI 호출 실측) — 아래 "비용·시간 측정" 참고. judge 채점도 데이터셋과 무관한 별도 모델(`databricks-gemini-3-1-pro`)이다.

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
- Phase 0 기반 구조(FMAPI 어댑터·설정·채점/비용 모듈)
- Phase 1 텍스트 안전 태스크(TXT-4/5/6/8)
- Phase 2 이미지 태스크(IMG-1~5)
- Phase 3 문서·표 태스크(TXT-1/2/3/7)
- Phase 4 시점별 리포트 축적·재현성 메타·인덱스

세부 설계·의사결정은 [`plan.md`](./plan.md), 결정 이력은 §2(D1–D14) 참고.
