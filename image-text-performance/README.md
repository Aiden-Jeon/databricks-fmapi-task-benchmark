# image-text-performance

Databricks Foundation Model API(FMAPI)로 서빙되는 LLM들의 **이미지·텍스트 처리 성능**을
표준 데이터셋 위에서 재현 가능하게 측정하고, 수치·그래프·정성 샘플·시간·비용을
**자동 리포트**로 생성하는 벤치마크.

> 설계·의사결정의 근거와 상세는 [`plan.md`](./plan.md) 참고. 이 README는 개요·사용법·핵심 주의사항 요약.

## 📊 리포트 바로가기

- **[▶ 전체 리포트 인덱스 (reports/index.md)](./reports/index.md)** — 모든 run을 시간순으로 링크. **최신 결과는 여기 맨 위 줄.**
- **[최신 리포트 (2026-08-07T13-42)](./reports/2026-08-07T13-42/report.md)** — 30샘플, opus,sol,glm,kimi. 리포트 내 "고객 설명용 프레젠테이션" 배너로 슬라이드(HTML)도 바로 볼 수 있다. *(이 줄은 새 run 때 runner가 자동 갱신)*

> 새 run을 돌리면 `reports/<run-id>/report.md`가 생기고, runner가 `reports/index.md`와 **위 "최신 리포트" 링크를 자동 갱신**한다(항상 실재하는 최신 run을 가리킴 — 수동 수정 불필요).

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
- **judge 실패는 점수로 메우지 않는다**: judge(gemini)가 응답을 잘리거나 형식을 벗어나면 그 샘플을
  평균에서 **제외**하고 실패 건수를 리포트(`judge_detail.n_failed`)에 남긴다. 예전에는 실패를 중간값
  3점으로 채워 judge 평균이 조용히 오염됐다(2026-08-05 수정 — 자세한 내용은 `CLAUDE.md`의 "judge 함정").
- **TXT-1은 ANLS**(DocVQA 공식 메트릭)를 대표 지표로 쓴다. 정답이 문서에 적힌 짧은 문자열이라
  exact_match는 한 글자 오타에도 0점, Token-F1은 표기 차이(`485` vs `$485`)에 과민하다.
  ANLS는 정규화 편집거리 기반(τ=0.5, 다중정답 max)으로 그 중간을 취한다.

---

## 대상 모델

`config/models.yaml`에서 언제든 추가·교체할 수 있다. 1차 대상(각 계열 최신):

| 별칭 | FMAPI 엔드포인트 | Vision | 비고 |
|------|-----------------|:------:|------|
| opus | `databricks-claude-opus-5` | ✅ | |
| sol | `databricks-gpt-5-6-sol` | ✅ | |
| glm | `databricks-glm-5-2` | ❌ | 이미지 입력 미지원 → 이미지 태스크는 **N/A**로 자동 스킵 |
| kimi | `databricks-kimi-k3` | ✅ | 동시 호출 제한을 고려해 concurrency 2 |
| judge | `databricks-gemini-3-1-pro` | ✅ | LLM-as-judge (평가 대상과 다른 계열로 bias 최소화, 텍스트·이미지 모두 채점) |

> vision 지원 여부는 `ai_devtools` 워크스페이스에서 **실제 이미지 호출로 검증**됨.

> **모델 추가 검증 범위(2026-08-06)**: `databricks-claude-sonnet-5`를 4번째 모델로 붙여
> ① 설정 검증 통과, ② dry-run 매트릭스 50셀 구성, ③ **opus+sonnet 2샘플 실호출 28셀 완주**
> (exit 0, 양쪽 비용 계산됨)까지 확인했다. **4모델 30샘플 전체 완주는 하지 않았다**
> (사용자 요청으로 3모델 비교로 되돌림). 현재 `config/models.yaml`에 주석으로 보존돼 있고
> 단가는 `pricing.yaml`에 등록된 상태라, 주석을 풀면 바로 4모델로 실행된다.

### 모델 추가하기

`config/models.yaml`의 `models:`에 항목을 더하면 된다. 아래 7가지가 필요하고,
빠지면 **실행 전 검증이 막는다**(`src/config.py:validate_models_config`).

| 항목 | 왜 필요한가 |
|---|---|
| `id` | 고유해야 한다. 중복이면 뒤 항목이 앞을 덮어써 한 모델이 조용히 사라진다 |
| `endpoint` | `databricks serving-endpoints list --profile <p>`로 확인 |
| `family` | 계열 표기(claude·openai·gemini …). 리포트·주석용 |
| `capabilities` | `[text]` 또는 `[text, vision]`. **실제 이미지 호출로 검증**할 것 — 오타·누락은 해당 태스크가 전부 N/A가 된다 |
| `reasoning` | 실행할 모든 모드를 정의. 빈 dict는 "모델 기본값"이라 reasoning이 켜진 채 돌면서 리포트는 OFF로 표기해 측정 조건이 어긋난다 |
| `config/pricing.yaml` 단가 | 없으면 비용이 계산되지 않고 리포트가 '단가 미등록'으로 표기한다(0으로 두면 **"가장 저렴한 모델"로 오선정**되므로 그렇게 하지 않는다) |
| `runtime`(선택) | 느리거나 긴 출력을 내는 모델이면 `timeout_seconds`·`max_tokens`를 모델별로 올린다 |

추가 후 확인:

```bash
python3 -m src.runner --dry-run          # 설정 검증 + 셀 수·모델별 실효 런타임 출력
python3 -m src.runner --samples 2 --models <새모델> --no-judge   # 소규모 실호출
```

> **주의**: 모델을 추가한 뒤 `--resume`으로 기존 run에 이어붙이면 **차단된다**. 서로 다른
> 구성의 결과가 한 run에 섞이고 manifest가 사실과 달라지기 때문이다. 새 run으로 실행한다
> (모델 추가 시 전체 재실행이 기본 정책 — 모든 모델을 동일 데이터·시점에서 비교해야 공정하다).

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

**전제**: Databricks CLI 프로파일이 설정돼 있어야 한다(FMAPI 호출용). 기본값은
`config/models.yaml`의 `profile`(현재 `ai_devtools`)이고, 실행할 때
**`--profile <name>`으로 덮어쓸 수 있다**. 어느 워크스페이스로 호출·과금되는지는 실행 로그와
각 run의 `manifest.json`(`profile` 필드)에 기록된다 — 다른 워크스페이스로 돌린 run은
엔드포인트 가용성·지연·단가가 달라 비교의 전제가 바뀐다.

### 1) 최초 설치·환경 준비

> "image-text-performance 벤치마크를 처음 실행할 수 있게 필요한 라이브러리를 설치하고 환경을 준비해줘.
> 한국어 형태소 분석(mecab)이나 BERTScore(torch)는 없으면 자동 fallback되니 무거우면 건너뛰어도 돼."

- 에이전트가 `pyproject.toml`의 의존성을 설치한다.
- 데이터셋은 HuggingFace **streaming**으로 필요한 샘플만 받는다(전체 다운로드 X, `.cache/`에 캐시).
- 선택 의존성: **mecab**(한국어 형태소 — 없으면 음절 fallback), **bert-score+torch**(의미유사도 — 없으면 `deferred`, ROUGE·judge로 대체). 리포트에 어떤 backend가 쓰였는지 표시된다.

### 2) 실험 실행 → 리포트 생성

> "image-text-performance 벤치마크를 10샘플로 돌려서 리포트를 새로 뽑아줘."

- 4모델(opus·sol·glm·kimi) × 14태스크 × reasoning OFF로 실행하고, `reports/<run-id>/`에 리포트·그래프·정성 갤러리·고객용 프레젠테이션(HTML)을 생성한다. glm은 vision 미지원이라 이미지 6태스크가 N/A → 실제 **50셀**.
- **종료 코드로 성패를 알린다**: 채점 오류·실패율 과다 셀이 있거나 전체 호출 실패율이 10%를 넘으면 `exit 1`(리포트는 그대로 생성됨). 자동화가 실패를 성공으로 오판하지 않게 하기 위함이다.
- **리포트를 리셋하고 새로 뽑을 때(`--fresh`) 이전 결과를 먼저 지우지 않는다**: 새 리포트가 완성되고 종료 코드 판정까지 통과한 뒤에 이전 run을 `.trash/`로 옮긴다(삭제 아님). 정리 도중이나 정리 직후 단계에서 실패하면 옮긴 것을 **전부 제자리로 되돌린다**. 그래서 "새 리포트도 없고 옛 리포트도 없는" 상태가 되지 않는다.
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

**현재 방식(실제 동작)**: 각 호출의 응답에 담긴 `usage` 토큰 수와 **클라이언트 벽시계 지연**
(`latency_ms_local`)을 기록하고, `config/pricing.yaml`의 DBU 단가로 USD를 환산한다.
리포트의 시간·비용은 이 값 기준의 **추정치**다.

**아직 미구현**: `system.ai_gateway.usage` 조인(`src/cost/usage.py:fetch_usage`가
`NotImplementedError`). 조인하면 서버 측 `latency_ms`·`token_details`(캐시·reasoning 분해)를
실측할 수 있다. 각 호출의 `request_id`는 이미 결과에 저장하고 있어 조인 준비는 돼 있다.
(`system.billing`은 워크스페이스 권한이 없어 쓸 수 없다.)

- **비용 공식**: `usd = usd_per_dbu × (billable_input×dbu_in + billable_output×dbu_out + cache…) / 1e6`
- **reasoning 토큰 주의**: `billable_output = completion_tokens + reasoning_tokens`. reasoning 토큰이
  `completion_tokens`에 포함되지 않으므로, 이를 빼먹으면 비용이 크게 과소 계상된다.
- **단가 확정 상태**: DBU 단가는 Databricks 공식 pricing 페이지 기준이며, Kimi K3도 2026-08-13 공식 Foundation Model Serving 가격표에서 입력 42.857·출력 214.286 DBU/백만 토큰으로 확인했다. `usd_per_dbu=0.07`은 Premium Model Serving 표준단가로 채택 — `system.billing`은 워크스페이스 권한이 없어 계약별 실단가 대조는 불가(권한 확보 시 사후 대조 가능). 계약 단가가 다르면 `pricing.yaml`의 이 한 줄만 바꾸면 전체 비용이 재계산된다.

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
| TXT-1 문서QA | DocVQA (OCR 텍스트 포함 1200건 서브셋) | [nielsr/docvqa_1200_examples](https://huggingface.co/datasets/nielsr/docvqa_1200_examples) | en | 미러 미표기 / 원본 research-use |
| TXT-2 표QA | WikiTableQuestions | [lighteval/wikitablequestions](https://huggingface.co/datasets/lighteval/wikitablequestions) | en | CC-BY-4.0 (mirror) |
| TXT-3 표구조추출 | PubTabNet (HTML) | [apoidea/pubtabnet-html](https://huggingface.co/datasets/apoidea/pubtabnet-html) | en | CDLA-Permissive-1.0 |
| TXT-4 한국어QA | KorQuAD v1 | [KorQuAD/squad_kor_v1](https://huggingface.co/datasets/KorQuAD/squad_kor_v1) | ko | CC-BY-ND-4.0 |
| TXT-5 요약 | CNN/DailyMail(en) · Naver News 요약(ko) | [abisee/cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) · [daekeun-ml/naver-news-summarization-ko](https://huggingface.co/datasets/daekeun-ml/naver-news-summarization-ko) | en·ko | apache-2.0 |
| TXT-6 감정 | SST-2(en) · NSMC(ko) | [stanfordnlp/sst2](https://huggingface.co/datasets/stanfordnlp/sst2) · [Blpeng/nsmc](https://huggingface.co/datasets/Blpeng/nsmc) | en·ko | GLUE · CC-BY-2.0 |
| TXT-7 키워드 | INSPEC | [memray/inspec](https://huggingface.co/datasets/memray/inspec) | en | research-use |
| TXT-8 비속어 | Jigsaw Toxic Comment(en) · APEACH(ko) | [thesofakillers/jigsaw-toxic-comment-classification-challenge](https://huggingface.co/datasets/thesofakillers/jigsaw-toxic-comment-classification-challenge) · [jason9693/APEACH](https://huggingface.co/datasets/jason9693/APEACH) | en·ko | CC-BY-SA / CC0 · CC-BY-SA-4.0 |

- **mirror 주의**: script-based 원본이 로드 불가하면 parquet mirror를 쓴다. mirror의 라이선스 태그가 원본과 다를 수 있어 registry에 **원본 라이선스를 함께 기록**한다(WikiTableQuestions·NSMC·Jigsaw·DocVQA 등).
- **TXT-1 데이터셋 교체 (2026-08-05)**: 이전에 쓴 `HuggingFaceM4/DocumentVQA`에는 **OCR 텍스트 컬럼이 없어**(제공: `questionId/question/question_types/image/docId/ucsf_*/answers`) 문서 컨텍스트 없이 질문만 모델에 전달됐다 — TXT-1이 문서 이해력이 아니라 **무문맥 QA**를 측정하고 있었다(실측: 30/30 샘플). full-size 미러(`lmms-lab/DocVQA` 등)도 페이지 이미지만 주고 OCR은 없어, OCR(`words`)을 가진 **1200건 서브셋**으로 교체했다. 대가로 모집단이 작아졌다(test 200 / train 1000). 자세한 경위는 `src/tasks/txt_1.py` 모듈 docstring.
- **TXT-2 split 수정 (2026-08-05)**: `lighteval/wikitablequestions` 미러는 split이 뒤집혀 **train이 10행뿐이고 실데이터 18486행은 test**에 있다. registry에 split이 `default`(=config 이름이지 split이 아님)로 적혀 있고 코드가 그걸 `train`으로 치환해, 30샘플을 요청해도 **항상 10샘플만** 채점됐다. split을 `test`로 고치고, 요청보다 적게 로드되면 예외를 던지게 했다.
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

**구현 완료 — 14개 태스크 전체가 end-to-end 동작** (데이터 로딩 → FMAPI 실행 → 채점 → 시간·비용·Executive Summary 리포트). 로드맵 Phase 0~4 완료:
- Phase 0 기반 구조(FMAPI 어댑터·설정·채점/비용 모듈)
- Phase 1 텍스트 안전 태스크(TXT-4/5/6/8)
- Phase 2 이미지 태스크(IMG-1~5)
- Phase 3 문서·표 태스크(TXT-1/2/3/7)
- Phase 4 시점별 리포트 축적·재현성 메타·인덱스
- 이후 추가: IMG-6(표이미지 구조추출)

세부 설계·의사결정은 [`plan.md`](./plan.md), 결정 이력은 §2(D1–D14) 참고.

### 리포트를 읽을 때 유의할 점

- **호출 실패는 채점에서 제외된다**(0점이 아니다). 정량표의 '실패' 열과 `n_evaluated`로
  드러나며, 실패율 50%를 넘는 셀은 순위·요약에서 제외하고 '신뢰불가'로 표시한다.
  엔드포인트 장애(특히 opus의 산발적 502)가 성능 저하로 오독되던 문제를 막기 위함이다.
- **judge 실패도 평균에서 제외**하고 건수를 표기한다(중간값으로 메우지 않는다).
- **표본이 요청보다 적으면** '실패' 열에 `표본 n/요청`으로 표기된다 — 그 태스크는 다른
  태스크와 같은 신뢰도로 비교하면 안 된다.
- **단가 미등록 모델**은 비용을 `⚠️ 단가 미등록`으로 표기하고 비용 비교에서 제외한다
  (0으로 두면 "가장 저렴한 모델"로 오선정되기 때문).
- **한국어 점수의 기준**은 리포트 "채점 조건"에 `형태소(mecab)` / `음절` 중 무엇인지 명시된다.
  음절 폴백은 글자 겹침만으로 점수를 주어 관대하다.
- **통계 유의성**은 judge 점수에만 적용된다(Wilcoxon). 정량 메트릭은 셀 단위 평균만 저장해
  (스트리밍 O(1) 설계) 샘플을 짝지을 수 없다.
- **IMG-2/TXT-2는 프롬프트를 2026-08-05에 바꿨다**(닫힌 라벨 집합 제공 / 짧은 답 강제).
  그 이전 run과 직접 비교하면 안 된다 — 측정 조건이 다르다.

### 남은 작업 (blocker 아님)

- **시간·비용은 추정치**: `system.ai_gateway.usage` 조인은 미구현(`src/cost/usage.py:fetch_usage`).
  현재는 클라이언트 벽시계 지연 + 응답 `usage` 토큰 × `pricing.yaml` 단가로 계산한다.
- **TEDS(TXT-3/IMG-6)**: 유지보수 패키지 부재로 Cell-F1을 대체 사용 중(plan D12).
- **judge 변별력**: TXT-4는 모델들이 모두 judge 5.0에 수렴해 순위를 못 낸다(정량 지표로 봐야 한다).
- **BERTScore 버퍼 상한 200쌍**: 그보다 큰 샘플 수로 돌리면 BERTScore만 앞 200쌍 기준이 된다
  (`bertscore_n`으로 확인 가능).
