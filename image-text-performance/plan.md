# Image & Text LLM 성능 벤치마크 — 구현 플랜

Databricks Foundation Model API(FMAPI)로 서빙되는 LLM들의 **이미지·텍스트 처리 성능**을
표준 데이터셋 위에서 꾸준히(재현 가능하게) 측정하고, 결과 수치·그래프·정성 샘플을
자동 리포트로 생성하는 프로젝트.

- **대상 모델**: `opus`, `sol`, `sonnet`, `glm` (설정으로 추가/교체 가능 — §6, README "모델 추가하기")
  - opus → `databricks-claude-opus-5` (vision ✅ — 실측 확인)
  - sol → `databricks-gpt-5-6-sol` (vision ✅ — 실측 확인)
  - sonnet → `databricks-claude-sonnet-5` (vision ✅ — 2026-08-06 실측 추가. claude 계열이라 `thinking` 파라미터, `reasoning_effort`는 HTTP 400 거부)
  - glm → `databricks-glm-5-2` (**vision ❌ text-only** — 이미지 입력 미지원, 실측 확인 → 이미지 태스크 전부 N/A)
- **Judge 모델**: `databricks-gemini-3-1-pro` 단일 (평가대상과 다른 Google 계열로 계열 bias 최소화). **실측 확인: 이미지 입력도 이해 가능 → 텍스트·이미지 채점 모두 이 모델 하나로 커버.** (`gemini-3-pro-image`는 image-*생성* 모델이라 judge로 부적합 — 초기 오기 정정)
- **워크스페이스/프로파일**: 기본 `ai_devtools` (host `dbc-a5d4177a-49dc.cloud.databricks.com`). 실행 시 `--profile`로 덮어쓸 수 있고 run의 manifest에 기록된다.
- **실행 스택**: 로컬 Python + 정적 리포트(repo 커밋)
- **최종 산출물**: `reports/<run-id>/` 아래 markdown/HTML 리포트 + 그래프 + 원시 결과(JSON/CSV) + **모델별 수행시간·비용** + Executive Summary. **벤치마크 시점(run)별로 분리·보존**(§12)
- **대상 모델이 전부 reasoning 모델** → reasoning **ON/OFF 둘 다 측정**(공정 비교). 제어 파라미터는 계열마다 다름(§11).

---

## 1. requirement 평가 (강점 / 보강점)

### 강점
- 태스크 분류(이미지 5 / 텍스트 8)가 구체적이고 실무적임.
- "설정으로 모델 추가", "자동 리포트", "표준·불변 데이터셋" 요구는 벤치마크의 정석.
- 대표 메트릭 + 수치 + 그래프라는 산출물 정의가 명확.

### 보강이 필요했던 점 (→ 아래 결정으로 해소)
1. **채점 방법론이 미정** — 생성 태스크(캡션·요약·태그)와 분류 태스크(violence·adult·감정·비속어)는
   메트릭이 완전히 다름. 자유형 생성물을 accuracy로 잴 수 없음.
2. **모델별 비전 지원 격차** — 실측 결과 **가정이 뒤집힘**: opus-5·sol은 vision ✅,
   **glm-5-2만 vision ❌**. 미지원 모델은 이미지 태스크 N/A로 처리.
3. **민감 데이터(violence/adult)의 라이선스·윤리·저장 리스크** — 초기엔 둘 다 민감으로 봤으나
   **실측 후 정리됨**: violence는 서열 데이터가 없어 **무기탐지 binary**(CC-BY-4.0, 일반 취급)로,
   adult는 **NSFW 분류**(ungated 확보, 이미지 비저장 예외만 유지)로 확정. (NSFW gated·violence
   video 문제를 still-image ungated 셋으로 우회 — §5, D13.)
4. **"표준·불변 데이터셋"의 재현성 확보** — 단순 다운로드가 아니라 버전·해시 고정,
   샘플 subset 고정(seed)이 있어야 "꾸준히 같은 기준으로" 비교 가능.

### 실측으로 드러난 추가 리스크 (구현 전 반드시 반영 — §11에 상세)
5. **대상 4개 모두 reasoning 모델** — reasoning ON/OFF가 성능·시간·비용을 크게 흔듦.
   제어 파라미터가 계열마다 다르고 일부는 완전 OFF 불가. 통제·문서화 필수.
6. **응답 스키마 불균일** — opus-5·gemini는 `content`가 리스트(reasoning 블록 포함),
   sol·glm은 문자열. glm은 답을 `reasoning_content`에 먼저 씀. 파서 정규화 필수.
7. **한국어 채점 함정** — ROUGE/Token-F1이 공백 토큰화로 한국어에서 조용히 왜곡됨.
   형태소 토큰화(Mecab) 필요. BERTScore도 다국어 모델 지정 필요.
8. **일부 계획 메트릭 실현 불가** — AUROC(IMG-4)는 chat LLM이 확률을 안 줘서 불가,
   TEDS(TXT-3)는 유지보수 패키지 부재. 대체 필요.

---

## 2. 확정된 설계 결정 (사용자 승인)

| # | 항목 | 결정 |
|---|------|------|
| D1 | **채점 방법** | **하이브리드**. 분류 태스크는 정답 라벨 대비 정량 메트릭(accuracy/F1/Cohen's κ), 생성 태스크는 LLM-as-judge + 참조기반(ROUGE/BERTScore) 병행. **추가로, 샘플 데이터에 대한 각 모델의 실제 출력·판정을 리포트에서 눈으로 비교할 수 있는 정성 갤러리를 반드시 포함.** |
| D2 | **비전 격차** | **모델별 capability 선언 → 미지원 모델은 이미지 태스크 자동 스킵(N/A)**. 리포트에 N/A로 명시. **실측 결과: opus-5·sol은 vision ✅, glm-5-2는 vision ❌ → glm은 이미지 태스크 전부 N/A.** |
| D3 | **민감 데이터** | **IMG-4(NSFW)만** 특별 취급: 이미지 **repo 저장 금지**(런타임 다운로드·로컬 캐시만), 정성 갤러리도 **썸네일 없이 판정값만**. IMG-3(무기, CC-BY-4.0)·나머지 이미지 태스크는 일반 취급(예외 없음). |
| D4 | **실행 스택** | **로컬 Python + 정적 리포트**. FMAPI를 로컬/CI에서 호출, 결과를 markdown/HTML + 그래프로 생성해 repo에 커밋. |
| D5 | **모델 버전 정책** | 각 계열의 **최신 모델**을 대상으로 함 (opus-5 / gpt-5-6-sol / glm-5-2). vision 지원 여부와 무관하게 최신 선택, 미지원은 N/A로 처리. |
| D6 | **Judge** | `databricks-gemini-3-1-pro` **단일**(계열 bias 최소화). 실측상 이미지 입력도 이해 → 텍스트·이미지 채점 모두 커버. 별도 vision judge 불필요. |
| D7 | **언어 범위** | **한국어 + 영어 병행**. 단 TXT-7(키워드 추출)은 한국어 표준셋 부재로 **영어 한정**. 한국어 이미지 데이터는 부재(이미지 태스크는 영어/시각 기반). |
| D8 | **샘플 규모** | 태스크당 기본 **~50 샘플**(seed 고정). 설정으로 확장 가능. v1은 파이프라인 검증·저비용 우선. |
| D9 | **라이선스 정책** | **연구용도 허용**까지 포함(출처·라이선스 명시). 상업 재배포 제약(KorQuAD CC-BY-ND 등)은 registry에 기록. |
| D10 | **reasoning 정책** | **[개정] reasoning OFF(minimal) 단일 고정.** (원안: ON/OFF 둘 다 측정.) 개정 이유: reasoning 효과가 특정 태스크에만 한정되고 full은 실험 시간을 크게 늘림(glm full은 15s 타임아웃 빈발, 실측 160호출 중 19 오류). 실행 매트릭스 = 모델 × 태스크 × 샘플(reasoning 축 제거). "OFF"는 각 모델 지원 **최소 reasoning**(opus·gemini는 완전 OFF 불가). full 비교는 `reasoning_modes:[minimal,full]`로 복원 가능. 리포트에 정책·이유 명시. |
| D11 | **비용·시간 리포트** | 최종 리포트에 **모델별 수행시간·USD 비용** 포함. 데이터 소스: `system.ai_gateway.usage`(권한 有) + `config/pricing.yaml` DBU 단가. `system.billing`은 권한 없음(§10). |
| D12 | **메트릭 대체** | AUROC(IMG-4)는 chat LLM 불가 → **F1+혼동행렬**로 대체. TEDS(TXT-3)는 v1에서 **Cell-F1**로 대체(TEDS는 후속). |
| D13 | **IMG-3/4 확정** | **still-image ungated 셋으로 확정, Phase 2에 통합**(IMG-1/2/5와 같은 이미지 이진분류 파이프라인). IMG-3=`Subh775/WeaponDetection`(CC-BY-4.0)을 **무기/위협 binary**로(서열 severity 데이터 부재 실측 확인 → binary, '무기' 측정임을 명시), **일반 취급**. IMG-4=`DarkyMan/nsfw-image-classification`(168MB, ungated), **NSFW라 D3 예외 적용**(이미지 비저장·판정값-only). 라벨은 다운로드 후 폴더명 확인. |
| D14 | **Executive Summary** | 리포트 최상단에 모델별 성능을 사람이 이해하기 쉬운 **자연어 요약** 포함. **하이브리드**(수치에서 핵심 사실 규칙 추출 → judge가 문단화)로 수치에 고정. fact sheet 저장·감사 가능(§7-0). |

---

## 3. 태스크 정의

각 태스크는 `입력 타입 · 데이터셋 · 대표 메트릭 · 채점 방식(D1)`으로 정의한다.

채점 유형·메트릭은 실측 검토(§부록)로 확정. **볼드 = 실측으로 계획 대비 변경된 부분.**

### 3.1 이미지 태스크 (glm-5-2는 vision 미지원 → 전부 N/A — D2)

| ID | 태스크 | 채점 유형 | 대표 메트릭 | 언어 |
|----|--------|-----------|-------------|------|
| IMG-1 | 이미지 설명(caption) | 생성 | BERTScore(다국어 모델), LLM-judge(1–5) | EN |
| IMG-2 | 이미지 태그 추출 | 멀티라벨 | Precision/Recall/F1 (set 매칭) | EN |
| IMG-3 | 무기/위협 존재 판별 | 이진 분류 | Accuracy, F1 (**서열 severity 데이터 부재 → binary 확정**) | EN(시각) |
| IMG-4 | adult(NSFW) 판별 | 이진 분류 | Accuracy, F1 (**AUROC 제거 — D12**) | EN(시각) |
| IMG-5 | 사람 포함 판별 | 이진 분류 | Accuracy, F1 | EN(시각) |

### 3.2 텍스트 태스크 (한국어는 형태소 토큰화 필수 — §부록)

| ID | 태스크 | 채점 유형 | 대표 메트릭 | 언어 |
|----|--------|-----------|-------------|------|
| TXT-1 | 문서(PDF) 이해 QA | 생성/QA | ANLS/Token-F1, LLM-judge | EN |
| TXT-2 | 표(엑셀) 이해 QA | 생성/QA | Accuracy(값 일치), LLM-judge | EN |
| TXT-3 | 표 구조 추출 | 구조화 추출 | **Cell-F1 (v1) — TEDS는 후속(D12)** | EN |
| TXT-4 | 한국어 이해 QA | 생성/QA | Token-F1(**Mecab**), LLM-judge | KO |
| TXT-5 | 텍스트 요약 | 생성 | ROUGE(**Mecab 전처리**), BERTScore, LLM-judge | KO+EN |
| TXT-6 | 감정 분석 | 분류 | Accuracy, Macro-F1 | KO+EN |
| TXT-7 | 키워드 추출 | 멀티라벨 | Precision/Recall/F1 | **EN 한정(D7)** |
| TXT-8 | 비속어/toxicity | 이진 분류 | Accuracy, F1 | KO+EN |

> 각 태스크는 `tasks/<id>.py` 플러그인으로 구현하고 공통 인터페이스(`build_prompt`, `parse_output`, `score`)를 따른다.
> 데이터셋은 §5, 채점 상세는 §부록(한국어 토큰화·judge 파싱·메트릭 대체) 참고.

---

## 4. 아키텍처

```
데이터셋 로더 ─► 태스크 러너 ─► 모델 어댑터(FMAPI) ─► 출력 파서 ─► 채점기 ─► 결과 저장 ─► 비용/시간 조인 ─► 리포트 생성기
   (§5)     (모델×태스크×샘플    (capability D2 /      (스키마     (정량      (JSON/CSV+  (ai_gateway.usage   (ExecSummary+정량
             ×reasoning D10)    reasoning 제어 D10)   정규화)     +judge)    request_id)  +pricing.yaml §10)  +정성+시간/비용)
```

- **모델 어댑터**: FMAPI 호출 단일 인터페이스. `capabilities`(vision/text)·reasoning 모드 선언, 응답 스키마 정규화, 타임아웃/재시도, `request_id` 기록.
- **태스크 러너**: (모델 × 태스크 × 샘플 × reasoning{on,off}) 매트릭스 순회. 미지원 조합 N/A 스킵.
- **채점기**: 정량 메트릭(한국어 Mecab 전처리) + LLM-judge(루브릭·리스트응답 파싱) + Wilcoxon 유의성.
- **비용/시간 조인**: `system.ai_gateway.usage`에서 request_id로 latency·토큰 조인 → `pricing.yaml`로 USD 환산(§10). **[상태: 미구현]** `src/cost/usage.py:fetch_usage`가 `NotImplementedError`. 현재 리포트는 응답 `usage` 토큰 + 클라이언트 벽시계(`latency_ms_local`) 기반 **추정치**를 쓴다. request_id는 이미 저장돼 조인 준비만 된 상태.
- **리포트 생성기**: Executive Summary(하이브리드 D14) + 정량 표·그래프 + 정성 갤러리 + 시간·비용 리포트.

---

## 5. 데이터셋 전략 (표준·불변 — requirement #5)

- 각 데이터셋은 `datasets/registry.yaml`에 **출처 URL·버전·라이선스·기대 해시·사용 샘플 수(seed 고정)**를 명시.
- 다운로드 스크립트가 해시를 검증해 "불변 표준"을 보장. 대용량/민감 데이터는 **repo에 커밋하지 않고** 로컬 캐시(`.cache/`, gitignore)로 관리.
- 아래는 **HuggingFace API로 실측 검증한 확정 데이터셋**. `registry.yaml`에 id·config·split·라이선스·해시를 고정.
- **로딩 함정 주의**: script-based(viewer disabled) 원본은 parquet mirror로 대체(아래 명시). mirror는 원본과 라이선스 태그가 다를 수 있어 원본 라이선스를 registry에 별도 기록.

| 태스크 | 확정 데이터셋 (HF id) | 라이선스 | 언어 | 비고 |
|--------|----------------------|----------|------|------|
| IMG-1 | `yerevann/coco-karpathy` (alt `nlphuji/flickr30k`) | COCO CC-BY-4.0(주석)/이미지는 Flickr 약관 | EN | 참조 캡션 5개/이미지 |
| IMG-2 | `detection-datasets/coco` (objects.category) | CC-BY-4.0(주석) | EN | 80클래스 멀티라벨 |
| IMG-5 | `detection-datasets/coco` (person=class 0 파생) | CC-BY-4.0(주석) | EN | 사람 포함 이진 |
| IMG-3 | `Subh775/WeaponDetection` (무기/위협 존재 binary) | **CC-BY-4.0** | EN | ungated·still-image·introspectable. **severity 라벨 부재 → binary("무기/위협 존재 여부")로 확정**. '폭력'이 아닌 '무기' 측정임을 리포트 명시 |
| IMG-4 | `DarkyMan/nsfw-image-classification` (168MB) | `cc`(변이 미상, registry 기록) | EN | ungated. 라벨 스키마는 다운로드 후 폴더명으로 확인 전제. gated `deepghs`/`strangerguardhf` 회피 |
| TXT-1 | `nielsr/docvqa_1200_examples` (**OCR 텍스트 `words` 보유**) | 미러 미표기/원본 연구용 | EN | query(dict, en)+words(OCR)+answers, **ANLS 채점 구현 완료**. 2026-08-05 교체 — 아래 주석 참고 |
| TXT-3 | `apoidea/pubtabnet-html` (`html_table` = 실제 HTML 구조 GT) | **CDLA-Permissive-1.0(상업 OK)** | EN | **TEDS-ready 확인**, PubLayNet은 부적합 |
| TXT-2 | `lighteval/wikitablequestions` (parquet) | CC-BY-4.0 | EN | 원본 `stanfordnlp/…`는 script-based |
| TXT-4 | `KorQuAD/squad_kor_v1` + `klue/klue`(mrc) | KorQuAD **CC-BY-ND-4.0**/KLUE CC-BY-SA | KO | ND=변형 재배포 금지(registry 기록) |
| TXT-5 | EN `abisee/cnn_dailymail`(apache-2.0) / KO `daekeun-ml/naver-news-summarization-ko`(apache-2.0) | apache-2.0 | EN+KO | **KLUE엔 요약 태스크 없음**(가정 정정) |
| TXT-6 | EN `stanfordnlp/sst2` / KO `Blpeng/nsmc`(parquet mirror) | GLUE 관례/CC-BY-2.0 | EN+KO | 원본 `e9t/nsmc`는 script-based |
| TXT-7 | `taln-ls2n/inspec` (keyphrases) | unknown(연구용) | **EN 한정** | 한국어 표준셋 부재(D7) |
| TXT-8 | EN `thesofakillers/jigsaw-…`(parquet) / KO `jason9693/APEACH`(CC-BY-SA-4.0) | CC-BY-SA | EN+KO | 원본 `google/jigsaw_…`는 script-based |

> **IMG-3/4**: still-image ungated 셋으로 확정, Phase 2에 통합(D13). IMG-3는 severity 라벨이 어디에도 없어 **무기/위협 binary**로 측정('폭력'이 아닌 '무기'임을 리포트 명시), 일반 취급. **IMG-4(NSFW)만** 미디어 repo 커밋 금지·런타임 다운로드·판정값-only(D3). 두 셋 모두 라벨 스키마는 다운로드 후 폴더명으로 검증.

> **TXT-1 데이터셋 교체 (2026-08-05, 실측)**: 최초 선택한 `HuggingFaceM4/DocumentVQA`와 대안으로 적어둔 `lmms-lab-encoder/DocVQA` 둘 다 **OCR 텍스트 컬럼이 없다**(제공: `questionId/question/question_types/image/docId/ucsf_*/answers`). TXT-1은 텍스트 태스크(is_vision=False)라 페이지 이미지를 못 쓰는데, 구현이 OCR 부재 시 "질문만" 프롬프트로 조용히 폴백해 **30/30 샘플이 문서 없이 전송**됐다(무문맥 QA 측정, token_f1 0.006). full-size 미러도 전부 이미지-only여서, OCR(`words`)을 가진 `nielsr/docvqa_1200_examples`(train 1000/test 200, 평가는 test)로 교체하고 **컨텍스트 부재 시 예외를 던지도록** 바꿨다(§부록 "합성/조용한 폴백 금지"와 동일 원칙). 트레이드오프: 모집단이 5349→200으로 작아졌다. 이 미러의 `query`는 언어별 dict(de/en/es/fr/it)라 en만 쓴다.

---

## 6. 설정 (모델·태스크 확장성 — requirement #3, #6)

`config/models.yaml` (예시 스키마):

```yaml
profile: ai_devtools                    # Databricks CLI profile
judge: databricks-gemini-3-1-pro        # 단일 judge, 텍스트+이미지 모두 (D6)
reasoning_modes: [minimal, full]        # 둘 다 측정 (D10). minimal = 각 모델 최소 reasoning
models:
  - id: opus
    endpoint: databricks-claude-opus-5
    capabilities: [text, vision]        # 실측 vision ✅
    reasoning:                          # 계열마다 파라미터 다름 (§11, 실측)
      minimal: {thinking: {type: disabled}}
      full:    {}                       # 기본(adaptive)
  - id: sol
    endpoint: databricks-gpt-5-6-sol
    capabilities: [text, vision]        # 실측 vision ✅
    reasoning:
      minimal: {reasoning_effort: none}
      full:    {reasoning_effort: high}
  - id: glm
    endpoint: databricks-glm-5-2
    capabilities: [text]                # 실측 vision ❌ → 이미지 태스크 N/A (D2)
    reasoning:
      minimal: {reasoning_effort: none}
      full:    {}
```

> ⚠️ 모드 키를 `on`/`off` 대신 **`minimal`/`full`**로 명명 — YAML에서 `on`/`off`는 boolean으로 파싱되는 예약어라 키로 쓰면 버그. 의미도 더 정확(§11: opus·gemini는 완전 OFF 불가라 "off"는 오해 소지).

- `config/tasks.yaml`: 태스크 ID, 샘플 수(기본 50, seed 고정 — D8), 언어(ko/en — D7), 프롬프트 템플릿.
- `config/pricing.yaml`: 모델별 DBU 단가 + `usd_per_dbu`(§10, D11). 비용 계산의 단일 소스.

> ✅ vision·reasoning 파라미터는 `ai_devtools`에서 실제 호출로 검증(opus-5·sol vision ✅,
> glm-5-2 `"Image input is not supported"`; reasoning off 파라미터는 §11 표 참고).

---

## 7. 리포트 생성 (requirement #4, D1)

리포트는 **Executive Summary + 정량 + 정성 + 성능(시간·비용)** 축으로 자동 생성. reasoning ON/OFF를 나란히 비교:

0. **Executive Summary (리포트 최상단 — 신규 요구, D14)**
   - 모델별 성능을 **사람이 한눈에 이해할 자연어 요약**. "어느 모델이 무엇을 잘하고/못하고, 시간·비용은 어떤지"를 문단으로.
   - 생성 방식 = **하이브리드(수치→LLM 문장)**: 집계 수치에서 핵심 사실(태스크별 1위·강점/약점·reasoning ON/OFF delta·시간·비용 트레이드오프)을 **규칙 기반으로 추출**해 구조화 → 그 사실들만 judge 모델에 넘겨 자연어 문단 생성.
   - 이렇게 하면 요약이 **실제 수치에 고정(hallucination 방지)**되면서도 읽기 쉬움. 추출된 사실(fact sheet)도 함께 저장해 문장-수치 대응을 감사 가능.
   - 결정론적 fallback: judge 실패 시 규칙 기반 템플릿 문장으로 대체.

1. **정량 요약**
   - 태스크 × 모델 점수 매트릭스(표), 대표 메트릭 강조. reasoning ON/OFF 열 분리.
   - 그래프: 태스크별 막대 차트, 모델별 종합 레이더, (가능 시) 히트맵.
   - 미지원 조합은 `N/A`로 명시. 생성 태스크는 통계 유의성(Wilcoxon) 병기해 과대해석 방지.
2. **정성 샘플 갤러리** (D1 핵심)
   - 태스크별 고정 샘플 N개에 **입력 + 각 모델 실제 출력/판정 + 정답 + 채점 + judge 근거**를 나란히.
   - 이미지 태스크: 썸네일 + 각 모델 출력. **단 IMG-4(NSFW)만 썸네일 없이 판정값·정답만 (D3).**
   - "샘플은 예시일 뿐 평균 성능이 아니다" 디스클레이머 명시.
3. **성능 리포트 (시간·비용 — D11, 신규 요구)**
   - 모델별 **수행시간**(태스크별·전체, median/p95), **USD 비용**(입력·출력·reasoning·캐시 분해), 토큰 사용량.
   - 소스: 벤치마크가 각 호출의 `request_id`를 기록 → `system.ai_gateway.usage`와 조인해 `latency_ms`·토큰 실측, `config/pricing.yaml`로 USD 환산. **[현재: 조인 미구현 — 응답 `usage` 토큰 + 벽시계 지연 기반 추정치]**
   - **reasoning ON vs OFF의 시간·비용 delta를 명시**(reasoning의 실질 비용을 드러냄).
   - **billable_output = completion_tokens + reasoning_tokens** (reasoning 토큰이 completion에 미포함 — 실측 확인, 누락 시 비용 과소계상).
4. **원시 결과**: 재현·감사용 JSON/CSV(`results/<run-id>/`) 커밋. usage 원본(raw) 보존.

리포트는 **벤치마크 시점(run)별로 분리 생성·보존**된다 — 상세 정책은 §12.

---

## 8. 디렉터리 구조 (초안)

```
image-text-performance/
├── plan.md
├── requirement.md
├── README.md
├── pyproject.toml            # 의존성·설정
├── config/
│   ├── models.yaml           # 모델·capability·judge·reasoning 모드 (§6)
│   ├── tasks.yaml            # 태스크·샘플수·언어·프롬프트 참조
│   ├── pricing.yaml          # DBU 단가 + usd_per_dbu (§10, 비용 계산)
│   └── judge_rubrics.yaml    # 태스크별 judge 채점 루브릭(1–5 앵커)
├── datasets/
│   ├── registry.yaml         # 출처·버전·해시·라이선스·mirror (§5)
│   └── download.py           # 해시 검증 다운로드 (민감데이터 캐시-only)
├── src/
│   ├── adapters/             # FMAPI 어댑터 (응답 스키마 정규화·reasoning 제어·타임아웃/재시도)
│   ├── tasks/                # 태스크별 플러그인 (build_prompt/parse/score)
│   ├── scoring/
│   │   ├── tokenizers.py     # 언어별 토큰화 (한국어 Mecab — §부록)
│   │   ├── metrics.py        # ROUGE(ko 전처리)/BERTScore/F1/Cell-F1
│   │   ├── judge.py          # LLM-judge (gemini 리스트 응답 파싱·루브릭)
│   │   └── stats.py          # Wilcoxon 유의성
│   ├── cost/                 # ai_gateway.usage 조인 + pricing.yaml → 시간·USD (§10)
│   ├── runner.py             # (모델×태스크×샘플×reasoning) 실행 + request_id 기록
│   └── report/               # 정량·정성·성능(시간·비용) 리포트 생성
├── results/<run-id>/         # run별 원시 결과 (JSON/CSV, usage raw) — append-only (§12)
├── reports/
│   ├── <run-id>/             # run별 리포트 (markdown/HTML + 그래프 png + manifest.json)
│   └── index.md              # 전체 run 목록·시간순 링크 (§12)
├── .cache/                   # 데이터셋·모델 로컬 캐시 (gitignore, NSFW 이미지도 여기만)
└── tests/                    # 한국어 메트릭·파서·비용계산 단위 테스트
```

---

## 9. 구현 단계 (로드맵)

- **Phase 0 — 스캐폴딩**: repo 구조, 설정 스키마(models/tasks/pricing/rubrics), FMAPI 어댑터(응답 정규화·reasoning 제어·타임아웃/재시도), 1개 모델 smoke test. 로컬 의존성 설치(§부록).
- **Phase 1 — 텍스트 안전 태스크**: TXT-4(한글)·TXT-5(요약)·TXT-6(감정)·TXT-8(비속어) end-to-end. **한국어 Mecab 토큰화·judge 파싱을 5샘플로 먼저 검증**(§부록 P0). reasoning ON/OFF·비용·시간·Executive Summary까지 이 단계에서 완성해 파이프라인 전체를 세움.
- **Phase 2 — 이미지 태스크 (전부)**: IMG-1/2/5 + **IMG-3(무기, `Subh775`) + IMG-4(NSFW, `DarkyMan`)**. glm N/A 처리, 정성 갤러리(이미지 포함, **단 IMG-4는 판정값-only·비저장 — D3**). 착수 시 IMG-3/4 라벨 폴더명 검증부터.
- **Phase 3 — 문서·표 태스크**: TXT-1/2 + TXT-3(**Cell-F1**; TEDS는 후속). TXT-7(EN 한정).
- **Phase 4 — 자동화·축적**: 정기·모델추가 재실행, 시점별 리포트 축적(§12). (선택) `system.billing` 권한 확보 시 비용 사후 검증.

---

## 10. 비용·시간 데이터 파이프라인 (D11, 실측 확정)

수행시간·비용 리포트의 데이터 경로:

1. **호출 시 `request_id` 기록** — 어댑터가 각 FMAPI 응답의 request_id를 결과에 저장.
2. **usage 조인** — `system.ai_gateway.usage` 테이블(현 권한으로 **접근 가능**, warehouse `2c4aa6fec2649553`)에서 request_id로 조인. 이 테이블이 제공(실측 컬럼): `latency_ms`, `time_to_first_byte_ms`, `input_tokens`, `output_tokens`, `total_tokens`, `token_details.{cache_read_input_tokens, cache_creation_input_tokens, output_reasoning_tokens}`, `endpoint_name`, `status_code`, `event_time`.
3. **USD 환산** — `config/pricing.yaml`의 모델별 DBU 단가 × `usd_per_dbu`.
   - 공식: `usd = usd_per_dbu × (billable_input×dbu_in + billable_output×dbu_out + cache_read×dbu_cache_rd + cache_write×dbu_cache_wr) / 1e6`
   - **billable_output = completion_tokens + reasoning_tokens** (reasoning 토큰이 completion에 미포함 — 실측 확인, 누락 시 비용 과소계상).
   - context 임계값(gemini 200K 등)에 따라 short/long 단가 행 선택.

**검증된 DBU 단가** (Databricks 공식 pricing, $0.07/DBU 가정 시 USD in/out):

| 모델 | input DBU | output DBU | ≈USD in/out (1M) |
|------|-----------|-----------|------------------|
| opus-5 | 71.4 | 357.1 | $5.00 / $25.00 |
| sol (short) | 71.4 | 428.6 | $5.00 / $30.00 |
| glm-5-2 | 20.0 | 62.9 | $1.40 / $4.40 |
| gemini-3-1-pro (judge) | 35.7 | 214.3 | $2.50 / $15.00 |

> ⚠️ **확인 필요(단가 정밀도)**: `usd_per_dbu`(=0.07 가정, 실제 계약 단가로 확정) / Global vs In-geo 라우팅(~10% 차) / gemini 20% 프로모(2027-01-31까지) / sol short-long 임계값. `system.billing`은 **권한 없음** → pricing.yaml이 1차 소스, 권한 확보 시 사후 검증.

---

## 11. reasoning 제어 (D10, README에 문서화 — 실측 확정)

대상 4개 모델이 **전부 reasoning 모델**이고 제어 파라미터가 **계열마다 다름**. reasoning ON/OFF를 둘 다 측정하므로 정확한 파라미터가 필수:

config 모드 키 `minimal`/`full`에 매핑되는 실제 파라미터(§6):

| 모델 | reasoning 제어 | `minimal` (최소/OFF) | `full` (ON) | 실측 |
|------|---------------|----------------|-----|------|
| **sol** (GPT식) | `reasoning_effort` | `none` | `high` | ✅ minimal 즉답, reasoning_tokens=0 |
| **opus-5** (Claude식) | `thinking.type` + `output_config.effort` | `thinking:{type:disabled}` | `thinking:{type:adaptive}` | ✅ `reasoning_effort` 거부, effort는 low~max(none 없음) |
| **glm** (GLM식) | `reasoning_effort` | `none` | (기본) | ✅ minimal 즉답 |
| **gemini** (judge) | `generation_config.thinking_config.thinking_level` | (최소만, 완전 OFF 불가) | (기본) | none/minimal 거부 |

**README 필수 서술**: (1) reasoning ON/OFF 둘 다 측정하는 정책과 근거(성능·시간·비용 trade-off), (2) 모델별 제어 파라미터, (3) **opus·gemini는 완전 OFF 불가** → "OFF 모드"는 "각 모델 지원 최소 reasoning"으로 정의, (4) glm의 함정(큰 max_tokens → reasoning에 소진, `reasoning_content` 필드).

**런타임 정책(실측)**: per-request timeout 15s + 지수 backoff 3회. 동시성 5–10 안전(rate limit 넉넉, 429 미관측). glm은 3–5배 느림.

---

## 12. 벤치마크 시점·버전 관리 (requirement #3·#4 — 시점별 리포트)

**요구**: 모델을 추가해(또는 데이터셋·프롬프트를 바꿔) 다시 벤치마크하면, 결과가 **벤치마크 시점(run)별로 나뉘어** 리포트로 남아야 한다. 과거 리포트는 덮어쓰지 않고 보존해 시점 간 비교가 가능해야 한다.

- **run 단위 격리**: 각 실행은 고유 `run-id`(예: `2026-07-31T14-00_v3` = 타임스탬프 + 벤치마크 버전)로 `results/<run-id>/`, `reports/<run-id>/`에 분리 저장. 기존 run은 절대 덮어쓰지 않음(append-only).
- **run 메타데이터**: 각 run의 `manifest.json`에 **그 시점의 모델 목록·엔드포인트·reasoning 모드·데이터셋 버전(해시)·pricing 버전·코드 커밋 SHA**를 기록. → "이 리포트는 어떤 구성으로 뽑혔나"가 자기설명적이고 재현 가능.
- **모델 추가 시나리오**: 설정에 모델을 추가하고 재실행하면 새 `run-id` 리포트가 생성됨. 그 리포트는 그 시점의 **전체 모델**(기존+신규)을 함께 비교. 기존 run 리포트는 그대로 보존.
  - 기본 정책: **모델 추가 시 전체 재실행**(모든 모델을 동일 데이터·시점에서 비교해야 공정). 비용 절감을 위해 기존 모델 결과 캐시 재사용은 옵션으로 두되, 데이터셋·프롬프트·pricing이 동일할 때만 허용(manifest 해시로 검증).
- **인덱스·추세**: `reports/index.md`(또는 HTML)가 전체 run 목록을 시간순으로 링크. (후속) 여러 run에 걸친 **동일 모델의 시계열 추세** 그래프 — 모델 버전이 올라가며 성능·비용이 어떻게 변했는지.
- **디렉터리 반영**: `results/<run-id>/`, `reports/<run-id>/`, `reports/index.md`, run별 `manifest.json`.

> 이 정책이 requirement #4의 "꾸준히 리포트"와 #3의 "모델 언제든 추가"를 시점 비교까지 되도록 구체화한다.

---

## 부록. 채점 방법론 상세 (구현 전 필독)

실측 검토로 확정된 채점 관련 필수 처리:

- **한국어 토큰화 (P0)**: ROUGE·Token-F1은 공백 토큰화 시 한국어에서 조용히 왜곡(형태소 미분리). **KoNLPy Mecab** 형태소 토큰화 후 채점. 시스템 의존성(`mecab`, `mecab-ko-dic`) 설치 필요. Phase 1에서 **5샘플로 먼저 검증** 후 스케일업.
- **BERTScore**: 한국어는 **다국어 모델**(`bert-base-multilingual-cased` 또는 `klue/roberta-base`) 지정. 첫 실행 시 모델 다운로드(오프라인 재현 위해 해시 고정).
- **judge 응답 파싱 (P0)**: gemini는 `content`가 리스트(`[{type,text,thoughtSignature}]`) → 텍스트 추출 후 점수 파싱. 루브릭(1–5 앵커)을 프롬프트에 포함. 위치·장황함 편향 완화(후보 순서 셔플), 한국어 태스크는 10% 이중 judge 감사(v1은 config-gated).
- **judge 잘림·오탐·폴백 (2026-08-05 수정, 실측)**: gemini는 reasoning을 못 꺼서 사고 토큰을 먼저 쓴다. `max_tokens=256`이면 IMG-1 캡션 판정에서 `reasoning 240 / completion 12 / finish=length`로 잘려 **30/30 파싱 실패**(리포트에 `judge_mean=0.0`으로 표시). 1024면 정상. 짧은 판정(TXT-4)은 256에서도 통과해 **길이에 따라 조용히 갈렸다**. 더해서 옛 `parse_judge_score`가 폴백으로 아무 숫자나 주워 **잘린 산문의 본문 숫자를 점수로** 오인했고(`"captures 1 of the 5 key elements"`→5), TXT-1/2/4/5는 파싱 실패를 **3점으로 메워** 실제 판정과 구분 불가였다(3점 비율: sol TXT-2 70% vs 1024를 쓴 TXT-5 0%). 조치: `JUDGE_MAX_TOKENS`(1024) 단일 상수, 명시적 점수 표현만 인정(못 찾으면 None), 실패는 평균에서 제외하고 `n_judge_failed`로 노출, 전 태스크가 `run_judge()`/`summarize_judge_scores()` 공유. 테스트: `tests/test_judge.py`.
- **AUROC 제거(IMG-4)**: chat LLM이 logprobs·확률 미제공 → Accuracy+F1+혼동행렬로 대체.
- **TEDS 대체(TXT-3)**: 유지보수 Python 패키지 부재 → v1은 **Cell-F1**(셀 위치+텍스트 fuzzy 매칭). GT는 `apoidea/pubtabnet-html`의 `html_table`.
- **ANLS 구현(TXT-1, 2026-08-05)**: DocVQA 공식 메트릭을 `src/scoring/metrics.py:anls()`로 구현(정규화 편집거리, **τ=0.5**, 다중정답 max, strip+lowercase 정규화). 근거: 정답이 문서에 적힌 짧은 문자열이라 exact_match는 한 글자 오타에 0점, Token-F1은 표기 차이(`485` vs `$485`)에 과민하다. τ 미만 유사도를 0으로 절단해 우연한 부분일치를 배제한다. 리포트 대표 메트릭 우선순위에서 `token_f1`보다 앞(`_METRIC_PRIORITY`)이라 TXT-1의 대표값이 공식 메트릭이 된다. 테스트: `tests/test_metrics.py`.
- **응답 스키마 정규화**: opus-5·gemini 리스트 / sol·glm 문자열 / glm은 `reasoning_content` 우선. 어댑터가 4종 모두 평문으로 정규화.
- **과대해석 방지**: 생성 태스크는 참조기반 메트릭이 품질과 약상관 → 정성 갤러리 + Wilcoxon 유의성으로 보완, 요약 문장은 fact sheet에 고정(§7-0).
- **로컬 의존성(설치 필요)**: `datasets huggingface_hub rouge_score bert_score sacrebleu evaluate konlpy transformers scipy fuzzywuzzy pdfplumber`. 이미 있음: matplotlib, pandas, openpyxl, jinja2, sklearn.

---

## 부록2. 확인이 남은 항목 (blocker 아님)

- **IMG-3/4 라벨 스키마**: DarkyMan(nsfw.zip)·Subh775는 다운로드 후 폴더명/클래스로 binary 라벨 확정. DarkyMan `cc` 라이선스 변이(구체 버전) 확인.
- **비용 단가 정밀화**: usd_per_dbu(=0.07 가정), 라우팅(Global/In-geo), gemini 프로모, sol 컨텍스트 임계값(§10).
- GPT/Claude의 cached·reasoning 토큰 필드명은 실부하에서 raw usage 로깅해 확정.

---

*이 플랜은 requirement.md와 사용자 승인 결정(D1–D14)을 기반으로 하며, 모델·vision·reasoning·비용소스·데이터셋은 `ai_devtools` workspace와 HuggingFace API에서 **실측 검증**됨. 5개 조사 트랙(런타임 계약·채점 방법론·데이터셋·가격·민감 still-image 탐색) 결과를 반영.*
