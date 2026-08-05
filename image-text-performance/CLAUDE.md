# CLAUDE.md — image-text-performance 벤치마크 운영 지침

이 파일은 이 프로젝트를 다루는 AI 에이전트(Claude Code / Vibe)와 사람 기여자 모두를 위한
운영 지식이다. clone 후 재수행·확장할 때 여기부터 읽으면 이미 확인된 함정을 반복하지 않는다.
프로젝트 개요·사용법은 [README.md](./README.md), 설계 결정은 [plan.md](./plan.md) 참고.

## 한 줄 요약
Databricks FMAPI 모델(opus·sol·glm)의 이미지·텍스트 성능을 표준 데이터셋으로 측정해
수치·그래프·정성 갤러리·고객용 HTML 프레젠테이션 리포트를 자동 생성하는 벤치마크.
**14개 태스크**(IMG-1~6, TXT-1~8) × **3모델**(opus·sol·glm) → glm vision 미지원으로
이미지 6태스크 N/A → 실행 **36셀**. 모델 추가는 `config/models.yaml`(+`pricing.yaml`)만 고치면 되고,
빠진 항목은 실행 전 검증이 막는다(README "모델 추가하기"). 2026-08-06에 sonnet-5를 붙여
설정검증·dry-run 50셀·**2샘플 실호출 28셀 완주**까지 확인했다(4모델 30샘플 전체 완주는 안 함).
이후 3모델로 되돌렸다 — models.yaml에 주석으로 남아 있어 풀면 바로 4모델이 된다.

## 전제
- Databricks CLI 프로파일 필요. 기본값은 `config/models.yaml`의 `profile`(현재 `ai_devtools`)이고
  실행 시 **`--profile <name>`으로 덮어쓴다**. 프로파일은 추측하지 말고 사용자에게 확인받을 것 —
  잘못된 프로파일은 엉뚱한 워크스페이스에 과금되고 실패 원인(IP ACL 403 등) 파악도 흐려진다.
  실행 로그와 run의 `manifest.json`(`profile`)에 어느 프로파일로 돌았는지 기록된다.
- 데이터셋은 HuggingFace **streaming**으로 필요한 샘플만 받음(전체 다운로드 X). `.cache/`(gitignore).
- 의존성은 `pip install -e .`. 선택: mecab(한국어 형태소, 없으면 음절 fallback), bert-score+torch(없으면 deferred).

## 벤치마크 실행 / 재수행
```bash
python -m src.runner --samples 30 --fresh        # 30샘플, 기존 리포트 삭제 후 새로
python -m src.runner --dry-run                    # 매트릭스만 미리보기(호출 없음)
python -m src.runner --models sol --samples 3     # 빠른 확인
```
- **`--fresh`**: 이전 run을 정리하고 index부터 새로 생성. 단 **삭제가 아니라 `.trash/`로
  이동**이고, **리포트가 다 만들어지고 종료 코드 판정까지 통과한 뒤에만** 치운다.
  실패로 판정되면(exit 1) 이전 run을 그대로 둔다 — 안 그러면 모델 전체가 403인 run도
  구조적으로는 report.md·chart를 만들어내므로, 검증된 유일한 리포트를 치운 뒤 exit 1을
  내는 최악의 조합이 나온다(실측 사고 2026-08-06: git에서 복구해야 했다).
  **정리는 트랜잭션이다**: 이동 도중 실패하거나 정리 후 index 재생성이 실패하면 옮긴 것을
  전부 제자리로 되돌린다. 판정을 정리보다 앞세우는 것만으로는 정리 *후* 단계의 실패를
  막을 수 없어서다(전부 성공 아니면 전부 원복).
- **`--samples N`**: 태스크당 N개(생략 시 config 기본 50). seed 고정이라 재현 가능.
  **최대 요청값일 뿐**이라 데이터셋이 그만큼 없으면 적게 채점된다 → 러너가 경고하고
  리포트의 '실패' 열에 `표본 n/요청`으로 표기한다.
- **`--profile <name>`**: Databricks 프로파일 명시(생략 시 config 기본값). 실행 로그·manifest에 기록.
- reasoning은 config에서 **OFF(minimal) 고정** → 옵션 불필요. ON까지 비교하려면 `--reasoning-modes minimal,full`(실행 배증·glm 타임아웃 주의).
- **오래 걸림**: 36셀 × N샘플 + judge. 30샘플 ≈ 1.5~2시간(모델 3개), 50샘플 수 시간+.
  → 반드시 백그라운드 실행(`nohup python3 -u -m src.runner ... &`) 후 프로세스 종료까지 대기.
  "실행 완료" 로그 뒤에도 리포트 생성(judge 요약·이미지 처리·BERTScore)이 수 분 더 걸림.
- **종료 코드를 확인할 것**: 채점 오류·실패율 과다 셀이 있거나 전체 호출 실패율 >10%면 `exit 1`.
  리포트는 그대로 생성되지만 수치를 신뢰할 수 없다는 뜻이다(옛 러너는 무조건 0을 반환해
  모델 전체가 403이어도 "실행 완료"로 보였다).

## 모델 추가 (검증이 막아주는 것들)
`config/models.yaml`에 항목 추가 + `config/pricing.yaml`에 단가 추가. 그다음 `--dry-run`.
실행 전 `validate_models_config()`가 아래를 **차단**한다(조용히 잘못 도는 걸 막기 위해):
- `reasoning` 모드 누락 → 빈 파라미터는 모델 기본값(=ON일 수 있음)인데 리포트는 OFF로 표기
- `capabilities` 누락·오타(`visoin`) → 해당 태스크가 전부 조용히 N/A
- `id` 중복 → 뒤 항목이 앞을 덮어써 한 모델이 사라짐
- judge endpoint를 평가 대상에 넣기 → 자기 채점
경고(실행은 됨): `pricing.yaml` 단가 누락 → 비용 계산 불가('단가 미등록' 표기, 비용 비교 제외).
- **`--resume`의 복구 단위**: 완료 셀은 건너뛰고(`resume-skip`), **중단된 셀은 이미 호출한
  샘플을 재사용**해 남은 것만 부른다(실측: 10샘플 셀에서 6개 재사용 → 4개만 호출).
  실패 응답(`finish_reason=error`)과 오염 셀(>50% 실패) 행은 재사용하지 않고 재호출한다.
  샘플 결과는 5개마다 디스크에 flush되므로 중단 시 재호출 손실은 최대 4개다.
- **모델 추가 후 `--resume` 금지**(러너가 차단함): 구성이 다른 결과가 한 run에 섞이고 manifest가
  거짓이 된다. 새 run으로 전체 재실행하는 것이 정책(동일 데이터·시점 비교).
  **manifest.json이 없거나·읽을 수 없거나·핵심 축이 비어 있으면 `--resume` 자체를 거부**한다
  (fail-closed). 핵심 축은 `models`·`task_ids`·`reasoning_modes`·`samples_per_task`·`profile`.
  예전엔 (a) 파일이 없으면 그냥 새로 써버렸고(= 확인 불가한 경우가 가장 느슨하게 통과),
  (b) 읽기 실패 시 빈 dict로 폴백했다. 드리프트 검사는 없는 필드를 건너뛰므로 `{}`는
  "차이 없음"이 되어 구성 가드가 통째로 무력화됐다(모델·샘플 수가 달라도 옛 결과와 섞임).
- **오염 셀(>50% 실패) 판정도 last-wins 기준**이다. 물리 행을 세면 같은 샘플의 옛 성공 행이
  최신 실패를 희석한다(실측: 옛 성공 30 + 최신 403 30 → 실제 30/30 실패인데 30/60=50%로
  계산돼 임계를 못 넘고 재실행 대상에서 빠졌다).
- **손상 행 처리는 모든 로드 경로가 같아야 한다**: `[]`·`null`처럼 유효 JSON이지만 object가
  아닌 행에서 `isinstance(d, dict)` 가드가 없으면 `_load_cell_rows`는 AttributeError,
  `_partial_progress`는 TypeError로 resume 전체가 죽는다(한쪽만 고치면 경로마다 갈린다).
- **완주 판정은 셀 개수가 아니라 키 집합**으로 한다. 개수만 보면 옛 run에서 남은 scores 항목이
  이번 매트릭스의 누락 셀을 벌충해 exit 0이 되고, 그 낡은 항목이 리포트 순위에도 들어간다.
- **samples.jsonl에 같은 (모델·태스크·모드·샘플) 행이 중복되면 마지막 것만 집계**한다
  (`load_sample_results`). append-only라 중단·재실행이 겹치면 중복이 생기는데, 그대로 세면
  시간·비용·토큰·실패율이 그 샘플만큼 부풀려진다. 채점에 쓰인 건 마지막 응답이다.
- 모델별 런타임: `models[].runtime.timeout_seconds`로 느린 모델만 올릴 수 있다(glm은 120s 설정).

## 실행 규모를 키울 때 (대규모 팁)
- **HF unauthenticated rate limit**으로 데이터 스트리밍이 느림 → `HF_TOKEN` 설정하면 빨라짐.
- **glm-5-2가 3~5배 느림** → `models.yaml`에서 glm만 `runtime.timeout_seconds: 120`으로 둠.
- **opus 엔드포인트 502 산발 발생**(실측 420호출 중 20건, `invalid response from an upstream
  server`). 같은 샘플을 다른 모델은 성공하므로 데이터가 아니라 업스트림 문제. 재시도 5회·
  backoff 2s로 완화했고, 남은 실패는 채점에서 제외돼 리포트에 드러난다.

## 실측 확정 사실 (검증됨 — 재조사 불필요)
### 모델 · vision · reasoning
- opus=`databricks-claude-opus-5`(vision✅), sol=`databricks-gpt-5-6-sol`(vision✅),
  glm=`databricks-glm-5-2`(**vision❌** "Image input is not supported"), judge=`databricks-gemini-3-1-pro`(vision✅).
- 넷 다 reasoning 모델. 제어 파라미터가 계열마다 다름:
  - sol/glm: `reasoning_effort: none` (OFF)
  - opus: `thinking: {type: disabled}` (`reasoning_effort` 거부)
  - gemini: `generation_config.thinking_config.thinking_level` (완전 OFF 불가, 최소 레벨만)
- **opus·gemini는 완전 OFF 불가** → "OFF"는 각 모델 지원 최소 reasoning으로 정의.
- 응답 스키마 2갈래: opus·gemini는 `content`가 list(reasoning 블록 포함), sol·glm은 string.
  glm은 답을 `reasoning_content`에 먼저 씀. 어댑터(`src/adapters/fmapi.py`)가 4종 다 정규화.

### 비용·시간 데이터 소스
- **`system.billing.*`는 워크스페이스 권한 없음**(USE SCHEMA 거부). 재시도 무의미.
- **`system.ai_gateway.usage`가 접근 가능** — request_id로 조인해 `latency_ms`·토큰 실측.
  USD 단가 컬럼은 없음 → 토큰→USD는 `config/pricing.yaml`(DBU 단가 × usd_per_dbu=0.07 표준). SQL warehouse 필요.
- **단, 이 조인은 아직 미구현**(`src/cost/usage.py:fetch_usage`가 NotImplementedError).
  현재 리포트의 시간은 **클라이언트 벽시계**(`latency_ms_local`), 비용은 **응답 `usage` 토큰** × pricing이다. 추정치임을 유의.

### judge 함정 — 조용히 오염된다 (2026-08-05 수정)
judge(gemini)는 **reasoning을 완전히 끌 수 없어** 사고 토큰을 먼저 쓴다. 세 겹의 버그가 있었다:
1. **max_tokens 부족**: 태스크가 각자 `256`을 하드코딩(TXT-5만 1024). 실측 IMG-1 캡션 판정:
   256 → `reasoning 240 / completion 12 / finish=length`로 문장 중간 잘림 → 점수 파싱 실패 30/30.
   1024 → `finish=stop`, 정상. **짧은 판정은 256에서도 통과해 길이에 따라 조용히 갈렸다.**
   → 지금은 `src/scoring/judge.py:JUDGE_MAX_TOKENS`(1024) **하나만** 쓴다. 태스크에 하드코딩 금지.
2. **파싱 오탐**: 옛 `parse_judge_score`가 폴백으로 아무 숫자나 주워 점수로 썼다. 잘린 산문의
   **본문 숫자**가 점수가 됐다(실측: `"captures 1 of the 5 key elements"`→5, `"2 sinks and 3 mirrors"`→3).
   → 지금은 **명시적 점수 표현만** 인정(`Score: N`, `N/5`, 숫자 단독). 못 찾으면 None.
3. **조용한 3점 폴백**: 실패 시 `judge_scores.append(3)`. 실제 판정과 섞여 사후 구분 불가였다
   (정황: 3점 비율이 sol TXT-2 70% vs 1024를 쓴 TXT-5는 0%).
   → 지금은 실패=None으로 **평균에서 제외**하고 `judge_detail.n_failed`로 노출.
- 전 태스크가 `run_judge()` + `summarize_judge_scores()`를 공유한다(복붙으로 갈라지던 원인 제거).
- **judge_mean이 없으면 0.0으로 채우지 말 것**: 정량표에서 "judge가 최악 평가"로 오독된다
  (옛 IMG-1 `judge_mean=0.0`이 실제로는 "전부 파싱 실패"였다).

### 데이터셋 함정 — 컬럼이 있는지 실제로 확인할 것
- **DocVQA 미러 대부분에 OCR 텍스트가 없다.** `HuggingFaceM4/DocumentVQA`·`lmms-lab/DocVQA`·
  `lmms-lab-encoder/DocVQA`는 페이지 **이미지만** 준다(`words` 컬럼 없음). TXT-1은 텍스트 태스크라
  이미지를 못 써서, 2026-08-05까지 **문서 없이 질문만 보내는 무문맥 QA**를 측정하고 있었다.
  → OCR(`words`)을 가진 `nielsr/docvqa_1200_examples`(test 200/train 1000)로 교체. `query`는
  언어별 **dict**(de/en/es/fr/it)라 `query["en"]`을 써야 한다(문자열로 착각하면 질문이 깨진다).
- **미러는 split도 뒤집어 놓는다.** `lighteval/wikitablequestions`(TXT-2)는 **train이 10행뿐**이고
  실데이터 18486행이 **test**에 있다. registry에 `split: "default"`(=config 이름이지 split이 아님)로
  적혀 있고 `txt_2.py`가 그걸 `train`으로 치환해, 30샘플을 요청해도 **항상 n=10**으로 채점됐다
  (2026-08-05 수정: split=test + 부족하면 예외). 확인: `load_dataset_builder(hf).info.splits`로
  **split별 행 수**를 본다 — 이름만 보고 train이 크다고 가정하지 말 것.
- 교훈: 새 데이터셋을 붙일 때 `load_dataset_builder(hf_id).info.features`로 **컬럼 존재를 먼저 확인**하고,
  `.info.splits`로 **행 수**까지 확인한다. `row.get("없는컬럼")`이 None을 돌려주고 코드가 폴백하면,
  또는 split이 작아 요청보다 적게 로드되면 실패가 조용히 숨는다.

## 리포트 구조 (섹션 순서 확정)
평가 대상 모델 → **Executive Summary**(모델 바로 뒤) → 비교 그래프 → 정량 결과 → 성능(시간·비용)
→ 정성 비교(모델 간 갈린 샘플 + 입력 이미지/텍스트) → **참고: Reasoning 정책(맨 뒤)** → Fact Sheet.
- 리포트마다 고객용 `presentation.html` 자동 생성 + `htmlpreview.github.io` 링크(GitHub에서 클릭 시 렌더).
- run별 `reports/<run-id>/`에 축적, `reports/index.md`가 전체 run 인덱스.

## 회귀 주의 (과거 버그 — 반복 금지)
1. **정성 갤러리 이미지는 실행 시 저장**(`results/<run>/images/`)에서 읽는다. streaming 재로드는
   shuffle 비재현성으로 sample_id가 다른 데이터를 가리켜 이미지↔정답이 어긋난다(옛 버그).
2. **리포트에 넣는 정답·입력 텍스트에 `<`(HTML)나 개행이 있으면 코드펜스로** 감싼다(인용문·인라인코드 금지).
   안 그러면 GitHub GFM이 raw `<table>` 등을 렌더해 표가 깨지고 다음 섹션을 빨아들인다(TXT-3 옛 버그).
   검증: `gh api /markdown`으로 렌더 후 `<table>` 열림/닫힘 균형·각 태스크 `<h3>` 확인.
3. **YAML의 reasoning 모드 키는 `minimal`/`full`** (on/off는 예약어라 boolean 파싱됨).
4. **데이터셋에 합성/더미 fallback 금지** — 실제 로드 실패 시 조용히 가짜 데이터 만들지 말고 명확히 실패(표준 데이터셋 원칙).
   **입력 일부가 비어도 마찬가지다**: TXT-1이 OCR 컨텍스트 부재 시 "질문만" 프롬프트로 폴백해
   측정 대상이 통째로 바뀐 버그가 있었다(위 "데이터셋 함정"). 지금은 컨텍스트가 없으면 예외를 던진다.
   새 태스크도 **필수 입력이 없으면 실패**하게 쓴다 — 0점은 모델 탓처럼 보여 원인 파악이 늦어진다.
5. **IMG-4 NSFW**: `DarkyMan/nsfw-image-classification`은 단일 클래스(nsfw만)라 SFW는 COCO에서 가져와 이진 구성.
   NSFW 이미지는 repo 커밋 금지(cache-only), 갤러리에서 입력 숨김·판정값만(D3).

## 커밋 규칙
- `results/`·`reports/`는 gitignore. 리포트를 커밋하려면 `git add -f`로 명시(report.md·facts.json·
  presentation.html·chart_*.png·gallery_*.png·manifest.json·scores.json). **samples.jsonl·images/는 커밋 제외**(용량·raw, NSFW).
- 태스크 코드 수정 후 `python -m pytest tests/` + `python tests/check_all_tasks.py` 통과 확인.

## 리포트 생성 후 커밋·푸시 절차 (재조사 불필요 — 이대로 실행)
리포트를 새로 뽑은 뒤 커밋/푸시 요청이 오면 아래를 그대로 따른다. repo 루트(`sme-llm-benchmark/`)에서 실행.
1. **run-id 확인**: 방금 생성된 `reports/<run-id>/`·`results/<run-id>/` (예: `2026-07-31T10-58`).
2. **강제 스테이징** (`reports/`·`results/`가 ignore라 `-f` 필수). gallery 파일명은 run마다 다름
   (`gallery_<TASK>_s<sample_id>.png` — sample_id가 매번 바뀜) → **하드코딩 말고 실제 존재하는 것만** 추가:
   ```bash
   R=image-text-performance; RID=<run-id>
   git add -f $R/reports/$RID/report.md $R/reports/$RID/facts.json \
     $R/reports/$RID/presentation.html $R/reports/$RID/chart_*.png \
     $R/reports/$RID/gallery_*.png \
     $R/results/$RID/manifest.json $R/results/$RID/scores.json
   git add $R/reports/index.md      # index.md는 tracked(비-ignore) → 일반 add. runner가 자동 갱신함
   ```
   - **커밋 제외 확인**: `samples.jsonl`·`images/`(용량·raw·NSFW), `logs/`, repo 루트 `.isaac/`. `git diff --cached --name-only`로 검수.
3. **브랜치**: 이 repo는 리포트를 **`main`에 직접 커밋**하는 관례(과거 모든 run·index.md가 main에 있음). 리포트 산출물 커밋은 별도 브랜치 만들지 말고 main에 그대로.
4. **커밋 메시지**: 한국어 요약체(과거 로그 스타일) — run-id·모델·샘플수·핵심결과·검증내용 1~2줄.
5. **푸시**: `git push origin main` (upstream=origin/main, fast-forward).
6. **훅 잡음(정상)**: Databricks pre-commit/commit-msg/pre-push secret-scan이 돌고, 푸시 때
   `Missing github token, skipping the repo check`가 뜨는데 **실패 아님**(exit 0 확인). `Unknown project name: None, skipping linting`도 정상.
