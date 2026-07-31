# CLAUDE.md — image-text-performance 벤치마크 운영 지침

이 파일은 이 프로젝트를 다루는 AI 에이전트(Claude Code / Vibe)와 사람 기여자 모두를 위한
운영 지식이다. clone 후 재수행·확장할 때 여기부터 읽으면 이미 확인된 함정을 반복하지 않는다.
프로젝트 개요·사용법은 [README.md](./README.md), 설계 결정은 [plan.md](./plan.md) 참고.

## 한 줄 요약
Databricks FMAPI 모델(opus·sol·glm)의 이미지·텍스트 성능을 표준 데이터셋으로 측정해
수치·그래프·정성 갤러리·고객용 HTML 프레젠테이션 리포트를 자동 생성하는 벤치마크. 13개 태스크.

## 전제
- Databricks CLI 프로파일 필요(기본 `ai_devtools`). FMAPI 호출 + `system.ai_gateway.usage` 조회용.
- 데이터셋은 HuggingFace **streaming**으로 필요한 샘플만 받음(전체 다운로드 X). `.cache/`(gitignore).
- 의존성은 `pip install -e .`. 선택: mecab(한국어 형태소, 없으면 음절 fallback), bert-score+torch(없으면 deferred).

## 벤치마크 실행 / 재수행
```bash
python -m src.runner --samples 30 --fresh        # 30샘플, 기존 리포트 삭제 후 새로
python -m src.runner --dry-run                    # 매트릭스만 미리보기(호출 없음)
python -m src.runner --models sol --samples 3     # 빠른 확인
```
- **`--fresh`**: 실행 전 `results/`·`reports/` 전부 삭제하고 index부터 새로 생성.
- **`--samples N`**: 태스크당 N개(생략 시 config 기본 50). seed 고정이라 재현 가능.
- reasoning은 config에서 **OFF(minimal) 고정** → 옵션 불필요. ON까지 비교하려면 `--reasoning-modes minimal,full`(실행 배증·glm 타임아웃 주의).
- **오래 걸림**: 34셀 × N샘플 + judge. 10샘플 ≈ 20~30분, 30샘플 ≈ 1시간+, 50샘플 수 시간+.
  → 반드시 백그라운드 실행(`nohup python3 -u -m src.runner ... &`) 후 프로세스 종료까지 대기.
  "실행 완료" 로그 뒤에도 리포트 생성(judge 요약·이미지 처리)이 수 분 더 걸림.

## 실행 규모를 키울 때 (대규모 팁)
- **HF unauthenticated rate limit**으로 데이터 스트리밍이 느림 → `HF_TOKEN` 설정하면 빨라짐.
- **glm-5-2가 3~5배 느림**. reasoning ON이면 15초 타임아웃을 자주 초과(실측: 160호출 중 19 오류).
  대규모·ON 실행 전 `config/models.yaml`의 `runtime.timeout_seconds` 상향(예: 30~60s) 고려.

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
