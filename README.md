# databricks-fmapi-task-benchmark

"**Databricks를 설명하는 HTML 슬라이드 만들기**" 태스크를 **여러 파운데이션 모델(FMAPI) × 여러 에이전트 하네스**로 기계적으로 실행하고, **동일 채점 + 사람 리뷰**로 비교하는 벤치마크.

같은 area의 `agent-ml/` 벤치마크(Kaggle 태스크)와 **동일한 뼈대**를 쓴다: "산출물을 만드는 단계"만 셀마다 교체하고, 프롬프트·지시문·예산·채점은 100% 동일하게 둔다.

## 무엇이 들어있나

| 파일 | 설명 |
|---|---|
| `task_spec.py` | **단일 진실 공급원** — `COMMON_PROMPT`, `build_instructions()`. 러너와 채점기가 함께 import |
| `tasks/explain-databricks/brief.md` | 사람이 읽는 태스크 브리프(청중·목표·스타일) |
| `tasks/explain-databricks/keywords.json` | 필수 토픽 키워드 + 슬라이드 수 [min,max] (프롬프트와 채점이 공유) |
| `run_task.py` | (하네스 × 모델) 한 셀을 실행하는 러너 |
| `grade_tasks.py` | 산출물 동일 채점 + 렌더링 + 사람 리뷰 갤러리 |
| `runs/run_<harness>_<model>/` | 셀별 산출물(`slides.html`·로그·`run_meta.json`·`screenshots/`) |

## 0) 준비 (한 번)

```bash
uv venv
uv pip install openai playwright lxml cssselect
uv run playwright install chromium   # 헤드리스 렌더링용 (~150MB)

# FMAPI 인증 (direct-fmapi / omnigent 에 필요)
export DATABRICKS_HOST=https://<workspace>.cloud.databricks.com
export DATABRICKS_TOKEN=dapi...
```

## 1) 하네스별 실행

**공통 프롬프트 (모든 하네스에 byte-identical, `runs/*/prompt.txt`에도 저장):**
```
Read ./instructions.txt and write a slide deck to ./slides.html that explains what
Databricks is. Produce ONE self-contained HTML file ... (전문은 task_spec.COMMON_PROMPT)
```

```bash
# direct-fmapi — 에이전트 없이 FMAPI 단발 chat completion (가장 단순한 baseline)
uv run python run_task.py --harness direct-fmapi --model databricks-claude-sonnet-4-6

# claude-code — `claude -p` 헤드리스
uv run python run_task.py --harness claude-code

# codex — `codex exec` 비대화형
uv run python run_task.py --harness codex --max-seconds 1200

# omnigent — Databricks 메타-하네스 (CLI 구문 확정 전까지 --manual 로)
uv run python run_task.py --harness omnigent --model databricks-claude-sonnet-4-6 --manual

# UI-only 에이전트 (Databricks Playground 등) — 사람은 키보드 프록시만
uv run python run_task.py --harness playground --model databricks-claude-sonnet-4-6 --manual
```

`run_task.py`는 각 셀의 작업 폴더에 **글자 그대로 동일한** `instructions.txt` + `prompt.txt`를 넣고, 지정한 하네스로 `./slides.html`을 만든 뒤 wall time·exit·모델을 `run_meta.json`에 기록한다.

## 2) 채점 + 사람 리뷰

```bash
uv run python grade_tasks.py          # 모든 run 채점 → 표 + grade_results.json + 갤러리
uv run python grade_tasks.py --no-render   # Playwright 없이 검증만
```

**Phase A (자동):** HTML 파싱/well-formedness, 슬라이드 수(`.slide` 또는 `<section>` 둘 다 인정), 필수 토픽 키워드 커버리지, 외부 http 참조 경고, 헤드리스 렌더링(콘솔 에러 수집 + 슬라이드별 스크린샷).
`auto_score = 0.4·키워드 + 0.3·슬라이드수OK + 0.3·콘솔에러없음`.

**Phase B (사람):** `runs/gallery/index.html`을 브라우저로 열어 덱을 나란히 보고 1-5점 + 메모 → `human_scores.json` 다운로드 → 병합:
```bash
uv run python grade_tasks.py --merge-human runs/gallery/human_scores.json
```

## 실전 유의점

- **모델명 footgun** — FMAPI 모델명은 반드시 `databricks-`로 시작. `claude-*`/`gpt-*` 는 벤더 백엔드로 라우팅됨. `direct-fmapi`가 이를 강제한다.
- **모델 축** — `direct-fmapi`/`omnigent`만 모델을 완전히 변화시킨다. `claude-code`/`codex`는 자기 설정 모델을 쓰므로 `run_meta.json`의 `effective_model`에 실제 모델을 기록한다.
- **HTML 추출** — chat completion이 HTML을 산문/```html 펜스로 감쌀 수 있어 `extract_html()`가 `<!DOCTYPE>…</html>` 구간을 뽑아낸다 (direct-fmapi 한정).
- **렌더링 의존성** — Playwright chromium 필요(`playwright install chromium`). 미설치 시 채점기는 렌더링만 건너뛰고 검증은 계속한다.
- **omnigent** — 정확한 비대화형 CLI 구문 미확정. `run_task.py:build_omnigent_argv()` 하나에 격리; 그 전까지는 `--manual`로 실행(작업 폴더/프롬프트 준비는 동일 → 코드 위험 0).

## 공정성 체크리스트

- 같은 태스크 · 같은 프롬프트/지시문(byte-identical) · 같은 **예산**(wall-clock) · 같은 채점기.
- **`direct-fmapi`는 단발 완성 1회**라 멀티턴 에이전트(자가수정 가능)와 apples-to-apples가 아님 → **"raw baseline"으로 명시**하고 표에서 그렇게 읽을 것.
- 기록표 컬럼: `harness | model | valid | slides | keywords | console_err | auto_score | wall`.
- 사람 점수는 갤러리에서 기계적으로 기록(`human_scores.json`) → 재현 가능.
