# databricks-fmapi-task-benchmark

여러 파운데이션 모델(FMAPI) / 에이전트 하네스를 **태스크별로** 실행하고 **동일 채점 + 사람 리뷰**로 비교하는 벤치마크. 같은 area의 `agent-ml/` 벤치마크와 **동일한 뼈대**를 쓴다: "산출물을 만드는 단계"만 후보마다 교체하고, 프롬프트·지시문·예산·채점은 100% 동일하게 둔다.

## 구조 (태스크-중심)

공용 로직은 `src/benchmark/` 패키지로 묶여 모든 태스크가 재사용한다. 태스크 디렉토리는 **실행 위치(repo 루트)** 기준으로 찾는다 (`BENCHMARK_ROOT` 환경변수로 override 가능).

```
<repo>/
├── pyproject.toml        # 패키지 + console_scripts (run-task / grade-task)
├── src/benchmark/        # 공용 소스 (태스크 무관, 재사용)
│   ├── task_spec.py          # 단일 진실 공급원 — COMMON_PROMPT + 태스크 로더
│   ├── run_task.py           # 후보 1개 실행 → <task>/<candidate>/slides.html
│   └── grade_tasks.py        # 후보들 동일 채점 + 렌더링 + 사람 리뷰 갤러리
└── <task_name>/          # 태스크 하나 = 디렉토리 하나
    ├── README.md             # 태스크 개요 + 실행법
    ├── TASK_DESCRIPTION.md    # 표준 브리프 + 포맷 계약 (각 후보에 instructions.txt로 복사)
    ├── keywords.json         # 채점 설정 (슬라이드 수 + 필수 토픽)
    ├── opus/                 # 후보 산출물 (slides.html, run_meta.json, screenshots/ …)
    ├── sol/                  # 후보 산출물
    └── glm/                  # 후보 산출물
```

`opus`/`sol`/`glm`은 **비교 대상 후보 모델**이다. 후보 이름은 임의 — `--candidate <이름>`으로 얼마든지 추가한다. 러너는 각 후보 디렉토리에 **글자 그대로 동일한** `instructions.txt`(= `TASK_DESCRIPTION.md`)와 `prompt.txt`를 넣은 뒤 `slides.html`을 만든다.

현재 태스크: **`explain-databricks`** — "Databricks를 설명하는 HTML 슬라이드 만들기".

## 0) 준비 (한 번)

```bash
uv venv
uv pip install -e .                  # 패키지 + 의존성 설치 → run-task / grade-task 명령 생성
uv run playwright install chromium   # 헤드리스 렌더링용 (~150MB)

# FMAPI 인증 (direct-fmapi / omnigent 에 필요)
export DATABRICKS_HOST=https://<workspace>.cloud.databricks.com
export DATABRICKS_TOKEN=dapi...
```

> 명령은 **repo 루트에서** 실행한다 (태스크 디렉토리를 작업 디렉토리 기준으로 찾음).
> 다른 위치에서 돌리려면 `BENCHMARK_ROOT=<repo>` 를 지정한다.

## 1) 후보별 실행

```bash
# direct-fmapi — 에이전트 없이 FMAPI 단발 chat completion (가장 단순한 baseline)
uv run run-task --task explain-databricks --candidate opus \
    --harness direct-fmapi --model databricks-claude-opus-4-1
uv run run-task --task explain-databricks --candidate glm \
    --harness direct-fmapi --model databricks-glm-...

# 에이전트 하네스 (candidate 는 산출물 폴더 라벨)
uv run run-task --task explain-databricks --candidate opus --harness claude-code
uv run run-task --task explain-databricks --candidate sol  --harness codex

# omnigent — Databricks 메타-하네스 (CLI 구문 확정 전까지 --manual 로)
uv run run-task --task explain-databricks --candidate sol \
    --harness omnigent --model databricks-... --manual

# (동등: uv run python -m benchmark.run_task --task ...)
```

## 2) 채점 + 사람 리뷰

```bash
uv run grade-task --task explain-databricks         # 모든 후보 채점
uv run grade-task --task explain-databricks --candidates opus glm
uv run grade-task --task explain-databricks --no-render   # 검증만
```

**Phase A (자동):** HTML 파싱, 슬라이드 수(`.slide` 또는 `<section>` 둘 다 인정), 필수 토픽 키워드 커버리지, 외부 http 참조 경고, 헤드리스 렌더링(콘솔 에러 수집 + 슬라이드별 스크린샷).
`auto_score = 0.4·키워드 + 0.3·슬라이드수OK + 0.3·콘솔에러없음`. 결과는 `<task>/grade_results.json` + `<task>/gallery/index.html`.

**Phase B (사람):** `<task>/gallery/index.html`을 브라우저로 열어 덱을 나란히 보고 1-5점 + 메모 → `human_scores.json` 다운로드 → 병합:
```bash
uv run grade-task --task explain-databricks \
    --merge-human explain-databricks/gallery/human_scores.json
```

## 새 태스크 추가

공용 소스(`src/benchmark/`)는 건드리지 않는다 — 태스크 디렉토리만 추가하면 된다.

1. `<new_task>/` 디렉토리 생성.
2. `TASK_DESCRIPTION.md`(브리프 + 포맷 계약)와 `keywords.json`(채점 설정) 작성.
3. `uv run run-task --task <new_task> --candidate <name> …`로 실행.

## 실전 유의점

- **모델명 footgun** — FMAPI 모델명은 반드시 `databricks-`로 시작. `claude-*`/`gpt-*`는 벤더 백엔드로 라우팅됨. `direct-fmapi`가 이를 강제한다.
- **모델 축** — `direct-fmapi`/`omnigent`만 모델을 완전히 변화시킨다. `claude-code`/`codex`는 자기 설정 모델을 쓰므로 `run_meta.json`의 `effective_model`에 실제 모델을 기록한다.
- **HTML 추출** — chat completion이 HTML을 산문/```html 펜스로 감쌀 수 있어 `extract_html()`가 `<!DOCTYPE>…</html>` 구간을 뽑아낸다 (direct-fmapi 한정).
- **렌더링 의존성** — Playwright chromium 필요. 미설치 시 채점기는 렌더링만 건너뛰고 검증은 계속한다.
- **omnigent** — 정확한 비대화형 CLI 구문 미확정. `run_task.py:build_omnigent_argv()` 하나에 격리; 그 전까지는 `--manual`로 실행.

## 공정성 체크리스트

- 같은 태스크 · 같은 프롬프트/지시문(byte-identical) · 같은 **예산**(wall-clock) · 같은 채점기.
- **`direct-fmapi`는 단발 완성 1회**라 멀티턴 에이전트(자가수정 가능)와 apples-to-apples가 아님 → **"raw baseline"으로 명시**하고 표에서 그렇게 읽을 것.
- 사람 점수는 갤러리에서 기계적으로 기록(`human_scores.json`) → 재현 가능.
