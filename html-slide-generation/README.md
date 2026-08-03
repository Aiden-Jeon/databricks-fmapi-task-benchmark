# html-slide-generation

여러 파운데이션 모델(FMAPI) / 에이전트 하네스를 **동일 프롬프트로** 실행하고 **동일 채점 + 사람 리뷰**로 비교하는 벤치마크 태스크: "Databricks를 설명하는 HTML 슬라이드 만들기". 같은 area의 `agent-ml/` 벤치마크와 **동일한 뼈대**를 쓴다: "산출물을 만드는 단계"만 후보마다 교체하고, 프롬프트·지시문·예산·채점은 100% 동일하게 둔다.

이 디렉토리는 **독립 실행 가능한 태스크 하나**다 (`image-text-performance/`처럼 자체 `pyproject.toml`/`src/`를 가진다). 공용 러너·채점기는 `src/benchmark/` 패키지에 있다.

## 구조

```
html-slide-generation/          # 태스크 = 디렉토리 하나 (= 벤치마크 루트)
├── pyproject.toml              # 패키지 + console_scripts (run-task / grade-task)
├── src/benchmark/              # 공용 소스 (재사용 가능)
│   ├── task_spec.py                # 단일 진실 공급원 — COMMON_PROMPT + 태스크 로더
│   ├── run_task.py                 # 후보 1개 실행 → <candidate>/slides.html
│   └── grade_tasks.py              # 후보들 동일 채점 + 렌더링 + 사람 리뷰 갤러리
├── TASK_DESCRIPTION.md         # 표준 브리프 + 포맷 계약 (각 후보에 instructions.txt로 복사)
├── keywords.json               # 채점 설정 (슬라이드 수 + 필수 토픽)
├── opus/                       # 후보 산출물 (slides.html, run_meta.json, screenshots/ …)
├── sol/                        # 후보 산출물
├── glm/                        # 후보 산출물
├── pi-opus/                    # 후보 산출물
└── gallery/                    # 채점 후 생성되는 사람 리뷰 갤러리
```

`opus`/`sol`/`glm`/`pi-opus`는 **비교 대상 후보 모델**이다. 후보 이름은 임의 — `--candidate <이름>`으로 얼마든지 추가한다. 러너는 각 후보 디렉토리에 **글자 그대로 동일한** `instructions.txt`(= `TASK_DESCRIPTION.md`)와 `prompt.txt`를 넣은 뒤 `slides.html`을 만든다.

태스크 id 기본값은 디렉토리 이름 **`html-slide-generation`**이며, 태스크 디렉토리는 **실행 위치(repo 루트)** 기준으로 찾는다 (`BENCHMARK_ROOT` 환경변수로 override 가능).

## 0) 준비 (한 번)

```bash
# repo 루트에서 (태스크 디렉토리를 작업 디렉토리 기준으로 찾음)
uv venv
uv pip install -e html-slide-generation   # 패키지 + 의존성 설치 → run-task / grade-task 명령 생성
uv run playwright install chromium         # 헤드리스 렌더링용 (~150MB)
```

**FMAPI 인증 (direct-fmapi / omnigent).** 별도 export 불필요 — `ucode`(Databricks AI Gateway CLI)가 설치돼 있으면 host/token을 자동으로 가져온다 (`~/.codex/ucode.config.toml`에서 host, `ucode auth-token`으로 매 실행마다 신선한 토큰 → 15분 만료 자동 해결). 명시적으로 덮어쓰려면 `DATABRICKS_HOST`/`DATABRICKS_TOKEN` 환경변수를 설정한다 (env가 항상 우선).

후보 이름은 기본 모델에 매핑된다 (`--model`로 덮어쓰기 가능):
`opus` → `databricks-claude-opus-4-8`, `sol` → `databricks-gpt-5-6-sol`, `glm` → `databricks-glm-5-2`.

> 명령은 **repo 루트에서** 실행한다 (태스크 디렉토리를 작업 디렉토리 기준으로 찾음).
> `--task` 기본값은 `html-slide-generation`이라 생략 가능. 다른 위치에서 돌리려면 `BENCHMARK_ROOT=<repo>` 를 지정한다.

## 1) 후보별 실행

```bash
# direct-fmapi — 에이전트 없이 FMAPI 단발 chat completion (가장 단순한 baseline)
# 모델은 후보 이름에서 자동 결정 (opus/sol/glm). --model 로 덮어쓰기 가능.
uv run run-task --candidate opus --harness direct-fmapi
uv run run-task --candidate glm  --harness direct-fmapi
uv run run-task --candidate sol  --harness direct-fmapi

# 에이전트 하네스 — 툴로 여러 턴 돌며 slides.html 을 직접 작성 (direct-fmapi 단발 호출과 대비)
uv run run-task --candidate opus --harness claude-code
uv run run-task --candidate sol  --harness codex

# pi — Pi 코딩 에이전트를 ucode(Databricks AI Gateway) 설정으로 헤드리스 실행.
# provider/model 은 후보에서 자동 결정 (opus/sol). ucode configure --agent pi 선행 필요.
uv run run-task --candidate pi-opus --harness pi \
    --pi-provider databricks-claude --model system.ai.claude-opus-5

# omnigent — Databricks 메타-하네스 (CLI 구문 확정 전까지 --manual 로)
uv run run-task --candidate sol \
    --harness omnigent --model databricks-... --manual

# (동등: uv run python -m benchmark.run_task …)
```

> `pi` 하네스는 ucode가 Pi provider를 설정해둬야 한다: `ucode configure --agent pi --skip-validate`.
> 러너는 `PI_CODING_AGENT_DIR`을 ucode의 pi-home(`~/.ucode/pi-home/.pi/agent`)으로 주입해 gateway provider(`databricks-claude` / `databricks-openai` / `databricks-gemini`)에 붙는다.

## 2) 채점 + 사람 리뷰

```bash
uv run grade-task                       # 모든 후보 채점
uv run grade-task --candidates opus glm
uv run grade-task --no-render           # 검증만
```

**Phase A (자동):** HTML 파싱, 슬라이드 수(`.slide` 또는 `<section>` 둘 다 인정), 필수 토픽 키워드 커버리지, 외부 http 참조 경고, 헤드리스 렌더링(콘솔 에러 수집 + 슬라이드별 스크린샷).
`auto_score = 0.4·키워드 + 0.3·슬라이드수OK + 0.3·콘솔에러없음`. 결과는 `html-slide-generation/grade_results.json` + `html-slide-generation/gallery/index.html`.

**Phase B (사람):** `gallery/index.html`을 브라우저로 열어 덱을 나란히 보고 1-5점 + 메모 → `human_scores.json` 다운로드 → 병합:
```bash
uv run grade-task \
    --merge-human html-slide-generation/gallery/human_scores.json
```

## 다른 태스크로 재사용

이 벤치마크 뼈대(`src/benchmark/`)는 태스크 무관하게 재사용 가능하다. 새 태스크는 **별도 디렉토리**로 독립시킨다 (`image-text-performance/`처럼):

1. 새 태스크 디렉토리를 만들고 `TASK_DESCRIPTION.md`(브리프 + 포맷 계약)와 `keywords.json`(채점 설정)을 둔다.
2. 공용 소스가 필요하면 `src/benchmark/`를 복사하거나 패키지로 공유한다.
3. `uv run run-task --task <task-dir-name> --candidate <name> …`로 실행 (기본값은 `html-slide-generation`).

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
