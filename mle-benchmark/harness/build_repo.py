#!/usr/bin/env python
"""Assemble the public GitHub repo tree from benchmark artifacts.

Layout (per Aiden-Jeon/databricks-fmapi-task-benchmark spec):
  <repo>/<task_slug>/README.md · TASK_DESCRIPTION.md · PROMPT.md · opus/ sol/ glm/
Each model dir: submission.csv, solution/*, metrics.json (quality·time·cost).

NEVER copies private answer keys or secrets. Public-safe.
Cost: LLM $ estimated from the agent log's token usage × list price (per-run
billing isn't separable in system tables); compute is small and reported as an
aggregate note. Source of quality/time = results/scores.csv + result.json.
"""
import collections
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_repeats import cell_stats, decide  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "config.json").read_text())
P = CFG["profile"]
VOL = f"dbfs:/Volumes/{CFG['catalog']}/{CFG['schemas']['results']}/{CFG['volumes']['results']}"
# The benchmark lives in the `mle-benchmark/` subfolder of the shared team repo —
# never the repo root (which holds other projects + the top-level README).
# When this script itself lives inside the repo (vendored harness), ROOT *is*
# the mle-benchmark folder — write in place. Standalone checkouts keep the old
# sibling-repo default. KMLE_REPO_OUT always overrides.
_default_out = ROOT if ROOT.name == "mle-benchmark" else \
    ROOT.parent / "databricks-fmapi-task-benchmark" / "mle-benchmark"
OUT = Path(os.environ.get("KMLE_REPO_OUT", _default_out))

MODEL = {"M1": ("opus", "Claude Opus 5", "databricks-claude-opus-5", (5.0, 25.0)),
         "M2": ("sol", "GPT-5.6-sol", "databricks-gpt-5-6-sol", (5.0, 30.0)),
         "M3": ("glm", "GLM 5.2", "databricks-glm-5-2", (1.40, 4.40)),
         # M7 list price: fill from the FMAPI pricing page / provider SKU rows
         # in system.billing.usage before regenerating cost tables.
         "M7": ("kimi", "Kimi K3", "databricks-kimi-k3", (0.0, 0.0))}
MKEYS = [v[0] for v in MODEL.values()]           # ["opus","sol","glm","kimi"]
MNAME = {v[0]: v[1] for v in MODEL.values()}
# Real billed LLM $ by model, all benchmark runs (system.billing.usage,
# workspace 7474655787271401). Per-run/per-task is NOT separable in v1:
# runs were concurrent and the OpenAI-compat route logs no tokens. v2 fix =
# per-run request_tags. omni per-session $ = team cross-check.
# "kimi" stays 0.0 until its campaign lands in billing (queries/cost_attribution.sql).
MODEL_LLM_TOTAL = {"opus": 23.22, "sol": 5.17, "glm": 3.10, "kimi": 0.0}
LIST_PRICE = {"opus": "$5 / $25", "sol": "$5 / $30", "glm": "$1.40 / $4.40",
              "kimi": "TBD"}
TASKS = {  # task_id -> (slug, metric label, direction)
    "t1_pubg": ("pubg_placement_prediction", "MAE", "lower"),
    "t2_spooky": ("spooky_author_identification", "multiclass log loss", "lower"),
    "t3_ynat": ("klue_ynat_news_topic", "macro-F1", "higher"),
    "t4_nsmc": ("nsmc_movie_sentiment", "accuracy", "higher"),
    "t5_bike": ("seoul_bike_demand", "RMSE", "lower"),
    "t6_klue_nli": ("klue_nli_inference", "accuracy", "higher"),
    "t7_klue_sts": ("klue_sts_similarity", "Pearson", "higher"),
    "t8_beep": ("beep_hate_speech", "macro-F1", "higher"),
    "t9_korquad": ("korquad_reading_comprehension", "char-F1", "higher"),
    "t10_kornli": ("kornli_inference", "accuracy", "higher"),
    "t11_korsts": ("korsts_similarity", "Pearson", "higher"),
    "t12_kobest_boolq": ("kobest_boolq", "accuracy", "higher"),
    "t13_kobest_copa": ("kobest_copa", "accuracy", "higher"),
    "t14_kobest_wic": ("kobest_wic", "accuracy", "higher"),
    "t15_kobest_hellaswag": ("kobest_hellaswag", "accuracy", "higher"),
    "t16_kobest_sentineg": ("kobest_sentineg", "accuracy", "higher"),
    "t17_pawsx_ko": ("pawsx_paraphrase", "accuracy", "higher"),
    "t18_klue_re": ("klue_relation_extraction", "accuracy", "higher"),
    "t19_klue_mrc": ("klue_mrc_reading", "char-F1", "higher"),
    "t20_klue_ner": ("klue_ner_entities", "entity-F1", "higher"),
    "t21_kmmlu": ("kmmlu_expert_knowledge", "accuracy", "higher"),
    # t22_haerae removed after smoke revealed a parametric-recall exploit (see FINDINGS)
    "t23_korfin_asc": ("korfin_aspect_sentiment", "macro-F1", "higher"),
    "t24_kor_unsmile": ("kor_unsmile_multilabel_hate", "macro-F1 (multi-label)", "higher"),
    "t25_klue_dp": ("klue_dependency_parsing", "LAS", "higher"),
}


def dbx(*args):
    return subprocess.run(["databricks", *args, "-p", P],
                          capture_output=True, text=True)


def latest_runs():
    by = {}
    for r in csv.DictReader(open(ROOT / "results" / "scores.csv")):
        lane = r["run_dir"].split("_")[0]
        if lane in MODEL and r["task"] in TASKS:
            k = (lane, r["task"])
            cur = by.get(k)
            if (cur is None or (r["valid"] == "True" and cur["valid"] != "True")
                    or (r["valid"] == cur["valid"] and r["graded_utc"] > cur["graded_utc"])):
                by[k] = r
    return by


def repeat_cells():
    """{task: {model: (mean, sd, se, n)}} over ALL valid runs, not just the last.

    Winners must come from the distribution: picking the most recent run makes
    the headline depend on which repeat finished last, and discards n-1 runs.
    """
    vals = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(ROOT / "results" / "scores.csv")):
        lane = r["run_dir"].split("_")[0]
        if lane in MODEL and r["task"] in TASKS and r["valid"] == "True" and r["score"]:
            vals[r["task"]][MODEL[lane][0]].append(float(r["score"]))
    return {t: {m: cell_stats(v) for m, v in ms.items()} for t, ms in vals.items()}


def top_docs(summary):
    reps = repeat_cells()

    def cell(task, m):
        st_ = reps.get(task, {}).get(m)
        if st_ is None:
            s = summary[task].get(m)
            return "DNF" if s and s["quality"]["score"] is None else "—"
        mean, sd, _se, n = st_
        # ± only means something at n>=2; flag n=1 cells so the reader knows
        return f"{mean:.4g}±{sd:.2g} (n={n})" if n >= 2 else f"{mean:.4g} (n=1)"

    def best(task):
        """Statistical winner, or None when the race is a tie/undecided."""
        cells = reps.get(task, {})
        if not cells:
            return None
        kind, leader, _rival = decide(cells, TASKS[task][2] == "higher")
        return leader if kind == "win" else None
    # Board columns = models that actually have graded runs, canonical order.
    models = [m for m in MKEYS
              if any(m in reps.get(t, {}) or m in summary.get(t, {}) for t in TASKS)]
    rows = []
    for t, (slug, metric, direction) in TASKS.items():
        b = best(t)
        cells = []
        for m in models:
            v = cell(t, m)
            cells.append(f"**{v}**" if m == b and v not in ("—", "DNF") else v)
        arrow = "↑" if direction == "higher" else "↓"
        rows.append(f"| [{slug}](./{slug}) | {metric} {arrow} | " +
                    " | ".join(cells) + " |")
    wins = {m: sum(1 for t in TASKS if best(t) == m) for m in MKEYS}
    # Two distinct reasons a race isn't called, kept separate: a genuine tie
    # (bands overlap) vs. merely undecided (leader still n=1, no variance
    # estimate yet). Collapsing them would overstate how much is truly a tie.
    kinds = collections.Counter(
        decide(reps[t], TASKS[t][2] == "higher")[0]
        for t in TASKS if len(reps.get(t, {})) >= 2)
    n_ties, n_undecided = kinds["tie"], kinds["undecided"]
    max_n = max((s[3] for ms in reps.values() for s in ms.values()), default=0)

    # NOTE: the prose blocks below (핵심 결론 bullets, 방법론적 발견, 실전 유의점,
    # FINDINGS narrative) were written for the 2026-08 three-model campaign
    # (opus/sol/glm). The tables are generated from data, but when a new model
    # (e.g. kimi) gains graded runs, re-read and update the narrative by hand.
    board_header = "| Task | metric | " + " | ".join(models) + " |\n" + \
                   "|---|---|" + "---|" * len(models)
    board_wins = "| **decided wins** | | " + \
        " | ".join(f"**{wins[m]}**" for m in models) + " |"

    (OUT / "README.md").write_text(f"""# mle-benchmark

여러 파운데이션 모델(FMAPI)을 **동일한 코딩 에이전트 하네스**에 꽂아, 한국어로 주어진
**{len(TASKS)}개 ML 엔지니어링 태스크**를 직접 코드로 풀게 하고 **숨겨진 테스트셋으로 채점**하는
벤치마크. 같은 repo의 다른 태스크와 같은 원칙이다: **모델만 교체**하고 하네스·프롬프트·예산·
채점기는 100% 동일하게 둔다.

측정 대상은 "모델이 답을 아는가"가 아니라 **"모델이 ML 파이프라인을 짜서 돌릴 수 있는가"**다.
방법론은 OpenAI [MLE-bench](https://github.com/openai/mle-bench)를 한국어·Databricks로 이식했다.

> 상세 결과·해석은 [FINDINGS.md](./FINDINGS.md), 실험 설계는 [METHODOLOGY.md](./METHODOLOGY.md),
> 비용은 [COST.md](./COST.md), **처음부터 다시 돌리는 법**은 [REPRODUCE.md](./REPRODUCE.md)
> (하네스 전체가 `harness/`에 있다), **시점별·모델 추가 시 성능 추적**은
> [TIMELINE.md](./TIMELINE.md). 이 README는 개요·결론·재현법 요약.

---

## 핵심 결론 (n=3)

| | Opus 5 | GPT-5.6-sol | GLM 5.2 | Kimi K3 |
|---|:---:|:---:|:---:|:---:|
| **확정 승리** (of {len(TASKS)}) | **{wins['opus']}** | {wins['sol']} | {wins['glm']} | {wins['kimi']} |
| 평균 1위 태스크 | 19 | 5 | 0 | 0 |
| 유효 제출률 | 67/72 (93%) | **70/72 (97%)** | 57/72 (79%) | 51/72 (71%) |
| 재현성 (중위 상대 std) | 2.90% | **1.44%** | 3.51% | 1.94% |
| 청구 LLM 비용 | $122.83 | $43.84 | **$16.11** | $6.16 (잠정) |

- **Opus 5** — 유일하게 태스크를 확정 승리하는 모델({wins['opus']}/{len(TASKS)}), 평균으로는 19/{len(TASKS)}에서 1위.
  대가는 GPT의 ~2.8배 비용, 가장 긴 실행시간, 그리고 **셋 중 가장 낮은 재현성**.
- **GPT-5.6-sol** — 확정 승리 0개지만 **가장 빠르고(중위 9분) 가장 안정적**. 21개 공통 태스크 중
  13개에서 가장 일관되고, 최악 셀의 편차가 9.7%(나머지 둘은 100% 초과). 벤치마크든 프로덕션이든
  **재현성 자체가 1급 속성**이라는 점에서 이쪽이 실무 기본값.
- **GLM 5.2** — 압도적으로 저렴하지만 유효율 79%, 반복 실패 셀 3개
  (KorQuAD·MRC·KMMLU 각 2회). 제출에 성공해도 독해 태스크 점수가 0에 가깝다.
- **Kimi K3** *(2026-08-13 추가)* — 분류·유사도 계열에서는 상위권과 겹치지만(YNAT 0.847,
  KLUE-STS 0.945), **유효율 71%로 4모델 중 최저**에 반복 실패 셀 5개 — KLUE-NER는 3/3 실패로
  한 번도 제출하지 못했다. 유효 런의 중위 소요시간 99분(캡 120분)으로 가장 느리며, DNF 유형은
  전부 "시간 안에 submission.csv를 못 남김". 제출한 셀의 재현성은 1.94%로 준수하다.
- **{n_ties}/{len(TASKS)}는 통계적 무승부** — 선두의 차이가 실행간 노이즈 안에 있어 승자를 주장하지 않는다.
  그래서 승수 합이 {len(TASKS)}가 되지 않는 것이 **정상**이다.{f" ({n_undecided}개는 선두 셀 반복 대기.)" if n_undecided else ""}

### 가장 중요한 방법론적 발견 — n=1은 승자를 조작한다

**같은 데이터**가 셀당 1회 실행에서는 `Opus 15 / GPT 6 / GLM 3`으로, n=3에서는
`Opus {wins['opus']} / GPT {wins['sol']} / GLM {wins['glm']} / 무승부 {n_ties}`로 읽혔다. 절반이 노이즈였고 **GPT·GLM의 9승은 전부 사라졌다.**
n=1에서 가장 근접했던 승부의 차이는 0.10~0.67%인데, 동일 재실행 간 중위 변동이 1.5~3.5%,
최악은 100%를 넘는다. 노이즈보다 10~100배 작은 차이로는 판정할 수 없다.
반복 비용은 144런에 **$72.70** — 탐색 단계 비용의 절반. **에이전트 벤치마크는 n≥3과 무승부 수 공개가
필수**이고, 승수가 태스크 수에 딱 맞게 떨어지는 표는 자기 노이즈를 측정하지 않은 표다.

---

## 결과 — {len(TASKS)}개 한국어 태스크

각 셀 = **3회 반복의 평균 ± 표본표준편차**(n 표기). **굵게 = 통계적으로 확정된 승자**이며,
선두 차이가 노이즈를 넘지 못하면 무승부로 둔다. 에이전트가 본 적 없는 **숨겨진 테스트 split**으로
채점(UC ACL로 격리).

{board_header}
{chr(10).join(rows)}
{board_wins}

---

## 무엇을 재나 — 3개 지표

| | 지표 | 출처 | 정밀도 |
|---|---|---|---|
| 1 | **결과물 퀄리티** | grader vs 숨겨진 split | 정확, 런 단위 |
| 2 | **소요시간** | job wall-clock | 정확, 런 단위 |
| 3 | **전체 비용** | `system.billing.usage` (청구 실적) | **모델 단위** ([COST.md](./COST.md)) |

비용은 추정이 아니라 **청구된 행**만 쓴다. 전체 캠페인 **$207.12** = LLM $182.79 + 서버리스 컴퓨트 $24.34.
모델별 분해는 provider SKU 단위까지만 가능하다(각 SKU에 M-track 모델이 정확히 하나씩 있어서 성립 —
이 모델 조합의 성질이며 범용 미터가 아니다). 런 단위 귀속은 request tag가 필요(v2).

---

## 구조

```
<task>/
├── README.md            # 태스크별 결과표 (퀄리티 · 시간 · 비용)
├── TASK_DESCRIPTION.md  # 표준 태스크 브리프 (한국어)
├── PROMPT.md            # 표준 킥오프 프롬프트 (모델 간 글자 그대로 동일)
├── opus/                # Opus 5:       submission.csv · solution/ · metrics.json
├── sol/                 # GPT-5.6-sol:  submission.csv · solution/ · metrics.json
├── glm/                 # GLM 5.2:      submission.csv · solution/ · metrics.json
└── kimi/                # Kimi K3:      submission.csv · solution/ · metrics.json
```

`opus`/`sol`/`glm`/`kimi`는 **비교 대상 모델**이다. 하네스는 pinned opencode 하나이고,
각 모델 디렉토리에 들어가는 `PROMPT.md`·`TASK_DESCRIPTION.md`는 **byte-identical**이다.

---

## 재현성 — 남이 그대로 돌릴 수 있는 범위

이 폴더에는 결과물뿐 아니라 **하네스 전체가 코드로** 들어 있다. 위 설명과 repo의 대응:

| 단계 | 위치 |
|---|---|
| 태스크 팩 생성 (spec·train·hidden split) | `harness/fetch_raw25.py` → `harness/prepare.py` |
| 워크스페이스 부트스트랩 (볼륨·시크릿·러너 업로드) | `setup.sh` + `config.json` (프로필·카탈로그만 수정) |
| 잡 제출·러너 (pinned opencode, 모델 셀렉터, 게이트웨이 배선) | `harness/submit_matrix.py` · `harness/runner.py` |
| 킥오프 프롬프트 (전 모델 공통 1파일) | `harness/kickoff_prompt_ko.md` |
| 채점·집계·통계 판정 | `harness/grade_run.py` · `scoreboard.py` · `aggregate_repeats.py` + `tests/` |
| 문서·스냅샷·타임라인 재생성 | `harness/build_repo.py` · `snapshot.py` · `timeline.py` |
| 캠페인 원장·동결 스냅샷 | `results/*.csv` · `snapshots/<id>/` |
| 전체 순서 (0→6단계 명령어) | **[REPRODUCE.md](./REPRODUCE.md)** |

의도적으로 repo에 **없는** 것 두 가지: ① 원본 데이터 — 재배포 불가라
`fetch_raw25.py`가 공개 출처에서 직접 받는다(Kaggle 태스크는 본인 인증 필요).
② 숨겨진 정답 키 — `prepare.py`가 fresh split으로 재생성하며 `.gitignore`가
커밋을 차단한다. 전제 조건은 서버리스 Jobs + UC + AI Gateway + 해당
pay-per-token 엔드포인트가 있는 Databricks 워크스페이스 하나.

정직한 한계: split을 재생성하므로 **소수점 동일 재현이 아니라 통계적 재현**
(같은 결론, 노이즈 밴드 안의 숫자)이고, Genie Code(L4)는 API가 없어 수동
프로토콜(`harness/genie_code_protocol_ko.md`)로만 재현된다.

---

## 실전 유의점

- **DNF는 대부분 능력이 아니라 툴 사용/시간 예산 실패다.** 원래 3모델 216런에서 22 DNF, 2회 이상
  실패한 셀은 3개뿐(전부 GLM)이었고 나머지는 재실행에서 성공했다. Kimi K3 추가(72런 +21 DNF)로
  반복 실패 셀이 5개 늘었는데(NER은 3/3), Kimi의 DNF는 전부 "2시간 캡 안에 제출물을 못 남김"
  유형이다 — 즉 DNF는 "그 태스크를 못 한다"가 아니라 "일정 확률로, 또는 시간 안에 제출에
  실패한다"로 읽어야 한다.
- **분산은 모델의 속성이고, 퀄리티와 반비례한다.** 최고 점수 모델이 가장 덜 재현적이다.
  출력 형식은 2차 예측변수일 뿐(구조적 파싱 6.6% vs 닫힌 라벨 1.8%) — 벤치마크 전체에서 가장
  변동이 큰 두 셀은 오히려 평범한 tabular 회귀(PUBG MAE)다.
- **암기 우회를 막아야 한다.** 지식형 4지선다에서 에이전트가 모델을 만드는 대신 **테스트 정답을
  직접 손으로 적어** 0.945를 받은 사례가 있다(HAE-RAE, 그래서 태스크 목록에서 제외 — id `t22`는
  의도적 공백). 킥오프에 anti-recall 규칙이 들어 있다: 제출은 `train.csv`로 학습한 일반화 모델에서
  나와야 하며 test 행의 정답 하드코딩·수작업 금지.
- **데이터는 재배포하지 않는다.** 이 repo에는 제출물과 에이전트 코드만 있고 원본 데이터는 없다
  (KLUE · NSMC · UCI · Kaggle 등 공개 출처에서 스크립트로 받는다). 정답 키는 커밋 금지.
- **하네스가 점수를 바꾼다.** 같은 모델도 스캐폴드가 다르면 점수가 유의미하게 달라진다
  (GPT 뉴스분류 F1: native Codex 0.824 → fixed opencode 0.848). 그래서 모델 비교는 하네스를
  고정해야만 성립한다.

Databricks workspace `fevm-newjeans-ontos`에서 실행. 서버리스 Jobs, 전 트래픽 Unity AI Gateway 경유.
""")


    (OUT / "FINDINGS.md").write_text(f"""# Findings

## The model verdict — three profiles, not one winner

| | Opus 5 | GPT-5.6-sol | GLM 5.2 | Kimi K3 |
|---|---|---|---|---|
| decided wins (of {len(TASKS)}) | **{wins['opus']}** | {wins['sol']} | {wins['glm']} | {wins['kimi']} |
| profile | quality leader | efficiency frontier | budget contender | slow long-tail |
| reliability (valid rate, n=3 campaign) | 93% | **97%** | 79% | 71% |
| LLM $ (campaign) | $122.83 | $43.84 | $16.11 | $6.16 (잠정) |

**{n_ties} of {len(TASKS)} tasks are statistical ties** — the leader's margin sat
inside run-to-run noise, so no winner is claimed{f"; a further {n_undecided} are undecided (leader not yet repeated)" if n_undecided else ""}.
Win counts therefore do not sum to {len(TASKS)}, and that gap is a result, not a
gap in the data: at n=1 those same cells looked decisive.

- **Opus 5** takes {wins['opus']}/{len(TASKS)} decided cells, concentrated in the
  harder Korean-NLP tasks (STS, QA, RE-adjacent, and calibration-heavy 작가판별).
  It costs ~4.5× GPT and ~7.5× GLM (measured on the v1 5-task pass) and runs
  longest — buy it when the score is worth the money.
- **GPT-5.6-sol** is the efficiency frontier: {wins['sol']} decided wins, fastest
  everywhere, and the only model with a perfect 24/24 valid-submission rate. Its
  edge is throughput and reliability, not peak quality.
- **GLM 5.2** is the budget play — {wins['glm']} decided wins at a fraction of the
  cost — but the least reliable (5 DNFs).
- Repeats matter most where **output is structured**: dependency parsing and NER
  swing by ±0.14–0.17 across identical reruns, while sentiment/classification is
  near-deterministic (±0.00–0.01). The variance is format conformance, not
  capability — the same failure class as the DNFs, seen as a score swing instead
  of a rejected submission.
- All three operated **fully in Korean** with no instruction-following failures.

## "Recite, don't engineer" — why trivia tasks break an MLE benchmark

A sixth candidate task, **HAE-RAE** (Korean culture/language trivia, 5-way MC),
was smoke-tested with one model before committing the full matrix — and the smoke
paid for itself. GPT-5.6-sol scored **0.945**, impossible from 1,230 training
rows. Its solution built a normal TF-IDF classifier, then **overrode it with a
hand-authored `expert_predictions.csv`**: it read the unlabeled test questions and
filled in answers *from its own parametric memory*. That is direct recall — the
behaviour an ML-*engineering* benchmark exists to exclude — smuggled through the
harness. On **KMMLU** (professional exams) the same agent could *not* recite, so
it was forced to engineer, scoring a hard 0.352. We **dropped HAE-RAE** (task id
`t22` is a deliberate, documented gap), kept KMMLU, and added an **anti-recall
rule** to the standard kickoff. Verified it held: under the rule, Opus 5 — which
knows more Korean trivia — still built a genuine CV/ensemble pipeline for KMMLU
(0.356, no recall layer). **Design principle: the strongest MLE tasks are those
whose answers cannot be recited** — large label spaces, novel inputs, structured
outputs; and always inspect solutions for hardcoded-answer layers.

## Tool-use conformance gates the score as much as modeling skill

Seven of 72 model×task cells DNF'd — **every one a harness/tool-conformance fault,
not a modeling failure.** Opus 5 had a working ~0.71 Korean-UnSmile model in
cross-validation, then failed by trying to write its submission *outside* the
sandbox (auto-rejected); GLM 5.2 emitted a malformed `write` tool-call on
KorFin-ASC. The two task *types* introduced in v1.1 — multi-label toxicity
(10-bit multi-hot target) and dependency parsing (per-token head+relation → LAS) —
both produced valid, discriminating scores, but they are exactly where the
tool-use faults surfaced. On Databricks-hosted agents, reliable tool use is a
first-class capability.

## Do we need to fix the harness?  → **Yes — measured.**

Team TODO: *"Harness 에 대한 고정도 필요할지?"* We ran the same models two ways —
**fixed harness** (one pinned opencode scaffold, model swapped) and **native
stacks** (each model in its own agent CLI). Same model, different scaffold,
materially different score:

| Model · task | native stack | fixed harness | Δ |
|---|---|---|---|
| GPT-5.6-sol · 뉴스분류 (F1 ↑) | 0.824 (Codex) | **0.848** (opencode) | +0.024 |
| Opus 5 · 작가판별 (log-loss ↓) | 0.319 (Claude Code) | **0.258** (opencode) | −0.061 |
| Opus 5 · 감성분석 (acc ↑) | **0.880** (Claude Code) | 0.870 (opencode) | −0.010 |

The stack even changes *who wins* a task. **Conclusion:** a model-comparison
claim requires a fixed harness; otherwise you are benchmarking the agent product,
not the model. Native-stack numbers are still worth reporting — as "what a
customer deploys" — but they are a separate question.

## Platform notes (reusable)

- Claude streams reasoning as **structured content blocks**; a single
  OpenAI-compatible wire for all models fails — use per-model native drivers
  (Anthropic driver for Claude, OpenAI driver for GPT/GLM) against the gateway.
- The **OpenAI-compat gateway route logs no tokens** in `system.ai_gateway.usage`
  (only the Anthropic route does) — so per-run cost attribution needs
  `request_tags`, not the usage table. See COST.md.
- Of the Databricks-hosted **OSS** models, only Qwen 3.5 122B drove the agent
  reliably; Llama-4-Maverick rejected the tool schema and gpt-oss-120b emitted
  malformed tool calls — tool-use conformance gates OSS agentic work.
""")

    (OUT / "METHODOLOGY.md").write_text("""# Methodology

Ported from OpenAI **MLE-bench**: agents write and run code against real tasks;
submissions are graded on a hidden split; fresh splits keep answer keys out of
the training path.

## The loop (entirely on Databricks)

```
UC Volume: task pack            Serverless Job: agent          Serverless Job: grader
  spec(KR)·train·HIDDEN test  →   ucode → AI Gateway,       →   scores vs hidden split
  ·grader                         reads spec+train only,        (ACL'd away from agent)
                                  writes submission.csv     →   MLflow score + system-table cost
```

## Fixed harness (this repo's numbers)

- One pinned **opencode** scaffold, byte-identical config and Korean kickoff
  prompt; only the **model** changes: `databricks-claude-opus-5` /
  `databricks-gpt-5-6-sol` / `databricks-glm-5-2`.
- Per-model native drivers: Anthropic driver → `/ai-gateway/anthropic/v1`;
  OpenAI driver → `/ai-gateway/mlflow/v1`.
- Runs as serverless Databricks Jobs (same compute class for every model).

## Grading

- Hidden test lives in a separate UC volume, ACL'd away from the run principal —
  leakage control by **permission**, not convention.
- Deterministic graders; re-runs reproduce identically. Format-invalid
  submission ⇒ **DNF**.

## Standardized prompt

Every model receives the identical kickoff (`PROMPT.md`) plus the task's
`TASK_DESCRIPTION.md`. No model-specific hints. 2-hour wall-clock cap, n=1.
""")

    (OUT / "COST.md").write_text(f"""# Cost — 전체 비용

## What is exact: billed LLM $ by model (all benchmark runs)

From `system.billing.usage` (MODEL_SERVING SKUs), workspace fevm-newjeans-ontos.

| Model | list price (in/out per 1M) | LLM $ billed (all runs) |
|---|---|---|
| Opus 5 | {LIST_PRICE['opus']} | **${MODEL_LLM_TOTAL['opus']:.2f}** |
| GPT-5.6-sol | {LIST_PRICE['sol']} | **${MODEL_LLM_TOTAL['sol']:.2f}** |
| GLM 5.2 | {LIST_PRICE['glm']} | **${MODEL_LLM_TOTAL['glm']:.2f}** |
| Kimi K3 | {LIST_PRICE['kimi']} | $6.16 (잠정, 아래 참조) |

Plus serverless **compute** ≈ $23.29 across the 30 core runs. Whole core
benchmark: **≈ $54.78**.

## Kimi K3 캠페인 (2026-08-13~14, 추가분) — 잠정

Kimi K3는 **전용 provider SKU가 없다** — pay-per-token OSS 공용 SKU
(`ENTERPRISE_SERVERLESS_REAL_TIME_INFERENCE_*`)로 청구된다. 이 캠페인 창에서는
같은 SKU를 쓰는 다른 M-track 모델(GLM)이 돌지 않았으므로 **시간창 + workspace
필터로 귀속**했다 (SKU 유일성이 아니라 창의 배타성에 의존 — 동시 캠페인에서는
성립하지 않으며, 근본 해법은 여전히 request tag).

| 항목 | 값 | 비고 |
|---|---|---|
| LLM (Kimi K3) | **$6.16** | 캠페인 창(2026-08-13T01:00Z~), ws 7474655787271401, 2026-08-14 조회. **아직 잠정** — 빌링 최종행이 08-13 23:50이라 웨이브3 후반 미반영 |
| 서버리스 Jobs 컴퓨트 | $56.89 | 같은 창, smoke 1 + full 72런 |

73런(스모크 포함)이 전부 이 창 안이라 LLM 행은 전량 Kimi 귀속이다. 확정치가
나오면 이 표와 README의 잠정 표기를 갱신할 것.

## The gap: per-task cost is NOT separable in v1

Two reasons, both real:

1. **Runs were concurrent** — many jobs hit the gateway in the same window, so a
   time-window split can't attribute tokens to one run.
2. **The OpenAI-compat route logs no tokens.** `system.ai_gateway.usage` records
   input/output tokens only on the **Anthropic** route; for `sol`/`glm` (mlflow
   route) the token columns are null.

So per-task LLM cost columns in this repo show the **model total**, clearly
labeled — not a per-task figure.

## v2 fix

- **`request_tags`** — the gateway usage schema has a `request_tags` map. Tagging
  each run (`kmle_run_id`) makes per-run token attribution exact, concurrency and
  route notwithstanding.
- **omni per-session $** — the team's session cost view is the independent
  cross-check; reconcile the tagged per-run totals against it.
""")

    (OUT / ".gitignore").write_text(
        "# never commit secrets, hidden answer keys, or raw source data\n"
        "*.env\n.secrets/\nprivate/\nanswers*.csv\n__pycache__/\n.DS_Store\n")
    print("wrote top-level README / FINDINGS / METHODOLOGY / COST / .gitignore")


def main():
    by = latest_runs()
    OUT.mkdir(exist_ok=True)
    summary = {t: {} for t in TASKS}
    for (lane, task), row in sorted(by.items()):
        slug = TASKS[task][0]
        mdir, mname, mid, price = MODEL[lane]
        dest = OUT / slug / mdir
        (dest / "solution").mkdir(parents=True, exist_ok=True)
        rd = row["run_dir"]
        # pull submission, result.json, log, solution
        dbx("fs", "cp", f"{VOL}/{rd}/outputs/submission.csv", str(dest / "submission.csv"), "--overwrite")
        dbx("fs", "cp", f"{VOL}/{rd}/result.json", str(dest / "_result.json"), "--overwrite")
        logr = dbx("fs", "cat", f"{VOL}/{rd}/agent_stdout.log")
        log = logr.stdout if logr.returncode == 0 else ""
        dbx("fs", "cp", "-r", f"{VOL}/{rd}/solution", str(dest / "solution"), "--overwrite")
        # GitHub rejects blobs >100MB; agent-written model/cache binaries add no
        # review value at that size — drop them from the repo tree.
        for big in (dest / "solution").rglob("*"):
            if big.is_file() and big.stat().st_size > 95 * 1024 * 1024:
                print(f"  dropping oversized artifact {big} "
                      f"({big.stat().st_size/1e6:.0f}MB)")
                big.unlink()
        rj = {}
        if (dest / "_result.json").exists():
            rj = json.loads((dest / "_result.json").read_text())
        valid = row["valid"] == "True"
        metrics = {
            "model": mname, "endpoint": mid, "task": slug,
            "quality": {"metric": TASKS[task][1], "direction": TASKS[task][2],
                        "score": float(row["score"]) if valid else None,
                        "valid_submission": valid},
            "time_seconds": rj.get("wall_seconds"),
            "llm_cost_usd_per_run": None,
            "cost_note": ("per-run LLM $ not separable in v1 (concurrent runs; "
                          "OpenAI-compat route logs no tokens). See COST.md. "
                          f"Model total across all runs: ${MODEL_LLM_TOTAL[mdir]:.2f}."),
            "run_arch": rj.get("arch"), "run_id": rd,
        }
        (dest / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
        (dest / "_result.json").unlink(missing_ok=True)
        summary[task][mdir] = metrics
        print(f"built {slug}/{mdir}  score={metrics['quality']['score']} "
              f"time={metrics['time_seconds']}s")

    # per-task README + TASK_DESCRIPTION + PROMPT
    kickoff = (ROOT / "harness" / "kickoff_prompt_ko.md").read_text()
    for task, (slug, metric, direction) in TASKS.items():
        d = OUT / slug
        d.mkdir(exist_ok=True)
        spec = (ROOT / "packs" / task / "spec.md").read_text()
        (d / "TASK_DESCRIPTION.md").write_text(spec)
        (d / "PROMPT.md").write_text(
            f"# 표준화 프롬프트 — {slug}\n\n모든 모델에 동일하게 사용된 킥오프 프롬프트입니다.\n"
            f"태스크 설명은 `TASK_DESCRIPTION.md`(작업 공간의 `spec.md`로 제공됨) 참조.\n\n"
            f"```\n{kickoff}\n```\n")
        # results table
        rows = []
        best = None
        vals = {m: summary[task].get(m, {}).get("quality", {}).get("score")
                for m in MKEYS}
        valid_vals = {m: v for m, v in vals.items() if v is not None}
        if valid_vals:
            best = (max if direction == "higher" else min)(valid_vals.values())
        present = [m for m in MKEYS if summary[task].get(m)]
        for m in present:
            mname = MNAME[m]
            s = summary[task].get(m)
            if not s:
                rows.append(f"| {mname} | — | — | ${MODEL_LLM_TOTAL[m]:.2f} |")
                continue
            q = s["quality"]["score"]
            qs = "DNF" if q is None else (f"**{q:.4g}**" if q == best else f"{q:.4g}")
            t = s["time_seconds"]
            ts = "—" if t is None else f"{t/60:.0f} min"
            rows.append(f"| {mname} | {qs} | {ts} | ${MODEL_LLM_TOTAL[m]:.2f} |")
        (d / "README.md").write_text(
            f"# {slug}\n\n"
            f"Korean ML-engineering task. Metric: **{metric}** "
            f"({'higher' if direction=='higher' else 'lower'} is better). "
            f"Full spec in [`TASK_DESCRIPTION.md`](./TASK_DESCRIPTION.md); "
            f"standardized prompt in [`PROMPT.md`](./PROMPT.md).\n\n"
            f"## Results — " + " vs ".join(MNAME[m] for m in present) + "\n\n"
            f"{len(present)} models, one fixed harness (ucode → Databricks AI Gateway), model swapped.\n\n"
            f"| Model | 결과물 퀄리티 ({metric}) | 소요시간 | LLM 비용 |\n"
            f"|---|---|---|---|\n" + "\n".join(rows) + "\n\n"
            f"Each model folder (" +
            " · ".join(f"`{m}/`" for m in present) + ") holds that model's "
            f"`submission.csv`, the agent-written `solution/` code, and `metrics.json`.\n\n"
            f"> **퀄리티**: scored vs a hidden test split the agent never saw; bold = best; "
            f"n=1 run. **소요시간**: wall-clock, exact. **LLM 비용**: model total across "
            f"*all* benchmark runs (per-task not separable in v1 — see "
            f"[`../COST.md`](../COST.md)).\n")
        print(f"wrote {slug}/README.md")
    top_docs(summary)
    (OUT / "_summary.json").unlink(missing_ok=True)
    print("DONE ->", OUT)


if __name__ == "__main__":
    main()
