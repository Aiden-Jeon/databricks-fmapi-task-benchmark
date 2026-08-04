# MLE Benchmark — Coding Agents on Korean ML-Engineering Tasks

Benchmarking **coding agents on Korean ML-engineering tasks**, run entirely on
Databricks via [ucode](https://github.com/databricks/ucode) → Unity AI Gateway.
Three Foundation-Model-API models compared head-to-head under one fixed harness:

**Claude Opus 5 · GPT-5.6-sol · GLM 5.2**

## Results — 24 Korean tasks

| Task | metric | opus | sol | glm |
|---|---|---|---|---|
| [pubg_placement_prediction](./pubg_placement_prediction) | MAE ↓ | 0.0572±0.059 (n=3) | 0.02448±0.0013 (n=3) | 0.1237±0.12 (n=3) |
| [spooky_author_identification](./spooky_author_identification) | multiclass log loss ↓ | 0.3319±0.077 (n=3) | 0.3632±0.013 (n=3) | 0.3843±0.037 (n=3) |
| [klue_ynat_news_topic](./klue_ynat_news_topic) | macro-F1 ↑ | **0.85±0.0012 (n=2)** | 0.8402±0.012 (n=3) | 0.8442±0.0014 (n=2) |
| [nsmc_movie_sentiment](./nsmc_movie_sentiment) | accuracy ↑ | 0.877±0.0067 (n=3) | 0.8767±0.0014 (n=3) | 0.8703±0.00042 (n=2) |
| [seoul_bike_demand](./seoul_bike_demand) | RMSE ↓ | 316.9±16 (n=3) | 223.2±22 (n=3) | 249.1±49 (n=3) |
| [klue_nli_inference](./klue_nli_inference) | accuracy ↑ | 0.8758±0.0034 (n=2) | 0.8745±0.034 (n=2) | 0.517±0.071 (n=3) |
| [klue_sts_similarity](./klue_sts_similarity) | Pearson ↑ | **0.9586±0.0047 (n=3)** | 0.9481±0.00069 (n=3) | 0.9158±0.045 (n=3) |
| [beep_hate_speech](./beep_hate_speech) | macro-F1 ↑ | **0.5867±0.013 (n=3)** | 0.5635±0.0017 (n=3) | 0.5511±0.012 (n=2) |
| [korquad_reading_comprehension](./korquad_reading_comprehension) | char-F1 ↑ | **0.5159±0.047 (n=3)** | 0.4227±0.015 (n=2) | 0.0484 (n=1) |
| [kornli_inference](./kornli_inference) | accuracy ↑ | 0.6235±0.021 (n=3) | 0.6152±0.019 (n=3) | 0.5278±0.033 (n=3) |
| [korsts_similarity](./korsts_similarity) | Pearson ↑ | **0.7914±0.01 (n=3)** | 0.7455±0.0083 (n=3) | 0.7534±0.0055 (n=2) |
| [kobest_boolq](./kobest_boolq) | accuracy ↑ | 0.6048±0.0099 (n=3) | 0.6025±0.0044 (n=3) | 0.5748±0.042 (n=3) |
| [kobest_copa](./kobest_copa) | accuracy ↑ | **0.6391±0.019 (n=3)** | 0.5904±0.0037 (n=3) | 0.6039±0.018 (n=2) |
| [kobest_wic](./kobest_wic) | accuracy ↑ | 0.6285±0.011 (n=3) | 0.5959±0.0087 (n=3) | 0.6182±0.027 (n=2) |
| [kobest_hellaswag](./kobest_hellaswag) | accuracy ↑ | **0.6786±0.047 (n=2)** | 0.6273±0.01 (n=3) | 0.5788±0.031 (n=3) |
| [kobest_sentineg](./kobest_sentineg) | accuracy ↑ | **0.9575±0.0024 (n=3)** | 0.9539±0.0042 (n=3) | 0.9527±0.0029 (n=2) |
| [pawsx_paraphrase](./pawsx_paraphrase) | accuracy ↑ | **0.7974±0.044 (n=3)** | 0.7602±0.011 (n=3) | 0.7013±0.025 (n=3) |
| [klue_relation_extraction](./klue_relation_extraction) | accuracy ↑ | **0.7713±0.0042 (n=2)** | 0.7292±0.012 (n=3) | 0.7151±0.0082 (n=3) |
| [klue_mrc_reading](./klue_mrc_reading) | char-F1 ↑ | 0.3364±0.014 (n=3) | 0.3375±0.0074 (n=3) | 0.02729 (n=1) |
| [klue_ner_entities](./klue_ner_entities) | entity-F1 ↑ | 0.786±0.11 (n=3) | 0.7692±0.027 (n=3) | 0.7562±0.019 (n=3) |
| [kmmlu_expert_knowledge](./kmmlu_expert_knowledge) | accuracy ↑ | **0.3394±0.014 (n=3)** | 0.3239±0.024 (n=3) | 0.3075 (n=1) |
| [korfin_aspect_sentiment](./korfin_aspect_sentiment) | macro-F1 ↑ | **0.7189±0.021 (n=3)** | 0.6757±0.0039 (n=3) | 0.6831±0.014 (n=2) |
| [kor_unsmile_multilabel_hate](./kor_unsmile_multilabel_hate) | macro-F1 (multi-label) ↑ | 0.706±0.038 (n=2) | 0.714±0.0066 (n=3) | 0.7026±0.0096 (n=2) |
| [klue_dependency_parsing](./klue_dependency_parsing) | LAS ↑ | 0.7195±0.13 (n=3) | 0.7674±0.03 (n=3) | 0.6982±0.066 (n=3) |
| **decided wins** | | **12** | **0** | **0** |

Each cell is **mean ± sample std over repeated runs** (n shown). Bold = the
statistically decided winner; **12 of 24 tasks are ties**, where the
leader's margin did not exceed run-to-run noise — so the win counts deliberately
do not sum to 24.
Scored against a **hidden test split** the agent never saw (leakage controlled by
Unity Catalog ACL). Up to n=3 runs per cell.

## Three metrics (per the team spec)

| | 지표 | Source | Granularity |
|---|---|---|---|
| 1 | **결과물 퀄리티** | grader vs hidden split | exact, per run |
| 2 | **소요시간** | job wall-clock | exact, per run |
| 3 | **전체 비용** | `system.billing.usage` | **per model** (see [COST.md](./COST.md)) |

## Layout

```
<task>/
├── README.md            # per-task result table (quality · time · cost)
├── TASK_DESCRIPTION.md  # the standardized task spec (Korean)
├── PROMPT.md            # the standardized kickoff prompt (identical across models)
├── opus/                # Opus 5:       submission.csv · solution/ · metrics.json
├── sol/                 # GPT-5.6-sol:  submission.csv · solution/ · metrics.json
└── glm/                 # GLM 5.2:      submission.csv · solution/ · metrics.json
```

## Read next

- **[FINDINGS.md](./FINDINGS.md)** — model verdict + the answer to *"do we need to
  fix the harness?"* (measured: yes).
- **[METHODOLOGY.md](./METHODOLOGY.md)** — fixed-harness design, grading, gateway routing.
- **[COST.md](./COST.md)** — billed cost by model + the per-task-attribution gap.

> Methodology ported from OpenAI's [MLE-bench](https://github.com/openai/mle-bench).
> Data is **not** redistributed here (submissions and agent code only); source datasets
> are public (KLUE · NSMC · UCI · Kaggle). Run on Databricks workspace fevm-newjeans-ontos.
