# kobest_copa

Korean ML-engineering task. Metric: **accuracy** (higher is better). Full spec in [`TASK_DESCRIPTION.md`](./TASK_DESCRIPTION.md); standardized prompt in [`PROMPT.md`](./PROMPT.md).

## Results — Claude Opus 5 vs GPT-5.6-sol vs GLM 5.2 vs Kimi K3

4 models, one fixed harness (ucode → Databricks AI Gateway), model swapped.

| Model | 결과물 퀄리티 (accuracy) | 소요시간 | LLM 비용 |
|---|---|---|---|
| Claude Opus 5 | **0.6607** | 46 min | $23.22 |
| GPT-5.6-sol | 0.5925 | 8 min | $5.17 |
| GLM 5.2 | 0.5909 | 52 min | $3.10 |
| Kimi K3 | 0.5925 | 88 min | $0.00 |

Each model folder (`opus/` · `sol/` · `glm/` · `kimi/`) holds that model's `submission.csv`, the agent-written `solution/` code, and `metrics.json`.

> **퀄리티**: scored vs a hidden test split the agent never saw; bold = best; n=1 run. **소요시간**: wall-clock, exact. **LLM 비용**: model total across *all* benchmark runs (per-task not separable in v1 — see [`../COST.md`](../COST.md)).
