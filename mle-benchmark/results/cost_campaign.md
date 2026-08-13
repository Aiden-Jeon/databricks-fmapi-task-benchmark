# Billed cost — full benchmark, all phases

Source: `system.billing.usage` × `system.billing.list_prices`, workspace
7474655787271401, from 2026-07-30. Phase split at 2026-08-03 (start of the
repeat campaign). Amounts are **billed**, not list-price estimates.

| | v1 (pre-repeat) | repeat campaign | total |
|---|---|---|---|
| Anthropic serving (Opus 5) | $82.02 | $40.81 | **$122.83** |
| OpenAI serving (GPT-5.6-sol) | $25.90 | $17.94 | **$43.84** |
| Serverless RT inference (GLM 5.2 + OSS) | $12.14 | $3.97 | **$16.11** |
| **LLM subtotal** | $120.06 | $62.72 | **$182.79** |
| Serverless JOBS (DBU) | $14.36 | $9.98 | **$24.34** |
| **TOTAL** | **$134.42** | **$72.70** | **$207.12** |

Notes:
- Per-model LLM $ is attributable only at **provider-SKU granularity**: the
  Anthropic route bills under `ENTERPRISE_ANTHROPIC_MODEL_SERVING`, GPT under
  `ENTERPRISE_OPENAI_MODEL_SERVING`, and GLM/OSS under
  `ENTERPRISE_SERVERLESS_REAL_TIME_INFERENCE_*`. Since each provider SKU carries
  exactly one benchmarked model in the M-track, the split is clean here — but it
  is a property of this model mix, not a general per-model meter. Per-run
  attribution still needs request tags (v2).
- The repeat campaign (144 runs) cost **$72.70** — about half of v1's $134.42 for
  ~5x the runs, because v1 included exploratory/failed lanes and per-task agent
  spend fell as tasks were reused.
- Opus is ~2.8x GPT's LLM spend and ~7.6x GLM's across the whole campaign,
  consistent with the v1 5-task ratio (~4.5x / ~7.5x) at a smaller Opus multiple
  once cheap repeat runs are included.
- Compute (DBU) is 11.8% of total spend; LLM tokens dominate at 88.2%.
