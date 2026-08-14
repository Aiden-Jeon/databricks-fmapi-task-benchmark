# Findings

## The model verdict — three profiles, not one winner

| | Opus 5 | GPT-5.6-sol | GLM 5.2 | Kimi K3 |
|---|---|---|---|---|
| decided wins (of 24) | **12** | 0 | 0 | 0 |
| profile | quality leader | efficiency frontier | budget contender | slow long-tail |
| reliability (valid rate, n=3 campaign) | 93% | **97%** | 79% | 71% |
| LLM $ (campaign) | $122.83 | $43.84 | $16.11 | $6.16 (잠정) |

**12 of 24 tasks are statistical ties** — the leader's margin sat
inside run-to-run noise, so no winner is claimed.
Win counts therefore do not sum to 24, and that gap is a result, not a
gap in the data: at n=1 those same cells looked decisive.

- **Opus 5** takes 12/24 decided cells, concentrated in the
  harder Korean-NLP tasks (STS, QA, RE-adjacent, and calibration-heavy 작가판별).
  It costs ~4.5× GPT and ~7.5× GLM (measured on the v1 5-task pass) and runs
  longest — buy it when the score is worth the money.
- **GPT-5.6-sol** is the efficiency frontier: 0 decided wins, fastest
  everywhere, and the only model with a perfect 24/24 valid-submission rate. Its
  edge is throughput and reliability, not peak quality.
- **GLM 5.2** is the budget play — 0 decided wins at a fraction of the
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
