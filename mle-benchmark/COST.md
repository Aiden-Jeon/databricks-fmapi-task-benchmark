# Cost — 전체 비용

## What is exact: billed LLM $ by model (all benchmark runs)

From `system.billing.usage` (MODEL_SERVING SKUs), workspace fevm-newjeans-ontos.

| Model | list price (in/out per 1M) | LLM $ billed (all runs) |
|---|---|---|
| Opus 5 | $5 / $25 | **$23.22** |
| GPT-5.6-sol | $5 / $30 | **$5.17** |
| GLM 5.2 | $1.40 / $4.40 | **$3.10** |
| Kimi K3 | TBD | $6.16 (잠정, 아래 참조) |

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
