-- K-MLE-Bench cost attribution queries (workspace: fevm-newjeans-ontos, ws id 7474655787271401)
-- Join keys: job run_ids from the run manifest (kmle/results/run_manifest.csv)
-- All amounts cross-checked against list prices in the report.

-- ── 1. LLM usage per gateway request (tokens + latency) ────────────────────
-- Requires account-admin (or fallback: endpoint payload/inference tables).
-- Filter by run window; the runner records started_utc + wall_seconds per run.
SELECT
  request_time,
  endpoint_name,
  requester,
  usage.prompt_tokens,
  usage.completion_tokens,
  usage.total_tokens,
  latency_ms
FROM system.ai_gateway.usage
WHERE request_time BETWEEN :run_start AND :run_end
ORDER BY request_time;

-- ── 2. LLM billed $ (MODEL_SERVING records enriched by gateway) ────────────
SELECT
  u.usage_start_time,
  u.sku_name,
  u.usage_quantity,
  u.usage_quantity * COALESCE(p.pricing.default, 0) AS usd,
  u.usage_metadata
FROM system.billing.usage u
LEFT JOIN system.billing.list_prices p
  ON u.sku_name = p.sku_name
 AND u.usage_start_time >= p.price_start_time
 AND (p.price_end_time IS NULL OR u.usage_start_time < p.price_end_time)
WHERE u.billing_origin_product = 'MODEL_SERVING'
  AND u.usage_start_time BETWEEN :run_start AND :run_end;

-- ── 3. Serverless jobs compute $ per benchmark run ─────────────────────────
-- Serverless-only workspace: attribute by job_run_id from the manifest.
SELECT
  u.usage_metadata.job_run_id,
  SUM(u.usage_quantity) AS dbus,
  SUM(u.usage_quantity * COALESCE(p.pricing.default, 0)) AS usd
FROM system.billing.usage u
LEFT JOIN system.billing.list_prices p
  ON u.sku_name = p.sku_name
 AND u.usage_start_time >= p.price_start_time
 AND (p.price_end_time IS NULL OR u.usage_start_time < p.price_end_time)
WHERE u.billing_origin_product = 'JOBS'
  AND u.usage_metadata.job_run_id IN (:run_id_list)
GROUP BY 1;

-- ── 4. Genie Code PAYG rows (L4 sessions, matched by time window) ──────────
SELECT
  u.usage_start_time, u.usage_end_time, u.sku_name,
  u.usage_quantity, u.usage_metadata, u.billing_origin_product
FROM system.billing.usage u
WHERE u.usage_start_time BETWEEN :l4_session_start AND :l4_session_end
  AND (UPPER(u.sku_name) LIKE '%GENIE%'
       OR UPPER(CAST(u.billing_origin_product AS STRING)) LIKE '%GENIE%')
ORDER BY u.usage_start_time;
