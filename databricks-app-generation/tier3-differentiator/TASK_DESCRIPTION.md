# Task: trip-monitor-repair (Tier 3 — differentiator)

You inherit a **legacy Databricks App** (`./legacy_app/` — a Streamlit dashboard over
`samples.nyctaxi.trips`) written by an engineer who left. Users report wrong numbers,
crashes, and slowness. Your job: **repair it, harden it, and extend it** — while
keeping its existing UI structure and keys.
This file is the CANONICAL task description — copied byte-for-byte into every
candidate's working directory as `instructions.txt`, together with a copy of
`legacy_app/`.

## What you must deliver
Write the fixed and extended app to `./app/` (`app.py`, `app.yaml`,
`requirements.txt`) — start from the legacy code, preserve what is correct.

### 1. Repair (defects are NOT listed — audit the code and fix everything you find)
Known symptom reports from users:
- "The KPI numbers don't match our SQL notebooks."
- "The app crashes on some date ranges."
- "Sometimes it just shows an empty dashboard with no error."
- "Filtering by ZIP behaves strangely with certain inputs."
- "It got slower as more people used it."
- "It only works in the dev workspace."
- "Trips on the end date sometimes don't show up."

Correct semantics (same as Tier 1 conventions):
- KPIs ("Total Trips", "Avg Fare", and keep a third metric of max fare) must be exact
  over ALL trips matching the filters — computed by SQL aggregation, never from a
  row-limited sample.
- Date range is half-open: `>= start 00:00` AND `< end + 1 day 00:00`.
- Top-10 chart: exclude NULL `pickup_zip`; ties by `pickup_zip` ASC.
- Table: first 100 matching trips ordered by `tpep_pickup_datetime` ASC.
- All user inputs reach SQL via bound parameters or validation — never string
  concatenation.
- Errors must surface to the user (`st.error` + retry button), never be silently
  swallowed and never show a stack trace. Empty results show a friendly message.
- Connection/warehouse configuration only via standard env
  (`DATABRICKS_WAREHOUSE_ID`, default SDK/connector auth) — nothing
  workspace-specific hardcoded.
- Cache what is safe to cache (`st.cache_data`) so a filter change triggers at most
  3 queries and an unchanged rerun triggers none.

### 2. Extend
- Add a "Top dropoff ZIPs" bar chart (`st.container(key="dropoff_chart")`), same
  semantics as the pickup chart.
- Add a CSV download (`st.download_button(key="download_csv")`) of the current
  filtered table view (same columns/ordering, cap 10,000 rows).

### 3. Harden (the grader actively attacks the app)
- Invalid `DATABRICKS_WAREHOUSE_ID` or unreachable warehouse at startup → app still
  renders, shows a clear error state and a retry button (no crash loop).
- Query timeout mid-session → error state + retry; app process stays alive.
- Malicious ZIP input (`' OR 1=1 --`, very long strings, unicode) → handled safely.
- A GUI fuzzer will click/type randomly for several minutes — zero unhandled
  exceptions is the bar.

## Hard format contract
Same as Tier 1/2: Streamlit fixed, single page with ALL widgets in the MAIN page body
(no `st.sidebar`), no `--server.port`/`--server.address` in the app.yaml command
(the runtime injects the bind address), exact file layout under `./app/`,
read-only queries against `samples.nyctaxi.trips` only, no hardcoded
host/token/warehouse/HTTP path. Keep existing widget keys (`start_date`, `end_date`,
`zip_filter`, `trips_table`) and metric labels; new elements use the keys specified
above. Key mismatches cost contract-compliance points only.

## Reminders
- The grader diffs behavior against ground-truth SQL (recomputed at grading time),
  replays the failure scenarios above, runs the fuzzer, and counts queries via
  warehouse query history.
- Fixing the symptoms without preserving working behavior (e.g. deleting features)
  scores zero for those features.
- Keep iterating until `./app/` satisfies the contract.
