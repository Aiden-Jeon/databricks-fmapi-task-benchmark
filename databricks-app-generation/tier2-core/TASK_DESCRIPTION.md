# Task: tpch-revenue-app (Tier 2 — core)

Build a **Databricks App** — a Streamlit revenue-analytics dashboard over the TPC-H
sample schema (`samples.tpch`), queried through a Databricks SQL warehouse.
This file is the CANONICAL task description — copied byte-for-byte into every
candidate's working directory as `instructions.txt`.

## Scenario
A sales-ops team needs a revenue dashboard over TPC-H orders. Queries must run live
against the warehouse, be parameterized, cached where specified, and stay responsive
on large tables (`lineitem` has millions of rows — never fetch it unbounded).

## Definitions (exact — ground truth uses these)
- **revenue** of a lineitem = `l_extendedprice * (1 - l_discount)`.
- **order month** = `date_trunc('MONTH', o_orderdate)`.
- An order is **in range** iff `o_orderdate >= <start date>` AND
  `o_orderdate < <end date + 1 day>` (half-open; end date inclusive as a calendar day).
- **return rate** = revenue of lineitems with `l_returnflag = 'R'` ÷ total revenue,
  within the selected filters.
- Join path: `customer ⋈ orders ON c_custkey = o_custkey ⋈ lineitem ON o_orderkey = l_orderkey`.
- Monetary values displayed with 2 decimal places; rates as percentages with 1 decimal.

## Functional requirements
1. **Title** — main heading "TPC-H Revenue Explorer".
2. **Filters (interacting — both apply to everything below):**
   - market segment multiselect over distinct `c_mktsegment` values
     (`key="segment_filter"`; empty selection = all segments)
   - order-date range, two date inputs (`key="start_date"`, `key="end_date"`)
3. **KPI cards** (within current filters): "Total Revenue", "Order Count",
   "Return Rate" (`st.metric` with exactly these labels).
4. **Monthly trend** (`st.container(key="trend_chart")`): line or bar chart of revenue
   by order month within filters, plus month-over-month % change for the latest
   complete month shown as a metric labeled "MoM Change" (window function or
   equivalent; months ordered ascending).
5. **Segment breakdown** (`st.container(key="segment_chart")`): bar chart of revenue by
   `c_mktsegment` within filters, ordered revenue DESC, ties by segment name ASC.
6. **Order detail table** (`st.dataframe(..., key="orders_table")`): orders in filters
   with columns `o_orderkey`, `o_orderdate`, `c_mktsegment`, `order_revenue`
   (sum of lineitem revenue per order), ordered by `o_orderdate` ASC then
   `o_orderkey` ASC, **paginated 50 rows per page** with prev/next buttons
   (`key="page_prev"`, `key="page_next"`) and a page indicator.
7. **CSV download** (`st.download_button(key="download_csv")`): current filtered order
   detail (all pages, cap 10,000 rows), same columns and ordering as the table.
8. **States** — loading spinner while querying; friendly empty state for zero-result
   filters; `st.error` + a retry button on query failure. Never an unhandled exception.

## Engineering contract (graded — these ARE the point of this tier)
- **Parameter binding:** user inputs (dates, segments) MUST reach SQL via bound
  parameters or validated whitelists — NEVER raw f-string/`%`/`+` interpolation of
  user-controlled values into SQL text. (Grader inspects source and probes inputs.)
- **Query budget:** changing a filter must trigger **at most 4 queries**; paging must
  trigger **at most 1** (page from cached/offset query, don't recompute KPIs); initial
  load at most 6. Distinct-segment list must be cached (`st.cache_data`) for the
  session. The grader counts real queries via warehouse query history.
- **Boundedness:** every query over `lineitem`/`orders` must aggregate or LIMIT —
  no unbounded full-table fetch into pandas.
- Aggregate in SQL, not in pandas over raw lineitems.

## Hard format contract
Same as Tier 1: write exactly `app/app.py`, `app/app.yaml`, `app/requirements.txt`;
Streamlit fixed; single page with ALL widgets in the MAIN page body (no `st.sidebar`);
warehouse ID only from `DATABRICKS_WAREHOUSE_ID`; no hardcoded host/token/warehouse/HTTP
path; default SDK/connector auth; read-only — query ONLY `samples.tpch.*`. Do NOT pass
`--server.port`/`--server.address` in the app.yaml command (the runtime injects the bind
address). Key mismatches cost contract-compliance points; the grader still locates
elements semantically for functional tests.

## Reminders
- KPI numbers are checked against ground-truth SQL (tolerance ±0.01 on money, ±0.1pp
  on rates) — exact aggregation, no sampling.
- The grader tests filter combinations, pagination boundaries (first/last page), empty
  results, and injection probes (e.g. a segment value containing `' OR 1=1 --` must be
  handled safely — parameterized or rejected, never concatenated).
- Keep iterating until `./app/` satisfies the contract.
