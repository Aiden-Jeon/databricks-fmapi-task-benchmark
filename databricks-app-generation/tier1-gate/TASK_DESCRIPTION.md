# Task: nyctaxi-dashboard-app

Build a **Databricks App** — a Streamlit dashboard that explores the `samples.nyctaxi.trips`
table through a Databricks SQL warehouse.
This file is the CANONICAL task description — it is copied byte-for-byte into every
candidate's working directory as `instructions.txt`, so every candidate reads the exact
same brief (the fairness rule of this benchmark).

## Scenario
A data team wants a lightweight internal dashboard, deployed as a Databricks App, to
explore NYC taxi trips. The app must query live data through a SQL warehouse — no CSV
exports, no hardcoded data.

## Functional requirements
The app MUST provide:

1. **Title** — page title "NYC Taxi Explorer" rendered as the main heading.
2. **KPI cards** — three metrics computed from the full `samples.nyctaxi.trips` table:
   - total trip count
   - average fare amount (2 decimal places)
   - average trip distance (2 decimal places)
3. **Date-range filter** — two date inputs (start / end, on `tpep_pickup_datetime`).
   Changing the range MUST re-query and update ONLY the trips table (requirement 5) and
   the chart (requirement 4). The KPI cards in requirement 2 always show full-table
   values and MUST NOT be re-queried on filter changes (cache them, e.g. `st.cache_data`).
   - **Boundary semantics (exact):** a trip is in range iff
     `tpep_pickup_datetime >= <start date 00:00:00>` AND
     `tpep_pickup_datetime < <end date + 1 day, 00:00:00>` (half-open interval; end date
     is inclusive as a calendar day). Timestamps are compared as stored in the table
     (no timezone conversion).
4. **Chart** — a bar chart of the top 10 pickup ZIP codes (`pickup_zip`) by trip count
   within the selected date range.
   - Exclude rows where `pickup_zip IS NULL`.
   - Ties: order by trip count DESC, then `pickup_zip` ASC.
5. **Trips table** — a data table showing the first 100 trips in the selected range
   ordered by `tpep_pickup_datetime` ASC (ties: any stable order is accepted), with at
   least these columns: `tpep_pickup_datetime`, `trip_distance`, `fare_amount`,
   `pickup_zip`, `dropoff_zip`.
6. **Empty-state handling** — if the selected range matches zero trips, show a friendly
   message; the app must not crash or show a stack trace.

## Hard format contract (the grader checks these mechanically)
- Write EXACTLY these files under `./app/` in this working directory:
  - `app/app.py` — the Streamlit entrypoint (single file, no other Python modules).
  - `app/app.yaml` — valid YAML with a `command` list that starts the app
    (e.g. `["streamlit", "run", "app.py"]`). Do NOT pass `--server.port` or
    `--server.address` in the command — the runtime (grader / Databricks Apps) injects
    the bind address; a hardcoded port breaks portability and the health check.
  - `app/requirements.txt` — all Python dependencies beyond the Databricks Apps
    pre-installed set. It MUST include the SQL client you use
    (`databricks-sql-connector` or `databricks-sdk`).
- **Framework is fixed: Streamlit.** Do not use Dash, Gradio, Flask, or Node.js.
- **Connection rules (least-privilege, portable):**
  - Read the SQL warehouse ID ONLY from the environment variable
    `DATABRICKS_WAREHOUSE_ID`.
  - Do NOT hardcode any workspace hostname, token, warehouse ID, or HTTP path in the
    source. Authentication is provided by the Databricks Apps runtime (the grader
    injects credentials via environment when running locally); use default SDK/connector
    auth resolution.
  - Query ONLY `samples.nyctaxi.trips`. Do not create, write, or drop any table.
- **Testability contract** — the grader locates elements primarily by Streamlit `key=` /
  labels below, with a semantic fallback (visible label text / element role). Using the
  exact keys is scored as contract compliance; functionality is still graded via the
  fallback if a key differs. Use EXACTLY these keys:
  - KPI metrics: `st.metric` labels "Total Trips", "Avg Fare", "Avg Distance"
  - date inputs: `key="start_date"`, `key="end_date"`
  - chart: rendered inside a container created with `st.container(key="zip_chart")`
  - table: `st.dataframe(..., key="trips_table")`
- Keep the app to ONE page: no multipage app, and render ALL widgets (filters, KPIs,
  charts, table) in the MAIN page body — do NOT use `st.sidebar`.
- Handle query errors gracefully (show `st.error`, never an unhandled exception).

## Reminders
- The grader will: statically check the contract → install `requirements.txt` in a clean
  venv → launch via `app.yaml`'s command with credentials in env → run GUI test cases
  against the running app → deploy it as a real Databricks App and smoke-test it.
- Numeric KPI values are checked against ground truth computed by SQL — make sure your
  aggregations are exact (no sampling, no LIMIT before aggregating).
- Keep iterating until `./app/` exists and satisfies the contract above.
