"""Trip Monitor — internal dashboard (repaired, hardened, extended).

Dashboard over samples.nyctaxi.trips. All KPIs computed exactly via SQL
aggregation; user inputs bound as parameters; errors surfaced to the user.
"""
import os
import datetime as dt

import pandas as pd
import streamlit as st
from databricks import sql
from databricks.sdk.core import Config

st.set_page_config(page_title="Trip Monitor", layout="wide")
st.title("Trip Monitor")

TABLE = "samples.nyctaxi.trips"
MAX_TABLE_ROWS = 100
MAX_CSV_ROWS = 10000


# --- connection -------------------------------------------------------------
def _warehouse_http_path() -> str:
    """Resolve the SQL warehouse HTTP path from standard env only."""
    warehouse_id = os.getenv("DATABRICKS_WAREHOUSE_ID", "").strip()
    if not warehouse_id:
        raise RuntimeError(
            "DATABRICKS_WAREHOUSE_ID is not set. Configure the app's SQL warehouse."
        )
    return f"/sql/1.0/warehouses/{warehouse_id}"


def _connect():
    cfg = Config()
    http_path = _warehouse_http_path()
    return sql.connect(
        server_hostname=cfg.host,
        http_path=http_path,
        credentials_provider=lambda: cfg.authenticate,
    )


def _run(query: str, params=None) -> pd.DataFrame:
    """Execute a read-only query with bound parameters and return a DataFrame."""
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(query, params or {})
            arrow = cur.fetchall_arrow()
            return arrow.to_pandas()
    finally:
        conn.close()


# --- cached query helpers ---------------------------------------------------
# Half-open date range: >= start 00:00 AND < end + 1 day 00:00.
def _bounds(start_date: dt.date, end_date: dt.date):
    start_ts = dt.datetime.combine(start_date, dt.time.min)
    end_ts = dt.datetime.combine(end_date + dt.timedelta(days=1), dt.time.min)
    return start_ts, end_ts


def _clean_zip(zip_filter: str) -> str:
    if zip_filter is None:
        return ""
    z = zip_filter.strip()
    if not z:
        return ""
    return z


def _where_and_params(start_date, end_date, zip_filter):
    start_ts, end_ts = _bounds(start_date, end_date)
    params = {"start_ts": start_ts, "end_ts": end_ts}
    where = (
        "tpep_pickup_datetime >= %(start_ts)s "
        "AND tpep_pickup_datetime < %(end_ts)s"
    )
    z = _clean_zip(zip_filter)
    if z:
        where += " AND pickup_zip = %(zip)s"
        params["zip"] = z
    return where, params


@st.cache_data(ttl=300, show_spinner=False)
def load_kpis(start_date, end_date, zip_filter):
    where, params = _where_and_params(start_date, end_date, zip_filter)
    q = (
        "SELECT COUNT(*) AS total_trips, "
        "AVG(fare_amount) AS avg_fare, "
        "MAX(fare_amount) AS max_fare "
        f"FROM {TABLE} WHERE {where}"
    )
    df = _run(q, params)
    row = df.iloc[0]
    total = int(row["total_trips"]) if pd.notna(row["total_trips"]) else 0
    avg = float(row["avg_fare"]) if pd.notna(row["avg_fare"]) else None
    mx = float(row["max_fare"]) if pd.notna(row["max_fare"]) else None
    return {"total_trips": total, "avg_fare": avg, "max_fare": mx}


@st.cache_data(ttl=300, show_spinner=False)
def load_top_pickup_zips(start_date, end_date, zip_filter):
    where, params = _where_and_params(start_date, end_date, zip_filter)
    q = (
        "SELECT pickup_zip, COUNT(*) AS trips "
        f"FROM {TABLE} WHERE {where} AND pickup_zip IS NOT NULL "
        "GROUP BY pickup_zip "
        "ORDER BY trips DESC, pickup_zip ASC "
        "LIMIT 10"
    )
    df = _run(q, params)
    return df


@st.cache_data(ttl=300, show_spinner=False)
def load_top_dropoff_zips(start_date, end_date, zip_filter):
    where, params = _where_and_params(start_date, end_date, zip_filter)
    q = (
        "SELECT dropoff_zip, COUNT(*) AS trips "
        f"FROM {TABLE} WHERE {where} AND dropoff_zip IS NOT NULL "
        "GROUP BY dropoff_zip "
        "ORDER BY trips DESC, dropoff_zip ASC "
        "LIMIT 10"
    )
    df = _run(q, params)
    return df


@st.cache_data(ttl=300, show_spinner=False)
def load_table(start_date, end_date, zip_filter, limit):
    where, params = _where_and_params(start_date, end_date, zip_filter)
    params = dict(params)
    params["row_limit"] = int(limit)
    q = (
        "SELECT tpep_pickup_datetime, tpep_dropoff_datetime, "
        "trip_distance, fare_amount, pickup_zip, dropoff_zip "
        f"FROM {TABLE} WHERE {where} "
        "ORDER BY tpep_pickup_datetime ASC "
        "LIMIT %(row_limit)s"
    )
    df = _run(q, params)
    return df


# --- filters ----------------------------------------------------------------
_default_start = dt.date(2016, 1, 1)
_default_end = dt.date(2016, 2, 29)

col1, col2, col3 = st.columns(3)
start_date = col1.date_input("Start date", key="start_date", value=_default_start)
end_date = col2.date_input("End date", key="end_date", value=_default_end)
zip_filter = col3.text_input("Pickup ZIP (optional)", key="zip_filter", max_chars=32)


def _render():
    if start_date > end_date:
        st.error("Start date must be on or before end date.")
        return

    try:
        kpis = load_kpis(start_date, end_date, zip_filter)
        pickup = load_top_pickup_zips(start_date, end_date, zip_filter)
        dropoff = load_top_dropoff_zips(start_date, end_date, zip_filter)
    except Exception:
        st.error(
            "Something went wrong loading the dashboard. The warehouse may be "
            "unreachable or misconfigured. Please try again."
        )
        st.button("Retry", key="retry_top")
        return

    # --- KPIs ---------------------------------------------------------------
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Trips", f"{kpis['total_trips']:,}")
    k2.metric(
        "Avg Fare",
        "N/A" if kpis["avg_fare"] is None else f"{round(kpis['avg_fare'], 2):,.2f}",
    )
    k3.metric(
        "Max Fare",
        "N/A" if kpis["max_fare"] is None else f"{round(kpis['max_fare'], 2):,.2f}",
    )

    if kpis["total_trips"] == 0:
        st.info("No trips match the selected filters.")
        return

    # --- pickup chart -------------------------------------------------------
    with st.container(key="pickup_chart"):
        st.subheader("Top pickup ZIPs")
        if pickup.empty:
            st.info("No pickup ZIP data for the selected filters.")
        else:
            st.bar_chart(pickup.set_index("pickup_zip")["trips"])

    # --- dropoff chart ------------------------------------------------------
    with st.container(key="dropoff_chart"):
        st.subheader("Top dropoff ZIPs")
        if dropoff.empty:
            st.info("No dropoff ZIP data for the selected filters.")
        else:
            st.bar_chart(dropoff.set_index("dropoff_zip")["trips"])

    # --- table --------------------------------------------------------------
    st.subheader("Trips")
    try:
        table_df = load_table(start_date, end_date, zip_filter, MAX_TABLE_ROWS)
    except Exception:
        st.error("Could not load the trips table. Please try again.")
        st.button("Retry", key="retry_table")
        return

    if table_df.empty:
        st.info("No trips match the selected filters.")
    else:
        st.dataframe(table_df, key="trips_table")

        try:
            csv_df = load_table(start_date, end_date, zip_filter, MAX_CSV_ROWS)
        except Exception:
            csv_df = table_df
        st.download_button(
            "Download CSV",
            data=csv_df.to_csv(index=False).encode("utf-8"),
            file_name="trips.csv",
            mime="text/csv",
            key="download_csv",
        )


_render()
