import os
from datetime import date, timedelta

import pandas as pd
import streamlit as st
from databricks import sql
from databricks.sdk import WorkspaceClient

st.set_page_config(page_title="NYC Taxi Explorer", layout="wide")
st.title("NYC Taxi Explorer")

WAREHOUSE_ID = os.environ.get("DATABRICKS_WAREHOUSE_ID")


def get_connection():
    """Create a Databricks SQL connection using SDK default auth resolution."""
    if not WAREHOUSE_ID:
        raise RuntimeError("DATABRICKS_WAREHOUSE_ID environment variable is not set.")
    w = WorkspaceClient()
    warehouse = w.warehouses.get(WAREHOUSE_ID)
    odbc = warehouse.odbc_params
    if odbc is None:
        raise RuntimeError("Warehouse ODBC params not available.")
    hostname = odbc.hostname
    http_path = getattr(odbc, "path", None) or getattr(odbc, "http_path", None)
    headers = w.config.authenticate()
    token = headers.get("Authorization", "").replace("Bearer ", "")
    if not token:
        raise RuntimeError("Failed to obtain auth token from Databricks SDK.")
    return sql.connect(
        server_hostname=hostname,
        http_path=http_path,
        access_token=token,
    )


@st.cache_data(show_spinner=False)
def fetch_kpis():
    """Full-table KPIs — cached so they are never re-queried on filter changes."""
    query = (
        "SELECT COUNT(*) AS total_trips, "
        "ROUND(AVG(fare_amount), 2) AS avg_fare, "
        "ROUND(AVG(trip_distance), 2) AS avg_distance "
        "FROM samples.nyctaxi.trips"
    )
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            row = cur.fetchone()
            return int(row[0]), float(row[1]), float(row[2])


@st.cache_data(show_spinner=False)
def fetch_date_bounds():
    """Min/max pickup dates used to set default filter range."""
    query = (
        "SELECT MIN(tpep_pickup_datetime), MAX(tpep_pickup_datetime) "
        "FROM samples.nyctaxi.trips"
    )
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            row = cur.fetchone()
            return row[0], row[1]


@st.cache_data(show_spinner=False)
def fetch_filtered(start_date, end_date):
    """Top-10 ZIP chart data and first-100 trips for the selected date range."""
    start_ts = start_date.strftime("%Y-%m-%d 00:00:00")
    end_ts = (end_date + timedelta(days=1)).strftime("%Y-%m-%d 00:00:00")

    chart_query = (
        "SELECT pickup_zip, COUNT(*) AS trip_count "
        "FROM samples.nyctaxi.trips "
        "WHERE tpep_pickup_datetime >= '{}' "
        "AND tpep_pickup_datetime < '{}' "
        "AND pickup_zip IS NOT NULL "
        "GROUP BY pickup_zip "
        "ORDER BY trip_count DESC, pickup_zip ASC "
        "LIMIT 10"
    ).format(start_ts, end_ts)

    trips_query = (
        "SELECT tpep_pickup_datetime, trip_distance, fare_amount, "
        "pickup_zip, dropoff_zip "
        "FROM samples.nyctaxi.trips "
        "WHERE tpep_pickup_datetime >= '{}' "
        "AND tpep_pickup_datetime < '{}' "
        "ORDER BY tpep_pickup_datetime ASC "
        "LIMIT 100"
    ).format(start_ts, end_ts)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(chart_query)
            chart_rows = cur.fetchall()
            chart_df = pd.DataFrame(chart_rows, columns=["pickup_zip", "trip_count"])
            cur.execute(trips_query)
            trips_rows = cur.fetchall()
            trips_df = pd.DataFrame(
                trips_rows,
                columns=[
                    "tpep_pickup_datetime",
                    "trip_distance",
                    "fare_amount",
                    "pickup_zip",
                    "dropoff_zip",
                ],
            )
    if not chart_df.empty:
        chart_df["pickup_zip"] = chart_df["pickup_zip"].astype(str)
    return chart_df, trips_df


# ---- KPI cards (full table, cached) ----
try:
    total_trips, avg_fare, avg_distance = fetch_kpis()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Trips", total_trips)
    c2.metric("Avg Fare", f"{avg_fare:.2f}")
    c3.metric("Avg Distance", f"{avg_distance:.2f}")
except Exception as e:
    st.error(f"Failed to load KPIs: {e}")

# ---- Date-range filter ----
try:
    min_dt, max_dt = fetch_date_bounds()
    if min_dt is None or max_dt is None:
        min_date = date(2016, 1, 1)
        max_date = date(2016, 12, 31)
    else:
        min_date = min_dt.date() if hasattr(min_dt, "date") else min_dt
        max_date = max_dt.date() if hasattr(max_dt, "date") else max_dt
except Exception:
    min_date = date(2016, 1, 1)
    max_date = date(2016, 12, 31)

col_s, col_e = st.columns(2)
with col_s:
    start_date = st.date_input(
        "Start Date",
        value=min_date,
        min_value=min_date,
        max_value=max_date,
        key="start_date",
    )
with col_e:
    end_date = st.date_input(
        "End Date",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
        key="end_date",
    )

# ---- Chart + trips table (re-queried on filter change) ----
if start_date > end_date:
    st.error("Start date must be on or before end date.")
else:
    try:
        chart_df, trips_df = fetch_filtered(start_date, end_date)
        if trips_df.empty:
            st.info(
                "No trips found in the selected date range. "
                "Try a different range."
            )
        else:
            with st.container(key="zip_chart"):
                st.subheader("Top 10 Pickup ZIP Codes by Trip Count")
                if not chart_df.empty:
                    chart_data = chart_df.set_index("pickup_zip")
                    st.bar_chart(chart_data)
                else:
                    st.info(
                        "No pickup ZIP data available for the selected range."
                    )
            st.subheader("Trips (First 100)")
            st.dataframe(
                trips_df,
                key="trips_table",
                use_container_width=True,
            )
    except Exception as e:
        st.error(f"Query failed: {e}")
