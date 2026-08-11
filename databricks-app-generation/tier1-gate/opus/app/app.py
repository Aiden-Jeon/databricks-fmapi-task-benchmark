import os
from datetime import date, timedelta

import pandas as pd
import streamlit as st
from databricks import sql
from databricks.sdk.core import Config

st.set_page_config(page_title="NYC Taxi Explorer", layout="wide")

TABLE = "samples.nyctaxi.trips"


def get_connection():
    warehouse_id = os.getenv("DATABRICKS_WAREHOUSE_ID")
    if not warehouse_id:
        raise RuntimeError(
            "DATABRICKS_WAREHOUSE_ID environment variable is not set."
        )
    cfg = Config()
    http_path = f"/sql/1.0/warehouses/{warehouse_id}"
    return sql.connect(
        server_hostname=cfg.host,
        http_path=http_path,
        credentials_provider=lambda: cfg.authenticate,
    )


def run_query(query, params=None):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(query, params or {})
            columns = [c[0] for c in cur.description]
            rows = cur.fetchall()
        return pd.DataFrame(rows, columns=columns)
    finally:
        conn.close()


@st.cache_data(show_spinner=False)
def load_kpis():
    query = f"""
        SELECT
            COUNT(*) AS total_trips,
            AVG(fare_amount) AS avg_fare,
            AVG(trip_distance) AS avg_distance
        FROM {TABLE}
    """
    return run_query(query)


@st.cache_data(show_spinner=False)
def load_top_zips(start_d, end_d):
    query = f"""
        SELECT pickup_zip, COUNT(*) AS trip_count
        FROM {TABLE}
        WHERE tpep_pickup_datetime >= :start_ts
          AND tpep_pickup_datetime < :end_ts
          AND pickup_zip IS NOT NULL
        GROUP BY pickup_zip
        ORDER BY trip_count DESC, pickup_zip ASC
        LIMIT 10
    """
    params = {
        "start_ts": f"{start_d} 00:00:00",
        "end_ts": f"{end_d + timedelta(days=1)} 00:00:00",
    }
    return run_query(query, params)


@st.cache_data(show_spinner=False)
def load_trips(start_d, end_d):
    query = f"""
        SELECT
            tpep_pickup_datetime,
            trip_distance,
            fare_amount,
            pickup_zip,
            dropoff_zip
        FROM {TABLE}
        WHERE tpep_pickup_datetime >= :start_ts
          AND tpep_pickup_datetime < :end_ts
        ORDER BY tpep_pickup_datetime ASC
        LIMIT 100
    """
    params = {
        "start_ts": f"{start_d} 00:00:00",
        "end_ts": f"{end_d + timedelta(days=1)} 00:00:00",
    }
    return run_query(query, params)


@st.cache_data(show_spinner=False)
def load_date_bounds():
    query = f"""
        SELECT
            MIN(tpep_pickup_datetime) AS min_ts,
            MAX(tpep_pickup_datetime) AS max_ts
        FROM {TABLE}
    """
    return run_query(query)


st.title("NYC Taxi Explorer")

try:
    kpis = load_kpis()
    total_trips = int(kpis["total_trips"].iloc[0])
    avg_fare = float(kpis["avg_fare"].iloc[0])
    avg_distance = float(kpis["avg_distance"].iloc[0])

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Trips", f"{total_trips:,}")
    c2.metric("Avg Fare", f"{avg_fare:.2f}")
    c3.metric("Avg Distance", f"{avg_distance:.2f}")
except Exception as e:
    st.error(f"Failed to load KPI data: {e}")
    st.stop()

try:
    bounds = load_date_bounds()
    min_ts = bounds["min_ts"].iloc[0]
    max_ts = bounds["max_ts"].iloc[0]
    default_start = pd.to_datetime(min_ts).date() if min_ts is not None else date(2016, 1, 1)
    default_end = pd.to_datetime(max_ts).date() if max_ts is not None else date(2016, 12, 31)
except Exception:
    default_start = date(2016, 1, 1)
    default_end = date(2016, 12, 31)

col_start, col_end = st.columns(2)
with col_start:
    start_date = st.date_input("Start date", value=default_start, key="start_date")
with col_end:
    end_date = st.date_input("End date", value=default_end, key="end_date")

if start_date > end_date:
    st.error("Start date must be on or before end date.")
    st.stop()

st.subheader("Top 10 Pickup ZIP Codes")
chart_container = st.container(key="zip_chart")

st.subheader("Trips")

try:
    zip_df = load_top_zips(start_date, end_date)
    trips_df = load_trips(start_date, end_date)
except Exception as e:
    st.error(f"Failed to query trips: {e}")
    st.stop()

with chart_container:
    if zip_df.empty:
        st.info("No trips found in the selected date range.")
    else:
        chart_data = zip_df.copy()
        chart_data["pickup_zip"] = chart_data["pickup_zip"].astype(str)
        chart_data = chart_data.set_index("pickup_zip")
        st.bar_chart(chart_data["trip_count"])

if trips_df.empty:
    st.info("No trips found in the selected date range.")
else:
    st.dataframe(trips_df, key="trips_table", use_container_width=True)
