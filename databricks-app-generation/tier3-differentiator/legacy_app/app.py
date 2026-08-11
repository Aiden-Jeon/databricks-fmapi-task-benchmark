"""Trip Monitor — internal dashboard. (v0.3, original author left the team)

NOTE FOR CANDIDATES: this file is the legacy code you must repair and extend.
See instructions.txt for requirements. Defects are NOT listed — find them.
"""
import os
import pandas as pd
import streamlit as st
from databricks import sql

st.set_page_config(page_title="Trip Monitor", layout="wide")
st.title("Trip Monitor")

# --- connection -------------------------------------------------------------
HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH", "/sql/1.0/warehouses/8f2e3a1b9c0d4e5f")


def run_query(query: str) -> pd.DataFrame:
    conn = sql.connect(
        server_hostname=os.getenv("DATABRICKS_HOST", ""),
        http_path=HTTP_PATH,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            return cur.fetchall_arrow().to_pandas()
    except Exception:
        # TODO: figure out why this sometimes fails
        return pd.DataFrame()
    finally:
        conn.close()


# --- filters ----------------------------------------------------------------
col1, col2, col3 = st.columns(3)
start_date = col1.date_input("Start date", key="start_date")
end_date = col2.date_input("End date", key="end_date")
zip_filter = col3.text_input("Pickup ZIP (optional)", key="zip_filter")

where = f"tpep_pickup_datetime >= '{start_date}' AND tpep_pickup_datetime <= '{end_date}'"
if zip_filter:
    where += f" AND pickup_zip = '{zip_filter}'"

# --- data -------------------------------------------------------------------
trips = run_query(
    f"SELECT * FROM samples.nyctaxi.trips WHERE {where} LIMIT 1000"
)

# --- KPIs -------------------------------------------------------------------
k1, k2, k3 = st.columns(3)
k1.metric("Total Trips", len(trips))
k2.metric("Avg Fare", round(trips["fare_amount"].mean(), 2))
k3.metric("Max Fare", trips["fare_amount"].iloc[trips["fare_amount"].idxmax()])

# --- chart ------------------------------------------------------------------
by_zip = trips.groupby("pickup_zip").size().sort_values(ascending=False).head(10)
st.subheader("Top pickup ZIPs")
st.bar_chart(by_zip)

# --- table ------------------------------------------------------------------
st.subheader("Trips")
st.dataframe(trips.head(100), key="trips_table")
