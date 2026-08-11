"""Trip Monitor — hardened Databricks App dashboard."""

import os
import re
from datetime import date, datetime, time, timedelta
from typing import Any, Optional

import pandas as pd
import streamlit as st
from databricks import sql
from databricks.sdk.core import Config


TABLE_NAME = "samples.nyctaxi.trips"
WAREHOUSE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
ZIP_PATTERN = re.compile(r"^[0-9]{1,10}$", re.ASCII)


st.set_page_config(page_title="Trip Monitor", layout="wide")
st.title("Trip Monitor")


def _warehouse_connection():
    """Create a SQL connection using Databricks default authentication."""
    warehouse_id = os.getenv("DATABRICKS_WAREHOUSE_ID", "").strip()
    if not warehouse_id:
        raise RuntimeError("DATABRICKS_WAREHOUSE_ID is not configured.")
    if not WAREHOUSE_ID_PATTERN.fullmatch(warehouse_id):
        raise RuntimeError("DATABRICKS_WAREHOUSE_ID is invalid.")

    config = Config()
    if not config.host:
        raise RuntimeError("Databricks workspace authentication is not configured.")

    return sql.connect(
        server_hostname=config.host,
        http_path=f"/sql/1.0/warehouses/{warehouse_id}",
        credentials_provider=lambda: config.authenticate,
    )


def _run_query(statement: str, parameters: dict[str, Any]) -> pd.DataFrame:
    connection = None
    try:
        connection = _warehouse_connection()
        with connection.cursor() as cursor:
            cursor.execute(statement, parameters=parameters)
            return cursor.fetchall_arrow().to_pandas()
    finally:
        if connection is not None:
            connection.close()


def _query_parts(
    start_ts: datetime,
    end_ts: datetime,
    pickup_zip: Optional[int],
) -> tuple[str, dict[str, Any]]:
    predicate = (
        "tpep_pickup_datetime >= :start_ts "
        "AND tpep_pickup_datetime < :end_ts"
    )
    parameters: dict[str, Any] = {
        "start_ts": start_ts,
        "end_ts": end_ts,
    }

    if pickup_zip is not None:
        predicate += " AND pickup_zip = :pickup_zip"
        parameters["pickup_zip"] = pickup_zip

    return predicate, parameters


@st.cache_data(show_spinner=False)
def load_kpis(
    start_ts: datetime,
    end_ts: datetime,
    pickup_zip: Optional[int],
) -> pd.DataFrame:
    predicate, parameters = _query_parts(start_ts, end_ts, pickup_zip)
    return _run_query(
        f"""
        SELECT
            COUNT(*) AS total_trips,
            AVG(fare_amount) AS avg_fare,
            MAX(fare_amount) AS max_fare
        FROM {TABLE_NAME}
        WHERE {predicate}
        """,
        parameters,
    )


@st.cache_data(show_spinner=False)
def load_zip_charts(
    start_ts: datetime,
    end_ts: datetime,
    pickup_zip: Optional[int],
) -> pd.DataFrame:
    predicate, parameters = _query_parts(start_ts, end_ts, pickup_zip)
    return _run_query(
        f"""
        WITH filtered AS (
            SELECT pickup_zip, dropoff_zip
            FROM {TABLE_NAME}
            WHERE {predicate}
        ),
        zip_counts AS (
            SELECT
                'pickup' AS chart_type,
                pickup_zip AS zip_code,
                COUNT(*) AS trip_count
            FROM filtered
            WHERE pickup_zip IS NOT NULL
            GROUP BY pickup_zip

            UNION ALL

            SELECT
                'dropoff' AS chart_type,
                dropoff_zip AS zip_code,
                COUNT(*) AS trip_count
            FROM filtered
            WHERE dropoff_zip IS NOT NULL
            GROUP BY dropoff_zip
        ),
        ranked AS (
            SELECT
                chart_type,
                zip_code,
                trip_count,
                ROW_NUMBER() OVER (
                    PARTITION BY chart_type
                    ORDER BY trip_count DESC, zip_code ASC
                ) AS zip_rank
            FROM zip_counts
        )
        SELECT chart_type, zip_code, trip_count
        FROM ranked
        WHERE zip_rank <= 10
        ORDER BY chart_type ASC, trip_count DESC, zip_code ASC
        """,
        parameters,
    )


@st.cache_data(show_spinner=False)
def load_trips(
    start_ts: datetime,
    end_ts: datetime,
    pickup_zip: Optional[int],
) -> pd.DataFrame:
    predicate, parameters = _query_parts(start_ts, end_ts, pickup_zip)
    return _run_query(
        f"""
        SELECT *
        FROM {TABLE_NAME}
        WHERE {predicate}
        ORDER BY tpep_pickup_datetime ASC
        LIMIT 10000
        """,
        parameters,
    )


def _as_date(value: Any) -> Optional[date]:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return None


def _parse_zip(value: str) -> tuple[Optional[int], Optional[str]]:
    cleaned = value.strip()
    if not cleaned:
        return None, None
    if len(cleaned) > 10 or not ZIP_PATTERN.fullmatch(cleaned):
        return None, "Pickup ZIP must contain only 1 to 10 ASCII digits."
    return int(cleaned), None


def _friendly_query_error(exc: Exception) -> str:
    message = str(exc).lower()
    if "timeout" in message or "timed out" in message:
        return "The warehouse query timed out. Please retry."
    if "warehouse" in message:
        return (
            "The SQL warehouse is unavailable or incorrectly configured. "
            "Check DATABRICKS_WAREHOUSE_ID and retry."
        )
    if "auth" in message or "credential" in message or "permission" in message:
        return (
            "Databricks authentication or table access failed. "
            "Check the app permissions and retry."
        )
    return "The trip data could not be loaded from Databricks. Please retry."


def _show_error(message: str) -> None:
    st.error(message)
    if st.button("Retry", key="retry_queries"):
        st.cache_data.clear()
        st.rerun()


def _metric_value(value: Any, decimals: Optional[int] = None) -> Any:
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
        number = float(value)
        if decimals is not None:
            return round(number, decimals)
        return number
    except (TypeError, ValueError, OverflowError):
        return "—"


def _chart_frame(chart_data: pd.DataFrame, chart_type: str) -> pd.DataFrame:
    required = {"chart_type", "zip_code", "trip_count"}
    if chart_data.empty or not required.issubset(chart_data.columns):
        return pd.DataFrame(columns=["Trips"])

    result = chart_data.loc[
        chart_data["chart_type"] == chart_type,
        ["zip_code", "trip_count"],
    ].copy()
    if result.empty:
        return pd.DataFrame(columns=["Trips"])

    result["zip_code"] = result["zip_code"].astype("string")
    result = result.rename(
        columns={"zip_code": "ZIP", "trip_count": "Trips"}
    ).set_index("ZIP")
    return result


def main() -> None:
    col1, col2, col3 = st.columns(3)
    raw_start_date = col1.date_input("Start date", key="start_date")
    raw_end_date = col2.date_input("End date", key="end_date")
    raw_zip_filter = col3.text_input(
        "Pickup ZIP (optional)",
        key="zip_filter",
        max_chars=256,
    )

    start_day = _as_date(raw_start_date)
    end_day = _as_date(raw_end_date)

    if start_day is None or end_day is None:
        _show_error("Choose one valid start date and one valid end date.")
        return

    if start_day > end_day:
        _show_error("Start date must be on or before end date.")
        return

    pickup_zip, zip_error = _parse_zip(raw_zip_filter)
    if zip_error:
        _show_error(zip_error)
        return

    start_ts = datetime.combine(start_day, time.min)
    end_ts = datetime.combine(end_day + timedelta(days=1), time.min)

    try:
        with st.spinner("Loading trip data..."):
            kpis = load_kpis(start_ts, end_ts, pickup_zip)
            chart_data = load_zip_charts(start_ts, end_ts, pickup_zip)
            trips = load_trips(start_ts, end_ts, pickup_zip)
    except Exception as exc:
        _show_error(_friendly_query_error(exc))
        return

    expected_kpis = {"total_trips", "avg_fare", "max_fare"}
    if kpis.empty or not expected_kpis.issubset(kpis.columns):
        _show_error("The warehouse returned an unexpected KPI result. Please retry.")
        return

    total_raw = kpis.iloc[0]["total_trips"]
    try:
        total_trips = int(total_raw)
    except (TypeError, ValueError, OverflowError):
        _show_error("The warehouse returned an unexpected KPI result. Please retry.")
        return

    k1, k2, k3 = st.columns(3)
    k1.metric("Total Trips", total_trips)
    k2.metric("Avg Fare", _metric_value(kpis.iloc[0]["avg_fare"], 2))
    k3.metric("Max Fare", _metric_value(kpis.iloc[0]["max_fare"]))

    if total_trips == 0:
        st.info("No trips match the selected filters.")

    pickup_chart = _chart_frame(chart_data, "pickup")
    st.subheader("Top pickup ZIPs")
    if pickup_chart.empty:
        st.caption("No pickup ZIP data is available for these filters.")
    else:
        st.bar_chart(pickup_chart)

    with st.container(key="dropoff_chart"):
        st.subheader("Top dropoff ZIPs")
        dropoff_chart = _chart_frame(chart_data, "dropoff")
        if dropoff_chart.empty:
            st.caption("No dropoff ZIP data is available for these filters.")
        else:
            st.bar_chart(dropoff_chart)

    st.subheader("Trips")
    visible_trips = trips.head(100)
    st.dataframe(
        visible_trips,
        key="trips_table",
        use_container_width=True,
        hide_index=True,
    )

    try:
        csv_data = trips.to_csv(index=False).encode("utf-8")
    except Exception:
        csv_data = b""

    st.download_button(
        "Download CSV",
        data=csv_data,
        file_name="filtered_trips.csv",
        mime="text/csv",
        key="download_csv",
        disabled=not bool(csv_data),
    )


main()
