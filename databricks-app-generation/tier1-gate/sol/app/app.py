import os
from datetime import date, datetime, time, timedelta

import pandas as pd
import streamlit as st
from databricks import sql
from databricks.sdk.core import Config


TABLE_NAME = "samples.nyctaxi.trips"
TRIP_COLUMNS = [
    "tpep_pickup_datetime",
    "trip_distance",
    "fare_amount",
    "pickup_zip",
    "dropoff_zip",
]


def open_connection():
    warehouse_id = os.environ.get("DATABRICKS_WAREHOUSE_ID", "").strip()
    if not warehouse_id:
        raise RuntimeError(
            "DATABRICKS_WAREHOUSE_ID is not configured for this app."
        )

    config = Config()
    if not config.host:
        raise RuntimeError("Databricks workspace authentication is not configured.")

    return sql.connect(
        server_hostname=config.host,
        http_path=f"/sql/1.0/warehouses/{warehouse_id}",
        credentials_provider=lambda: config.authenticate,
    )


def fetch_dataframe(statement, parameters=None):
    connection = None
    cursor = None
    try:
        connection = open_connection()
        cursor = connection.cursor()
        cursor.execute(statement, parameters or {})
        rows = cursor.fetchall()
        columns = [column[0] for column in cursor.description]
        return pd.DataFrame.from_records(rows, columns=columns)
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()


@st.cache_data(show_spinner=False)
def load_full_table_summary():
    statement = f"""
        SELECT
            COUNT(*) AS total_trips,
            ROUND(AVG(fare_amount), 2) AS avg_fare,
            ROUND(AVG(trip_distance), 2) AS avg_distance,
            CAST(MIN(tpep_pickup_datetime) AS DATE) AS min_pickup_date,
            CAST(MAX(tpep_pickup_datetime) AS DATE) AS max_pickup_date
        FROM {TABLE_NAME}
    """
    summary = fetch_dataframe(statement)
    if summary.empty:
        return 0, None, None, None, None

    row = summary.iloc[0]
    return (
        row["total_trips"],
        row["avg_fare"],
        row["avg_distance"],
        row["min_pickup_date"],
        row["max_pickup_date"],
    )


def as_date(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def load_trips(start_timestamp, end_timestamp):
    statement = f"""
        SELECT
            tpep_pickup_datetime,
            trip_distance,
            fare_amount,
            pickup_zip,
            dropoff_zip
        FROM {TABLE_NAME}
        WHERE tpep_pickup_datetime >= :start_ts
          AND tpep_pickup_datetime < :end_ts
        ORDER BY
            tpep_pickup_datetime ASC,
            pickup_zip ASC,
            dropoff_zip ASC
        LIMIT 100
    """
    return fetch_dataframe(
        statement,
        {
            "start_ts": start_timestamp,
            "end_ts": end_timestamp,
        },
    )


def load_zip_counts(start_timestamp, end_timestamp):
    statement = f"""
        SELECT
            pickup_zip,
            COUNT(*) AS trip_count
        FROM {TABLE_NAME}
        WHERE tpep_pickup_datetime >= :start_ts
          AND tpep_pickup_datetime < :end_ts
          AND pickup_zip IS NOT NULL
        GROUP BY pickup_zip
        ORDER BY trip_count DESC, pickup_zip ASC
        LIMIT 10
    """
    return fetch_dataframe(
        statement,
        {
            "start_ts": start_timestamp,
            "end_ts": end_timestamp,
        },
    )


def empty_trips_dataframe():
    return pd.DataFrame(columns=TRIP_COLUMNS)


def main():
    st.set_page_config(page_title="NYC Taxi Explorer", layout="wide")
    st.title("NYC Taxi Explorer")

    try:
        with st.spinner("Loading trip summary..."):
            (
                total_trips,
                avg_fare,
                avg_distance,
                minimum_date,
                maximum_date,
            ) = load_full_table_summary()
    except Exception as exc:
        st.error(f"Unable to load the trip summary: {exc}")
        st.stop()

    metric_columns = st.columns(3)
    with metric_columns[0]:
        st.metric(
            "Total Trips",
            f"{int(total_trips or 0):,}",
        )
    with metric_columns[1]:
        st.metric(
            "Avg Fare",
            "N/A" if avg_fare is None else f"${float(avg_fare):.2f}",
        )
    with metric_columns[2]:
        st.metric(
            "Avg Distance",
            "N/A" if avg_distance is None else f"{float(avg_distance):.2f}",
        )

    minimum_date = as_date(minimum_date)
    maximum_date = as_date(maximum_date)
    today = date.today()
    default_start = minimum_date or today
    default_end = maximum_date or today

    st.subheader("Date range")
    date_columns = st.columns(2)
    with date_columns[0]:
        start_date = st.date_input(
            "Start date",
            value=default_start,
            key="start_date",
        )
    with date_columns[1]:
        end_date = st.date_input(
            "End date",
            value=default_end,
            key="end_date",
        )

    trips = empty_trips_dataframe()
    zip_counts = pd.DataFrame(columns=["pickup_zip", "trip_count"])
    query_succeeded = False

    if start_date > end_date:
        st.warning("Start date must be on or before end date.")
    else:
        try:
            start_timestamp = datetime.combine(start_date, time.min)
            end_timestamp = datetime.combine(end_date, time.min) + timedelta(days=1)

            with st.spinner("Loading trips for the selected dates..."):
                trips = load_trips(start_timestamp, end_timestamp)
                zip_counts = load_zip_counts(start_timestamp, end_timestamp)
            query_succeeded = True
        except OverflowError:
            st.error("The selected end date is outside the supported date range.")
        except Exception as exc:
            st.error(f"Unable to load trips for the selected date range: {exc}")

    with st.container(key="zip_chart"):
        st.subheader("Top 10 Pickup ZIP Codes")
        if query_succeeded and trips.empty:
            st.info("No trips were found in the selected date range.")
        elif query_succeeded and zip_counts.empty:
            st.info("No pickup ZIP code data is available for the selected trips.")
        elif query_succeeded:
            chart_data = zip_counts.copy()
            chart_data["pickup_zip"] = chart_data["pickup_zip"].astype(str)
            st.bar_chart(
                chart_data,
                x="pickup_zip",
                y="trip_count",
                use_container_width=True,
            )

    st.subheader("Trips")
    st.dataframe(
        trips,
        key="trips_table",
        use_container_width=True,
        hide_index=True,
    )


if __name__ == "__main__":
    main()
