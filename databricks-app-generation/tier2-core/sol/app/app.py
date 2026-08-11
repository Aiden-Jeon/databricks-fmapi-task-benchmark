import math
import os
from datetime import date
from decimal import Decimal
from typing import Any

import pandas as pd
import streamlit as st
from databricks import sql
from databricks.sdk.core import Config


st.set_page_config(page_title="TPC-H Revenue Explorer", layout="wide")

PAGE_SIZE = 50
DOWNLOAD_LIMIT = 10_000

AGGREGATE_SQL = """
WITH filtered AS (
    SELECT
        o.o_orderkey AS order_key,
        date_trunc('MONTH', o.o_orderdate) AS order_month,
        c.c_mktsegment AS segment,
        l.l_extendedprice * (1 - l.l_discount) AS revenue,
        l.l_returnflag AS return_flag
    FROM samples.tpch.customer AS c
    INNER JOIN samples.tpch.orders AS o
        ON c.c_custkey = o.o_custkey
    INNER JOIN samples.tpch.lineitem AS l
        ON o.o_orderkey = l.l_orderkey
    WHERE o.o_orderdate >= CAST(:start_date AS DATE)
      AND o.o_orderdate < date_add(CAST(:end_date AS DATE), 1)
      /*SEGMENT_FILTER*/
)
SELECT
    CASE
        WHEN grouping(order_month) = 0 THEN 'MONTH'
        WHEN grouping(segment) = 0 THEN 'SEGMENT'
        ELSE 'TOTAL'
    END AS row_type,
    order_month,
    segment,
    SUM(revenue) AS revenue,
    COUNT(DISTINCT order_key) AS order_count,
    SUM(CASE WHEN return_flag = 'R' THEN revenue ELSE 0 END) AS returned_revenue
FROM filtered
GROUP BY GROUPING SETS (
    (),
    (order_month),
    (segment)
)
"""

ORDER_DETAIL_SQL = """
SELECT
    o.o_orderkey AS o_orderkey,
    o.o_orderdate AS o_orderdate,
    c.c_mktsegment AS c_mktsegment,
    SUM(l.l_extendedprice * (1 - l.l_discount)) AS order_revenue
FROM samples.tpch.customer AS c
INNER JOIN samples.tpch.orders AS o
    ON c.c_custkey = o.o_custkey
INNER JOIN samples.tpch.lineitem AS l
    ON o.o_orderkey = l.l_orderkey
WHERE o.o_orderdate >= CAST(:start_date AS DATE)
  AND o.o_orderdate < date_add(CAST(:end_date AS DATE), 1)
  /*SEGMENT_FILTER*/
GROUP BY
    o.o_orderkey,
    o.o_orderdate,
    c.c_mktsegment
ORDER BY
    o.o_orderdate ASC,
    o.o_orderkey ASC
LIMIT :row_limit OFFSET :row_offset
"""

SEGMENT_SQL = """
SELECT DISTINCT c_mktsegment
FROM samples.tpch.customer
WHERE c_mktsegment IS NOT NULL
ORDER BY c_mktsegment ASC
"""


def get_connection():
    warehouse_id = os.environ.get("DATABRICKS_WAREHOUSE_ID")
    if not warehouse_id:
        raise RuntimeError("DATABRICKS_WAREHOUSE_ID is not configured.")

    config = Config()
    if not config.host:
        raise RuntimeError("Databricks workspace authentication is not configured.")

    return sql.connect(
        server_hostname=config.host,
        http_path="/sql/1.0/warehouses/" + warehouse_id,
        credentials_provider=lambda: config.authenticate,
    )


def run_query(statement: str, parameters: dict[str, Any] | None = None) -> pd.DataFrame:
    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute(statement, parameters=parameters or {})
        rows = cursor.fetchall()
        columns = [description[0].lower() for description in cursor.description]
        return pd.DataFrame.from_records(rows, columns=columns)
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()


def segment_filter_clause(segment_count: int) -> str:
    if segment_count == 0:
        return ""

    markers = [":segment_" + str(index) for index in range(segment_count)]
    return "AND c.c_mktsegment IN (" + ", ".join(markers) + ")"


def query_parameters(
    start_date: date,
    end_date: date,
    segments: tuple[str, ...],
) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "start_date": start_date,
        "end_date": end_date,
    }
    for index, segment in enumerate(segments):
        parameters["segment_" + str(index)] = segment
    return parameters


def apply_segment_clause(statement: str, segment_count: int) -> str:
    return statement.replace(
        "/*SEGMENT_FILTER*/",
        segment_filter_clause(segment_count),
    )


@st.cache_data(show_spinner=False, ttl=3600)
def load_segments(retry_nonce: int) -> tuple[str, ...]:
    del retry_nonce
    frame = run_query(SEGMENT_SQL)
    if frame.empty:
        return ()
    return tuple(str(value) for value in frame["c_mktsegment"].dropna().tolist())


@st.cache_data(show_spinner=False, ttl=600)
def load_aggregates(
    start_date: date,
    end_date: date,
    segments: tuple[str, ...],
    retry_nonce: int,
) -> pd.DataFrame:
    del retry_nonce
    statement = apply_segment_clause(AGGREGATE_SQL, len(segments))
    return run_query(
        statement,
        query_parameters(start_date, end_date, segments),
    )


@st.cache_data(show_spinner=False, ttl=600)
def load_order_page(
    start_date: date,
    end_date: date,
    segments: tuple[str, ...],
    page_number: int,
    retry_nonce: int,
) -> pd.DataFrame:
    del retry_nonce
    statement = apply_segment_clause(ORDER_DETAIL_SQL, len(segments))
    parameters = query_parameters(start_date, end_date, segments)
    parameters["row_limit"] = PAGE_SIZE
    parameters["row_offset"] = page_number * PAGE_SIZE
    return run_query(statement, parameters)


@st.cache_data(show_spinner=False, ttl=600)
def load_download_orders(
    start_date: date,
    end_date: date,
    segments: tuple[str, ...],
    retry_nonce: int,
) -> pd.DataFrame:
    del retry_nonce
    statement = apply_segment_clause(ORDER_DETAIL_SQL, len(segments))
    parameters = query_parameters(start_date, end_date, segments)
    parameters["row_limit"] = DOWNLOAD_LIMIT
    parameters["row_offset"] = 0
    return run_query(statement, parameters)


def numeric_value(value: Any) -> float:
    if value is None or pd.isna(value):
        return 0.0
    if isinstance(value, Decimal):
        return float(value)
    return float(value)


def previous_page() -> None:
    st.session_state.order_page = max(0, st.session_state.order_page - 1)


def next_page() -> None:
    st.session_state.order_page += 1


def retry_queries() -> None:
    st.session_state.retry_nonce += 1


def empty_orders_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "o_orderkey": pd.Series(dtype="int64"),
            "o_orderdate": pd.Series(dtype="datetime64[ns]"),
            "c_mktsegment": pd.Series(dtype="object"),
            "order_revenue": pd.Series(dtype="float64"),
        }
    )


def prepare_order_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return empty_orders_frame()

    prepared = frame[
        ["o_orderkey", "o_orderdate", "c_mktsegment", "order_revenue"]
    ].copy()
    prepared["o_orderdate"] = pd.to_datetime(prepared["o_orderdate"])
    prepared["order_revenue"] = prepared["order_revenue"].map(numeric_value)
    return prepared


def render_app() -> None:
    if "retry_nonce" not in st.session_state:
        st.session_state.retry_nonce = 0
    if "order_page" not in st.session_state:
        st.session_state.order_page = 0
    if "filter_signature" not in st.session_state:
        st.session_state.filter_signature = None

    st.title("TPC-H Revenue Explorer")

    with st.spinner("Loading available market segments..."):
        segment_options = load_segments(st.session_state.retry_nonce)

    selected_segments = st.multiselect(
        "Market segment",
        options=list(segment_options),
        key="segment_filter",
        help="Leave empty to include all market segments.",
    )

    date_columns = st.columns(2)
    with date_columns[0]:
        selected_start = st.date_input(
            "Start date",
            value=date(1992, 1, 1),
            key="start_date",
        )
    with date_columns[1]:
        selected_end = st.date_input(
            "End date",
            value=date(1998, 12, 31),
            key="end_date",
        )

    if selected_start > selected_end:
        st.error("Start date must be on or before end date.")
        return

    invalid_segments = [
        segment for segment in selected_segments if segment not in segment_options
    ]
    if invalid_segments:
        st.error("One or more selected market segments are invalid.")
        return

    segments = tuple(sorted(selected_segments))
    filter_signature = (
        selected_start.isoformat(),
        selected_end.isoformat(),
        segments,
    )
    if st.session_state.filter_signature != filter_signature:
        st.session_state.filter_signature = filter_signature
        st.session_state.order_page = 0

    with st.spinner("Querying revenue analytics..."):
        aggregate_frame = load_aggregates(
            selected_start,
            selected_end,
            segments,
            st.session_state.retry_nonce,
        )

    total_rows = aggregate_frame[aggregate_frame["row_type"] == "TOTAL"]
    if total_rows.empty:
        total_revenue = 0.0
        order_count = 0
        returned_revenue = 0.0
    else:
        total_row = total_rows.iloc[0]
        total_revenue = numeric_value(total_row["revenue"])
        order_count = int(numeric_value(total_row["order_count"]))
        returned_revenue = numeric_value(total_row["returned_revenue"])

    return_rate = returned_revenue / total_revenue if total_revenue else 0.0
    page_count = max(1, math.ceil(order_count / PAGE_SIZE))
    st.session_state.order_page = min(
        st.session_state.order_page,
        page_count - 1,
    )

    with st.spinner("Loading order details..."):
        page_frame = load_order_page(
            selected_start,
            selected_end,
            segments,
            st.session_state.order_page,
            st.session_state.retry_nonce,
        )
        if order_count:
            download_frame = load_download_orders(
                selected_start,
                selected_end,
                segments,
                st.session_state.retry_nonce,
            )
        else:
            download_frame = empty_orders_frame()

    metric_columns = st.columns(3)
    metric_columns[0].metric("Total Revenue", "${:,.2f}".format(total_revenue))
    metric_columns[1].metric("Order Count", "{:,}".format(order_count))
    metric_columns[2].metric("Return Rate", "{:.1f}%".format(return_rate * 100))

    if order_count == 0:
        st.info("No orders match the selected filters.")

    monthly = aggregate_frame[aggregate_frame["row_type"] == "MONTH"].copy()
    if not monthly.empty:
        monthly["order_month"] = pd.to_datetime(monthly["order_month"])
        monthly["revenue"] = monthly["revenue"].map(numeric_value)
        monthly = monthly.sort_values("order_month", ascending=True)
        monthly["mom_change"] = monthly["revenue"].pct_change()

    with st.container(key="trend_chart"):
        st.subheader("Monthly Revenue Trend")
        if monthly.empty:
            st.info("No monthly revenue is available for the selected filters.")
            mom_display = "N/A"
        else:
            st.line_chart(
                monthly[["order_month", "revenue"]],
                x="order_month",
                y="revenue",
                x_label="Order month",
                y_label="Revenue",
            )
            current_month = pd.Timestamp(date.today().replace(day=1))
            complete_months = monthly[monthly["order_month"] < current_month]
            if complete_months.empty:
                mom_display = "N/A"
            else:
                latest_change = complete_months.iloc[-1]["mom_change"]
                if pd.isna(latest_change) or not math.isfinite(float(latest_change)):
                    mom_display = "N/A"
                else:
                    mom_display = "{:.1f}%".format(float(latest_change) * 100)
        st.metric("MoM Change", mom_display)

    segment_frame = aggregate_frame[
        aggregate_frame["row_type"] == "SEGMENT"
    ].copy()
    if not segment_frame.empty:
        segment_frame["segment"] = segment_frame["segment"].astype(str)
        segment_frame["revenue"] = segment_frame["revenue"].map(numeric_value)
        segment_frame = segment_frame.sort_values(
            ["revenue", "segment"],
            ascending=[False, True],
        )

    with st.container(key="segment_chart"):
        st.subheader("Revenue by Market Segment")
        if segment_frame.empty:
            st.info("No segment revenue is available for the selected filters.")
        else:
            st.bar_chart(
                segment_frame[["segment", "revenue"]],
                x="segment",
                y="revenue",
                x_label="Market segment",
                y_label="Revenue",
                sort=False,
            )

    st.subheader("Order Details")
    prepared_page = prepare_order_frame(page_frame)
    st.dataframe(
        prepared_page,
        key="orders_table",
        hide_index=True,
        use_container_width=True,
        column_config={
            "o_orderkey": st.column_config.NumberColumn(
                "o_orderkey",
                format="%d",
            ),
            "o_orderdate": st.column_config.DateColumn(
                "o_orderdate",
                format="YYYY-MM-DD",
            ),
            "c_mktsegment": st.column_config.TextColumn("c_mktsegment"),
            "order_revenue": st.column_config.NumberColumn(
                "order_revenue",
                format="$%.2f",
            ),
        },
    )

    navigation_columns = st.columns([1, 1, 3])
    navigation_columns[0].button(
        "Previous",
        key="page_prev",
        disabled=st.session_state.order_page == 0,
        on_click=previous_page,
        use_container_width=True,
    )
    navigation_columns[1].button(
        "Next",
        key="page_next",
        disabled=(
            order_count == 0
            or st.session_state.order_page >= page_count - 1
        ),
        on_click=next_page,
        use_container_width=True,
    )
    navigation_columns[2].markdown(
        "Page **{}** of **{}**".format(
            st.session_state.order_page + 1,
            page_count,
        )
    )

    prepared_download = prepare_order_frame(download_frame)
    csv_data = prepared_download.to_csv(
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.2f",
    )
    st.download_button(
        "Download order details as CSV",
        data=csv_data,
        file_name="tpch_order_details.csv",
        mime="text/csv",
        key="download_csv",
        on_click="ignore",
    )
    if order_count > DOWNLOAD_LIMIT:
        st.caption(
            "The CSV download is capped at the first {:,} filtered orders.".format(
                DOWNLOAD_LIMIT
            )
        )


try:
    render_app()
except Exception:
    st.error(
        "We couldn't query the revenue data. Check the warehouse connection and try again."
    )
    st.button(
        "Retry",
        key="retry_queries",
        on_click=retry_queries,
    )
