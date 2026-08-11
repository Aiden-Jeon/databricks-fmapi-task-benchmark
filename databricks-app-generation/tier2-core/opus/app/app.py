import os
from datetime import date, timedelta

import pandas as pd
import streamlit as st
from databricks import sql
from databricks.sdk.core import Config

st.set_page_config(page_title="TPC-H Revenue Explorer", layout="wide")

PAGE_SIZE = 50
DOWNLOAD_CAP = 10000

VALID_RETURN_FLAG = "R"


def get_connection():
    warehouse_id = os.getenv("DATABRICKS_WAREHOUSE_ID")
    if not warehouse_id:
        raise RuntimeError("DATABRICKS_WAREHOUSE_ID environment variable is not set.")
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
            cols = [c[0] for c in cur.description]
            rows = cur.fetchall()
        return pd.DataFrame(rows, columns=cols)
    finally:
        conn.close()


@st.cache_data(ttl=3600, show_spinner=False)
def get_segments():
    df = run_query(
        "SELECT DISTINCT c_mktsegment FROM samples.tpch.customer "
        "WHERE c_mktsegment IS NOT NULL ORDER BY c_mktsegment"
    )
    return df["c_mktsegment"].tolist()


def build_filter_clause(segments, start_date, end_date, params):
    clauses = [
        "o_orderdate >= :start_date",
        "o_orderdate < :end_excl",
    ]
    params["start_date"] = start_date.isoformat()
    params["end_excl"] = (end_date + timedelta(days=1)).isoformat()

    if segments:
        seg_placeholders = []
        for i, seg in enumerate(segments):
            key = f"seg_{i}"
            seg_placeholders.append(f":{key}")
            params[key] = seg
        clauses.append(f"c_mktsegment IN ({', '.join(seg_placeholders)})")
    return " AND ".join(clauses)


@st.cache_data(ttl=600, show_spinner=False)
def query_kpis(segments, start_date, end_date):
    params = {}
    where = build_filter_clause(segments, start_date, end_date, params)
    q = f"""
        SELECT
            COALESCE(SUM(l_extendedprice * (1 - l_discount)), 0) AS total_revenue,
            COUNT(DISTINCT o_orderkey) AS order_count,
            COALESCE(
                SUM(CASE WHEN l_returnflag = '{VALID_RETURN_FLAG}'
                    THEN l_extendedprice * (1 - l_discount) ELSE 0 END), 0
            ) AS returned_revenue
        FROM samples.tpch.customer
        JOIN samples.tpch.orders ON c_custkey = o_custkey
        JOIN samples.tpch.lineitem ON o_orderkey = l_orderkey
        WHERE {where}
    """
    return run_query(q, params)


@st.cache_data(ttl=600, show_spinner=False)
def query_monthly(segments, start_date, end_date):
    params = {}
    where = build_filter_clause(segments, start_date, end_date, params)
    q = f"""
        SELECT
            date_trunc('MONTH', o_orderdate) AS order_month,
            SUM(l_extendedprice * (1 - l_discount)) AS revenue
        FROM samples.tpch.customer
        JOIN samples.tpch.orders ON c_custkey = o_custkey
        JOIN samples.tpch.lineitem ON o_orderkey = l_orderkey
        WHERE {where}
        GROUP BY date_trunc('MONTH', o_orderdate)
        ORDER BY order_month ASC
    """
    return run_query(q, params)


@st.cache_data(ttl=600, show_spinner=False)
def query_segment_breakdown(segments, start_date, end_date):
    params = {}
    where = build_filter_clause(segments, start_date, end_date, params)
    q = f"""
        SELECT
            c_mktsegment,
            SUM(l_extendedprice * (1 - l_discount)) AS revenue
        FROM samples.tpch.customer
        JOIN samples.tpch.orders ON c_custkey = o_custkey
        JOIN samples.tpch.lineitem ON o_orderkey = l_orderkey
        WHERE {where}
        GROUP BY c_mktsegment
        ORDER BY revenue DESC, c_mktsegment ASC
    """
    return run_query(q, params)


@st.cache_data(ttl=600, show_spinner=False)
def query_orders(segments, start_date, end_date, limit, offset):
    params = {}
    where = build_filter_clause(segments, start_date, end_date, params)
    params["lim"] = int(limit)
    params["off"] = int(offset)
    q = f"""
        SELECT
            o_orderkey,
            o_orderdate,
            c_mktsegment,
            SUM(l_extendedprice * (1 - l_discount)) AS order_revenue
        FROM samples.tpch.customer
        JOIN samples.tpch.orders ON c_custkey = o_custkey
        JOIN samples.tpch.lineitem ON o_orderkey = l_orderkey
        WHERE {where}
        GROUP BY o_orderkey, o_orderdate, c_mktsegment
        ORDER BY o_orderdate ASC, o_orderkey ASC
        LIMIT :lim OFFSET :off
    """
    return run_query(q, params)


@st.cache_data(ttl=600, show_spinner=False)
def query_orders_download(segments, start_date, end_date, cap):
    params = {}
    where = build_filter_clause(segments, start_date, end_date, params)
    params["lim"] = int(cap)
    q = f"""
        SELECT
            o_orderkey,
            o_orderdate,
            c_mktsegment,
            SUM(l_extendedprice * (1 - l_discount)) AS order_revenue
        FROM samples.tpch.customer
        JOIN samples.tpch.orders ON c_custkey = o_custkey
        JOIN samples.tpch.lineitem ON o_orderkey = l_orderkey
        WHERE {where}
        GROUP BY o_orderkey, o_orderdate, c_mktsegment
        ORDER BY o_orderdate ASC, o_orderkey ASC
        LIMIT :lim
    """
    return run_query(q, params)


def main():
    st.title("TPC-H Revenue Explorer")

    try:
        segment_options = get_segments()
    except Exception as e:
        st.error(f"Failed to load market segments: {e}")
        if st.button("Retry", key="retry_segments"):
            st.cache_data.clear()
            st.rerun()
        return

    segments = st.multiselect(
        "Market Segment", options=segment_options, default=[], key="segment_filter"
    )

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date", value=date(1992, 1, 1), key="start_date"
        )
    with col2:
        end_date = st.date_input("End Date", value=date(1998, 12, 31), key="end_date")

    if isinstance(start_date, (list, tuple)):
        start_date = start_date[0]
    if isinstance(end_date, (list, tuple)):
        end_date = end_date[0]

    if start_date > end_date:
        st.error("Start date must be on or before end date.")
        return

    # Reset page when filters change
    filter_signature = (tuple(sorted(segments)), start_date, end_date)
    if st.session_state.get("_filter_sig") != filter_signature:
        st.session_state["_filter_sig"] = filter_signature
        st.session_state["page"] = 0

    if "page" not in st.session_state:
        st.session_state["page"] = 0

    try:
        with st.spinner("Loading KPIs..."):
            kpi_df = query_kpis(segments, start_date, end_date)
    except Exception as e:
        st.error(f"Query failed: {e}")
        if st.button("Retry", key="retry_kpi"):
            st.cache_data.clear()
            st.rerun()
        return

    total_revenue = float(kpi_df["total_revenue"].iloc[0] or 0)
    order_count = int(kpi_df["order_count"].iloc[0] or 0)
    returned_revenue = float(kpi_df["returned_revenue"].iloc[0] or 0)
    return_rate = (returned_revenue / total_revenue * 100) if total_revenue else 0.0

    k1, k2, k3 = st.columns(3)
    k1.metric("Total Revenue", f"{total_revenue:,.2f}")
    k2.metric("Order Count", f"{order_count:,}")
    k3.metric("Return Rate", f"{return_rate:.1f}%")

    if order_count == 0:
        st.info("No orders match the selected filters. Try broadening your selection.")
        return

    # Monthly trend
    try:
        with st.spinner("Loading monthly trend..."):
            monthly_df = query_monthly(segments, start_date, end_date)
    except Exception as e:
        st.error(f"Query failed: {e}")
        if st.button("Retry", key="retry_monthly"):
            st.cache_data.clear()
            st.rerun()
        return

    with st.container(key="trend_chart"):
        st.subheader("Monthly Revenue Trend")
        if monthly_df.empty:
            st.info("No revenue data for the selected filters.")
        else:
            md = monthly_df.copy()
            md["order_month"] = pd.to_datetime(md["order_month"])
            md = md.sort_values("order_month")
            md["revenue"] = md["revenue"].astype(float)
            chart_df = md.set_index("order_month")[["revenue"]]
            st.bar_chart(chart_df)

            if len(md) >= 2:
                latest = md["revenue"].iloc[-1]
                prev = md["revenue"].iloc[-2]
                mom = ((latest - prev) / prev * 100) if prev else 0.0
                st.metric("MoM Change", f"{mom:.1f}%")
            else:
                st.metric("MoM Change", "N/A")

    # Segment breakdown
    try:
        with st.spinner("Loading segment breakdown..."):
            seg_df = query_segment_breakdown(segments, start_date, end_date)
    except Exception as e:
        st.error(f"Query failed: {e}")
        if st.button("Retry", key="retry_segment"):
            st.cache_data.clear()
            st.rerun()
        return

    with st.container(key="segment_chart"):
        st.subheader("Revenue by Market Segment")
        if seg_df.empty:
            st.info("No segment data for the selected filters.")
        else:
            sd = seg_df.copy()
            sd["revenue"] = sd["revenue"].astype(float)
            st.bar_chart(sd.set_index("c_mktsegment")[["revenue"]])

    # Order detail table with pagination
    st.subheader("Order Detail")
    page = st.session_state.get("page", 0)
    offset = page * PAGE_SIZE

    try:
        with st.spinner("Loading orders..."):
            orders_df = query_orders(
                segments, start_date, end_date, PAGE_SIZE, offset
            )
    except Exception as e:
        st.error(f"Query failed: {e}")
        if st.button("Retry", key="retry_orders"):
            st.cache_data.clear()
            st.rerun()
        return

    if orders_df.empty and page > 0:
        st.session_state["page"] = max(0, page - 1)
        st.rerun()

    display_df = orders_df.copy()
    if not display_df.empty:
        display_df["order_revenue"] = display_df["order_revenue"].astype(float).round(2)

    st.dataframe(display_df, key="orders_table", use_container_width=True)

    pcol1, pcol2, pcol3 = st.columns([1, 2, 1])
    with pcol1:
        if st.button("Previous", key="page_prev", disabled=(page == 0)):
            st.session_state["page"] = max(0, page - 1)
            st.rerun()
    with pcol3:
        next_disabled = len(orders_df) < PAGE_SIZE
        if st.button("Next", key="page_next", disabled=next_disabled):
            st.session_state["page"] = page + 1
            st.rerun()
    with pcol2:
        st.markdown(f"**Page {page + 1}**")

    # CSV download
    try:
        dl_df = query_orders_download(segments, start_date, end_date, DOWNLOAD_CAP)
        if not dl_df.empty:
            dl_df = dl_df.copy()
            dl_df["order_revenue"] = dl_df["order_revenue"].astype(float).round(2)
        csv_bytes = dl_df.to_csv(index=False).encode("utf-8")
    except Exception as e:
        st.error(f"Failed to prepare download: {e}")
        csv_bytes = b""

    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name="order_detail.csv",
        mime="text/csv",
        key="download_csv",
    )


if __name__ == "__main__":
    main()
