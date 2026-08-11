-- Tier 2 ground truth. 채점 시 동일 warehouse에서 재계산. 파라미터:
--   :segments  (배열; 비어 있으면 전체), :start_d, :end_d (반개방 [start, end+1d))
-- revenue = l_extendedprice * (1 - l_discount)

-- gt_kpis
SELECT
  ROUND(SUM(l.l_extendedprice * (1 - l.l_discount)), 2)                        AS total_revenue,
  COUNT(DISTINCT o.o_orderkey)                                                 AS order_count,
  ROUND(100 * SUM(CASE WHEN l.l_returnflag = 'R'
                       THEN l.l_extendedprice * (1 - l.l_discount) END)
            / SUM(l.l_extendedprice * (1 - l.l_discount)), 1)                  AS return_rate_pct
FROM samples.tpch.customer c
JOIN samples.tpch.orders   o ON c.c_custkey = o.o_custkey
JOIN samples.tpch.lineitem l ON o.o_orderkey = l.l_orderkey
WHERE (:segments IS NULL OR c.c_mktsegment IN (:segments))
  AND o.o_orderdate >= :start_d AND o.o_orderdate < :end_d;

-- gt_monthly_revenue (+ MoM % change, months ASC)
WITH monthly AS (
  SELECT date_trunc('MONTH', o.o_orderdate) AS order_month,
         SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue
  FROM samples.tpch.customer c
  JOIN samples.tpch.orders   o ON c.c_custkey = o.o_custkey
  JOIN samples.tpch.lineitem l ON o.o_orderkey = l.l_orderkey
  WHERE (:segments IS NULL OR c.c_mktsegment IN (:segments))
    AND o.o_orderdate >= :start_d AND o.o_orderdate < :end_d
  GROUP BY 1
)
SELECT order_month, ROUND(revenue, 2) AS revenue,
       ROUND(100 * (revenue - LAG(revenue) OVER (ORDER BY order_month))
                 / LAG(revenue) OVER (ORDER BY order_month), 1) AS mom_pct
FROM monthly ORDER BY order_month ASC;

-- gt_segment_revenue (revenue DESC, ties segment ASC)
SELECT c.c_mktsegment, ROUND(SUM(l.l_extendedprice * (1 - l.l_discount)), 2) AS revenue
FROM samples.tpch.customer c
JOIN samples.tpch.orders   o ON c.c_custkey = o.o_custkey
JOIN samples.tpch.lineitem l ON o.o_orderkey = l.l_orderkey
WHERE (:segments IS NULL OR c.c_mktsegment IN (:segments))
  AND o.o_orderdate >= :start_d AND o.o_orderdate < :end_d
GROUP BY 1 ORDER BY revenue DESC, c.c_mktsegment ASC;

-- gt_orders_page (parameters: + :limit, :offset) — 테이블/CSV 검증용
SELECT o.o_orderkey, o.o_orderdate, c.c_mktsegment,
       ROUND(SUM(l.l_extendedprice * (1 - l.l_discount)), 2) AS order_revenue
FROM samples.tpch.customer c
JOIN samples.tpch.orders   o ON c.c_custkey = o.o_custkey
JOIN samples.tpch.lineitem l ON o.o_orderkey = l.l_orderkey
WHERE (:segments IS NULL OR c.c_mktsegment IN (:segments))
  AND o.o_orderdate >= :start_d AND o.o_orderdate < :end_d
GROUP BY o.o_orderkey, o.o_orderdate, c.c_mktsegment
ORDER BY o.o_orderdate ASC, o.o_orderkey ASC
LIMIT :limit OFFSET :offset;

-- gt_date_span — 채점기가 테스트 윈도우 선정용
SELECT MIN(o_orderdate) AS min_d, MAX(o_orderdate) AS max_d FROM samples.tpch.orders;
