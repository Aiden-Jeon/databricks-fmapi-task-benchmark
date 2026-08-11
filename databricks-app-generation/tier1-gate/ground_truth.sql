-- Ground truth queries. The grader runs these against the SAME warehouse the
-- candidate apps use, at grading time (never hardcode expected values).

-- gt_total_trips
SELECT COUNT(*) AS total_trips FROM samples.nyctaxi.trips;

-- gt_avg_fare
SELECT ROUND(AVG(fare_amount), 2) AS avg_fare FROM samples.nyctaxi.trips;

-- gt_avg_distance
SELECT ROUND(AVG(trip_distance), 2) AS avg_distance FROM samples.nyctaxi.trips;

-- gt_window_rowcount (parameters: :start_ts, :end_ts)
SELECT COUNT(*) AS window_trips
FROM samples.nyctaxi.trips
WHERE tpep_pickup_datetime >= :start_ts AND tpep_pickup_datetime < :end_ts;

-- gt_top_pickup_zip (parameters: :start_ts, :end_ts) — NULL 제외, 동률은 zip ASC
SELECT pickup_zip, COUNT(*) AS trips
FROM samples.nyctaxi.trips
WHERE tpep_pickup_datetime >= :start_ts AND tpep_pickup_datetime < :end_ts
  AND pickup_zip IS NOT NULL
GROUP BY pickup_zip
ORDER BY trips DESC, pickup_zip ASC
LIMIT 10;

-- gt_window_table (parameters: :start_ts, :end_ts) — 테이블 검증용 정렬 고정
SELECT tpep_pickup_datetime, trip_distance, fare_amount, pickup_zip, dropoff_zip
FROM samples.nyctaxi.trips
WHERE tpep_pickup_datetime >= :start_ts AND tpep_pickup_datetime < :end_ts
ORDER BY tpep_pickup_datetime ASC
LIMIT 100;

-- gt_window_pick: the grader picks a non-empty one-week window for T5/T6 by
-- querying min/max pickup dates first, so the test never depends on a fixed date.
SELECT MIN(tpep_pickup_datetime) AS min_ts, MAX(tpep_pickup_datetime) AS max_ts
FROM samples.nyctaxi.trips;
