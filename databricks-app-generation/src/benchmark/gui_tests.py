#!/usr/bin/env python3
"""
gui_tests.py — Playwright GUI functional tests against a locally booted candidate app.

Selector policy (the "contract compliance vs functional defect" split):
  primary   — the contract's Streamlit key= (rendered as a `st-key-<key>` class) or
              exact st.metric label;
  fallback  — semantic lookup (heading/label text, element role, chart svg/canvas
              presence). A key mismatch is recorded in `contract_misses` and costs
              contract-compliance points only; the functional verdict uses whichever
              selector found the element.

Implemented concretely: tier1 T1–T7 (+T5b), tier2 C1–C9, tier3 fuzzer. st.dataframe
renders a canvas grid, so row-level VALUE checks go through the CSV download (C7);
grids are otherwise verified structurally (presence + aria-rowcount when exposed).
`report.windows` records epoch-ms brackets around interactions so grade_tasks can
audit real query counts via the warehouse Query History API (tier2 E1–E3).
"""
import csv as csvmod
import random
import re
import time
from dataclasses import dataclass, field
from datetime import date, timedelta

APP_URL = "http://localhost:8501"
SETTLE_MS = 1200
# Streamlit runs its script (issuing warehouse SQL) AFTER the HTTP 200 + networkidle
# that a bare goto waits on — widgets stream in over the websocket seconds later. So
# every load/interaction waits for real content to appear and the DOM to stop changing
# before asserting, capped at this budget (cold warehouse queries can take seconds).
APP_READY_TIMEOUT_MS = 45000
_CONTENT_SELECTOR = (
    'h1, h2, [data-testid="stMetric"], [data-testid="stDateInput"], '
    '[data-testid="stMultiSelect"], [data-testid="stDataFrame"], '
    '[data-testid="stVegaLiteChart"], [data-testid="stArrowVegaLiteChart"], '
    '[data-testid="stException"]'
)


def _script_idle(page) -> bool:
    """True when Streamlit's script runner is not running and no spinner is active.
    The app renders in waves as each cached warehouse query returns; element counts
    go stale-then-grow, so counting elements is unreliable. Streamlit exposes the
    real signal: stApp's `data-test-script-state` ('running' while executing) plus the
    per-query stSpinner. We treat 'notRunning' + no spinner as done. If the attribute
    is absent (older Streamlit), fall back to spinner-only."""
    try:
        state = page.evaluate(
            "() => document.querySelector('[data-testid=\"stApp\"]')"
            "?.getAttribute('data-test-script-state')")
    except Exception:
        state = None
    spinners = page.locator('[data-testid="stSpinner"]').count()
    if state is not None:
        return state != "running" and spinners == 0
    return spinners == 0


def wait_for_app(page, timeout_ms: int = APP_READY_TIMEOUT_MS) -> None:
    """Wait for Streamlit's script run to finish. networkidle fires before the app
    script has produced widgets (the warehouse queries run afterward), so wait until
    the script runner reports idle for two consecutive polls, or an exception shows."""
    try:
        page.wait_for_load_state("networkidle")
    except Exception:
        pass
    deadline = time.time() + timeout_ms / 1000.0
    idle_streak = 0
    while time.time() < deadline:
        if page.locator('[data-testid="stException"]').count() > 0:
            return
        idle_streak = idle_streak + 1 if _script_idle(page) else 0
        if idle_streak >= 2 and page.locator(_CONTENT_SELECTOR).count() > 0:
            break
        page.wait_for_timeout(600)
    page.wait_for_timeout(SETTLE_MS)


def _settle(page) -> None:
    """After an interaction that triggers a Streamlit rerun (which re-queries the
    warehouse): wait for the script runner to go idle again, then a fixed settle.

    First wait for the rerun to actually *start*. A widget change schedules the rerun
    over the websocket a beat later, so the script state is still idle for a short
    window right after the interaction — checking idle immediately would return before
    the rerun even began (the classic race where a segment-multiselect rerun clobbers a
    date typed too soon after it). Poll briefly for state=='running'; if it never
    flips (fast/no-op rerun) fall through after a short grace period."""
    # Fast exit if the app is already showing an exception — a crashing app (e.g. one
    # that raises on every rerun) would otherwise make every settle burn its full
    # timeout, turning a rule-based pass into minutes. The crash is captured by the
    # caller's stException check; no need to wait for a rerun that will just re-crash.
    if page.locator('[data-testid="stException"]').count() > 0:
        page.wait_for_timeout(SETTLE_MS)
        return
    started_deadline = time.time() + 5
    while time.time() < started_deadline:
        try:
            state = page.evaluate(
                "() => document.querySelector('[data-testid=\"stApp\"]')"
                "?.getAttribute('data-test-script-state')")
        except Exception:
            state = None
        if state == "running" or page.locator('[data-testid="stSpinner"]').count() > 0:
            break
        if page.locator('[data-testid="stException"]').count() > 0:
            return
        page.wait_for_timeout(200)
    try:
        page.wait_for_load_state("networkidle", timeout=15000)
    except Exception:
        pass
    deadline = time.time() + 20
    idle_streak = 0
    while time.time() < deadline:
        if page.locator('[data-testid="stException"]').count() > 0:
            break
        idle_streak = idle_streak + 1 if _script_idle(page) else 0
        if idle_streak >= 2:
            break
        page.wait_for_timeout(500)
    page.wait_for_timeout(SETTLE_MS)


@dataclass
class GuiReport:
    results: dict = field(default_factory=dict)       # test id -> pass|fail|skip
    notes: dict = field(default_factory=dict)
    contract_misses: list = field(default_factory=list)
    crashes: int = 0
    windows: dict = field(default_factory=dict)       # name -> (start_ms, end_ms)
    timings: dict = field(default_factory=dict)       # name -> seconds

    def record(self, tid: str, ok: bool | None, note: str = ""):
        self.results[tid] = "pass" if ok else ("skip" if ok is None else "fail")
        if note:
            self.notes[tid] = note

    def pass_rate(self) -> float:
        graded = [v for v in self.results.values() if v != "skip"]
        return round(sum(v == "pass" for v in graded) / len(graded), 3) if graded else 0.0


# ------------------------------------------------------------------ primitives ---
def _find_by_key(page, key: str):
    loc = page.locator(f".st-key-{key}")
    return loc if loc.count() > 0 else None


def _find_semantic(page, kind: str, label: str | None = None):
    """Fallback lookup by role/testid/visible text."""
    if kind == "metric" and label:
        m = page.locator('[data-testid="stMetric"]', has_text=label)
        return m if m.count() > 0 else None
    if kind == "date_input":
        d = page.locator('[data-testid="stDateInput"]')
        return d if d.count() > 0 else None
    if kind == "dataframe":
        d = page.locator('[data-testid="stDataFrame"]')
        return d if d.count() > 0 else None
    if kind == "multiselect":
        d = page.locator('[data-testid="stMultiSelect"]')
        return d if d.count() > 0 else None
    if kind == "chart":
        c = page.locator('[data-testid="stVegaLiteChart"], [data-testid="stArrowVegaLiteChart"], .stPlotlyChart, svg.marks, canvas')
        return c if c.count() > 0 else None
    return None


def locate(page, report: GuiReport, key: str | None, kind: str, label: str | None = None):
    """Two-stage selector; records a contract miss when only the fallback hits."""
    if key:
        el = _find_by_key(page, key)
        if el:
            return el
    el = _find_semantic(page, kind, label)
    if el is not None and key:
        report.contract_misses.append(key)
    return el


def metric_value(page, report: GuiReport, label: str) -> str | None:
    m = _find_semantic(page, "metric", label)
    if m is None:
        return None
    v = m.first.locator('[data-testid="stMetricValue"]')
    return v.inner_text().strip() if v.count() else None


def parse_number(text: str | None) -> float | None:
    if not text:
        return None
    m = re.search(r"-?[\d,]+(?:\.\d+)?", text)
    return float(m.group(0).replace(",", "")) if m else None


def _date_input_box(page, report: GuiReport, key: str, nth: int):
    """Locate a st.date_input's <input>, re-resolved fresh (containers go stale across
    reruns). Records a contract miss when the key container is absent and we fall back
    to the nth date input."""
    container = _find_by_key(page, key)
    if container is None:
        allin = page.locator('[data-testid="stDateInput"]')
        if allin.count() <= nth:
            return None
        report.contract_misses.append(key)
        container = allin.nth(nth)
    return container.locator("input").first


def set_date(page, report: GuiReport, key: str, value: date, nth: int = 0) -> bool:
    """Set a st.date_input and VERIFY the value committed, retrying on failure.

    Why the retry+verify: an immediately-preceding interaction (e.g. the segment
    multiselect) triggers a Streamlit rerun; typing into the date input before that
    rerun finishes lets baseweb revert to the default date, so the filter is silently
    dropped and only shows up as a wrong KPI much later. We type the date as real
    keystrokes (press_sequentially — a programmatic fill() is ignored by baseweb and
    reverted on blur), clear first with fill("") (more reliable than Ctrl+A/Delete),
    close the popover with a neutral click (NOT Escape — that discards the value), then
    read the input back and retry if it didn't stick."""
    want = value.strftime("%Y/%m/%d")
    for attempt in range(3):
        # A crashed app (stException on every rerun) can't accept input; don't burn all
        # three attempts × timeouts on it — bail so the case fails fast.
        if page.locator('[data-testid="stException"]').count() > 0:
            return False
        box = _date_input_box(page, report, key, nth)
        if box is None:
            return False
        try:
            box.click(timeout=8000)
            # Select-all then type, rather than fill(""). baseweb's date field is a
            # masked input: fill("") clears unreliably (an existing value like
            # 2016/02/29 can survive), so the typed digits collide with leftover mask
            # chars and the commit fails. Ctrl/Cmd+A selects the whole field so the
            # sequential typing overwrites it cleanly.
            page.keyboard.press("ControlOrMeta+a")
            page.wait_for_timeout(120)
            box.press_sequentially(want, delay=70, timeout=8000)
            box.press("Enter")  # commit
        except Exception:
            _settle(page)
            continue
        # Close the datepicker popover WITHOUT discarding the value; its overlay would
        # otherwise intercept pointer events on the next widget (30s click timeout).
        try:
            page.mouse.click(5, 5)
        except Exception:
            pass
        _settle(page)
        # Verify it stuck (re-resolve — the node is stale after the rerun).
        chk = _date_input_box(page, report, key, nth)
        try:
            if chk is not None and chk.input_value(timeout=3000) == want:
                return True
        except Exception:
            pass
    return False


def has_exception(page) -> bool:
    return page.locator('[data-testid="stException"]').count() > 0


def set_multiselect(page, report: GuiReport, key: str, values: list[str]) -> bool:
    """Select values in a st.multiselect (primary: key container; fallback: first
    multiselect on the page)."""
    container = _find_by_key(page, key)
    if container is None:
        container = _find_semantic(page, "multiselect")
        if container is None:
            return False
        report.contract_misses.append(key)
    container = container.first
    for v in values:
        try:
            container.locator("input").first.click(timeout=3000)
            opt = page.locator('li[role="option"]', has_text=v)
            if not opt.count():
                page.keyboard.press("Escape")
                return False
            opt.first.click(timeout=3000)
        except Exception:
            return False
    page.keyboard.press("Escape")
    _settle(page)
    return True


def _find_key_button(page, report: GuiReport, key: str, fallback_text: str | None):
    """Locate a button by key container (fallback: visible text). Returns the button
    locator or None. Records a contract miss when only the text fallback matched."""
    container = _find_by_key(page, key)
    if container is not None and container.locator("button").count():
        return container.locator("button").first
    if fallback_text:
        cand = page.locator("button", has_text=fallback_text)
        if cand.count():
            report.contract_misses.append(key)
            return cand.first
    return None


def button_disabled(page, report: GuiReport, key: str, fallback_text: str | None = None):
    """True/False if the button exists (disabled state), or None if not found. A
    disabled prev/next at a pagination boundary is CORRECT behavior, not a failure."""
    btn = _find_key_button(page, report, key, fallback_text)
    if btn is None:
        return None
    try:
        return btn.is_disabled(timeout=2000)
    except Exception:
        return None


def click_key_button(page, report: GuiReport, key: str, fallback_text: str | None = None,
                     expect_download: bool = False):
    """Click a button inside a key container (fallback: visible text). Returns the
    Download object when expect_download, else True/False. A disabled button returns
    False/None (callers decide whether that's expected, e.g. prev on page 1)."""
    btn = _find_key_button(page, report, key, fallback_text)
    if btn is None:
        return None if expect_download else False
    try:
        if expect_download:
            with page.expect_download(timeout=20000) as dl:
                btn.click()
            return dl.value
        btn.click(timeout=5000)
        _settle(page)
        return True
    except Exception:
        return None if expect_download else False


def grid_rowcount(page, report: GuiReport, key: str) -> float | None:
    df = locate(page, report, key, "dataframe")
    if df is None:
        return None
    rc = df.first.locator("[aria-rowcount]")
    return parse_number(rc.first.get_attribute("aria-rowcount")) if rc.count() else None


class _Window:
    """Bracket an interaction and store its epoch-ms window on the report."""

    def __init__(self, report: GuiReport, name: str):
        self.report, self.name = report, name

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        t1 = time.time()
        self.report.windows[self.name] = (int(self.t0 * 1000), int(t1 * 1000))
        self.report.timings[self.name] = round(t1 - self.t0, 1)
        return False


# ------------------------------------------------------------------ tier1 T1–T7 ---
def run_tier1(page, gt: dict, report: GuiReport) -> None:
    """gt: precomputed ground truth  {total, avg_fare, avg_dist, window:(s,e),
    window_rows, top_zip} — computed by grade_tasks via warehouse.py, or {} when the
    warehouse is unavailable (numeric checks then skip, structural checks still run)."""
    page.goto(APP_URL, wait_until="domcontentloaded")
    wait_for_app(page)

    # T1 title
    h = page.locator("h1, h2").filter(has_text="NYC Taxi Explorer")
    report.record("T1-title", h.count() > 0)

    # T2–T4 KPIs vs ground truth
    for tid, label, gt_key, tol in (("T2-kpi-count", "Total Trips", "total", 0.5),
                                    ("T3-kpi-fare", "Avg Fare", "avg_fare", 0.01),
                                    ("T4-kpi-distance", "Avg Distance", "avg_dist", 0.01)):
        got = parse_number(metric_value(page, report, label))
        want = gt.get(gt_key)
        if want is None:
            report.record(tid, None, "warehouse unavailable — skipped")
        elif got is None:
            report.record(tid, False, f"metric '{label}' not found")
        else:
            report.record(tid, abs(got - float(want)) <= tol, f"got={got} want={want}")

    # T5 filter window. Bracket the interaction so grade_tasks can audit Query History
    # for T5b (a compliant app re-queries only the date-filtered chart+table here; a
    # full-table KPI aggregation in this window means the KPIs were NOT cached).
    win = gt.get("window")
    if win:
        s, e = win
        with _Window(report, "filter_change"):
            ok_s = set_date(page, report, "start_date", s, nth=0)
            ok_e = set_date(page, report, "end_date", e, nth=1)
        want_rows = gt.get("window_rows")
        # Poll aria-rowcount until the grid reflects the NEW query, not the pre-filter
        # data. set_date's settle waits for the script to go idle, but on a cold
        # warehouse the filtered query can still be in flight when we first read, so the
        # grid briefly shows the old (full-range) rowcount — the root of T5 flakiness.
        # Poll toward the expected count (capped) for up to ~15s; accept the last read
        # if it never converges (structural pass path below still applies).
        expected = min(want_rows, 100) if want_rows is not None else None
        df = locate(page, report, "trips_table", "dataframe")
        rowcount = None
        deadline = time.time() + 15
        while time.time() < deadline:
            df = locate(page, report, "trips_table", "dataframe")
            if df is not None:
                rc = df.first.locator("[aria-rowcount]")
                if rc.count():
                    rowcount = parse_number(rc.first.get_attribute("aria-rowcount"))
            if expected is None or rowcount is None:
                break
            if abs(rowcount - expected) <= 1:
                break
            page.wait_for_timeout(1000)
        if not (ok_s and ok_e) or df is None:
            report.record("T5-filter-updates-table", False, "filter inputs or table missing")
        elif want_rows is not None and rowcount is not None:
            # +1: grid header row is typically included in aria-rowcount
            report.record("T5-filter-updates-table",
                          abs(rowcount - min(want_rows, 100)) <= 1,
                          f"rows={rowcount} want≈{min(want_rows, 100)}")
        else:
            report.record("T5-filter-updates-table", True,
                          "structural pass (row values unverified — aria rowcount unavailable)")

        # T5b KPI cache. Values must not change after filtering (necessary but NOT
        # sufficient — a full-table KPI re-query returns the same numbers). The
        # authoritative check is the Query History audit over the filter_change window
        # above, applied by grade_tasks.tier1_kpi_cache_audit(); it upgrades a
        # value-only pass to a fail if it sees a full-table aggregation there.
        got_total = parse_number(metric_value(page, report, "Total Trips"))
        want_total = gt.get("total")
        if want_total is None or got_total is None:
            report.record("T5b-kpi-not-requeried", None, "unverifiable (value check)")
        elif abs(got_total - float(want_total)) > 0.5:
            report.record("T5b-kpi-not-requeried", False, "KPI value changed after filter")
        else:
            report.record("T5b-kpi-not-requeried", True,
                          "KPI value stable (pending query-history audit)")

        # T6 chart
        chart = locate(page, report, "zip_chart", "chart")
        report.record("T6-chart-present", chart is not None,
                      "" if chart is not None else "no chart element found")
        # OPEN ITEM: verify top-category == gt['top_zip'] by parsing the Vega spec
        # (page.evaluate on the chart's data) — structural presence only for now.
    else:
        for tid in ("T5-filter-updates-table", "T5b-kpi-not-requeried", "T6-chart-present"):
            report.record(tid, None, "warehouse unavailable — skipped")

    # T7 empty state (year 1999 — safely before any NYC taxi sample data)
    set_date(page, report, "start_date", date(1999, 1, 1), nth=0)
    set_date(page, report, "end_date", date(1999, 1, 7), nth=1)
    page.wait_for_timeout(SETTLE_MS)
    report.record("T7-empty-state", not has_exception(page),
                  "unhandled Streamlit exception on empty window" if has_exception(page) else "")


# ------------------------------------------------------------------ tier2 C1–C9 ---
def _check_kpis(page, report: GuiReport, tid: str, want: tuple | None) -> None:
    """want = (total_revenue, order_count, return_rate_pct) or None to skip."""
    if want is None:
        report.record(tid, None, "warehouse unavailable — skipped")
        return
    rev = parse_number(metric_value(page, report, "Total Revenue"))
    cnt = parse_number(metric_value(page, report, "Order Count"))
    rate = parse_number(metric_value(page, report, "Return Rate"))
    if None in (rev, cnt, rate):
        report.record(tid, False, "one or more KPI metrics not found")
        return
    ok = (abs(rev - float(want[0])) <= max(0.5, abs(float(want[0])) * 1e-6)
          and abs(cnt - float(want[1])) <= 0.5
          and abs(rate - float(want[2])) <= 0.15)
    report.record(tid, ok, f"got=({rev},{cnt},{rate}) want={want}")


def run_tier2(page, gt: dict, report: GuiReport) -> None:
    """gt (from grade_tasks.tier2_ground_truth, {} if warehouse unavailable):
      kpis_full, kpis_filtered: (revenue, order_count, return_rate_pct)
      window: (start_date, end_date)   segments: [..]   mom: float|None
      csv_rows: expected downloadable row count (cap 10000)
    """
    with _Window(report, "initial"):
        page.goto(APP_URL, wait_until="domcontentloaded")
        wait_for_app(page)

    h = page.locator("h1, h2").filter(has_text="TPC-H Revenue Explorer")
    report.record("C1-title", h.count() > 0)

    _check_kpis(page, report, "C2-kpi-initial", gt.get("kpis_full"))

    if gt.get("window"):
        s, e = gt["window"]
        # Apply all three filters for the C3 correctness check. This is NOT the E1
        # query-budget window: it makes three separate widget changes (segment, start,
        # end), each its own rerun+requery, so counting it against the "one filter
        # change ≤ 4 queries" cap would triple-count and wrongly fail a compliant app.
        # The E1 budget is measured separately below with a single isolated change.
        with _Window(report, "filter_setup"):
            ok_seg = set_multiselect(page, report, "segment_filter", gt["segments"])
            ok_s = set_date(page, report, "start_date", s, nth=0)
            ok_e = set_date(page, report, "end_date", e, nth=1)
        if not (ok_seg and ok_s and ok_e):
            report.record("C3-filter-combined", False,
                          f"filter widgets missing/unusable (seg={ok_seg} dates={ok_s and ok_e})")
        else:
            _check_kpis(page, report, "C3-filter-combined", gt.get("kpis_filtered"))

        # C4 MoM + trend chart
        mom = parse_number(metric_value(page, report, "MoM Change"))
        trend = locate(page, report, "trend_chart", "chart")
        want_mom = gt.get("mom")
        if trend is None:
            report.record("C4-trend-mom", False, "trend chart not found")
        elif want_mom is None or mom is None:
            report.record("C4-trend-mom", trend is not None,
                          "structural pass (MoM value unverified)")
        else:
            report.record("C4-trend-mom", abs(mom - float(want_mom)) <= 0.15,
                          f"mom got={mom} want={want_mom}")

        seg_chart = locate(page, report, "segment_chart", "chart")
        report.record("C5-segment-chart", seg_chart is not None,
                      "" if seg_chart else "segment chart not found")
        # OPEN ITEM: bar ORDER verification via Vega spec parse — presence only.

        # C6 pagination. Correct scenario (per test_cases): on page 1, prev is EXPECTED
        # to be disabled (boundary) — that is not a failure. Click next to advance (prev
        # becomes enabled), then click prev to return. Only a next that won't advance, or
        # an exception, fails the case.
        rc0 = grid_rowcount(page, report, "orders_table")
        prev_disabled_at_start = button_disabled(page, report, "page_prev", "Prev")
        with _Window(report, "page_change"):
            nxt = click_key_button(page, report, "page_next", fallback_text="Next")
        prv = click_key_button(page, report, "page_prev", fallback_text="Prev")
        if nxt is False and button_disabled(page, report, "page_next", "Next") is None:
            report.record("C6-pagination", False, "page_next button not found")
        elif has_exception(page):
            report.record("C6-pagination", False, "exception while paging")
        elif prev_disabled_at_start is False:
            # prev was clickable on page 1 → pagination boundary not enforced
            report.record("C6-pagination", False,
                          "page_prev was enabled on page 1 (boundary not enforced)")
        else:
            # A single-page result set legitimately leaves next disabled; accept that.
            note = f"page1 rows≈{rc0}" if rc0 is not None else "rowcount unexposed"
            note += f"; next={'clicked' if nxt else 'disabled/1page'}, prev_return={'ok' if prv else 'n/a'}"
            page_ok = rc0 is None or abs(rc0 - 51) <= 1  # 50 rows + header
            report.record("C6-pagination", page_ok, note)

        # C7 CSV download — the row-level VALUE check
        dl = click_key_button(page, report, "download_csv",
                              fallback_text="CSV", expect_download=True)
        if dl is None:
            report.record("C7-csv", False, "download button missing or no download event")
        else:
            try:
                with open(dl.path(), newline="", encoding="utf-8") as f:
                    rows = list(csvmod.reader(f))
                header = [c.strip().lower() for c in rows[0]] if rows else []
                need = {"o_orderkey", "o_orderdate", "c_mktsegment", "order_revenue"}
                cols_ok = need <= set(header)
                want_n = gt.get("csv_rows")
                n_ok = want_n is None or abs((len(rows) - 1) - want_n) <= 1
                report.record("C7-csv", cols_ok and n_ok,
                              f"cols_ok={cols_ok} rows={len(rows)-1} want={want_n}")
            except Exception as ex:  # noqa: BLE001
                report.record("C7-csv", False, f"csv parse failed: {ex}")

        # E1 query-budget probe: measure ONE isolated filter change. The contract caps
        # a single filter change at 4 queries; done last (after all correctness checks)
        # so the 1-day nudge can't disturb C3–C7. grade_tasks reads this window.
        if e:
            with _Window(report, "filter_change"):
                set_date(page, report, "end_date", e - timedelta(days=1), nth=1)
    else:
        for tid in ("C3-filter-combined", "C4-trend-mom", "C5-segment-chart",
                    "C6-pagination", "C7-csv"):
            report.record(tid, None, "warehouse unavailable — skipped")

    # C8 empty state (TPC-H orders start 1992)
    set_date(page, report, "start_date", date(1970, 1, 1), nth=0)
    set_date(page, report, "end_date", date(1970, 1, 7), nth=1)
    page.wait_for_timeout(SETTLE_MS)
    report.record("C8-empty-state", not has_exception(page))

    # C9 injection probe: every text input gets the payload; the app must neither
    # raise nor surface a SQL parser error. (Static interpolation scan complements
    # this in grade_tasks.static_checks.)
    payload = "' OR 1=1 --"
    boxes = page.locator('[data-testid="stTextInput"] input')
    probed = False
    for i in range(min(boxes.count(), 5)):
        try:
            boxes.nth(i).fill(payload, timeout=3000)
            boxes.nth(i).press("Enter")
            page.wait_for_timeout(SETTLE_MS)
            probed = True
        except Exception:
            pass
    sql_err = page.get_by_text(re.compile("SQLSTATE|PARSE_SYNTAX_ERROR|SyntaxError")).count() > 0
    if probed:
        report.record("C9-injection-probe", not has_exception(page) and not sql_err)
    else:
        report.record("C9-injection-probe", None,
                      "no text inputs to probe — rely on the static interpolation scan")


# ---------------------------------------------------------------------- fuzzer ---
def run_fuzzer(page, minutes: float = 5.0, seed: int = 20260807) -> int:
    """tier3 R3: random but SEEDED interactions; returns crash (stException) count."""
    rng = random.Random(seed)
    crashes = 0
    deadline_ms = minutes * 60 * 1000
    elapsed = 0
    inputs = page.locator("input")
    buttons = page.locator("button")
    while elapsed < deadline_ms:
        try:
            action = rng.random()
            if action < 0.4 and inputs.count():
                box = inputs.nth(rng.randrange(inputs.count()))
                box.click(timeout=2000)
                box.fill(rng.choice(["", "1999/13/40", "' OR 1=1 --", "𝕫" * 300,
                                     "2016/02/15", "-1", "10007"]), timeout=2000)
                box.press("Enter", timeout=2000)
            elif buttons.count():
                buttons.nth(rng.randrange(buttons.count())).click(timeout=2000)
        except Exception:
            pass  # unclickable elements are not app crashes
        page.wait_for_timeout(400)
        elapsed += 400 + 2000 * 0  # settle time only; playwright ops add real time
        if has_exception(page):
            crashes += 1
            page.goto(APP_URL, wait_until="networkidle")  # reset and keep fuzzing
    return crashes


# ------------------------------------------------------------------- entrypoint ---
def run_suite_for_tier(tier: str, gt: dict, fuzz_minutes: float = 0.0) -> GuiReport:
    """Open a browser against APP_URL and run the tier's implemented checks."""
    report = GuiReport()
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:  # noqa: BLE001
        report.record("playwright", None, f"unavailable: {e}")
        return report

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as e:  # noqa: BLE001
            report.record("chromium", None, f"launch failed: {e} — uv run playwright install chromium")
            return report
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.on("pageerror", lambda _e: None)
        try:
            if tier == "tier1-gate":
                run_tier1(page, gt, report)
            elif tier == "tier2-core":
                run_tier2(page, gt, report)
            elif tier == "tier3-differentiator":
                page.goto(APP_URL, wait_until="domcontentloaded")
                wait_for_app(page)
                run_tier1(page, gt, report)  # repaired app must satisfy tier1 semantics
                if fuzz_minutes > 0:
                    report.crashes = run_fuzzer(page, minutes=fuzz_minutes)
                    report.record("R3-fuzzer", report.crashes == 0,
                                  f"{report.crashes} crash(es)")
        except Exception as e:  # noqa: BLE001
            report.record("suite", False, f"GUI run aborted: {type(e).__name__}: {e}")
        finally:
            browser.close()
    return report
