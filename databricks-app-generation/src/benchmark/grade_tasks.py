#!/usr/bin/env python3
"""
grade_tasks.py — Grade every candidate of the suite with an IDENTICAL pipeline,
aggregate tier scores under suite.json's weights + gate rule, apply the efficiency
axis (time + cost), and build the human-review gallery.

Per candidate per tier (Phase A):
  A1 static   — files/app.yaml/py_compile/hardcoding scan  (+ contract_compliance,
                reported separately: key spellings never gate functional scores)
  A2 boot     — venv install + launch via app.yaml + HTTP health   (app_runner)
  A3 gui      — Playwright functional tests vs recomputed ground truth (gui_tests)
  A4 deploy   — tier1 only, real `databricks apps deploy` + smoke   (deploy)
  tier3 extra — robustness scenarios (R1 fault boot, R3 fuzzer)

Suite roll-up:
  suite_quality = Σ tier_weight · tier_auto_score,  gated on tier1 ≥ 0.5
  final_score   = 0.70·quality + 0.15·time + 0.15·cost   (0 if quality is 0)

Usage:
  grade-task                              # all tiers, all candidates
  grade-task --tier tier1-gate --candidates opus glm
  grade-task --no-deploy --no-gui         # fast static+boot pass
  grade-task --blind                      # anonymized gallery
  grade-task --merge-human databricks-app-generation/gallery/human_scores.json
"""
import argparse
import ast
import json
import re
from datetime import timedelta
from pathlib import Path

import yaml

from benchmark import app_runner, gui_tests, task_spec, warehouse
from benchmark.warehouse import WarehouseUnavailable

RESERVED = {"gallery", "__pycache__", "legacy_app", ".graderenv", "screenshots"}

# Contract keys checked for compliance (per tier). Missing keys are a compliance
# miss, NOT a functional failure (gui_tests falls back to semantic selectors).
CONTRACT_KEYS = {
    "tier1-gate": ["start_date", "end_date", "zip_chart", "trips_table"],
    "tier2-core": ["segment_filter", "start_date", "end_date", "trend_chart",
                   "segment_chart", "orders_table", "page_prev", "page_next",
                   "download_csv"],
    "tier3-differentiator": ["start_date", "end_date", "zip_filter", "trips_table",
                             "dropoff_chart", "download_csv"],
}


# ---------------------------------------------------------------- discovery ---
def discover_candidates(suite: str, tier: str, filter_names) -> list[dict]:
    tdir = task_spec.tier_dir(tier, suite)
    if not tdir.exists():
        return []
    out = []
    for d in sorted(tdir.iterdir()):
        if not d.is_dir() or d.name in RESERVED:
            continue
        if not (d / task_spec.ARTIFACT_DIR).exists() and not (d / "run_meta.json").exists():
            continue
        if filter_names and d.name not in filter_names:
            continue
        meta = {}
        mp = d / "run_meta.json"
        if mp.exists():
            try:
                meta = json.loads(mp.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        out.append({"dir": d, "name": d.name, "meta": meta})
    return out


# ------------------------------------------------------------- A1: static -----
def static_checks(workdir: Path, cfg: dict, tier: str) -> dict:
    app = workdir / task_spec.ARTIFACT_DIR
    sc = cfg.get("static_checks", {})
    out = {"files_ok": False, "yaml_ok": False, "compiles": False, "reqs_ok": False,
           "hardcoding_hits": [], "env_pattern_ok": False,
           "contract_hits": 0, "contract_total": 0, "score": 0.0, "notes": []}

    required = sc.get("required_files", [f"app/{f}" for f in task_spec.REQUIRED_FILES])
    missing = [f for f in required if not (workdir / f).exists()]
    out["files_ok"] = not missing
    if missing:
        out["notes"].append(f"missing files: {', '.join(missing)}")
        return out  # nothing else is checkable

    # app.yaml
    try:
        y = yaml.safe_load((app / "app.yaml").read_text(encoding="utf-8")) or {}
        out["yaml_ok"] = all(k in y for k in sc.get("app_yaml_must_have", ["command"]))
    except yaml.YAMLError as e:
        out["notes"].append(f"app.yaml parse error: {e}")

    # python compiles
    try:
        ast.parse((app / "app.py").read_text(encoding="utf-8"))
        out["compiles"] = True
    except SyntaxError as e:
        out["notes"].append(f"app.py syntax error: line {e.lineno}")

    # requirements mention a SQL client
    reqs = (app / "requirements.txt").read_text(encoding="utf-8").lower()
    out["reqs_ok"] = any(m in reqs for m in
                         sc.get("requirements_must_match_any", []))

    # hardcoding scan (user-visible defect: breaks deploy portability)
    source_blob = "\n".join(
        f.read_text(encoding="utf-8", errors="replace")
        for f in app.rglob("*") if f.is_file() and f.suffix in (".py", ".yaml", ".yml", ".txt"))
    for pat in sc.get("forbidden_patterns", []):
        if re.search(pat, source_blob):
            out["hardcoding_hits"].append(pat)
    out["env_pattern_ok"] = all(re.search(p, source_blob)
                                for p in sc.get("required_patterns", []))

    # contract compliance (reported separately; small weight inside static)
    keys = CONTRACT_KEYS.get(tier, [])
    out["contract_total"] = len(keys)
    out["contract_hits"] = sum(1 for k in keys
                               if re.search(rf"key\s*=\s*[\"']{re.escape(k)}[\"']", source_blob))

    # SQL-interpolation heuristic (C9/D1 static complement): f-strings or %/+
    # concatenation that build SQL text. Heuristic — reported, and used as a
    # functional signal only where the tier config asks for it.
    out["sql_interpolation_hits"] = len(re.findall(
        r"f[\"'][^\n]*\b(?:SELECT|FROM|WHERE|AND|OR)\b[^\n]*\{", source_blob,
        re.IGNORECASE))
    if out["sql_interpolation_hits"]:
        out["notes"].append(
            f"possible SQL string interpolation ({out['sql_interpolation_hits']} site(s))")

    hard = [out["files_ok"], out["yaml_ok"], out["compiles"], out["reqs_ok"],
            not out["hardcoding_hits"], out["env_pattern_ok"]]
    hard_score = sum(hard) / len(hard)
    w_cc = cfg.get("contract_compliance", {}).get("weight_in_static", 0.3)
    cc = out["contract_hits"] / out["contract_total"] if out["contract_total"] else 1.0
    out["score"] = round((1 - w_cc) * hard_score + w_cc * cc, 3)
    if out["hardcoding_hits"]:
        out["notes"].append(f"hardcoded host/token/warehouse ({len(out['hardcoding_hits'])} pattern hit(s))")
    return out


# ----------------------------------------------------- ground truth (tier1) ---
def tier1_ground_truth(suite: str, tier: str) -> dict:
    """Recompute GT at grading time. Empty dict when the warehouse is unavailable
    (numeric GUI checks then skip; structural checks still run)."""
    try:
        q = task_spec.load_ground_truth_queries(tier, suite)
        total = warehouse.scalar(q, "gt_total_trips")
        avg_fare = warehouse.scalar(q, "gt_avg_fare")
        avg_dist = warehouse.scalar(q, "gt_avg_distance")
        span = warehouse.run_gt(q, "gt_window_pick")[0]
        min_ts, max_ts = span[0], span[1]
        mid = min_ts + (max_ts - min_ts) / 2
        start = mid.date() if hasattr(mid, "date") else mid
        end = start + timedelta(days=6)
        params = {"start_ts": str(start), "end_ts": str(end + timedelta(days=1))}
        window_rows = warehouse.scalar(q, "gt_window_rowcount", params)
        top = warehouse.run_gt(q, "gt_top_pickup_zip", params)
        return {"total": total, "avg_fare": avg_fare, "avg_dist": avg_dist,
                "window": (start, end), "window_rows": window_rows,
                "top_zip": top[0][0] if top else None}
    except WarehouseUnavailable as e:
        print(f"[grade] warehouse unavailable — numeric checks will skip ({e})")
        return {}
    except Exception as e:  # noqa: BLE001
        print(f"[grade] ground-truth computation failed ({type(e).__name__}: {e})")
        return {}


_SEG_TOKEN = re.compile(r"\(:segments IS NULL OR c\.c_mktsegment IN \(:segments\)\)")


def _with_segments(sql: str, segments: list[str] | None) -> str:
    """The tier2 GT queries document the segment filter as a pseudo-parameter;
    array binding is not portable across connector versions, so the grader
    substitutes a validated literal list (grader-controlled values only)."""
    if segments:
        for s in segments:
            if not re.fullmatch(r"[A-Z ]+", s):
                raise ValueError(f"unsafe segment literal: {s!r}")
        clause = "c.c_mktsegment IN (" + ", ".join(f"'{s}'" for s in segments) + ")"
    else:
        clause = "1=1"
    return _SEG_TOKEN.sub(clause, sql)


def tier2_ground_truth(suite: str) -> dict:
    """Recompute tier2 GT: full-span KPIs, a 6-month window × 2 segments, monthly
    trend MoM, and the expected CSV row count."""
    try:
        q = task_spec.load_ground_truth_queries("tier2-core", suite)
        span = warehouse.run(_with_segments(q["gt_date_span"], None))[0]
        min_d, max_d = span[0], span[1]
        mid = min_d + (max_d - min_d) / 2
        start = mid.replace(day=1)
        end = (start + timedelta(days=185)).replace(day=1) - timedelta(days=1)
        segments = ["BUILDING", "MACHINERY"]
        full = {"start_d": str(min_d), "end_d": str(max_d + timedelta(days=1))}
        win = {"start_d": str(start), "end_d": str(end + timedelta(days=1))}

        kpis_full = warehouse.run(_with_segments(q["gt_kpis"], None), full)[0]
        kpis_filtered = warehouse.run(_with_segments(q["gt_kpis"], segments), win)[0]
        monthly = warehouse.run(_with_segments(q["gt_monthly_revenue"], segments), win)
        mom = monthly[-1][2] if monthly and monthly[-1][2] is not None else None
        return {
            "kpis_full": tuple(kpis_full), "kpis_filtered": tuple(kpis_filtered),
            "window": (start, end), "segments": segments, "mom": mom,
            "csv_rows": min(int(kpis_filtered[1]), 10000),
        }
    except WarehouseUnavailable as e:
        print(f"[grade] warehouse unavailable — tier2 numeric checks will skip ({e})")
        return {}
    except Exception as e:  # noqa: BLE001
        print(f"[grade] tier2 ground truth failed ({type(e).__name__}: {e})")
        return {}


def tier1_kpi_cache_audit(report) -> str | None:
    """T5b authoritative check via Query History over the gui's filter_change window.

    The contract (test_cases T5b): changing the date filter must re-query only the
    chart + trips table, NOT the full-table KPI aggregation (KPIs are cached). Value
    stability alone can't prove this — a re-query returns the same numbers. Here we
    inspect the queries actually issued during the filter interaction: a full-table KPI
    query is one with an AVG aggregate and NO date-filter predicate (the KPI queries are
    the only ones using AVG; chart/table queries carry a tpep_pickup_datetime bound).

    Mutates report.results['T5b-kpi-not-requeried'] in place. Returns a note, or None
    when the audit could not run (warehouse/history unavailable) — the value-based
    verdict then stands, never a vacuous pass. Precondition: a DEDICATED grading
    warehouse (README) so history in the window is this app's traffic only."""
    if "filter_change" not in report.windows:
        return None
    if report.results.get("T5b-kpi-not-requeried") == "fail":
        return None  # value already changed — definitive fail, no need to audit
    try:
        from benchmark import query_audit
        w = query_audit.AuditWindow()
        w.start_ms, w.end_ms = report.windows["filter_change"]
        qs = w.queries()
    except WarehouseUnavailable:
        return "T5b: query-history audit skipped (warehouse/history unavailable) — value check stands"
    except Exception as e:  # noqa: BLE001
        return f"T5b: query-history audit failed ({type(e).__name__}: {e}) — value check stands"
    kpi_requeries = 0
    for q in qs:
        text = (q.get("query_text") or "").lower()
        if "avg(" in text and "tpep_pickup_datetime" not in text:
            kpi_requeries += 1
    ok = kpi_requeries == 0
    report.record("T5b-kpi-not-requeried", ok,
                  f"{kpi_requeries} full-table KPI re-quer(ies) in filter window"
                  if not ok else "no KPI re-query in filter window (audited)")
    return (f"T5b: {kpi_requeries} full-table KPI re-query in filter window — KPIs not cached"
            if not ok else "T5b: KPI cache verified via query history")


def tier2_engineering(report, st: dict) -> tuple[float | None, list[str]]:
    """E1 query budgets + E2 caching + E3 boundedness via Query History over the
    interaction windows recorded by gui_tests; E4 latency from Playwright timing.
    Returns (score or None if nothing evaluable, notes)."""
    notes, checks = [], []
    try:
        from benchmark import query_audit
        budgets = {"initial": 6, "filter_change": 4, "page_change": 1}
        windows = {}
        for name, cap in budgets.items():
            if name not in report.windows:
                continue
            w = query_audit.AuditWindow()
            w.start_ms, w.end_ms = report.windows[name]
            qs = w.queries()
            windows[name] = qs
            ok = len(qs) <= cap
            checks.append(ok)
            notes.append(f"E1[{name}]: {len(qs)} queries (cap {cap}) {'OK' if ok else 'OVER'}")
        if windows:  # E2/E3 only when real audit data exists — never vacuous passes
            # E2 caching: the distinct-segment lookup must not repeat across windows
            seg_lookups = sum(
                1 for qs in windows.values() for qq in qs
                if "distinct" in (qq.get("query_text") or "").lower()
                and "mktsegment" in (qq.get("query_text") or "").lower())
            checks.append(seg_lookups <= 1)
            notes.append(f"E2: {seg_lookups} distinct-segment lookup(s)")
            # E3 boundedness over all audited queries
            unbounded = 0
            for qs in windows.values():
                for qq in qs:
                    text = (qq.get("query_text") or "").lower()
                    if any(t in text for t in ("lineitem", "orders")) \
                            and "limit" not in text \
                            and not any(f in text for f in
                                        ("count(", "sum(", "avg(", "group by")):
                        unbounded += 1
            checks.append(unbounded == 0)
            notes.append(f"E3: {unbounded} unbounded big-table quer(ies)")
    except WarehouseUnavailable:
        notes.append("E1–E3 skipped (query history unavailable)")
    except Exception as e:  # noqa: BLE001
        notes.append(f"E1–E3 audit failed ({type(e).__name__}: {e})")
    # E4 latency (measured regardless of warehouse availability)
    lat = report.timings.get("filter_change")
    if lat is not None:
        checks.append(lat < 15)
        notes.append(f"E4: filter change {lat}s (cap 15s)")
    # static complement: interpolation sites count against engineering, not gui
    if st.get("sql_interpolation_hits"):
        checks.append(False)
        notes.append("E-inj: SQL string interpolation found in source")
    if not checks:
        return None, notes
    return round(sum(checks) / len(checks), 3), notes


# ------------------------------------------------------------ per-candidate ---
def snap(page_url: str, out_path: Path) -> bool:
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page(viewport={"width": 1440, "height": 900})
            pg.goto(page_url, wait_until="networkidle")
            pg.wait_for_timeout(1500)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pg.screenshot(path=str(out_path), full_page=True)
            b.close()
        return True
    except Exception:
        return False


def grade_one(suite: str, tier: str, cand: dict, cfg: dict, gt: dict,
              do_gui: bool, do_deploy: bool, fuzz_minutes: float) -> dict:
    d = cand["dir"]
    meta = cand["meta"]
    app = d / task_spec.ARTIFACT_DIR
    weights = cfg.get("weights", {})
    row = {
        "tier": tier, "candidate": cand["name"],
        "harness": meta.get("harness", "-"),
        "model": meta.get("effective_model") or meta.get("model") or "-",
        "timed_out": meta.get("timed_out", False),
        "wall_seconds": meta.get("wall_seconds"),
        "static": None, "boot": None, "gui": None, "deploy": None,
        "engineering": None, "robustness": None, "contract_compliance": None,
        "gui_results": {}, "auto_score": 0.0, "screenshots": [], "note": "",
    }
    notes = []

    if meta.get("timed_out"):
        notes.append("run timed out — tier scored 0 (suite rule)")
        row["note"] = "; ".join(notes)
        return row

    # A1 static
    st = static_checks(d, cfg, tier)
    row["static"] = st["score"]
    row["contract_compliance"] = (round(st["contract_hits"] / st["contract_total"], 2)
                                  if st["contract_total"] else None)
    notes += st["notes"]
    if not st["files_ok"]:
        row["note"] = "; ".join(notes)
        return row  # gate within the tier: nothing to boot

    # A2 boot
    boot_cfg = cfg.get("boot", {})
    br = app_runner.boot(app, boot_cfg.get("health_url", app_runner.DEFAULT_HEALTH),
                         int(boot_cfg.get("timeout_seconds", 60)))
    row["boot"] = 1.0 if br.booted else 0.0
    if not br.booted:
        notes.append(f"boot failed: {br.note}")

    # A3 gui (+ tier2 engineering audit, tier3 fuzzer/robustness)
    gui_score = 0.0
    engineering = None
    if br.booted and do_gui:
        report = gui_tests.run_suite_for_tier(tier, gt, fuzz_minutes=fuzz_minutes
                                              if tier == "tier3-differentiator" else 0.0)
        # T5b authoritative check: audit Query History for a full-table KPI re-query in
        # the filter_change window (value-stability alone can't detect a re-query).
        if tier == "tier1-gate":
            t5b_note = tier1_kpi_cache_audit(report)
            if t5b_note:
                notes.append(t5b_note)
        row["gui_results"] = report.results
        gui_score = report.pass_rate()   # after the audit may have flipped T5b
        if report.contract_misses:
            notes.append(f"contract misses (fallback used): {sorted(set(report.contract_misses))}")
        shot = d / "screenshots" / "app_home.png"
        if snap(app_runner.DEFAULT_HEALTH, shot):
            row["screenshots"].append(f"screenshots/{shot.name}")
        if tier == "tier2-core":
            engineering, eng_notes = tier2_engineering(report, st)
            notes += eng_notes
        # tier3 robustness R1: reboot with a bad warehouse id — must still render.
        # R2 (mid-session timeout) needs a local SQL proxy — OPEN ITEM; robustness is
        # renormalized over R1+R3 so no candidate is penalized for our missing harness.
        if tier == "tier3-differentiator":
            br.stop()
            br1 = app_runner.boot(app, timeout_s=60, fault_bad_warehouse=True)
            r1_ok = br1.booted and not br1.crashed
            br1.stop()
            crashes_ok = report.crashes == 0
            row["robustness"] = round((0.3 * r1_ok + 0.4 * crashes_ok) / 0.7, 3)
            notes.append("R2 skipped (OPEN ITEM) — robustness renormalized over R1+R3")
    elif br.booted:
        notes.append("gui skipped (--no-gui)")
    row["gui"] = gui_score
    row["engineering"] = engineering
    br.stop()

    # A4 deploy (tier1 only)
    if tier == "tier1-gate" and cfg.get("deploy", {}).get("app_name_pattern") and do_deploy \
            and br.install_ok and row["boot"]:
        from benchmark import deploy as deploy_mod
        dr = deploy_mod.deploy_and_verify(app, cand["name"])
        row["deploy"] = round(0.5 * dr.running + 0.5 * dr.url_ok, 2)
        if dr.note:
            notes.append(f"deploy: {dr.note}")
        if not dr.stopped:
            notes.append(f"WARNING: app {dr.app_name} may still be running — stop it manually!")
    elif tier == "tier1-gate":
        row["deploy"] = None
        notes.append("deploy skipped")

    # tier auto_score under the tier's declared weights (None components -> weight
    # renormalized over what actually ran, noted for transparency)
    comp = {"static": row["static"], "boot": row["boot"], "gui_tests": row["gui"],
            "deploy": row["deploy"], "engineering": row.get("engineering"),
            "repair": row["gui"] if tier == "tier3-differentiator" else None,
            "extend": None,   # OPEN ITEM: X1/X2 need tier3-specific GT wiring
            "robustness": row["robustness"]}
    num, den = 0.0, 0.0
    for k, w in weights.items():
        v = comp.get(k)
        if v is not None:
            num += w * v
            den += w
    row["auto_score"] = round(num / den, 3) if den else 0.0
    if den < sum(weights.values()) - 1e-9:
        skipped = [k for k, w in weights.items() if comp.get(k) is None]
        notes.append(f"renormalized over available components (skipped: {', '.join(skipped)})")
    row["note"] = "; ".join(notes)
    return row


# -------------------------------------------------------- suite aggregation ---
def tokens_cost_usd(meta: dict, pricing: dict) -> float | None:
    model = meta.get("effective_model") or meta.get("model")
    p = (pricing.get("models") or {}).get(model or "", {})
    if not p or p.get("input") is None or meta.get("prompt_tokens") is None:
        return None
    return round((meta["prompt_tokens"] * p["input"]
                  + (meta.get("completion_tokens") or 0) * p["output"]) / 1e6, 4)


def aggregate_suite(suite: str, all_rows: list[dict], suite_cfg: dict, pricing: dict) -> list[dict]:
    tiers = {t["dir"]: t for t in suite_cfg.get("tiers", [])}
    eff = suite_cfg.get("efficiency", {})
    budget_total = sum(t.get("budget_minutes", 0) for t in tiers.values())
    by_cand: dict[str, dict] = {}
    for r in all_rows:
        by_cand.setdefault(r["candidate"], {})[r["tier"]] = r

    out = []
    for cand, tier_rows in sorted(by_cand.items()):
        quality, elapsed_min, cost, cost_known = 0.0, 0.0, 0.0, True
        for tdir, tcfg in tiers.items():
            r = tier_rows.get(tdir)
            score = r["auto_score"] if r else 0.0
            quality += tcfg["weight"] * score
            if r:
                w = r.get("wall_seconds") or 0
                cap = tcfg.get("budget_minutes", 0) * 60
                elapsed_min += min(w, cap) / 60 if r.get("timed_out") else w / 60
                # cost needs run_meta — carried on the row at grade time
                c = r.get("_cost_usd")
                if c is None:
                    cost_known = False
                else:
                    cost += c
        gate = tier_rows.get("tier1-gate")
        gated = gate is None or gate["auto_score"] < 0.5
        if gated:
            quality = 0.0
        t_score = max(0.0, 1 - elapsed_min / budget_total) if budget_total else 0.0
        c_score = (max(0.0, 1 - cost / eff.get("reference_cost_usd", 5.0))
                   if cost_known else None)
        fw = eff.get("final_score_weights", {"quality": 0.7, "time": 0.15, "cost": 0.15})
        if quality <= 0:
            final = 0.0
        elif c_score is None:
            final = round((fw["quality"] * quality + fw["time"] * t_score)
                          / (fw["quality"] + fw["time"]), 3)
        else:
            final = round(fw["quality"] * quality + fw["time"] * t_score
                          + fw["cost"] * c_score, 3)
        out.append({
            "candidate": cand, "gated_out": gated,
            "quality": round(quality, 3),
            "tier_scores": {t: tier_rows[t]["auto_score"] for t in tier_rows},
            "elapsed_minutes": round(elapsed_min, 1), "time_score": round(t_score, 3),
            "cost_usd": round(cost, 4) if cost_known else None,
            "cost_score": round(c_score, 3) if c_score is not None else None,
            "final_score": final,
        })
    return out


# ----------------------------------------------------------------- reporting ---
def print_tables(rows: list[dict], suite_rows: list[dict]) -> None:
    hdr = (f"{'tier':<22}{'candidate':<12}{'static':<8}{'boot':<6}{'gui':<6}"
           f"{'deploy':<8}{'robust':<8}{'cc':<6}{'auto':<7}{'wall(s)':<9}")
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        f = lambda v: "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))
        print(f"{r['tier']:<22}{r['candidate']:<12}{f(r['static']):<8}{f(r['boot']):<6}"
              f"{f(r['gui']):<6}{f(r['deploy']):<8}{f(r['robustness']):<8}"
              f"{f(r['contract_compliance']):<6}{f(r['auto_score']):<7}"
              f"{f(r['wall_seconds']):<9}")
    print("=" * len(hdr))
    for r in rows:
        if r["note"]:
            print(f"note[{r['tier']}/{r['candidate']}]: {r['note']}")

    if suite_rows:
        hdr2 = (f"{'candidate':<12}{'quality':<9}{'t1':<7}{'t2':<7}{'t3':<7}"
                f"{'time':<7}{'cost($)':<9}{'FINAL':<7}{'gated':<6}")
        print("\n" + "=" * len(hdr2))
        print("Suite roll-up  (quality/time/cost 3열 병기 — README 참조)")
        print("-" * len(hdr2))
        for s in suite_rows:
            ts = s["tier_scores"]
            g = lambda t: f"{ts[t]:.2f}" if t in ts else "-"
            print(f"{s['candidate']:<12}{s['quality']:<9}{g('tier1-gate'):<7}"
                  f"{g('tier2-core'):<7}{g('tier3-differentiator'):<7}"
                  f"{s['time_score']:<7}{str(s['cost_usd'] or '-'):<9}"
                  f"{s['final_score']:<7}{str(s['gated_out']):<6}")
        print("=" * len(hdr2))


def build_gallery(suite: str, rows: list[dict], out_path: Path, blind: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    alias = {}
    if blind:
        for i, name in enumerate(sorted({r["candidate"] for r in rows})):
            alias[name] = f"candidate-{chr(65 + i)}"
    cards = []
    for r in rows:
        shown = alias.get(r["candidate"], r["candidate"])
        rel = f"../{r['tier']}/{r['candidate']}"
        thumbs = "".join(
            f'<a href="{rel}/{s}" target="_blank"><img src="{rel}/{s}" loading="lazy"></a>'
            for s in r["screenshots"]) or '<p class="muted">(no screenshot)</p>'
        gui = " ".join(f"{k}:{v}" for k, v in r["gui_results"].items())
        cards.append(f"""
    <div class="card" data-candidate="{r['candidate']}" data-tier="{r['tier']}">
      <h2>{r['tier']} / {shown} <span class="muted">{'' if blind else '/ ' + str(r['harness']) + ' / ' + str(r['model'])}</span></h2>
      <p class="metrics">static={r['static']} boot={r['boot']} gui={r['gui']}
         deploy={r['deploy']} robust={r['robustness']} cc={r['contract_compliance']}
         auto={r['auto_score']}</p>
      <p class="metrics">{gui}</p>
      {'<p class="note">' + r['note'] + '</p>' if r['note'] else ''}
      <div class="thumbs">{thumbs}</div>
      <label>UX (1-5): <select class="h-ux"><option value=""></option>
        <option>1</option><option>2</option><option>3</option><option>4</option><option>5</option></select></label>
      <label>Code quality (1-5): <select class="h-code"><option value=""></option>
        <option>1</option><option>2</option><option>3</option><option>4</option><option>5</option></select></label>
      <label>Notes: <input class="h-note" size="40"></label>
    </div>""")
    doc = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{suite} — review gallery</title>
<style>
 body{{font-family:system-ui,sans-serif;margin:2rem;background:#fafafa;color:#222}}
 .card{{background:#fff;border:1px solid #ddd;border-radius:10px;padding:1rem 1.25rem;margin:1rem 0}}
 .muted{{color:#888;font-weight:normal}} .metrics{{color:#555;font-size:.9rem}}
 .note{{color:#b45;font-size:.85rem}} .thumbs img{{height:160px;border:1px solid #ccc;border-radius:4px}}
 label{{margin-right:1rem;font-size:.9rem}}
 button{{font-size:1rem;padding:.5rem 1rem;border-radius:8px;border:0;background:#0a7;color:#fff;cursor:pointer}}
</style></head><body>
<h1>{suite} — review gallery {'(blind)' if blind else ''}</h1>
<button onclick="dl()">⬇ download human_scores.json</button>
{''.join(cards)}
<script>
function dl(){{const o={{}};document.querySelectorAll('.card').forEach(c=>{{
 const k=c.dataset.tier+'/'+c.dataset.candidate;
 const ux=c.querySelector('.h-ux').value,cq=c.querySelector('.h-code').value,n=c.querySelector('.h-note').value;
 if(ux||cq||n)o[k]={{ux:ux?+ux:null,code:cq?+cq:null,note:n}};}});
 const b=new Blob([JSON.stringify(o,null,2)],{{type:'application/json'}});
 const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='human_scores.json';a.click();}}
</script></body></html>"""
    out_path.write_text(doc, encoding="utf-8")
    print(f"[grade] gallery -> {out_path}")


def merge_human(suite: str, human_path: Path) -> None:
    results_path = task_spec.suite_dir(suite) / "grade_results.json"
    if not results_path.exists():
        raise SystemExit(f"ERROR: {results_path} not found — run grading first.")
    data = json.loads(results_path.read_text(encoding="utf-8"))
    human = json.loads(human_path.read_text(encoding="utf-8"))
    for r in data.get("tier_rows", []):
        h = human.get(f"{r['tier']}/{r['candidate']}", {})
        r["human_ux"], r["human_code"] = h.get("ux"), h.get("code")
        r["human_note"] = h.get("note", "")
    results_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    print(f"[grade] merged human scores -> {results_path}")


# ---------------------------------------------------------------------- main ---
def main() -> None:
    ap = argparse.ArgumentParser(description="Grade the databricks-app-generation suite.")
    ap.add_argument("--suite", default=task_spec.DEFAULT_SUITE)
    ap.add_argument("--tier", default="all",
                    help=f"one of {task_spec.TIERS} or 'all'")
    ap.add_argument("--candidates", nargs="*", default=None)
    ap.add_argument("--no-gui", action="store_true", help="skip Playwright checks")
    ap.add_argument("--no-deploy", action="store_true", help="skip tier1 deployment")
    ap.add_argument("--fuzz-minutes", type=float, default=5.0)
    ap.add_argument("--blind", action="store_true", help="anonymize the gallery")
    ap.add_argument("--merge-human", type=Path, default=None)
    args = ap.parse_args()

    if args.merge_human is not None:
        merge_human(args.suite, args.merge_human)
        return

    suite_cfg = task_spec.load_suite(args.suite)
    pricing = task_spec.load_pricing(args.suite)
    tiers = list(task_spec.TIERS) if args.tier == "all" else [args.tier]

    all_rows = []
    for tier in tiers:
        cfg = task_spec.load_tier(tier, args.suite)
        cands = discover_candidates(args.suite, tier, args.candidates)
        if not cands:
            print(f"[grade] {tier}: no candidates — skipping")
            continue
        gt = (tier2_ground_truth(args.suite) if tier == "tier2-core"
              else tier1_ground_truth(args.suite, tier))
        for c in cands:
            row = grade_one(args.suite, tier, c, cfg, gt,
                            do_gui=not args.no_gui, do_deploy=not args.no_deploy,
                            fuzz_minutes=args.fuzz_minutes)
            row["_cost_usd"] = tokens_cost_usd(c["meta"], pricing)
            all_rows.append(row)

    suite_rows = aggregate_suite(args.suite, all_rows, suite_cfg, pricing)
    print_tables(all_rows, suite_rows)

    sdir = task_spec.suite_dir(args.suite)
    out = sdir / "grade_results.json"
    out.write_text(json.dumps({"tier_rows": all_rows, "suite": suite_rows},
                              indent=2, default=str), encoding="utf-8")
    print(f"\n[grade] wrote {out}")
    build_gallery(args.suite, all_rows, sdir / "gallery" / "index.html", args.blind)


if __name__ == "__main__":
    main()
