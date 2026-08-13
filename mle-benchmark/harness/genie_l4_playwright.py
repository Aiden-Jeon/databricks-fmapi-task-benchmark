#!/usr/bin/env python
"""L4 (Genie Code) automation — the REFERENCE lane helper.

Genie Code has no public API, so this lane is UI-driven and therefore *not* part
of the reproducible core (the four AI-Gateway lanes M1-M4 / L1-L3 are). This
script is a best-effort convenience for the Genie Code product-comparison point.
It depends ONLY on standard open-source Playwright — NOT on Claude-in-Chrome or
any Claude-specific tooling — so any reproducer can run it:
  * genie_l4_login.py captures an authenticated session (storage_state)
  * --calibrate uses Playwright's own inspector to confirm/fix selectors once
Caveat: UI automation is brittle across Databricks UI versions; selectors may
need recalibration. Treat L4 results as a reference, not a core benchmark cell.

Drives the Genie Code UI per task, one session per task, following the manual
protocol (harness/genie_code_protocol_ko.md) but automated.

Prereq: run genie_l4_login.py once to capture an authenticated session.

Per task:
  1. copy the pack (spec/train/test/sample) into a workspace folder via CLI
     (private answers are NEVER staged)
  2. open Genie Code full page, start a new thread, enable auto-approve
  3. paste the standardized Korean kickoff, send
  4. wait (poll the workspace folder via CLI) until outputs/submission.csv
     appears or the 2h cap elapses; no human intervention
  5. download submission + capture the thread text + timing to the artifacts
     volume as L4_<task>_<ts>/

Usage:
  .venv/bin/python kmle/harness/genie_l4_playwright.py --tasks t3_ynat [--headed] [--calibrate]

FIRST RUN: pass --headed --calibrate. The Genie Code DOM is not visible from
the dev machine, so SELECTORS below are best-effort (role/text based) and are
expected to need one calibration pass. --calibrate pauses at each UI step and
uses Playwright's inspector so you can confirm/fix each selector; the working
values are written back to kmle/.secrets/genie_selectors.json for later
unattended runs.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "config.json").read_text())
STATE = ROOT / ".secrets" / "genie_state.json"
SEL_FILE = ROOT / ".secrets" / "genie_selectors.json"
HOST = "https://fevm-newjeans-ontos.cloud.databricks.com"
PROFILE = CFG["profile"]
CAT = CFG["catalog"]
KICKOFF = (ROOT / "harness" / "kickoff_prompt_ko.md").read_text()
CAP_SECONDS = CFG["caps"]["wall_seconds"]

# Genie Code UI entry points and controls. Overridable via genie_selectors.json
# after the first --calibrate pass. Values are role/name/text based so they
# survive minor markup changes better than CSS/xpath.
SELECTORS = {
    "genie_code_url": f"{HOST}/genie/code",
    "new_thread": {"role": "button", "name_re": "New|새 (스레드|대화)"},
    "approve_mode_menu": {"role": "button", "name_re": "approv|승인|permission"},
    "auto_approve_item": {"role": "menuitem", "name_re": "current thread|이 (스레드|대화)|Auto"},
    "prompt_box": {"role": "textbox"},
    "send": {"role": "button", "name_re": "Send|전송|보내기"},
    "context_folder": {"role": "button", "name_re": "context|폴더|folder|attach"},
}
if SEL_FILE.exists():
    SELECTORS.update(json.loads(SEL_FILE.read_text()))

WS_BASE = "/Users/hyunsung.kim@databricks.com/kmle_genie"


def sh(args, **kw):
    return subprocess.run(args, capture_output=True, text=True, **kw)


def stage_pack(task: str) -> str:
    """Copy the public pack into a fresh workspace folder for Genie Code.
    Returns the workspace path. Never stages kmle/private."""
    wsdir = f"{WS_BASE}/{task}"
    sh(["databricks", "workspace", "mkdirs", wsdir, "-p", PROFILE])
    src = f"dbfs:/Volumes/{CAT}/{CFG['schemas']['packs']}/{CFG['volumes']['packs']}/{task}"
    local = ROOT / "packs" / task
    for f in ["spec.md", "train.csv", "test.csv", "sample_submission.csv"]:
        p = local / f
        if p.exists():
            sh(["databricks", "workspace", "import", f"{wsdir}/{f}",
                "--file", str(p), "--format", "AUTO", "--overwrite", "-p", PROFILE])
    return wsdir


def submission_exists(task: str) -> bool:
    out = sh(["databricks", "workspace", "list", f"{WS_BASE}/{task}/outputs",
              "-p", PROFILE]).stdout
    return "submission" in out


def click(page, sel, timeout=15000):
    spec = SELECTORS[sel]
    loc = page.get_by_role(spec["role"], name=__import__("re").compile(
        spec["name_re"], __import__("re").I)) if "name_re" in spec \
        else page.locator(spec)
    loc.first.click(timeout=timeout)


def run_task(page, task: str, calibrate: bool):
    wsdir = stage_pack(task)
    result = {"lane": "L4", "task": task,
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    t0 = time.time()

    page.goto(SELECTORS["genie_code_url"], wait_until="domcontentloaded")
    if calibrate:
        print(f"\n[CALIBRATE {task}] Genie Code page open. Use the Playwright "
              "inspector to confirm selectors. Resume when ready.")
        page.pause()

    for step in ("new_thread", "approve_mode_menu", "auto_approve_item"):
        try:
            click(page, step)
            time.sleep(1)
        except PWTimeout:
            if calibrate:
                print(f"[CALIBRATE] selector '{step}' not found — fix in inspector.")
                page.pause()
            else:
                result.setdefault("warnings", []).append(f"{step} not clickable")

    prompt = (f"작업 폴더: {wsdir} — 이 폴더 안에서만 작업하십시오. "
              f"결과 제출은 {wsdir}/outputs/submission.csv 로 저장하십시오.\n\n" + KICKOFF)
    box = page.get_by_role("textbox").first
    box.click()
    box.fill(prompt)
    if calibrate:
        print("[CALIBRATE] prompt filled — confirm the send control, then resume.")
        page.pause()
    try:
        click(page, "send")
    except PWTimeout:
        box.press("Enter")

    # Poll for the submission (no intervention) up to the cap.
    deadline = t0 + CAP_SECONDS
    got = False
    while time.time() < deadline:
        if submission_exists(task):
            got = True
            break
        time.sleep(60)
    result["wall_seconds"] = round(time.time() - t0, 1)
    result["submission_created"] = got
    result["timed_out"] = not got

    # Capture thread text (best effort) + persist.
    try:
        result["thread_text"] = page.inner_text("body")[:20000]
    except Exception:
        pass
    outdir = ROOT / "results" / "l4" / f"L4_{task}_{time.strftime('%Y%m%d_%H%M%S')}"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    if got:
        sh(["databricks", "workspace", "export", f"{WS_BASE}/{task}/outputs/submission.csv",
            "--file", str(outdir / "submission.csv"), "--format", "AUTO",
            "--overwrite", "-p", PROFILE])
    print(json.dumps({k: result[k] for k in
                      ("task", "wall_seconds", "submission_created")}, indent=1))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()
    if not STATE.exists():
        raise SystemExit("No saved session. Run genie_l4_login.py first.")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not (args.headed or args.calibrate))
        ctx = browser.new_context(storage_state=str(STATE))
        page = ctx.new_page()
        for task in args.tasks:
            try:
                run_task(page, task, args.calibrate)
            except Exception as e:
                print(f"[{task}] ERROR {e}")
        # Persist any selectors edited during calibration for later runs.
        if args.calibrate:
            SEL_FILE.write_text(json.dumps(SELECTORS, ensure_ascii=False, indent=2))
            print(f"selectors saved -> {SEL_FILE}")
        browser.close()


if __name__ == "__main__":
    main()
