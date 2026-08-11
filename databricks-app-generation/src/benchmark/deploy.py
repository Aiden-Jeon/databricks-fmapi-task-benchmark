#!/usr/bin/env python3
"""
deploy.py — tier1-only real deployment check (A4): sync the candidate app to the
workspace, `databricks apps deploy`, poll until RUNNING, smoke-test the URL, then
ALWAYS stop (and optionally delete) the app so billing stops.

Uses the `databricks` CLI (must be installed + authenticated: `databricks auth login`
or DATABRICKS_HOST/TOKEN env). App names follow bench-<task>-<candidate>-<ts> and are
unique per grading run because app URLs are name-based and immutable.
"""
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import requests


@dataclass
class DeployResult:
    created: bool = False
    running: bool = False
    url_ok: bool = False
    app_name: str = ""
    app_url: str = ""
    stopped: bool = False
    deleted: bool = False
    note: str = ""


def _cli(*args: str, timeout: int = 120) -> tuple[int, str]:
    try:
        r = subprocess.run(["databricks", *args], capture_output=True, text=True,
                           timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except FileNotFoundError:
        return 127, "databricks CLI not found"
    except subprocess.TimeoutExpired:
        return 124, f"databricks {' '.join(args[:2])} timed out"


def deploy_and_verify(app_dir: Path, candidate: str, task_slug: str = "nyctaxi",
                      warehouse_env: str = "DATABRICKS_WAREHOUSE_ID",
                      poll_s: int = 300, delete_after: bool = True) -> DeployResult:
    res = DeployResult()
    ts = time.strftime("%m%d%H%M")
    res.app_name = f"bench-{task_slug}-{candidate}-{ts}"[:30].rstrip("-")
    ws_path = f"/Workspace/Shared/bench-apps/{res.app_name}"

    # 1) create app shell
    code, out = _cli("apps", "create", res.app_name, timeout=300)
    if code != 0:
        res.note = f"apps create failed: {out[-400:]}"
        return res
    res.created = True
    try:
        # 2) sync source + deploy
        code, out = _cli("workspace", "import-dir", str(app_dir), ws_path, "--overwrite",
                         timeout=300)
        if code != 0:
            res.note = f"workspace import failed: {out[-400:]}"
            return res
        # Bind the SQL warehouse resource non-interactively BEFORE deploy so the
        # app's service principal gets CAN_USE and DATABRICKS_WAREHOUSE_ID resolves.
        # `databricks apps update --json` carries the resources spec on recent CLI
        # versions; if this CLI predates it, fall back gracefully: note it and rely
        # on a pre-granted grading service principal (documented in the README).
        wh = os.getenv("DATABRICKS_WAREHOUSE_ID", "")
        if wh:
            spec = json.dumps({"resources": [{
                "name": "sql-warehouse",
                "sql_warehouse": {"id": wh, "permission": "CAN_USE"},
            }]})
            code, out = _cli("apps", "update", res.app_name, "--json", spec, timeout=120)
            if code != 0:
                res.note = ("resource binding via `apps update --json` failed "
                            f"(CLI too old?) — falling back to pre-granted SP: {out[-200:]}")
        else:
            res.note = "DATABRICKS_WAREHOUSE_ID unset — deployed without resource binding"

        code, out = _cli("apps", "deploy", res.app_name,
                         "--source-code-path", ws_path, timeout=600)
        if code != 0:
            res.note = (res.note + "; " if res.note else "") + \
                f"apps deploy failed: {out[-400:]}"
            return res

        # 3) poll status
        deadline = time.monotonic() + poll_s
        while time.monotonic() < deadline:
            code, out = _cli("apps", "get", res.app_name)
            if code == 0:
                try:
                    info = json.loads(out)
                except json.JSONDecodeError:
                    info = {}
                state = ((info.get("compute_status") or {}).get("state")
                         or (info.get("app_status") or {}).get("state") or "")
                res.app_url = info.get("url", res.app_url)
                if state.upper() == "RUNNING" and res.app_url:
                    res.running = True
                    break
                if state.upper() in ("CRASHED", "ERROR"):
                    res.note = f"app state={state}"
                    break
            time.sleep(10)

        # 4) smoke: deployed URL answers (OAuth redirect 200/302 both accepted —
        #    a full authenticated GUI smoke needs a browser profile; OPEN ITEM)
        if res.running:
            try:
                r = requests.get(res.app_url, timeout=30, allow_redirects=False)
                res.url_ok = r.status_code in (200, 302, 303, 307)
            except requests.RequestException as e:
                res.note = f"url check failed: {e}"
    finally:
        # 5) ALWAYS stop billing
        code, _ = _cli("apps", "stop", res.app_name, timeout=300)
        res.stopped = code == 0
        if delete_after:
            code, _ = _cli("apps", "delete", res.app_name, timeout=300)
            res.deleted = code == 0
    return res
