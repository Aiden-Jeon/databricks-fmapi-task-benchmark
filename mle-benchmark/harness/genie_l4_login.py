#!/usr/bin/env python
"""One-time: open a headed browser, let the human log into Databricks, save the
authenticated session to disk so the L4 runner can reuse it unattended.

Run once (or whenever the cookie expires):
  .venv/bin/python kmle/harness/genie_l4_login.py

Opens the workspace, waits for you to finish SSO/MFA, then persists
storage_state to kmle/.secrets/genie_state.json (gitignored).
"""
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "config.json").read_text())
STATE = ROOT / ".secrets" / "genie_state.json"
HOST = "https://fevm-newjeans-ontos.cloud.databricks.com"


def main():
    STATE.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        ctx = browser.new_context()
        page = ctx.new_page()
        page.goto(HOST, wait_until="domcontentloaded")
        print("\n>>> Log in fully (SSO/MFA) until you see the Databricks workspace.")
        print(">>> Then return here and press Enter.")
        input()
        ctx.storage_state(path=str(STATE))
        print(f"Saved session -> {STATE}")
        browser.close()


if __name__ == "__main__":
    main()
