#!/usr/bin/env python
"""Headed capture: pop up a visible browser, let the human SSO, then save the
authenticated session AND dump the Genie Code DOM/screenshot so we can calibrate
selectors from the real UI. Non-interactive (polls for login; no input()).

Run backgrounded. Watch this script's stdout for progress.
"""
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SEC = ROOT / ".secrets"
SEC.mkdir(parents=True, exist_ok=True)
STATE = SEC / "genie_state.json"
HOST = "https://fevm-newjeans-ontos.cloud.databricks.com"
HOSTNAME = HOST.split("//", 1)[1]
GENIE_CANDIDATES = [f"{HOST}/genie/code", f"{HOST}/genie", f"{HOST}/ml/genie", HOST]

LOGIN_MARKERS = ("login", "oidc", "signin", "sign-in", "auth", "accounts.",
                 "sso", "saml", "microsoftonline", "okta")


def looks_logged_in(page):
    u = page.url.lower()
    if HOSTNAME not in u:
        return False
    if any(m in u for m in LOGIN_MARKERS):
        return False
    try:
        if page.query_selector("input[type=password]"):
            return False
    except Exception:
        pass
    return True


PROFILE_DIR = SEC / "genie_chrome_profile"  # dedicated, persistent, reusable


def main():
    with sync_playwright() as p:
        # Use the real Google Chrome app (familiar UI) with a dedicated persistent
        # profile — login is saved here for every future run.
        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        ctx = p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR), channel="chrome", headless=False,
            no_viewport=True, args=["--new-window"])
        browser = ctx.browser
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        try:
            page.bring_to_front()
        except Exception:
            pass
        print(">>> A GOOGLE CHROME window is opening — complete SSO IN THAT WINDOW.",
              flush=True)
        page.goto(HOST, wait_until="domcontentloaded")
        try:
            page.bring_to_front()
        except Exception:
            pass

        logged = False
        for i in range(400):  # up to ~20 min
            time.sleep(3)
            if looks_logged_in(page):
                # settle: confirm it stays logged in for two consecutive checks
                time.sleep(2)
                if looks_logged_in(page):
                    logged = True
                    break
            if i % 5 == 0:
                print(f"    waiting for login… ({i*3}s) url={page.url[:70]}", flush=True)

        if not logged:
            print("LOGIN_TIMEOUT — no authenticated workspace detected.", flush=True)
            browser.close()
            return

        ctx.storage_state(path=str(STATE))
        print(f">>> Logged in. Session saved -> {STATE}", flush=True)

        # Probe Genie Code entry points; dump the first that loads app content.
        for url in GENIE_CANDIDATES:
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                page.wait_for_timeout(6000)
                html = page.content()
                tag = url.rsplit("/", 1)[-1] or "root"
                (SEC / f"genie_dom_{tag}.html").write_text(html[:800000])
                page.screenshot(path=str(SEC / f"genie_view_{tag}.png"))
                blocked = "blocked by your organization" in html.lower()
                print(f"    {url} -> {'ORG-BLOCKED' if blocked else 'captured'} "
                      f"(dom+png saved, tag={tag})", flush=True)
            except Exception as e:
                print(f"    {url} -> error {e}", flush=True)
        print("CAPTURE_DONE", flush=True)
        page.wait_for_timeout(2000)
        browser.close()


if __name__ == "__main__":
    main()
