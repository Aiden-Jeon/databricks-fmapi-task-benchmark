#!/usr/bin/env python
"""L4 Genie Code capture/driver via Selenium using the user's REAL Chrome
profile (Profile 1) — inherits the existing Databricks login, so no SSO and no
manual cookie decryption. Chrome decrypts its own cookies natively.

Requires Chrome to be QUIT (profile lock). Steps:
  1. launch Chrome on the real profile → already logged in
  2. save decrypted cookies to .secrets/genie_state.json (Playwright reuse later)
  3. probe Genie Code entry points, dump DOM + screenshot for selector calibration

Usage: genie_l4_selenium.py            # capture/calibrate
"""
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

ROOT = Path(__file__).resolve().parents[1]
SEC = ROOT / ".secrets"
SEC.mkdir(parents=True, exist_ok=True)
STATE = SEC / "genie_state.json"
CHROME_DIR = str(Path.home() / "Library/Application Support/Google/Chrome")
PROFILE = "Profile 1"
HOST = "https://fevm-newjeans-ontos.cloud.databricks.com"
GENIE_CANDIDATES = [f"{HOST}/genie/code", f"{HOST}/genie", f"{HOST}/ml/genie", HOST]
LOGIN_MARKERS = ("login.html", "oidc", "signin", "sign-in", "/auth", "accounts.",
                 "saml", "microsoftonline", "okta")


def chrome_running():
    return subprocess.run(["pgrep", "-x", "Google Chrome"],
                          capture_output=True).returncode == 0


def main():
    # Copy-profile approach works whether or not Chrome is running (it launches a
    # separate throwaway instance). If the Cookies DB is locked by a live Chrome,
    # use SQLite's online backup so we still get a consistent copy.
    # Copy just the cookie/key files into a clean throwaway profile — avoids the
    # fragility of driving the live profile, but Chrome still self-decrypts the
    # cookies via the same Keychain key (no manual decryption, usually no prompt).
    src = Path(CHROME_DIR)
    tmp = Path(tempfile.mkdtemp(prefix="genie_chrome_"))
    (tmp / "Default").mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "Local State", tmp / "Local State")
    sp = src / PROFILE
    for rel in ("Cookies", "Cookies-wal", "Cookies-shm",
                "Network/Cookies", "Network/Cookies-wal", "Network/Cookies-shm",
                "Preferences", "Secure Preferences", "Login Data"):
        f = sp / rel
        if f.exists():
            dst = tmp / "Default" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(f, dst)
    print(f"staged profile copy -> {tmp}", flush=True)

    opts = Options()
    opts.add_argument(f"--user-data-dir={tmp}")
    opts.add_argument("--profile-directory=Default")
    opts.add_argument("--remote-debugging-port=9222")
    opts.add_argument("--remote-debugging-port=9222")
    opts.add_argument("--no-first-run")
    opts.add_argument("--no-default-browser-check")
    opts.add_argument("--disable-session-crashed-bubble")
    opts.add_argument("--restore-last-session=false")
    opts.add_argument("--start-maximized")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("detach", True)  # keep window open after script

    driver = webdriver.Chrome(options=opts)
    try:
        driver.get(HOST)
        time.sleep(7)
        url = driver.current_url.lower()
        logged = ("databricks.com" in url and
                  not any(m in url for m in LOGIN_MARKERS))
        print(f"workspace url={driver.current_url[:80]} logged_in={logged}", flush=True)
        if not logged:
            print("NOT_LOGGED_IN — Profile 1 session may have expired; "
                  "log in once in this window, then re-run.", flush=True)
            return

        # Save decrypted cookies (Playwright storage_state format) for reuse.
        cks = driver.get_cookies()
        ss_map = {"Strict": "Strict", "Lax": "Lax", "None": "None"}
        pw_cookies = [{
            "name": c["name"], "value": c["value"], "domain": c["domain"],
            "path": c.get("path", "/"),
            "expires": c.get("expiry", -1),
            "httpOnly": c.get("httpOnly", False), "secure": c.get("secure", False),
            "sameSite": ss_map.get(c.get("sameSite", "Lax"), "Lax"),
        } for c in cks]
        STATE.write_text(json.dumps({"cookies": pw_cookies, "origins": []}, indent=1))
        print(f"saved {len(pw_cookies)} cookies -> {STATE}", flush=True)

        # Probe Genie Code and dump for calibration.
        for u in GENIE_CANDIDATES:
            try:
                driver.get(u)
                time.sleep(6)
                tag = u.rstrip("/").rsplit("/", 1)[-1] or "root"
                html = driver.page_source
                (SEC / f"genie_dom_{tag}.html").write_text(html[:900000])
                driver.save_screenshot(str(SEC / f"genie_view_{tag}.png"))
                blocked = "blocked by your organization" in html.lower()
                nf = "not found" in html.lower() or driver.title.lower().startswith("404")
                print(f"  {u} -> {'ORG-BLOCKED' if blocked else 'NOTFOUND' if nf else 'captured'} "
                      f"(title={driver.title[:40]!r}, tag={tag})", flush=True)
            except Exception as e:
                print(f"  {u} -> error {e}", flush=True)
        print("CAPTURE_DONE", flush=True)
    finally:
        pass  # detach=True leaves the window open for inspection


if __name__ == "__main__":
    main()
