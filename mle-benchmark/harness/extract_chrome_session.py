#!/usr/bin/env python
"""Extract the user's existing Databricks session cookies from Chrome (macOS)
into a Playwright storage_state.json — so L4 automation reuses the login the
user already has, no re-SSO. Scoped to the workspace domain only.

Reads only *.databricks.com cookies. macOS Keychain will prompt once to allow
reading Chrome's encryption key.
"""
import base64
import json
import shutil
import sqlite3
import subprocess
import tempfile
import time
from pathlib import Path

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / ".secrets" / "genie_state.json"
STATE.parent.mkdir(parents=True, exist_ok=True)
PROFILE = "Profile 1"
HOST_MATCH = "databricks.com"
CHROME = Path.home() / "Library/Application Support/Google/Chrome"


def keychain_key():
    pw = subprocess.check_output(
        ["security", "find-generic-password", "-w", "-s", "Chrome Safe Storage",
         "-a", "Chrome"]).strip()
    kdf = PBKDF2HMAC(algorithm=hashes.SHA1(), length=16, salt=b"saltysalt",
                     iterations=1003, backend=default_backend())
    return kdf.derive(pw)


def decrypt(enc, key):
    if not enc:
        return ""
    if enc[:3] in (b"v10", b"v11"):
        iv = b" " * 16
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        dec = cipher.decryptor()
        out = dec.update(enc[3:]) + dec.finalize()
        pad = out[-1]
        out = out[:-pad]
        # macOS Chrome (v10+) prefixes 32 bytes of SHA256 domain hash
        try:
            return out[32:].decode("utf-8")
        except UnicodeDecodeError:
            return out.decode("utf-8", "ignore")
    return enc.decode("utf-8", "ignore")


def main():
    src = CHROME / PROFILE / "Cookies"
    if not src.exists():
        src = CHROME / PROFILE / "Network" / "Cookies"
    tmp = Path(tempfile.mkdtemp()) / "Cookies"
    shutil.copy(src, tmp)
    key = keychain_key()

    con = sqlite3.connect(str(tmp))
    rows = con.execute(
        "SELECT host_key,name,encrypted_value,value,path,expires_utc,"
        "is_secure,is_httponly,samesite FROM cookies WHERE host_key LIKE ?",
        (f"%{HOST_MATCH}%",)).fetchall()
    con.close()

    ss_map = {2: "Strict", 1: "Lax", 0: "None"}
    cookies, ok, fail = [], 0, 0
    for host, name, enc, val, path, exp, sec, http, ss in rows:
        v = val if val else decrypt(enc, key)
        if not v:
            fail += 1
            continue
        ok += 1
        expires = -1 if not exp else max(-1, exp / 1_000_000 - 11644473600)
        cookies.append({
            "name": name, "value": v, "domain": host, "path": path or "/",
            "expires": expires, "httpOnly": bool(http), "secure": bool(sec),
            "sameSite": ss_map.get(ss, "Lax"),
        })
    STATE.write_text(json.dumps({"cookies": cookies, "origins": []}, indent=1))
    print(f"decrypted {ok} databricks cookies ({fail} skipped) -> {STATE}")
    # quick sanity: any auth-ish cookies?
    names = {c["name"] for c in cookies}
    print("has session cookies:", any(n for n in names if "auth" in n.lower()
          or "session" in n.lower() or n in ("_dbc", "JSESSIONID")))


if __name__ == "__main__":
    main()
