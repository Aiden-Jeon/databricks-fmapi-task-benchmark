#!/usr/bin/env python3
"""
app_runner.py — Boot a candidate app locally the way Databricks Apps would:
isolated venv, `pip install -r requirements.txt`, launch app.yaml's command with
grader credentials injected via env, poll an HTTP health URL, and tear down.

Also provides the fault-injection knobs for tier3 robustness scenarios (R1: bad
warehouse id at boot). R2 (mid-session timeout) is an OPEN ITEM — it needs a local
TCP proxy in front of the warehouse endpoint; scaffolded but not wired.
"""
import os
import shutil
import signal
import subprocess
import sys
import time
import venv
from dataclasses import dataclass, field
from pathlib import Path

import requests
import yaml

from benchmark import fmapi_auth

DEFAULT_HEALTH = "http://localhost:8501"
# Force Streamlit to be headless & deterministic regardless of user config.
STREAMLIT_ENV = {
    "STREAMLIT_SERVER_HEADLESS": "true",
    "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
    "STREAMLIT_SERVER_PORT": "8501",
    "STREAMLIT_SERVER_ADDRESS": "127.0.0.1",
}


@dataclass
class BootResult:
    booted: bool = False
    install_ok: bool = False
    crashed: bool = False
    seconds_to_healthy: float | None = None
    note: str = ""
    proc: subprocess.Popen | None = field(default=None, repr=False)
    log_path: Path | None = None

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self.proc.wait(timeout=10)
            except Exception:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except Exception:
                    pass
        self.proc = None

    def alive(self) -> bool:
        return bool(self.proc and self.proc.poll() is None)


def load_app_yaml(app_dir: Path) -> dict:
    p = app_dir / "app.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def command_from_yaml(app_dir: Path) -> list[str] | None:
    cfg = load_app_yaml(app_dir)
    cmd = cfg.get("command")
    if isinstance(cmd, list) and all(isinstance(c, str) for c in cmd):
        return cmd
    if isinstance(cmd, str):
        return cmd.split()
    return None


def make_venv(app_dir: Path, work_root: Path) -> tuple[Path | None, str]:
    """Fresh venv + install requirements.txt. Returns (python_path, note).

    Prefer `uv` for both the venv and the install: uv is already the harness's hard
    prerequisite (dryrun.sh / README step 0), and stdlib `venv.create(with_pip=True)`
    aborts with SIGABRT when the interpreter is a uv-managed standalone CPython (its
    bundled `ensurepip` crashes). Fall back to stdlib venv + pip when uv is absent so
    the runner stays portable."""
    venv_dir = work_root / ".graderenv"
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    py = venv_dir / ("Scripts" if os.name == "nt" else "bin") / "python"
    req = app_dir / "requirements.txt"

    uv = shutil.which("uv")
    try:
        if uv:
            c = subprocess.run([uv, "venv", str(venv_dir)],
                               capture_output=True, text=True, timeout=120)
            if c.returncode != 0:
                return None, f"uv venv failed: {c.stderr[-800:]}"
            # Try the uv cache first (--offline): it's fast, deterministic, and immune
            # to the pypi proxy being slow or down (a real, observed failure mode that
            # made boots time out and grading flaky). Fall back to an online install
            # only when the cache can't satisfy every requirement.
            base = [uv, "pip", "install", "-q", "--python", str(py), "-r", str(req),
                    "streamlit"]
            r = subprocess.run(base + ["--offline"], capture_output=True, text=True,
                               timeout=600)
            if r.returncode != 0:
                r = subprocess.run(base, capture_output=True, text=True, timeout=600)
        else:
            venv.create(venv_dir, with_pip=True)
            r = subprocess.run(
                [str(py), "-m", "pip", "install", "-q", "-r", str(req), "streamlit"],
                capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            return None, f"pip install failed: {r.stderr[-800:]}"
    except subprocess.TimeoutExpired:
        return None, "pip install timed out (600s)"
    return py, ""


def grader_env(fault_bad_warehouse: bool = False) -> dict:
    """Env the Databricks Apps runtime would provide, minted from grader creds.
    fault_bad_warehouse implements tier3 scenario R1."""
    env = {**os.environ, **STREAMLIT_ENV}
    host, token, _ = fmapi_auth.resolve_host_token()
    if host:
        env["DATABRICKS_HOST"] = host.replace("https://", "")
        env["DATABRICKS_SERVER_HOSTNAME"] = host.replace("https://", "")
    if token:
        env["DATABRICKS_TOKEN"] = token
    wh = os.getenv("DATABRICKS_WAREHOUSE_ID", "")
    env["DATABRICKS_WAREHOUSE_ID"] = "0000deadbeef0000" if fault_bad_warehouse else wh
    return env


def boot(app_dir: Path, health_url: str = DEFAULT_HEALTH, timeout_s: int = 60,
         fault_bad_warehouse: bool = False) -> BootResult:
    """Install deps, launch via app.yaml's command, poll health_url until 200."""
    res = BootResult()
    cmd = command_from_yaml(app_dir)
    if not cmd:
        res.note = "app.yaml missing or has no usable `command`"
        return res

    py, note = make_venv(app_dir, app_dir.parent)
    if py is None:
        res.note = note
        return res
    res.install_ok = True

    # Rewrite `streamlit ...` / `python ...` to the venv binaries.
    bin_dir = py.parent
    argv = list(cmd)
    exe = bin_dir / argv[0]
    if exe.exists():
        argv[0] = str(exe)

    # Normalize the listen port/address to the grader's fixed values. The contract
    # does NOT pin a port (real Databricks Apps inject DATABRICKS_APP_PORT and the app
    # binds to it), so candidates legitimately differ: some omit it (Streamlit then
    # honors STREAMLIT_SERVER_PORT=8501 from env), but one that hardcodes
    # `--server.port=8000` in app.yaml would bind off the health-check port and be
    # wrongly scored as a boot failure. CLI flags override env in Streamlit, so strip
    # any candidate-supplied server.port/address and append the grader's — the health
    # check and GUI (both fixed to 8501) then always find the app. Only applies to a
    # streamlit command; other launchers are left untouched.
    if "streamlit" in Path(cmd[0]).name:
        cleaned, skip = [], False
        for tok in argv:
            if skip:
                skip = False
                continue
            if tok in ("--server.port", "--server.address"):
                skip = True  # drop this flag and its following value
                continue
            if tok.startswith(("--server.port=", "--server.address=")):
                continue
            cleaned.append(tok)
        argv = cleaned + ["--server.port=8501", "--server.address=127.0.0.1"]

    res.log_path = app_dir.parent / "app_boot.log"
    logf = open(res.log_path, "w", encoding="utf-8")
    start = time.monotonic()
    try:
        res.proc = subprocess.Popen(
            argv, cwd=str(app_dir), stdout=logf, stderr=subprocess.STDOUT,
            env=grader_env(fault_bad_warehouse), start_new_session=True)
    except OSError as e:
        res.note = f"launch failed: {e}"
        return res

    deadline = start + timeout_s
    while time.monotonic() < deadline:
        if res.proc.poll() is not None:
            res.crashed = True
            res.note = f"process exited early (code {res.proc.returncode}) — see app_boot.log"
            return res
        try:
            if requests.get(health_url, timeout=3).status_code == 200:
                res.booted = True
                res.seconds_to_healthy = round(time.monotonic() - start, 1)
                return res
        except requests.RequestException:
            pass
        time.sleep(1.5)
    res.note = f"no HTTP 200 on {health_url} within {timeout_s}s"
    res.stop()
    return res


def main() -> None:  # manual smoke: python -m benchmark.app_runner <app_dir>
    app_dir = Path(sys.argv[1]).resolve()
    r = boot(app_dir)
    print(r)
    r.stop()


if __name__ == "__main__":
    main()
