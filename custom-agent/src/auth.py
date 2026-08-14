"""인증 — Databricks CLI 로 host/token 을 얻고 만료 시 갱신한다.

function-calling-json/src/runner.py 의 Auth·is_auth_expiry 를 이식했다.
토큰 수명(~1시간)이 실행 시간보다 짧을 수 있어, 만료를 만나면 갱신 후 재시도한다.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time


class Auth:
    def __init__(self, profile: str) -> None:
        self.profile = profile
        self._lock = threading.Lock()
        self.host, self.token = self._fetch()
        self._last_refresh = time.time()

    def _fetch(self) -> tuple[str, str]:
        def cli(*a: str) -> str:
            p = subprocess.run(
                ["databricks", *a, "--profile", self.profile],
                capture_output=True, text=True,
            )
            if p.returncode != 0:
                raise RuntimeError(f"databricks {' '.join(a)}: {p.stderr.strip()}")
            return p.stdout

        host = json.loads(cli("auth", "env"))["env"]["DATABRICKS_HOST"].rstrip("/")
        return host, json.loads(cli("auth", "token"))["access_token"]

    def refresh(self) -> None:
        """스레드 안전. 방금 갱신했으면 건너뛴다(동시 만료 시 CLI 폭주 방지)."""
        with self._lock:
            if time.time() - self._last_refresh < 20:
                return
            self.host, self.token = self._fetch()
            self._last_refresh = time.time()
            print("    [토큰 갱신]", flush=True)


_PERMISSION_MARKERS = ("permission", "not authorized", "forbidden")
_EXPIRY_MARKERS = ("credential", "invalid token", "expired", "not authenticated")


def is_auth_expiry(status: int, body: str) -> bool:
    """갱신하면 풀릴 인증 실패인가(권한 거부와 구분)."""
    low = body.lower()
    if any(s in low for s in _PERMISSION_MARKERS):
        return False
    if status == 403:
        return "invalid token" in low
    if status == 401:
        return any(s in low for s in _EXPIRY_MARKERS)
    return False
