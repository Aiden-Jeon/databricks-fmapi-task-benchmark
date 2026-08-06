"""FMAPI 클라이언트 인증 갱신 테스트 — 토큰 만료로 실행 절반이 날아가는 것을 막는다.

실측 사고(2026-08-06): 토큰을 `__init__`에서 한 번 받아 실행 내내 캐시했는데, OAuth 토큰
수명(~1시간)이 전체 실행 시간(30샘플 3모델 ≈ 1.5~2시간)보다 짧아 **52분 지점에 세 모델이
동시에 `403 Invalid Token`**을 맞았다. 403은 4xx라 즉시 실패로 분류돼 재시도조차 안 됐고,
1080호출 중 480이 실패해 run 전체가 폐기됐다(exit 1).

여기서 고정하는 계약:
- 인증 만료는 **토큰을 갱신해 재시도**한다(권한 문제가 아니다). 상태코드가 두 갈래로
  관측됐다: 만료 토큰은 403 `Invalid Token`, 형식이 깨진/빈 토큰은 401 `Credential was
  not sent or was of an unsupported type`(실호출로 확인). 둘 다 갱신 대상이다.
- 갱신은 **호출당 한 번만** — 진짜 권한 오류일 때 재시도 횟수를 갱신으로 태우지 않는다.
- 갱신 후에도 같은 응답이면 즉시 실패한다(진짜 권한 문제).
- `PERMISSION_DENIED`·IP ACL 403은 갱신해도 안 풀리므로 갱신 없이 즉시 실패한다.
- 다른 4xx(400 등)는 갱신 없이 즉시 실패한다(요청 자체가 잘못된 것).
- 5xx·타임아웃 재시도는 토큰을 그대로 쓴다(CLI를 헛되게 부르지 않는다).
"""

from __future__ import annotations

import httpx
import pytest

from src.adapters import fmapi
from src.adapters.fmapi import FMAPIClient, FMAPIError


class _Resp:
    """httpx.Response 대역 — 상태코드·본문·json만 쓰인다."""

    def __init__(self, status_code: int, text: str = "", payload: dict | None = None):
        self.status_code = status_code
        self.text = text
        self._payload = payload or {}
        self.headers: dict[str, str] = {}

    def json(self) -> dict:
        return self._payload


_OK_PAYLOAD = {
    "choices": [{"message": {"content": "정상 응답"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    "id": "req-1",
}


@pytest.fixture
def client(monkeypatch):
    """토큰 조회를 가짜로 바꾼 클라이언트. 갱신할 때마다 새 토큰 문자열을 준다."""
    tokens = iter([f"token-{i}" for i in range(1, 10)])
    monkeypatch.setattr(
        fmapi, "_get_workspace_auth", lambda profile: ("https://host", next(tokens))
    )
    c = FMAPIClient("ai_devtools", max_retries=3, backoff_initial_seconds=0)
    monkeypatch.setattr(c, "_sleep_backoff", lambda attempt: None)
    return c


def test_expired_token_is_refreshed_and_call_succeeds(client, monkeypatch, capsys):
    """403 Invalid Token → 토큰 갱신 후 재시도해 성공한다(옛 동작은 즉시 실패)."""
    seen_tokens: list[str] = []
    responses = [
        _Resp(403, "Invalid Token"),
        _Resp(200, payload=_OK_PAYLOAD),
    ]

    def fake_post(url, json=None, headers=None, timeout=None):
        seen_tokens.append(headers["Authorization"])
        return responses.pop(0)

    monkeypatch.setattr(client._client, "post", fake_post)
    resp = client.chat("databricks-claude-opus-5", [{"role": "user", "content": "안녕"}])

    assert resp.text == "정상 응답"
    assert seen_tokens == ["Bearer token-1", "Bearer token-2"], \
        f"갱신된 토큰으로 재시도해야 한다: {seen_tokens}"
    assert "토큰 갱신" in capsys.readouterr().out


def test_401_malformed_credential_is_also_refreshed(client, monkeypatch):
    """401 `Credential was not sent...`도 갱신 대상이다.

    실측: 만료 토큰은 403 `Invalid Token`, 형식이 깨진/빈 토큰은 401을 돌려준다.
    한쪽만 다루면 나머지 경로에서 실행 전체가 날아간다(실호출로 401 확인).
    """
    seen: list[str] = []
    responses = [
        _Resp(401, '{"error_code":401,"message":"Credential was not sent or was of an '
                   'unsupported type for this API."}'),
        _Resp(200, payload=_OK_PAYLOAD),
    ]

    def fake_post(url, json=None, headers=None, timeout=None):
        seen.append(headers["Authorization"])
        return responses.pop(0)

    monkeypatch.setattr(client._client, "post", fake_post)
    assert client.chat("databricks-claude-opus-5", [{"role": "user", "content": "x"}]).text
    assert seen == ["Bearer token-1", "Bearer token-2"], f"갱신 후 재시도해야 한다: {seen}"


def test_refresh_happens_once_per_call(client, monkeypatch):
    """계속 403이면 갱신은 한 번만 하고 즉시 실패한다 — 재시도를 갱신으로 태우지 않는다."""
    calls = {"n": 0}

    def always_403(url, json=None, headers=None, timeout=None):
        calls["n"] += 1
        return _Resp(403, "Invalid Token")

    monkeypatch.setattr(client._client, "post", always_403)
    with pytest.raises(FMAPIError, match="403"):
        client.chat("databricks-glm-5-2", [{"role": "user", "content": "x"}])

    assert calls["n"] == 2, f"최초 1회 + 갱신 후 1회여야 한다(실제 {calls['n']}회)"


def test_permission_403_without_invalid_token_fails_immediately(client, monkeypatch):
    """만료가 아닌 403(IP ACL·권한)은 갱신하지 않고 즉시 실패한다."""
    calls = {"n": 0}

    def denied(url, json=None, headers=None, timeout=None):
        calls["n"] += 1
        return _Resp(403, "PERMISSION_DENIED: IP ACL 차단")

    monkeypatch.setattr(client._client, "post", denied)
    with pytest.raises(FMAPIError, match="PERMISSION_DENIED"):
        client.chat("databricks-gpt-5-6-sol", [{"role": "user", "content": "x"}])

    assert calls["n"] == 1, "만료가 아니면 갱신·재시도 없이 즉시 실패"


def test_other_4xx_still_fails_immediately(client, monkeypatch):
    """400 등은 요청 자체가 잘못된 것 — 갱신·재시도 대상이 아니다."""
    calls = {"n": 0}

    def bad_request(url, json=None, headers=None, timeout=None):
        calls["n"] += 1
        return _Resp(400, "Unsupported message role")

    monkeypatch.setattr(client._client, "post", bad_request)
    with pytest.raises(FMAPIError, match="400"):
        client.chat("databricks-glm-5-2", [{"role": "user", "content": "x"}])

    assert calls["n"] == 1


def test_refresh_failure_falls_through_to_error(client, monkeypatch, capsys):
    """토큰 갱신 자체가 실패하면(CLI 오류) 조용히 넘기지 않고 403으로 실패한다."""
    def boom(profile):
        raise FMAPIError("databricks CLI 실패: auth token")

    monkeypatch.setattr(fmapi, "_get_workspace_auth", boom)
    monkeypatch.setattr(
        client._client, "post",
        lambda url, json=None, headers=None, timeout=None: _Resp(403, "Invalid Token"),
    )
    with pytest.raises(FMAPIError, match="403"):
        client.chat("databricks-claude-opus-5", [{"role": "user", "content": "x"}])

    assert "토큰 갱신 실패" in capsys.readouterr().out


def test_5xx_retries_without_refreshing_token(client, monkeypatch):
    """5xx는 업스트림 문제 — 토큰을 갱신하지 않고 같은 토큰으로 재시도한다.

    opus 엔드포인트의 산발적 502(실측 420호출 중 20건)가 이 경로다.
    """
    seen: list[str] = []
    responses = [
        _Resp(502, "invalid response from an upstream server"),
        _Resp(200, payload=_OK_PAYLOAD),
    ]

    def flaky(url, json=None, headers=None, timeout=None):
        seen.append(headers["Authorization"])
        return responses.pop(0)

    monkeypatch.setattr(client._client, "post", flaky)
    assert client.chat("databricks-claude-opus-5", [{"role": "user", "content": "x"}]).text
    assert seen == ["Bearer token-1", "Bearer token-1"], \
        f"5xx에 토큰을 갱신하면 CLI를 불필요하게 호출한다: {seen}"


def test_timeout_retries_preserve_token(client, monkeypatch):
    """타임아웃 재시도도 토큰을 그대로 쓴다."""
    seen: list[str] = []
    state = {"first": True}

    def timeout_once(url, json=None, headers=None, timeout=None):
        seen.append(headers["Authorization"])
        if state["first"]:
            state["first"] = False
            raise httpx.TimeoutException("timeout")
        return _Resp(200, payload=_OK_PAYLOAD)

    monkeypatch.setattr(client._client, "post", timeout_once)
    assert client.chat("databricks-glm-5-2", [{"role": "user", "content": "x"}]).text
    assert seen == ["Bearer token-1", "Bearer token-1"]
