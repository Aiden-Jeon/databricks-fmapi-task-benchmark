"""Databricks FMAPI serving-endpoint 어댑터.

여러 모델 계열(claude/openai/glm)을 단일 인터페이스로 추상화한다. 아래 동작은
모두 ai_devtools 워크스페이스에서 실측 검증된 사실을 반영한다 (plan §11, 부록):

- 응답 `content`가 계열마다 다름: opus-5·gemini는 리스트(reasoning 블록 포함),
  sol·glm은 문자열. glm은 답을 `reasoning_content`에 먼저 쓴다. → `_normalize_content`.
- reasoning 제어 파라미터가 계열마다 다름 → config의 minimal/full 딕셔너리를 그대로 병합.
- request_id를 응답에서 뽑아 두면 나중에 system.ai_gateway.usage와 조인해 시간·비용 산출.
- glm은 3~5배 느리므로 넉넉한 timeout + 지수 backoff 재시도.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

import httpx


class FMAPIError(RuntimeError):
    """복구 불가능한 FMAPI 호출 실패."""


@dataclass
class ChatResponse:
    """정규화된 채팅 응답. 채점기·비용 계산이 이 형태만 알면 된다."""

    text: str                       # 평문 응답 (reasoning 블록·구조 제거 후)
    request_id: str | None          # ai_gateway.usage 조인 키
    finish_reason: str | None
    usage: dict[str, Any]           # prompt/completion/total tokens 등 원본
    raw: dict[str, Any] = field(repr=False, default_factory=dict)  # 감사용 원본 보존


def _get_workspace_auth(profile: str) -> tuple[str, str]:
    """databricks CLI로 host와 bearer token을 얻는다.

    CLI가 토큰 갱신을 처리하므로, **만료를 만나면 이 함수를 다시 불러** 새 토큰을 받는다.
    클라이언트 생성 시 한 번만 부르고 캐시하면 안 된다 — OAuth 토큰 수명(~1시간)이
    전체 실행 시간(30샘플 3모델 ≈ 1.5~2시간)보다 짧아서 중간에 전부 403이 된다
    (실측 2026-08-06: 52분 지점에 세 모델이 동시에 `403 Invalid Token`, 480/1080 실패).
    """
    import json

    def _cli(*args: str) -> str:
        proc = subprocess.run(
            ["databricks", *args, "--profile", profile],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise FMAPIError(f"databricks CLI 실패: {' '.join(args)}\n{proc.stderr.strip()}")
        return proc.stdout

    host = json.loads(_cli("auth", "env"))["env"]["DATABRICKS_HOST"]
    token = json.loads(_cli("auth", "token"))["access_token"]
    return host.rstrip("/"), token


def _is_auth_expiry(status_code: int, body: str) -> bool:
    """이 응답이 '토큰을 갱신하면 풀릴' 인증 실패인가(권한 거부와 구분).

    실측된 두 갈래:
    - 403 `Invalid Token` — 유효했던 토큰이 만료된 경우(장시간 실행 중간에 발생)
    - 401 `Credential was not sent or was of an unsupported type` — 토큰이 없거나 형식 불량

    반대로 `PERMISSION_DENIED`·IP ACL 차단은 403이지만 **갱신해도 안 풀린다** → False.
    구분하지 않으면 권한 문제에서 CLI를 헛되게 호출하고, 실패 원인도 흐려진다.
    """
    low = body.lower()
    if status_code == 403:
        return "invalid token" in low
    if status_code == 401:
        return "credential" in low or "token" in low
    return False


def _normalize_content(message: dict[str, Any]) -> str:
    """계열별로 다른 message 형태를 평문 텍스트로 정규화한다 (실측 기반).

    - content가 문자열(sol): 그대로.
    - content가 리스트(opus-5·gemini): type=="text" 파트만 이어붙이고 reasoning 파트는 버림.
    - content가 비었고 reasoning_content만 있음(glm이 사고 도중 잘림): reasoning_content로 폴백.
    """
    content = message.get("content")

    if isinstance(content, list):
        parts = [
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        text = "".join(parts).strip()
        if text:
            return text
        # 리스트에 text 파트가 없으면(전부 reasoning) reasoning_content 폴백
        return (message.get("reasoning_content") or "").strip()

    if isinstance(content, str) and content.strip():
        return content.strip()

    # content가 비었으면 glm처럼 reasoning_content에 답이 남아있을 수 있음
    return (message.get("reasoning_content") or "").strip()


class FMAPIClient:
    """단일 워크스페이스에 대한 FMAPI 호출 클라이언트."""

    def __init__(
        self,
        profile: str,
        timeout_seconds: float = 15.0,
        max_retries: int = 3,
        backoff_initial_seconds: float = 0.5,
    ) -> None:
        self.profile = profile
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_initial_seconds = backoff_initial_seconds
        self._host, self._token = _get_workspace_auth(profile)
        self._refreshed_this_call = False
        self._client = httpx.Client(timeout=timeout_seconds)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "FMAPIClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def chat(
        self,
        endpoint: str,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 1024,
        extra_params: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
    ) -> ChatResponse:
        """chat completion 호출 후 정규화된 응답을 돌려준다.

        extra_params에 reasoning 제어 딕셔너리(config의 minimal/full)를 그대로 넘긴다.

        `timeout_seconds`·`max_retries`는 **이 호출만** 클라이언트 기본값을 덮어쓴다
        (config의 모델별 runtime 오버라이드용). 모델마다 속도·출력 길이가 크게 달라
        (glm은 3~5배 느리고, 표→HTML 생성은 출력이 길다) 공통값을 강요하면 느린 모델에서
        타임아웃 실패가 쏟아지고 그 실패가 성능으로 오해된다.
        """
        url = f"{self._host}/serving-endpoints/{endpoint}/invocations"
        payload: dict[str, Any] = {"messages": messages, "max_tokens": max_tokens}
        if extra_params:
            payload.update(extra_params)
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }
        attempts = max_retries if max_retries is not None else self.max_retries
        req_timeout = timeout_seconds if timeout_seconds is not None else self.timeout_seconds
        self._refreshed_this_call = False   # 이 호출에서 토큰 갱신을 이미 썼는지

        last_err: Exception | None = None
        for attempt in range(attempts):
            try:
                resp = self._client.post(url, json=payload, headers=headers, timeout=req_timeout)
            except httpx.TimeoutException as e:
                last_err = e
                self._sleep_backoff(attempt)
                continue

            # 5xx·429는 재시도, 그 외 4xx는 즉시 실패(요청 자체가 잘못)
            if resp.status_code >= 500 or resp.status_code == 429:
                last_err = FMAPIError(f"{endpoint} HTTP {resp.status_code}: {resp.text[:200]}")
                self._sleep_backoff(attempt)
                continue
            # 인증 만료는 토큰을 갱신해 재시도한다(권한 문제가 아니다). 4xx를 일괄 즉시 실패로
            # 두면, 실행 중간에 토큰이 죽는 순간 남은 모든 셀이 통째로 실패한다
            # (실측 2026-08-06: 52분 지점부터 480/1080 실패).
            # 상태코드가 두 갈래로 관측됐다: 만료 토큰은 **403 "Invalid Token"**,
            # 형식이 깨진/빈 토큰은 **401 "Credential was not sent or was of an unsupported
            # type"**. 둘 다 갱신 대상이다. 갱신해도 같은 응답이면 아래 일반 4xx 경로로
            # 떨어져 즉시 실패한다(진짜 권한·설정 문제).
            if _is_auth_expiry(resp.status_code, resp.text):
                if self._refresh_auth(resp.status_code):
                    headers["Authorization"] = f"Bearer {self._token}"
                    last_err = FMAPIError(
                        f"{endpoint} HTTP {resp.status_code}: 인증 만료(갱신 후 재시도)"
                    )
                    continue
            if resp.status_code != 200:
                raise FMAPIError(f"{endpoint} HTTP {resp.status_code}: {resp.text[:300]}")

            data = resp.json()
            return self._parse(data, resp)

        raise FMAPIError(f"{endpoint} 재시도 {attempts}회 모두 실패: {last_err}")

    def _parse(self, data: dict[str, Any], resp: httpx.Response) -> ChatResponse:
        try:
            choice = data["choices"][0]
        except (KeyError, IndexError) as e:
            raise FMAPIError(f"예상치 못한 응답 형태: {str(data)[:300]}") from e
        message = choice.get("message", {})
        # request_id: 응답 헤더(x-request-id) 우선, 없으면 body의 id
        request_id = resp.headers.get("x-request-id") or data.get("id")
        return ChatResponse(
            text=_normalize_content(message),
            request_id=request_id,
            finish_reason=choice.get("finish_reason"),
            usage=data.get("usage", {}) or {},
            raw=data,
        )

    def _refresh_auth(self, status_code: int) -> bool:
        """토큰을 다시 받아 온다(만료 대응). 갱신에 성공했으면 True.

        **호출당 한 번만** 시도한다(`_refreshed_this_call`). 안 그러면 진짜 권한 문제일 때
        재시도 횟수를 갱신으로 다 태우고, CLI를 max_retries번 호출해 느려진다.
        갱신에 실패하면(CLI 오류 등) False를 돌려 호출부가 일반 4xx 실패로 처리하게 한다.
        """
        if self._refreshed_this_call:
            return False
        self._refreshed_this_call = True
        try:
            self._host, self._token = _get_workspace_auth(self.profile)
        except Exception as e:
            print(f"    [토큰 갱신 실패] {type(e).__name__}: {e}")
            return False
        print(f"    [토큰 갱신] HTTP {status_code} 인증 만료 → 새 토큰으로 재시도")
        return True

    def _sleep_backoff(self, attempt: int) -> None:
        time.sleep(self.backoff_initial_seconds * (2**attempt))


def build_text_message(prompt: str) -> list[dict[str, Any]]:
    """텍스트 전용 메시지."""
    return [{"role": "user", "content": prompt}]


def build_image_message(prompt: str, image_data_url: str) -> list[dict[str, Any]]:
    """이미지+텍스트 멀티모달 메시지 (vision 지원 모델용)."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        }
    ]
