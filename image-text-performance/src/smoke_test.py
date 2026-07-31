"""Phase 0 smoke test: 어댑터가 실제 FMAPI로 동작하는지 라이브 검증.

실행: python -m src.smoke_test [--config config/models.yaml]

검증 항목 (plan §11 실측 재확인):
- 각 모델에 텍스트 호출 → 평문 응답·usage·request_id가 정규화되어 나오는가
- reasoning minimal/full 모드가 모델별 파라미터로 실제 적용되는가
- vision 지원 모델은 이미지 입력을 받고, glm은 거부되는가(capability와 일치)
"""

from __future__ import annotations

import argparse
import base64
import struct
import zlib

from src.adapters.fmapi import (
    FMAPIClient,
    FMAPIError,
    build_image_message,
    build_text_message,
)
from src.config import load_models_config


def _tiny_red_png_data_url() -> str:
    """의존성 없이 8x8 빨강 PNG를 생성해 data URL로 반환."""
    w = h = 8
    raw = b""
    for _ in range(h):
        raw += b"\x00" + b"\xff\x00\x00" * w  # filter byte + RGB

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/models.yaml")
    args = parser.parse_args()

    cfg = load_models_config(args.config)
    print(f"프로파일: {cfg.profile} | judge: {cfg.judge} | 모드: {cfg.reasoning_modes}")
    print("=" * 70)

    failures = 0
    with FMAPIClient(
        profile=cfg.profile,
        timeout_seconds=cfg.runtime.timeout_seconds,
        max_retries=cfg.runtime.max_retries,
        backoff_initial_seconds=cfg.runtime.backoff_initial_seconds,
    ) as client:
        # judge도 함께 검증 대상에 포함
        targets = list(cfg.models)

        for model in targets:
            print(f"\n### {model.id} ({model.endpoint}) caps={model.capabilities}")

            # 1) 텍스트 + reasoning 모드별
            for mode in cfg.reasoning_modes:
                params = model.reasoning_params(mode)
                try:
                    resp = client.chat(
                        model.endpoint,
                        build_text_message("What is 17*23? Answer with the number only."),
                        max_tokens=cfg.runtime.max_tokens,
                        extra_params=params,
                    )
                    ok = "391" in resp.text
                    rtok = _reasoning_tokens(resp.usage)
                    flag = "✅" if ok else "⚠️ "
                    print(
                        f"  {flag} text/{mode:<7} → {resp.text[:40]!r} "
                        f"| finish={resp.finish_reason} | reasoning_tok={rtok} "
                        f"| req_id={'있음' if resp.request_id else '없음'}"
                    )
                    if not ok:
                        failures += 1
                except FMAPIError as e:
                    print(f"  ❌ text/{mode}: {e}")
                    failures += 1

            # 2) 이미지 (capability와 실제 동작 일치 확인)
            try:
                resp = client.chat(
                    model.endpoint,
                    build_image_message("What color is this image? One word.", _tiny_red_png_data_url()),
                    max_tokens=cfg.runtime.max_tokens,
                    extra_params=model.reasoning_params("minimal"),
                )
                got_image = True
                detail = f"{resp.text[:30]!r}"
            except FMAPIError as e:
                got_image = False
                detail = str(e)[:60]

            expected = model.supports("vision")
            match = got_image == expected
            print(
                f"  {'✅' if match else '❌'} vision: 실제={got_image} 기대={expected} | {detail}"
            )
            if not match:
                failures += 1

    print("\n" + "=" * 70)
    if failures == 0:
        print("SMOKE TEST 통과 — 어댑터가 모든 모델·모드에서 동작")
        return 0
    print(f"SMOKE TEST 실패 {failures}건 — 위 ❌/⚠️ 확인")
    return 1


def _reasoning_tokens(usage: dict) -> int | None:
    """usage에서 reasoning 토큰을 최대한 뽑아본다(계열마다 필드 다름)."""
    if "reasoning_tokens" in usage:
        return usage["reasoning_tokens"]
    details = usage.get("completion_tokens_details") or {}
    return details.get("reasoning_tokens")


if __name__ == "__main__":
    raise SystemExit(main())
