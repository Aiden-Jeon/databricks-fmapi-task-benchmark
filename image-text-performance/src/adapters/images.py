"""이미지 → data URL 변환 유틸 (이미지 태스크 공용).

HF 이미지 데이터셋은 PIL.Image를 준다. FMAPI 멀티모달 입력은 data URL을 받으므로
base64로 인코딩한다. 비용·대역폭을 위해 긴 변을 max_side로 리사이즈한다.
"""

from __future__ import annotations

import base64
import io
from typing import Any


def pil_to_data_url(image: Any, max_side: int = 768, fmt: str = "JPEG") -> str:
    """PIL.Image를 data URL로. 긴 변이 max_side를 넘으면 비율 유지 리사이즈.

    RGBA/P 등은 RGB로 변환(JPEG 호환). PIL 미설치 시 ImportError.
    """
    from PIL import Image

    if not isinstance(image, Image.Image):
        raise TypeError(f"PIL.Image가 필요하지만 {type(image)}를 받음")

    img = image
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    w, h = img.size
    longest = max(w, h)
    if longest > max_side:
        scale = max_side / longest
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))

    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    mime = "image/jpeg" if fmt.upper() == "JPEG" else f"image/{fmt.lower()}"
    return f"data:{mime};base64,{b64}"


def bytes_to_data_url(data: bytes, mime: str = "image/jpeg") -> str:
    """이미 인코딩된 이미지 bytes를 data URL로."""
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"
