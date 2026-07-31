"""데이터셋 로더 (plan §5). registry.yaml 기반, seed 고정 subset, 로컬 캐시.

- HuggingFace `datasets`로 로드. 다운로드는 .cache/(gitignore)에만.
- seed 고정 subset으로 "표준·불변" 재현성 확보(D8).
- 민감 데이터(NSFW 등 sensitive)는 캐시 전용 — 원본 미디어를 repo에 남기지 않음(D3).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# 캐시 루트 (gitignore됨)
CACHE_DIR = Path(".cache")


def load_registry(path: str | Path = "datasets/registry.yaml") -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _hf_cache_dir() -> str:
    d = CACHE_DIR / "hf"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def load_hf_split(
    hf_id: str,
    split: str,
    n: int,
    seed: int,
    config: str | None = None,
):
    """HF 데이터셋에서 seed 고정 subset n개를 로드.

    datasets 라이브러리를 지연 임포트(무거움). 캐시는 .cache/hf.
    반환: datasets.Dataset (subset).
    """
    from datasets import load_dataset

    ds = load_dataset(
        hf_id,
        name=config,
        split=split,
        cache_dir=_hf_cache_dir(),
    )
    # seed 고정 셔플 후 앞 n개 (재현 가능한 subset)
    if n and n < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n))
    return ds


def resolve_dataset_entry(registry: dict[str, Any], key: str) -> dict[str, Any]:
    """registry에서 데이터셋 항목을 꺼낸다. 없으면 KeyError."""
    if key not in registry:
        raise KeyError(f"데이터셋 '{key}'가 registry.yaml에 없음")
    return registry[key]


def is_sensitive(registry: dict[str, Any], key: str) -> bool:
    """민감 데이터셋 여부(NSFW 등). True면 미디어 repo 미저장·갤러리 숨김(D3)."""
    entry = registry.get(key, {})
    return bool(entry.get("sensitive", False))


# 오프라인/네트워크 차단 환경 대비 플래그
def offline_mode() -> bool:
    return os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("ITP_OFFLINE") == "1"
