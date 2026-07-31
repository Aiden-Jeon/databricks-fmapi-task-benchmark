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

    **streaming 우선**: 대형 데이터셋(COCO 19GB, PubTabNet 16GB 등)을 전체 다운로드하면
    디스크가 폭발하므로, streaming으로 필요한 만큼만 받는다(50샘플에 수십 GB 방지).
    streaming이 불가한 경우(로컬 parquet 등) 일반 로드로 폴백.

    재현성: streaming은 buffer shuffle(seed 고정) 후 take(n). buffer를 넉넉히 잡아
    앞쪽 편향을 줄인다. 반환: list[dict] (streaming) 또는 datasets.Dataset(폴백).
    """
    from datasets import load_dataset

    # 1) streaming 시도 (다운로드 최소화)
    try:
        ds = load_dataset(hf_id, name=config, split=split, streaming=True)
        # buffer shuffle로 앞쪽 편향 완화(전체 셔플은 streaming서 불가).
        # buffer가 크면 다운로드가 많아 느려짐 → n의 소배수로 제한(속도·편향 균형).
        buffer = min(max(50, n * 3), 500)
        ds = ds.shuffle(seed=seed, buffer_size=buffer)
        rows = list(ds.take(n)) if n else list(ds)
        if rows:
            return rows
        # streaming이 빈 결과면 폴백
    except Exception:
        pass

    # 2) 폴백: 일반 로드 (작은 데이터셋·mirror parquet). 캐시는 .cache/hf.
    ds = load_dataset(hf_id, name=config, split=split, cache_dir=_hf_cache_dir())
    if n and n < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n))
    # list[dict]로 정규화(streaming 경로와 반환 형식 통일 → 태스크 코드 단순화)
    return [dict(row) for row in ds]


def get_label_names(hf_id: str, split: str, config: str | None, column: str) -> list[str] | None:
    """분류 데이터셋의 라벨 이름 목록을 얻는다(예: COCO category names).

    streaming은 features를 안 주므로, 라벨 이름이 필요하면 이 함수로 비-streaming
    메타만 짧게 조회한다(데이터 다운로드 없이 features만). 실패 시 None.
    중첩 컬럼(예: 'objects.category')도 지원.
    """
    from datasets import load_dataset_builder

    try:
        builder = load_dataset_builder(hf_id, name=config, cache_dir=_hf_cache_dir())
        feats = builder.info.features
        if feats is None:
            return None
        # 중첩 경로 탐색
        cur: Any = feats
        for part in column.split("."):
            if hasattr(cur, "feature"):  # Sequence
                cur = cur.feature
            cur = cur[part]
        # ClassLabel or Sequence[ClassLabel]
        if hasattr(cur, "feature"):
            cur = cur.feature
        return list(getattr(cur, "names", []) or []) or None
    except Exception:
        return None


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
