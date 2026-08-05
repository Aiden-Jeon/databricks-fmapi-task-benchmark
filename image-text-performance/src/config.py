"""설정 로더 (config/models.yaml 등).

pydantic으로 스키마를 검증해, 오타나 빠진 필드를 실행 초반에 잡는다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class RuntimeConfig(BaseModel):
    timeout_seconds: float = 15.0
    max_retries: int = 3
    backoff_initial_seconds: float = 0.5
    max_concurrency: int = 8   # 미사용(러너는 순차 실행) — 병렬화 시 사용 예정
    max_tokens: int = 1024


class ModelConfig(BaseModel):
    id: str
    endpoint: str                                # Databricks model name (예: databricks-claude-opus-5)
    family: str                                  # claude | openai (정규화·분기 키)
    capabilities: list[str] = Field(default_factory=list)  # text, vision
    reasoning: dict[str, dict[str, Any]] = Field(default_factory=dict)  # minimal/full → 파라미터

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities

    def reasoning_params(self, mode: str) -> dict[str, Any]:
        """모드(minimal/full)에 해당하는 API 파라미터. 없으면 빈 dict(기본 동작)."""
        return self.reasoning.get(mode, {})


class ModelsConfig(BaseModel):
    profile: str
    judge: str
    reasoning_modes: list[str] = Field(default_factory=lambda: ["minimal", "full"])
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    models: list[ModelConfig]

    def get_model(self, model_id: str) -> ModelConfig:
        for m in self.models:
            if m.id == model_id:
                return m
        raise KeyError(f"모델 '{model_id}'를 config에서 찾을 수 없음")


def load_models_config(path: str | Path = "config/models.yaml") -> ModelsConfig:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return ModelsConfig(**data)
