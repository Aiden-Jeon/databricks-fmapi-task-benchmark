"""설정 로더 (config/models.yaml 등).

pydantic으로 스키마를 검증해, 오타나 빠진 필드를 실행 초반에 잡는다.

**모델 추가 시 이 파일이 1차 방어선이다.** 새 모델을 붙일 때 조용히 잘못 도는 경우가
많았기 때문에(아래), 실행 전에 설정을 교차 검증해 **명확한 오류로 실패**시킨다:
- reasoning 모드 누락 → 빈 파라미터를 보내 기본 reasoning이 켜지는데 리포트는 "OFF"로 표시
- capabilities 누락/오타 → vision 태스크가 조용히 전부 N/A가 되거나 미지원 모델에 이미지 전송
- 모델 ID 중복 → 뒤 항목이 앞을 덮어써 한 모델이 사라짐
- pricing.yaml 미등록 → 비용이 0으로 집계돼 "가장 저렴한 모델"로 오선정
검증은 `validate_models_config()`가 수행하고, 러너가 시작 직후 호출한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

# capabilities에 쓸 수 있는 값. 오타(`visoin`)를 잡기 위해 화이트리스트로 둔다 —
# 오타는 "그 모델은 vision 미지원"으로 해석돼 이미지 태스크가 전부 조용히 N/A가 된다.
KNOWN_CAPABILITIES = {"text", "vision"}


class RuntimeConfig(BaseModel):
    """전역 호출 정책. 모델별로 덮어쓰려면 ModelConfig.runtime을 쓴다(느린 모델 대응)."""

    timeout_seconds: float = 15.0
    max_retries: int = 3
    backoff_initial_seconds: float = 0.5
    max_concurrency: int = 8   # 미사용(러너는 순차 실행) — 병렬화 시 사용 예정
    max_tokens: int = 1024


class ModelRuntimeOverride(BaseModel):
    """모델별 런타임 오버라이드(선택). 지정한 키만 전역 설정을 덮어쓴다.

    모든 모델이 같은 timeout·max_tokens를 쓰면, 느린 모델(glm은 3~5배)이나 긴 출력을 내는
    모델(표→HTML 생성)에서 타임아웃 실패가 쏟아지고 그 실패가 점수로 오해된다
    (실측: IMG-6가 15s에서 opus 19/30 실패 → cell_f1 0.290 vs 실제 0.841).
    """

    timeout_seconds: float | None = None
    max_retries: int | None = None
    backoff_initial_seconds: float | None = None
    max_tokens: int | None = None


class ModelConfig(BaseModel):
    id: str
    endpoint: str                                # Databricks model name (예: databricks-claude-opus-5)
    # 계열 표기(manifest·문서용). **코드가 이 값으로 분기하지 않는다** — 응답 정규화는 어댑터가
    # 스키마를 보고 처리하고, reasoning 파라미터는 아래 reasoning에 직접 쓴다. family를 맞췄다고
    # 동작이 따라오지 않으니 새 모델은 reasoning을 실측해 채울 것.
    family: str                                  # claude | openai | gemini | glm 등
    capabilities: list[str] = Field(default_factory=list)  # text, vision
    reasoning: dict[str, dict[str, Any]] = Field(default_factory=dict)  # minimal/full → 파라미터
    runtime: ModelRuntimeOverride | None = None  # 모델별 timeout/max_tokens 등(선택)

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities

    def reasoning_params(self, mode: str) -> dict[str, Any]:
        """모드(minimal/full)에 해당하는 API 파라미터. 없으면 빈 dict(기본 동작).

        ⚠️ 빈 dict는 "모델 기본값"이라 **reasoning이 켜져 있을 수 있다.** 모드가 정의되지
        않은 모델은 `validate_models_config()`가 실행 전에 걸러내므로, 여기서는 설정된
        값을 그대로 돌려준다. 검증을 우회해 호출하는 경우(테스트 등)만 빈 dict가 나온다.
        """
        return self.reasoning.get(mode, {})

    def effective_runtime(self, base: RuntimeConfig) -> RuntimeConfig:
        """전역 런타임에 이 모델의 오버라이드를 적용한 값. 오버라이드가 없으면 전역 그대로."""
        if self.runtime is None:
            return base
        merged = base.model_dump()
        for k, v in self.runtime.model_dump().items():
            if v is not None:
                merged[k] = v
        return RuntimeConfig(**merged)


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


class ConfigValidationError(ValueError):
    """설정 교차 검증 실패. 여러 문제를 한 번에 모아 보고한다."""


def validate_models_config(
    cfg: ModelsConfig,
    *,
    pricing_path: str | Path = "config/pricing.yaml",
    strict_pricing: bool = False,
) -> list[str]:
    """설정을 교차 검증한다. 치명적 문제는 예외, 경고는 리스트로 반환.

    **치명적(예외)** — 이걸 통과시키면 리포트 수치가 사실과 달라진다:
    - 모델 ID 중복: 뒤 항목이 앞을 덮어써 한 모델이 조용히 사라진다
    - capabilities 비어 있음/미지의 값: 태스크 전체가 N/A가 되거나 미지원 모델에 이미지 전송
    - 실행할 reasoning 모드가 정의되지 않음: 빈 파라미터 → 기본 reasoning이 켜질 수 있는데
      리포트는 "OFF 고정"으로 서술한다(측정 조건과 문서가 어긋남)
    - judge가 models에 같은 endpoint로 들어 있음: 자기 자신을 채점하게 된다

    **경고(반환)** — 실행은 가능하나 리포트 해석에 영향:
    - pricing.yaml에 endpoint 미등록 → 비용 계산 불가(리포트가 '단가 미등록'으로 표기)
    - family가 비어 있음

    Args:
        cfg: 로드된 설정.
        pricing_path: 단가 파일 경로.
        strict_pricing: True면 pricing 누락도 치명적으로 취급(CI에서 유용).

    Returns:
        경고 메시지 리스트(치명적 문제는 ConfigValidationError로 raise).
    """
    fatal: list[str] = []
    warnings: list[str] = []

    # 1) 모델 ID·endpoint 중복
    seen_ids: dict[str, int] = {}
    for m in cfg.models:
        seen_ids[m.id] = seen_ids.get(m.id, 0) + 1
    dups = [k for k, v in seen_ids.items() if v > 1]
    if dups:
        fatal.append(f"모델 ID 중복: {dups} — 뒤 항목이 앞을 덮어써 한 모델이 사라집니다")

    seen_eps: dict[str, list[str]] = {}
    for m in cfg.models:
        seen_eps.setdefault(m.endpoint, []).append(m.id)
    for ep, ids in seen_eps.items():
        if len(ids) > 1:
            warnings.append(f"같은 endpoint를 여러 모델이 사용: {ep} → {ids}(의도적이면 무시)")

    # 2) capabilities
    for m in cfg.models:
        if not m.capabilities:
            fatal.append(
                f"모델 '{m.id}'에 capabilities가 없습니다 — 모든 태스크가 N/A로 스킵됩니다. "
                f"`[text]` 또는 `[text, vision]`을 명시하세요"
            )
        unknown = set(m.capabilities) - KNOWN_CAPABILITIES
        if unknown:
            fatal.append(
                f"모델 '{m.id}'의 capabilities에 알 수 없는 값 {sorted(unknown)} "
                f"(허용: {sorted(KNOWN_CAPABILITIES)}) — 오타면 해당 태스크가 조용히 전부 스킵됩니다"
            )

    # 3) reasoning 모드 — 실행할 모드가 모델에 정의돼 있어야 한다
    for mode in cfg.reasoning_modes:
        for m in cfg.models:
            if mode not in m.reasoning:
                fatal.append(
                    f"모델 '{m.id}'에 reasoning 모드 '{mode}'가 정의되지 않았습니다. "
                    f"빈 파라미터를 보내면 모델 기본값(=reasoning ON일 수 있음)으로 돌면서 "
                    f"리포트는 '{mode}'로 표기해 측정 조건이 어긋납니다. "
                    f"완전히 끌 수 없는 모델이면 지원되는 최소 설정을 명시하세요"
                    f"(예: `minimal: {{reasoning_effort: none}}`)"
                )

    # 4) judge가 평가 대상에 포함되면 자기 채점이 된다
    judge_eps = {m.endpoint for m in cfg.models if m.endpoint == cfg.judge}
    if judge_eps:
        fatal.append(
            f"judge endpoint '{cfg.judge}'가 평가 대상 models에도 있습니다 — 자기 자신을 "
            f"채점하게 되어 bias가 생깁니다(plan D6: 계열이 다른 judge 사용)"
        )

    # 5) family (경고)
    for m in cfg.models:
        if not (m.family or "").strip():
            warnings.append(f"모델 '{m.id}'의 family가 비어 있습니다(리포트 표기용)")

    # 6) pricing 등록 여부
    try:
        with open(pricing_path, encoding="utf-8") as f:
            pricing = yaml.safe_load(f) or {}
        priced = set((pricing.get("models") or {}).keys())
    except Exception as e:
        warnings.append(f"pricing.yaml 로드 실패({type(e).__name__}) — 비용 계산 불가")
        priced = set()
    missing_price = [m.id for m in cfg.models if m.endpoint not in priced]
    if cfg.judge not in priced:
        missing_price.append(f"judge({cfg.judge})")
    if missing_price:
        msg = (
            f"pricing.yaml에 단가가 없는 모델: {missing_price} — 비용이 계산되지 않습니다"
            f"(리포트는 '단가 미등록'으로 표기하고 비용 비교에서 제외). "
            f"`config/pricing.yaml`의 `models:`에 dbu_in/dbu_out을 추가하세요"
        )
        (fatal if strict_pricing else warnings).append(msg)

    if fatal:
        raise ConfigValidationError(
            "설정 검증 실패 — 아래를 고친 뒤 다시 실행하세요:\n"
            + "\n".join(f"  ❌ {f}" for f in fatal)
        )
    return warnings


def load_models_config(path: str | Path = "config/models.yaml") -> ModelsConfig:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return ModelsConfig(**data)
