"""config 로더·검증 테스트.

**모델 수·ID를 하드코딩하지 않는다.** 예전엔 `len(config.models) == 3`과
`set(ids) == {"opus","sol","glm"}`로 고정해, 모델을 하나 추가하면 코드가 멀쩡한데도
pytest가 실패했다(모델 추가가 목적인 프로젝트에서 잘못된 계약). 대신 **설정이 지켜야 할
성질**을 검증한다: 필수 필드, capabilities 유효성, 실행할 reasoning 모드 정의,
ID 유일성, pricing 등록, 모델별 런타임 오버라이드 병합.

현재 3모델(opus·sol·glm)에 대한 개별 값 검증은 `test_known_models_*`에 모아
"현재 구성 스냅샷"으로 두고, 그 모델이 config에 없으면 자동으로 skip한다.
"""

import pytest
import yaml

from src.config import (
    KNOWN_CAPABILITIES,
    ConfigValidationError,
    ModelsConfig,
    RuntimeConfig,
    load_models_config,
    validate_models_config,
)


@pytest.fixture(scope="module")
def config():
    return load_models_config("config/models.yaml")


# ──────────────────────────────────────────────── 구조 계약(모델 수와 무관)


def test_basic_fields(config):
    """필수 상위 필드가 채워져 있다."""
    assert config.profile, "profile이 비어 있으면 어느 워크스페이스로 호출되는지 알 수 없다"
    assert config.judge, "judge 엔드포인트가 필요하다"
    assert config.reasoning_modes, "실행할 reasoning 모드가 최소 1개 필요하다"
    assert config.models, "평가 대상 모델이 최소 1개 필요하다"


def test_model_ids_unique(config):
    """모델 ID는 유일해야 한다 — 중복이면 뒤 항목이 앞을 덮어써 한 모델이 사라진다."""
    ids = [m.id for m in config.models]
    assert len(ids) == len(set(ids)), f"ID 중복: {[i for i in ids if ids.count(i) > 1]}"


def test_every_model_has_required_fields(config):
    """모든 모델에 endpoint·capabilities가 있고 capability 값이 유효하다."""
    for m in config.models:
        assert m.endpoint, f"{m.id}: endpoint 누락"
        assert m.capabilities, f"{m.id}: capabilities 누락 → 모든 태스크가 N/A로 스킵된다"
        unknown = set(m.capabilities) - KNOWN_CAPABILITIES
        assert not unknown, f"{m.id}: 알 수 없는 capability {unknown}(오타면 태스크가 조용히 스킵된다)"
        assert "text" in m.capabilities, f"{m.id}: text는 모든 모델의 기본 capability다"


def test_every_model_defines_active_reasoning_modes(config):
    """실행할 모드가 모든 모델에 정의돼 있다.

    빈 파라미터를 보내면 모델 기본값(=reasoning ON일 수 있음)으로 돌면서 리포트는
    'OFF'로 표기해 측정 조건과 문서가 어긋난다.
    """
    for mode in config.reasoning_modes:
        for m in config.models:
            assert mode in m.reasoning, (
                f"{m.id}: reasoning 모드 '{mode}' 미정의 — 완전히 끌 수 없는 모델이면 "
                f"지원되는 최소 설정을 명시할 것"
            )


def test_judge_not_in_evaluated_models(config):
    """judge는 평가 대상이 아니어야 한다(자기 채점 방지 — plan D6)."""
    assert config.judge not in {m.endpoint for m in config.models}


# 확정 단가를 구하지 못해 **의도적으로** pricing.yaml에서 뺀 모델(비용 비교에서만 제외,
# 성능 비교는 정상). 추측 단가를 넣으면 "가장 저렴/비싼"으로 오선정되므로 비우는 게 맞다.
# 여기에 명시된 모델만 미등록이 허용된다 — 그 외 누락은 여전히 실패한다(가드 유지).
# 확정 단가를 구하면 pricing.yaml에 추가하고 이 목록에서 빼면 된다.
KNOWN_UNPRICED = {"kimi"}  # databricks-kimi-k3: 공개 단가 미확인(2026-08-07). pricing.yaml 주석 참고.


def test_all_models_have_pricing(config):
    """평가 모델은 pricing.yaml에 등록돼 있어야 한다(단, KNOWN_UNPRICED는 명시적 예외).

    누락되면 비용이 계산되지 않고, 0으로 두면 "가장 저렴한 모델"로 오선정된다.
    **judge는 예외 없이 반드시 등록**돼야 한다 — 비용 계산 경로에 항상 들어가므로 $0이면
    모든 비용 수치가 오염된다.
    """
    pricing = yaml.safe_load(open("config/pricing.yaml", encoding="utf-8"))
    priced = set((pricing.get("models") or {}).keys())
    missing = [m.id for m in config.models if m.endpoint not in priced]
    # 예상치 못한 누락(명시적 예외가 아닌 것)만 실패시킨다.
    unexpected = [mid for mid in missing if mid not in KNOWN_UNPRICED]
    assert not unexpected, (
        f"pricing.yaml 미등록(예상치 못함): {unexpected}. "
        f"의도적 미등록이면 tests의 KNOWN_UNPRICED에 추가하고 pricing.yaml에 사유를 남길 것"
    )
    assert config.judge in priced, f"judge({config.judge}) 단가 미등록 — judge는 예외 없이 필수"


def test_validate_models_config_passes_on_repo_config(config):
    """repo의 현재 설정은 교차 검증을 통과한다(경고는 허용)."""
    warnings = validate_models_config(config)
    assert isinstance(warnings, list)


# ──────────────────────────────────────────────── 검증기가 오설정을 잡는지


def _cfg_with_extra_model(**overrides):
    base = yaml.safe_load(open("config/models.yaml", encoding="utf-8"))
    model = {
        "id": "newmodel",
        "endpoint": "databricks-new-model",
        "family": "newfamily",
        "capabilities": ["text"],
        "reasoning": {m: {"reasoning_effort": "none"} for m in base["reasoning_modes"]},
    }
    model.update(overrides)
    base["models"].append(model)
    return ModelsConfig(**base)


def test_validator_rejects_missing_reasoning_mode():
    """새 모델에 실행 모드가 없으면 차단 — 조용히 기본 reasoning이 켜지는 것을 막는다."""
    with pytest.raises(ConfigValidationError, match="reasoning 모드"):
        validate_models_config(_cfg_with_extra_model(reasoning={}))


def test_validator_rejects_missing_capabilities():
    with pytest.raises(ConfigValidationError, match="capabilities"):
        validate_models_config(_cfg_with_extra_model(capabilities=[]))


def test_validator_rejects_unknown_capability():
    with pytest.raises(ConfigValidationError, match="알 수 없는 값"):
        validate_models_config(_cfg_with_extra_model(capabilities=["text", "visoin"]))


def test_validator_rejects_duplicate_id(config):
    with pytest.raises(ConfigValidationError, match="ID 중복"):
        validate_models_config(_cfg_with_extra_model(id=config.models[0].id))


def test_validator_rejects_judge_as_target(config):
    with pytest.raises(ConfigValidationError, match="judge"):
        validate_models_config(_cfg_with_extra_model(endpoint=config.judge))


def test_validator_warns_on_missing_pricing():
    """단가 미등록은 경고(실행 가능) — 리포트가 '단가 미등록'으로 표기한다."""
    warnings = validate_models_config(_cfg_with_extra_model())
    assert any("pricing.yaml" in w for w in warnings), warnings


def test_validator_strict_pricing_raises():
    """strict_pricing=True면 단가 누락도 차단(CI용)."""
    with pytest.raises(ConfigValidationError, match="pricing.yaml"):
        validate_models_config(_cfg_with_extra_model(), strict_pricing=True)


# ──────────────────────────────────────────────── 모델별 런타임 오버라이드


def test_effective_runtime_without_override(config):
    """오버라이드가 없으면 전역 런타임 그대로."""
    base = RuntimeConfig(timeout_seconds=60, max_retries=5, max_tokens=1024)
    m = next(m for m in config.models if m.runtime is None)
    assert m.effective_runtime(base).timeout_seconds == 60


def test_effective_runtime_merges_only_given_keys():
    """지정한 키만 덮어쓰고 나머지는 전역값을 유지한다."""
    cfg = _cfg_with_extra_model(runtime={"timeout_seconds": 300})
    m = cfg.get_model("newmodel")
    base = RuntimeConfig(timeout_seconds=60, max_retries=5, max_tokens=1024)
    eff = m.effective_runtime(base)
    assert eff.timeout_seconds == 300      # 오버라이드 적용
    assert eff.max_retries == 5            # 전역 유지
    assert eff.max_tokens == 1024          # 전역 유지


# ──────────────────────────────────────────────── 현재 구성 스냅샷(있으면 검증)

# 모델이 config에서 빠지거나 이름이 바뀌면 자동 skip — 테스트가 모델 추가·교체를 막지 않는다.
_KNOWN = {
    "opus": {"endpoint": "databricks-claude-opus-5", "vision": True,
             "minimal": {"thinking": {"type": "disabled"}}},
    "sol": {"endpoint": "databricks-gpt-5-6-sol", "vision": True,
            "minimal": {"reasoning_effort": "none"}},
    "glm": {"endpoint": "databricks-glm-5-2", "vision": False,
            "minimal": {"reasoning_effort": "none"}},
}


@pytest.mark.parametrize("model_id", sorted(_KNOWN))
def test_known_models_snapshot(config, model_id):
    """현재 구성의 실측값 스냅샷(endpoint·vision·minimal 파라미터)."""
    try:
        m = config.get_model(model_id)
    except KeyError:
        pytest.skip(f"{model_id}가 현재 config에 없음(제거·교체된 경우 정상)")
    exp = _KNOWN[model_id]
    assert m.endpoint == exp["endpoint"]
    assert m.supports("vision") is exp["vision"]
    assert m.reasoning_params("minimal") == exp["minimal"]


def test_model_not_found_raises(config):
    with pytest.raises(KeyError):
        config.get_model("nonexistent-model-id")
