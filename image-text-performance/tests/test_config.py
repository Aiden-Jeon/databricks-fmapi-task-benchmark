"""config 로더 테스트 (Phase 0).

config/models.yaml 로드, 모델 정보, reasoning 파라미터 검증.
"""

import pytest
from src.config import load_models_config, ModelConfig


def test_load_models_config():
    """models.yaml 로드 및 기본 구조 검증."""
    config = load_models_config("config/models.yaml")

    # 기본 필드
    assert config.profile == "ai_devtools"
    assert config.judge == "databricks-gemini-3-1-pro"
    assert config.reasoning_modes == ["minimal", "full"]

    # 모델 개수
    assert len(config.models) == 3
    model_ids = [m.id for m in config.models]
    assert set(model_ids) == {"opus", "sol", "glm"}


def test_model_capabilities():
    """모델별 capability 선언 (vision 지원 여부)."""
    config = load_models_config("config/models.yaml")

    opus = config.get_model("opus")
    assert opus.supports("text")
    assert opus.supports("vision")

    sol = config.get_model("sol")
    assert sol.supports("text")
    assert sol.supports("vision")

    glm = config.get_model("glm")
    assert glm.supports("text")
    assert not glm.supports("vision")  # glm은 vision 미지원


def test_reasoning_params_minimal():
    """minimal 모드의 reasoning 파라미터."""
    config = load_models_config("config/models.yaml")

    # opus: thinking disabled
    opus = config.get_model("opus")
    opus_minimal = opus.reasoning_params("minimal")
    assert "thinking" in opus_minimal
    assert opus_minimal["thinking"]["type"] == "disabled"

    # sol: reasoning_effort none
    sol = config.get_model("sol")
    sol_minimal = sol.reasoning_params("minimal")
    assert sol_minimal["reasoning_effort"] == "none"

    # glm: reasoning_effort none
    glm = config.get_model("glm")
    glm_minimal = glm.reasoning_params("minimal")
    assert glm_minimal["reasoning_effort"] == "none"


def test_reasoning_params_full():
    """full 모드의 reasoning 파라미터."""
    config = load_models_config("config/models.yaml")

    # opus: thinking adaptive
    opus = config.get_model("opus")
    opus_full = opus.reasoning_params("full")
    assert "thinking" in opus_full
    assert opus_full["thinking"]["type"] == "adaptive"

    # sol: reasoning_effort high
    sol = config.get_model("sol")
    sol_full = sol.reasoning_params("full")
    assert sol_full["reasoning_effort"] == "high"

    # glm: 기본(empty dict = 기본값 사용)
    glm = config.get_model("glm")
    glm_full = glm.reasoning_params("full")
    assert glm_full == {}


def test_model_not_found():
    """존재하지 않는 모델 조회 시 KeyError."""
    config = load_models_config("config/models.yaml")
    with pytest.raises(KeyError):
        config.get_model("nonexistent")


def test_model_endpoints():
    """모델의 FMAPI 엔드포인트명."""
    config = load_models_config("config/models.yaml")

    opus = config.get_model("opus")
    assert opus.endpoint == "databricks-claude-opus-5"

    sol = config.get_model("sol")
    assert sol.endpoint == "databricks-gpt-5-6-sol"

    glm = config.get_model("glm")
    assert glm.endpoint == "databricks-glm-5-2"
