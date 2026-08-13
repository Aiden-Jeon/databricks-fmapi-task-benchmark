"""비용 계산 테스트 (Phase 0).

pricing.yaml 로드, compute_usd 공식 검증, reasoning_tokens 포함 검증 등.
"""

import pytest
from src.cost.pricing import load_pricing, compute_usd


def test_load_pricing():
    """pricing.yaml 로드 및 기본 구조 검증."""
    pricing = load_pricing("config/pricing.yaml")

    assert "usd_per_dbu" in pricing
    assert "models" in pricing
    assert pricing["usd_per_dbu"] == 0.07

    # 평가 모델 4개와 judge 모두 있어야 함
    models = pricing["models"]
    assert "databricks-claude-opus-5" in models
    assert "databricks-gpt-5-6-sol" in models
    assert "databricks-glm-5-2" in models
    assert "databricks-kimi-k3" in models
    assert "databricks-gemini-3-1-pro" in models


def test_compute_usd_kimi_k3_official_flat_rate():
    """Kimi K3 공식 Global 단가(2026-08-13 확인)를 적용한다."""
    pricing = load_pricing("config/pricing.yaml")
    usage = {"prompt_tokens": 100_000, "completion_tokens": 10_000}

    usd = compute_usd("databricks-kimi-k3", usage, pricing)

    expected = 0.07 * (100_000 * 42.857 + 10_000 * 214.286) / 1e6
    assert usd is not None
    assert abs(usd - expected) < 0.001


def test_compute_usd_glm_flat_rate():
    """flat-rate 모델(glm)의 USD 계산.

    glm-5-2: dbu_in=20.0, dbu_out=62.857
    입력 100k 토큰, 출력 50k 토큰, reasoning 0, 캐시 0
    → billable_input=100k, billable_output=50k
    → billable_dbu = 100k*20.0 + 50k*62.857 = 2M + 3.14M = 5.14M DBU
    → usd = 0.07 * 5.14M / 1e6 = 0.36 (약)
    """
    pricing = load_pricing("config/pricing.yaml")

    usage = {
        "prompt_tokens": 100_000,
        "completion_tokens": 50_000,
    }

    usd = compute_usd("databricks-glm-5-2", usage, pricing)
    assert usd is not None
    assert isinstance(usd, float)
    assert usd > 0

    # 손계산 검증
    expected = 0.07 * (100_000 * 20.0 + 50_000 * 62.857) / 1e6
    assert abs(usd - expected) < 0.01, f"Expected ~{expected}, got {usd}"


def test_compute_usd_reasoning_tokens_included():
    """reasoning_tokens이 billable_output에 포함되는지 검증.

    같은 usage인데 reasoning_tokens만 다르면 비용이 달라야 한다 (plan §10).
    """
    pricing = load_pricing("config/pricing.yaml")

    # 기본 usage
    usage_no_reasoning = {
        "prompt_tokens": 100_000,
        "completion_tokens": 10_000,
    }

    # reasoning_tokens 포함
    usage_with_reasoning = {
        "prompt_tokens": 100_000,
        "completion_tokens": 10_000,
        "reasoning_tokens": 5_000,  # 추가
    }

    usd_no_reasoning = compute_usd("databricks-glm-5-2", usage_no_reasoning, pricing)
    usd_with_reasoning = compute_usd("databricks-glm-5-2", usage_with_reasoning, pricing)

    assert usd_no_reasoning is not None
    assert usd_with_reasoning is not None

    # reasoning_tokens이 있으면 더 비싸야 함
    assert usd_with_reasoning > usd_no_reasoning

    # reasoning_tokens 5k의 cost delta
    # 추가 5k tokens @ dbu_out=62.857 → 5k * 62.857 * 0.07 / 1e6 ≈ 0.022
    delta_expected = 0.07 * 5_000 * 62.857 / 1e6
    delta_actual = usd_with_reasoning - usd_no_reasoning
    assert abs(delta_actual - delta_expected) < 0.001


def test_compute_usd_tiered_short():
    """tiered 모델(sol)의 short 단가 선택.

    sol short 조건: prompt_tokens < 200000
    """
    pricing = load_pricing("config/pricing.yaml")

    usage_short = {
        "prompt_tokens": 100_000,  # < 200k
        "completion_tokens": 10_000,
    }

    usd = compute_usd("databricks-gpt-5-6-sol", usage_short, pricing)
    assert usd is not None
    assert isinstance(usd, float)
    assert usd > 0

    # short 단가로 계산
    short_rates = pricing["models"]["databricks-gpt-5-6-sol"]["short"]
    expected = 0.07 * (100_000 * short_rates["dbu_in"] + 10_000 * short_rates["dbu_out"]) / 1e6
    assert abs(usd - expected) < 0.01


def test_compute_usd_tiered_long():
    """tiered 모델(sol)의 long 단가 선택.

    sol long 조건: prompt_tokens >= 200000
    """
    pricing = load_pricing("config/pricing.yaml")

    usage_long = {
        "prompt_tokens": 250_000,  # > 200k
        "completion_tokens": 10_000,
    }

    usd = compute_usd("databricks-gpt-5-6-sol", usage_long, pricing)
    assert usd is not None
    assert isinstance(usd, float)
    assert usd > 0

    # long 단가로 계산
    long_rates = pricing["models"]["databricks-gpt-5-6-sol"]["long"]
    expected = 0.07 * (250_000 * long_rates["dbu_in"] + 10_000 * long_rates["dbu_out"]) / 1e6
    assert abs(usd - expected) < 0.01


def test_compute_usd_cached_tokens():
    """캐시 읽기 토큰의 비용 절감.

    billable_input = prompt_tokens - cached_tokens (floor 0)
    """
    pricing = load_pricing("config/pricing.yaml")

    usage_with_cache = {
        "prompt_tokens": 100_000,
        "completion_tokens": 10_000,
        "prompt_tokens_details": {"cached_tokens": 30_000},
    }

    usd = compute_usd("databricks-glm-5-2", usage_with_cache, pricing)
    assert usd is not None

    # billable_input = 100k - 30k = 70k (캐시 읽기로 30k 토큰 비용 면함)
    expected = 0.07 * (70_000 * 20.0 + 10_000 * 62.857) / 1e6
    assert abs(usd - expected) < 0.01


def test_compute_usd_unknown_endpoint():
    """존재하지 않는 엔드포인트 → None 반환."""
    pricing = load_pricing("config/pricing.yaml")

    usage = {
        "prompt_tokens": 100_000,
        "completion_tokens": 10_000,
    }

    usd = compute_usd("databricks-unknown-model", usage, pricing)
    assert usd is None


def test_compute_usd_opus_with_promo():
    """promo_multiplier 적용 (gemini 20% 할인).

    gemini-3-1-pro는 promo_multiplier: 0.80 (20% 할인)
    """
    pricing = load_pricing("config/pricing.yaml")

    usage = {
        "prompt_tokens": 100_000,
        "completion_tokens": 10_000,
    }

    usd = compute_usd("databricks-gemini-3-1-pro", usage, pricing)
    assert usd is not None

    # gemini short 단가로 계산
    short_rates = pricing["models"]["databricks-gemini-3-1-pro"]["short"]
    base_usd = 0.07 * (100_000 * short_rates["dbu_in"] + 10_000 * short_rates["dbu_out"]) / 1e6
    expected_with_promo = base_usd * 0.80

    assert abs(usd - expected_with_promo) < 0.001


def test_compute_usd_reasoning_nested_paths():
    """reasoning_tokens 추출: 여러 중첩 경로 지원.

    token_details에 output_reasoning_tokens이 있는 경우 (ai_gateway 형태).
    """
    pricing = load_pricing("config/pricing.yaml")

    # ai_gateway 형태: token_details.output_reasoning_tokens
    usage_ai_gateway = {
        "prompt_tokens": 100_000,
        "completion_tokens": 10_000,
        "token_details": {"output_reasoning_tokens": 5_000},
    }

    usd = compute_usd("databricks-glm-5-2", usage_ai_gateway, pricing)
    assert usd is not None

    # billable_output = 10k + 5k = 15k
    expected = 0.07 * (100_000 * 20.0 + 15_000 * 62.857) / 1e6
    assert abs(usd - expected) < 0.01
