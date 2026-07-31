"""비용 계산: FMAPI 비용 모델 로드 및 USD 환산 (plan §10).

비용 계산의 단일 소스. `config/pricing.yaml`로부터 DBU 단가를 로드하고,
ai_gateway.usage 데이터(토큰, 캐시 통계)와 결합해 실제 USD 비용을 계산한다.

**핵심 공식** (plan §10):
    usd = usd_per_dbu × (
        billable_input × dbu_in +
        billable_output × dbu_out +
        cached_read × dbu_cache_read +
        cache_write × dbu_cache_write
    ) / 1e6

여기서:
    - billable_output = completion_tokens + reasoning_tokens
      (reasoning 토큰이 completion_tokens에 포함되지 않으므로 별도 추가 필수)
    - billable_input = prompt_tokens - cached_tokens (floor 0)
    - cached_read, cache_write는 선택적 필드

**context 임계값**: 일부 모델(sol, gemini)은 context 크기에 따라 short/long 가격이 다름.
    → prompt_tokens vs context_threshold_tokens로 비교해 맞는 행 선택.

**promo_multiplier**: 제한 기간 프로모션(예: gemini 20% 할인) 적용.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_pricing(path: str | Path = "config/pricing.yaml") -> dict[str, Any]:
    """config/pricing.yaml을 읽어 가격 테이블 반환.

    Args:
        path: pricing.yaml 경로 (기본값: config/pricing.yaml)

    Returns:
        {
            'usd_per_dbu': float,
            'routing': str,
            'models': {
                'endpoint_name': {
                    'dbu_in': float,
                    'dbu_out': float,
                    'dbu_cache_read': float (optional),
                    'dbu_cache_write': float (optional),
                    'context_threshold_tokens': int (optional, tiered models),
                    'short': {...} (optional, if tiered),
                    'long': {...} (optional, if tiered),
                    'promo_multiplier': float (optional)
                }
            }
        }

    Raises:
        FileNotFoundError: pricing.yaml를 찾을 수 없으면.
        yaml.YAMLError: YAML 파싱 오류.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"pricing.yaml not found at {path}")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data


def compute_usd(
    model_endpoint: str,
    usage: dict[str, Any],
    pricing: dict[str, Any],
) -> float | None:
    """model_endpoint의 usage 기반 USD 비용 계산.

    Args:
        model_endpoint: 엔드포인트명 (예: "databricks-claude-opus-5")
        usage: ai_gateway.usage에서 나온 dict. 필드:
            - prompt_tokens (또는 input_tokens): 입력 토큰
            - completion_tokens (또는 output_tokens): 출력 토큰
            - reasoning_tokens: 사고(reasoning) 토큰 (선택적, 여러 경로에서 올 수 있음)
              * usage['reasoning_tokens']
              * usage['completion_tokens_details']['reasoning_tokens']
              * usage['token_details']['output_reasoning_tokens']
            - cached_tokens: 캐시 읽기 토큰 (선택적, 여러 경로)
              * usage['prompt_tokens_details']['cached_tokens']
              * usage['token_details']['cache_read_input_tokens']
              * usage['cache_read_input_tokens']
            - cache_creation_input_tokens: 캐시 쓰기 (선택적)
        pricing: load_pricing() 반환값

    Returns:
        float: USD 비용 (양수). 모델을 찾을 수 없으면 None.

    Note:
        plan §10에 따라, billable_output는 completion_tokens + reasoning_tokens.
        일부 usage dict에는 reasoning_tokens이 부모 completion_tokens에 포함되어 있을 수 있고,
        일부는 완전히 분리되어 있음. 방어적으로 여러 경로에서 추출.
    """

    # 모델을 가격표에서 찾기
    models = pricing.get("models", {})
    if model_endpoint not in models:
        logger.warning(f"Endpoint '{model_endpoint}' not found in pricing table")
        return None

    model_rates = models[model_endpoint]
    usd_per_dbu = pricing.get("usd_per_dbu", 0.07)

    # === 토큰 추출 (방어적: 여러 필드명 대응) ===
    prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens", 0)
    completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens", 0)

    # reasoning_tokens: 여러 경로에서 탐색
    reasoning_tokens = 0
    if "reasoning_tokens" in usage:
        reasoning_tokens = usage["reasoning_tokens"]
    elif "completion_tokens_details" in usage and "reasoning_tokens" in usage["completion_tokens_details"]:
        reasoning_tokens = usage["completion_tokens_details"]["reasoning_tokens"]
    elif "token_details" in usage and "output_reasoning_tokens" in usage["token_details"]:
        reasoning_tokens = usage["token_details"]["output_reasoning_tokens"]

    # cached_tokens: 여러 경로에서 탐색
    cached_tokens = 0
    if "prompt_tokens_details" in usage and "cached_tokens" in usage["prompt_tokens_details"]:
        cached_tokens = usage["prompt_tokens_details"]["cached_tokens"]
    elif "token_details" in usage and "cache_read_input_tokens" in usage["token_details"]:
        cached_tokens = usage["token_details"]["cache_read_input_tokens"]
    elif "cache_read_input_tokens" in usage:
        cached_tokens = usage["cache_read_input_tokens"]

    cache_write_tokens = usage.get("cache_creation_input_tokens", 0)

    # === billable 토큰 계산 ===
    billable_input = max(0, prompt_tokens - cached_tokens)  # 캐시 읽기는 비용 차감
    billable_output = completion_tokens + reasoning_tokens  # 핵심: reasoning 포함

    # === DBU 단가 결정 (tiered 모델 처리) ===
    # tiered 모델: short/long 키가 있고, context_threshold_tokens로 분기
    if "short" in model_rates and "long" in model_rates:
        threshold = model_rates.get("context_threshold_tokens", 200000)
        if prompt_tokens < threshold:
            rates = model_rates["short"]
        else:
            rates = model_rates["long"]
    else:
        # flat-rate 모델
        rates = model_rates

    dbu_in = rates.get("dbu_in", 0)
    dbu_out = rates.get("dbu_out", 0)
    dbu_cache_read = rates.get("dbu_cache_read", 0)
    dbu_cache_write = rates.get("dbu_cache_write", 0)

    # === USD 계산 ===
    billable_dbu = (
        billable_input * dbu_in +
        billable_output * dbu_out +
        cached_tokens * dbu_cache_read +
        cache_write_tokens * dbu_cache_write
    )

    usd = usd_per_dbu * billable_dbu / 1e6

    # === promo_multiplier 적용 (있으면) ===
    if "promo_multiplier" in model_rates:
        usd *= model_rates["promo_multiplier"]

    return usd
