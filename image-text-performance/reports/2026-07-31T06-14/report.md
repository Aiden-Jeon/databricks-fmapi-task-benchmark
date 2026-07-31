# 벤치마크 리포트 — 2026-07-31T06-14

## Executive Summary

제공된 데이터에 따르면 Opus는 이미지(IMG) 작업에서 강세를 보이며 총 8개 항목에서 우승한 반면, Sol(7승)과 GLM(5승)은 주로 텍스트(TXT) 작업에서 두각을 나타냈습니다. 속도와 비용 면에서 Sol은 지연 시간 중앙값이 1740.6ms로 가장 빠른 모델이며, Opus는 총비용 1.59달러로 가장 비싸지만 오류 없이 안정적인 성능을 보여줍니다. 반면 GLM은 총비용이 0.25달러로 가장 저렴하지만, 지연 시간이 4014.1ms로 가장 느리고 유일하게 19건의 오류가 발생하여 속도와 안정성을 희생해 비용을 절감하는 트레이드오프를 보입니다.

<sub>규칙 기반 요약(대조용): 태스크별 1위 횟수는 opus 8회, sol 7회, glm 5회로 **opus**가 가장 많다. 응답 속도는 **sol**가 가장 빠르다(median 1740.6ms). 비용은 **glm**가 가장 낮다($0.252097).</sub>

## 정량 결과 (태스크 × 모델 × reasoning)

| 태스크 | 모델 | reasoning | 대표 메트릭 |
|---|---|---|---|
| TXT-1 | glm | full | token_f1=0.0, exact_match=0.0, n_evaluated=10 |
| TXT-1 | glm | minimal | token_f1=0.002, exact_match=0.0, n_evaluated=10 |
| TXT-2 | glm | full | accuracy=0.3, token_f1=0.399, n_evaluated=10 |
| TXT-2 | glm | minimal | accuracy=0.3, token_f1=0.414, n_evaluated=10 |
| TXT-3 | glm | full | cell_f1=0.218, n_evaluated=10 |
| TXT-3 | glm | minimal | cell_f1=0.791, n_evaluated=10 |
| TXT-4 | glm | full | token_f1=0.867, exact_match=0.8, n_evaluated=10 |
| TXT-4 | glm | minimal | token_f1=0.853, exact_match=0.8, n_evaluated=10 |
| TXT-5 | glm | full | rouge1=0.162, rouge2=0.077, rougeL=0.136, n_evaluated=10 |
| TXT-5 | glm | minimal | rouge1=0.256, rouge2=0.1, rougeL=0.212, n_evaluated=10 |
| TXT-6 | glm | full | accuracy=1.0, macro_f1=1.0, n_evaluated=9, n_unparsed=1 |
| TXT-6 | glm | minimal | accuracy=1.0, macro_f1=1.0, n_evaluated=10, n_unparsed=0 |
| TXT-7 | glm | full | precision=0.313, recall=0.32, f1=0.317, n_evaluated=10 |
| TXT-7 | glm | minimal | precision=0.54, recall=0.422, f1=0.474, n_evaluated=10 |
| TXT-8 | glm | full | accuracy=0.889, f1=0.0, n_evaluated=9, n_unparsed=1 |
| TXT-8 | glm | minimal | accuracy=1.0, f1=1.0, n_evaluated=10, n_unparsed=0 |
| IMG-1 | opus | full | caption_token_f1=0.287, n_evaluated=10 |
| IMG-1 | opus | minimal | caption_token_f1=0.302, n_evaluated=10 |
| IMG-2 | opus | full | micro_precision=0.136, micro_recall=0.471, micro_f1=0.211, macro_precision=0.144 |
| IMG-2 | opus | minimal | micro_precision=0.12, micro_recall=0.412, micro_f1=0.185, macro_precision=0.135 |
| IMG-3 | opus | full | accuracy=0.5, f1=0.667, n_evaluated=10, n_unparsed=0 |
| IMG-3 | opus | minimal | accuracy=0.5, f1=0.667, n_evaluated=10, n_unparsed=0 |
| IMG-4 | opus | full | accuracy=0.8, f1=0.75, n_evaluated=10, n_unparsed=0 |
| IMG-4 | opus | minimal | accuracy=0.8, f1=0.75, n_evaluated=10, n_unparsed=0 |
| IMG-5 | opus | full | accuracy=1.0, f1=1.0, n_evaluated=10, n_unparsed=0 |
| IMG-5 | opus | minimal | accuracy=0.9, f1=0.909, n_evaluated=10, n_unparsed=0 |
| TXT-1 | opus | full | token_f1=0.0, exact_match=0.0, n_evaluated=10 |
| TXT-1 | opus | minimal | token_f1=0.001, exact_match=0.0, n_evaluated=10 |
| TXT-2 | opus | full | accuracy=0.0, token_f1=0.042, n_evaluated=10 |
| TXT-2 | opus | minimal | accuracy=0.1, token_f1=0.194, n_evaluated=10 |
| TXT-3 | opus | full | cell_f1=0.946, n_evaluated=10 |
| TXT-3 | opus | minimal | cell_f1=0.966, n_evaluated=10 |
| TXT-4 | opus | full | token_f1=0.551, exact_match=0.4, n_evaluated=10 |
| TXT-4 | opus | minimal | token_f1=0.706, exact_match=0.5, n_evaluated=10 |
| TXT-5 | opus | full | rouge1=0.274, rouge2=0.134, rougeL=0.235, n_evaluated=10 |
| TXT-5 | opus | minimal | rouge1=0.285, rouge2=0.153, rougeL=0.238, n_evaluated=10 |
| TXT-6 | opus | full | accuracy=1.0, macro_f1=1.0, n_evaluated=10, n_unparsed=0 |
| TXT-6 | opus | minimal | accuracy=1.0, macro_f1=1.0, n_evaluated=10, n_unparsed=0 |
| TXT-7 | opus | full | precision=0.345, recall=0.445, f1=0.389, n_evaluated=10 |
| TXT-7 | opus | minimal | precision=0.4, recall=0.484, f1=0.438, n_evaluated=10 |
| TXT-8 | opus | full | accuracy=0.7, f1=0.4, n_evaluated=10, n_unparsed=0 |
| TXT-8 | opus | minimal | accuracy=0.7, f1=0.4, n_evaluated=10, n_unparsed=0 |
| IMG-1 | sol | full | caption_token_f1=0.383, n_evaluated=10 |
| IMG-1 | sol | minimal | caption_token_f1=0.387, n_evaluated=10 |
| IMG-2 | sol | full | micro_precision=0.19, micro_recall=0.324, micro_f1=0.239, macro_precision=0.238 |
| IMG-2 | sol | minimal | micro_precision=0.175, micro_recall=0.294, micro_f1=0.22, macro_precision=0.226 |
| IMG-3 | sol | full | accuracy=0.5, f1=0.667, n_evaluated=10, n_unparsed=0 |
| IMG-3 | sol | minimal | accuracy=0.5, f1=0.667, n_evaluated=10, n_unparsed=0 |
| IMG-4 | sol | full | accuracy=0.8, f1=0.75, n_evaluated=10, n_unparsed=0 |
| IMG-4 | sol | minimal | accuracy=0.8, f1=0.75, n_evaluated=10, n_unparsed=0 |
| IMG-5 | sol | full | accuracy=0.9, f1=0.909, n_evaluated=10, n_unparsed=0 |
| IMG-5 | sol | minimal | accuracy=0.9, f1=0.909, n_evaluated=10, n_unparsed=0 |
| TXT-1 | sol | full | token_f1=0.01, exact_match=0.0, n_evaluated=10 |
| TXT-1 | sol | minimal | token_f1=0.013, exact_match=0.0, n_evaluated=10 |
| TXT-2 | sol | full | accuracy=0.0, token_f1=0.33, n_evaluated=10 |
| TXT-2 | sol | minimal | accuracy=0.0, token_f1=0.265, n_evaluated=10 |
| TXT-3 | sol | full | cell_f1=0.7, n_evaluated=10 |
| TXT-3 | sol | minimal | cell_f1=0.957, n_evaluated=10 |
| TXT-4 | sol | full | token_f1=0.869, exact_match=0.7, n_evaluated=10 |
| TXT-4 | sol | minimal | token_f1=0.883, exact_match=0.7, n_evaluated=10 |
| TXT-5 | sol | full | rouge1=0.276, rouge2=0.124, rougeL=0.234, n_evaluated=10 |
| TXT-5 | sol | minimal | rouge1=0.318, rouge2=0.161, rougeL=0.27, n_evaluated=10 |
| TXT-6 | sol | full | accuracy=1.0, macro_f1=1.0, n_evaluated=10, n_unparsed=0 |
| TXT-6 | sol | minimal | accuracy=1.0, macro_f1=1.0, n_evaluated=10, n_unparsed=0 |
| TXT-7 | sol | full | precision=0.442, recall=0.477, f1=0.459, n_evaluated=10 |
| TXT-7 | sol | minimal | precision=0.4, recall=0.438, f1=0.418, n_evaluated=10 |
| TXT-8 | sol | full | accuracy=0.8, f1=0.5, n_evaluated=10, n_unparsed=0 |
| TXT-8 | sol | minimal | accuracy=0.8, f1=0.5, n_evaluated=10, n_unparsed=0 |

## 성능: 수행시간·비용 (모델별)

| 모델 | 호출 | 오류 | latency median(ms) | p95(ms) | 입력토큰 | 출력토큰 | 비용(USD) |
|---|---|---|---|---|---|---|---|
| glm | 160 | 19 | 4014.1 | 21611.8 | 50900 | 41751 | 0.252097 |
| opus | 260 | 0 | 2953.0 | 8823.6 | 125936 | 38649 | 1.595909 |
| sol | 260 | 0 | 1740.6 | 4754.4 | 90854 | 20083 | 1.220562 |

> 비용은 `config/pricing.yaml` DBU 단가 기반 추정(usd_per_dbu 가정값). 정밀 비용은 `system.ai_gateway.usage` 조인 필요(§10).

## Fact Sheet (Executive Summary 근거 — 감사용)

```json
{
  "per_task_scores": {
    "IMG-3/minimal": {
      "opus": 0.5,
      "sol": 0.5
    },
    "IMG-3/full": {
      "opus": 0.5,
      "sol": 0.5
    },
    "IMG-4/minimal": {
      "opus": 0.8,
      "sol": 0.8
    },
    "IMG-4/full": {
      "opus": 0.8,
      "sol": 0.8
    },
    "IMG-5/minimal": {
      "opus": 0.9,
      "sol": 0.9
    },
    "IMG-5/full": {
      "opus": 1.0,
      "sol": 0.9
    },
    "TXT-1/minimal": {
      "opus": 0.0011,
      "sol": 0.0125,
      "glm": 0.0022
    },
    "TXT-1/full": {
      "opus": 0.0,
      "sol": 0.01,
      "glm": 0.0
    },
    "TXT-2/minimal": {
      "opus": 0.1,
      "sol": 0.0,
      "glm": 0.3
    },
    "TXT-2/full": {
      "opus": 0.0,
      "sol": 0.0,
      "glm": 0.3
    },
    "TXT-4/minimal": {
      "opus": 0.7059,
      "sol": 0.8833,
      "glm": 0.8526
    },
    "TXT-4/full": {
      "opus": 0.551,
      "sol": 0.8693,
      "glm": 0.8667
    },
    "TXT-5/minimal": {
      "opus": 0.2854,
      "sol": 0.3175,
      "glm": 0.2565
    },
    "TXT-5/full": {
      "opus": 0.2744,
      "sol": 0.2758,
      "glm": 0.1624
    },
    "TXT-6/minimal": {
      "opus": 1.0,
      "sol": 1.0,
      "glm": 1.0
    },
    "TXT-6/full": {
      "opus": 1.0,
      "sol": 1.0,
      "glm": 1.0
    },
    "TXT-7/minimal": {
      "opus": 0.4382,
      "sol": 0.4179,
      "glm": 0.4737
    },
    "TXT-7/full": {
      "opus": 0.3891,
      "sol": 0.4586,
      "glm": 0.3166
    },
    "TXT-8/minimal": {
      "opus": 0.7,
      "sol": 0.8,
      "glm": 1.0
    },
    "TXT-8/full": {
      "opus": 0.7,
      "sol": 0.8,
      "glm": 0.8889
    }
  },
  "task_winners": {
    "IMG-3/minimal": "opus",
    "IMG-3/full": "opus",
    "IMG-4/minimal": "opus",
    "IMG-4/full": "opus",
    "IMG-5/minimal": "opus",
    "IMG-5/full": "opus",
    "TXT-1/minimal": "sol",
    "TXT-1/full": "sol",
    "TXT-2/minimal": "glm",
    "TXT-2/full": "glm",
    "TXT-4/minimal": "sol",
    "TXT-4/full": "sol",
    "TXT-5/minimal": "sol",
    "TXT-5/full": "sol",
    "TXT-6/minimal": "opus",
    "TXT-6/full": "opus",
    "TXT-7/minimal": "glm",
    "TXT-7/full": "sol",
    "TXT-8/minimal": "glm",
    "TXT-8/full": "glm"
  },
  "win_counts": {
    "opus": 8,
    "sol": 7,
    "glm": 5
  },
  "cheapest_model": "glm",
  "fastest_model": "sol",
  "perf": {
    "opus": {
      "n_calls": 260,
      "errors": 0,
      "latency_ms_median": 2953.0,
      "latency_ms_p95": 8823.6,
      "total_usd": 1.595909,
      "in_tokens": 125936,
      "out_tokens": 38649
    },
    "sol": {
      "n_calls": 260,
      "errors": 0,
      "latency_ms_median": 1740.6,
      "latency_ms_p95": 4754.4,
      "total_usd": 1.220562,
      "in_tokens": 90854,
      "out_tokens": 20083
    },
    "glm": {
      "n_calls": 160,
      "errors": 19,
      "latency_ms_median": 4014.1,
      "latency_ms_p95": 21611.8,
      "total_usd": 0.252097,
      "in_tokens": 50900,
      "out_tokens": 41751
    }
  }
}
```
