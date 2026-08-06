# 벤치마크 리포트 — 2026-08-06T02-06

> 📊 고객 설명용 프레젠테이션: **[▶ 브라우저로 바로 보기](https://htmlpreview.github.io/?https://github.com/Aiden-Jeon/databricks-fmapi-task-benchmark/blob/main/image-text-performance/reports/2026-08-06T02-06/presentation.html)** · [HTML 소스](./presentation.html) — 이 리포트 결과를 슬라이드로 정리

## 평가 대상 모델 (Databricks hosted)

| 별칭 | Databricks model name | vision | reasoning 파라미터 | timeout |
|---|---|---|---|---|
| opus | `databricks-claude-opus-5` | ✅ | `minimal`: `{'thinking': {'type': 'disabled'}}` | 60s |
| sol | `databricks-gpt-5-6-sol` | ✅ | `minimal`: `{'reasoning_effort': 'none'}` | 60s |
| glm | `databricks-glm-5-2` | ❌ | `minimal`: `{'reasoning_effort': 'none'}` | 120s |

> Judge: `databricks-gemini-3-1-pro`

## Executive Summary

제공된 수치에 따르면 'sol' 모델은 총 9개 태스크에서 1위를 차지하며 특히 이미지(IMG) 영역에서 압도적인 강점을 보였고, 'opus'와 'glm'은 각각 4개 태스크에서 우위를 기록했습니다. 텍스트(TXT) 영역에만 참여한 'glm'은 전체 모델 중 가장 저렴하고 처리 속도(중간 지연시간 724.2ms)가 가장 빨라 비용 및 속도 트레이드오프 측면에서 최고의 효율을 보여줍니다. 반면 'opus'는 비용이 가장 비싸고 속도(중간 지연시간 2527.4ms)가 가장 느릴 뿐만 아니라, 유일하게 5건의 오류(errors)가 발생해 효율성과 안정성이 가장 떨어지는 것으로 나타났습니다.

<sub>규칙 기반 요약(대조용): 태스크별 1위 횟수는 sol 9회, opus 4회, glm 4회로 **sol**가 가장 많다. 이 중 2개 태스크는 동점이라 공동 1위로 집계했다. 응답 속도는 **glm**가 가장 빠르다(median 724.2ms). 비용은 **glm**가 가장 낮다($0.246797).</sub>

## 비교 그래프

### 태스크별 모델 성능 비교

![태스크별 모델 성능 비교](chart_scores.png)

### 모델별 속도 vs 비용

![모델별 속도 vs 비용](chart_perf.png)

**태스크 ID 범례:** IMG-1=이미지 캡션 생성 · IMG-2=이미지 태그(객체) 추출 · IMG-3=무기/위협 존재 판별 · IMG-4=성인/NSFW 이미지 판별 · IMG-5=사람 포함 여부 판별 · IMG-6=표 이미지 구조 추출 · TXT-1=문서(PDF) 이해 QA · TXT-2=표(엑셀) 이해 QA · TXT-3=표 구조 추출 · TXT-4=한국어 독해 QA · TXT-5=텍스트 요약 · TXT-6=감정 분석 · TXT-7=키워드 추출 · TXT-8=비속어/유해성 판별

## 정량 결과 (태스크 × 모델)

| 태스크 | 모델 | reasoning | 대표 메트릭 | 실패 |
|---|---|---|---|---|
| TXT-1 · 문서(PDF) 이해 QA | glm | minimal | anls=0.863, token_f1=0.843, exact_match=0.8, n_evaluated=30, judge_mean=4.5 | — |
| TXT-2 · 표(엑셀) 이해 QA | glm | minimal | accuracy=0.533, token_f1=0.572, n_evaluated=30, judge_mean=3.5 | — |
| TXT-3 · 표 구조 추출 | glm | minimal | cell_f1=0.972, n_evaluated=30 | — |
| TXT-4 · 한국어 독해 QA | glm | minimal | token_f1=0.94, exact_match=0.8, n_evaluated=30, judge_mean=5.0 | — |
| TXT-5 · 텍스트 요약 | glm | minimal | rouge1=0.285, rouge2=0.126, rougeL=0.227, n_evaluated=30, bertscore_f1=0.734, bertscore_n=30 | judge 1 |
| TXT-6 · 감정 분석 | glm | minimal | accuracy=0.833, macro_f1=0.829, n_evaluated=30, n_unparsed=0 | — |
| TXT-7 · 키워드 추출 | glm | minimal | precision=0.497, recall=0.511, f1=0.504, n_evaluated=30, macro_precision=0.483, macro_recall=0.509 | — |
| TXT-8 · 비속어/유해성 판별 | glm | minimal | accuracy=0.867, f1=0.778, n_evaluated=30, n_unparsed=0 | — |
| IMG-1 · 이미지 캡션 생성 | opus | minimal | caption_token_f1=0.331, n_evaluated=30, bertscore_f1=0.744, bertscore_n=30, judge_mean=4.3 | — |
| IMG-2 · 이미지 태그(객체) 추출 | opus | minimal | micro_precision=0.708, micro_recall=0.941, micro_f1=0.808, macro_precision=0.774, macro_recall=0.927, macro_f1=0.819 | — |
| IMG-3 · 무기/위협 존재 판별 | opus | minimal | accuracy=0.667, f1=0.783, n_evaluated=30, n_unparsed=0 | — |
| IMG-4 · 성인/NSFW 이미지 판별 | opus | minimal | accuracy=0.929, f1=0.923, n_evaluated=28, n_unparsed=0, n_skipped=2 | 호출 2/30 |
| IMG-5 · 사람 포함 여부 판별 | opus | minimal | accuracy=0.931, f1=0.938, n_evaluated=29, n_unparsed=0, n_skipped=1 | 호출 1/30 |
| IMG-6 · 표 이미지 구조 추출 | opus | minimal | cell_f1=0.868, n_evaluated=29, n_skipped=1 | 호출 1/30 |
| TXT-1 · 문서(PDF) 이해 QA | opus | minimal | anls=0.947, token_f1=0.927, exact_match=0.9, n_evaluated=30, judge_mean=4.965 | judge 1 |
| TXT-2 · 표(엑셀) 이해 QA | opus | minimal | accuracy=0.533, token_f1=0.589, n_evaluated=30, judge_mean=4.367 | — |
| TXT-3 · 표 구조 추출 | opus | minimal | cell_f1=0.978, n_evaluated=29, n_skipped=1 | 호출 1/30 |
| TXT-4 · 한국어 독해 QA | opus | minimal | token_f1=0.785, exact_match=0.633, n_evaluated=30, judge_mean=5.0 | — |
| TXT-5 · 텍스트 요약 | opus | minimal | rouge1=0.325, rouge2=0.18, rougeL=0.268, n_evaluated=30, bertscore_f1=0.74, bertscore_n=30 | judge 2 |
| TXT-6 · 감정 분석 | opus | minimal | accuracy=0.833, macro_f1=0.829, n_evaluated=30, n_unparsed=0 | — |
| TXT-7 · 키워드 추출 | opus | minimal | precision=0.311, recall=0.464, f1=0.372, n_evaluated=30, macro_precision=0.304, macro_recall=0.467 | — |
| TXT-8 · 비속어/유해성 판별 | opus | minimal | accuracy=0.867, f1=0.778, n_evaluated=30, n_unparsed=0 | — |
| IMG-1 · 이미지 캡션 생성 | sol | minimal | caption_token_f1=0.429, n_evaluated=30, bertscore_f1=0.78, bertscore_n=30, judge_mean=3.833 | — |
| IMG-2 · 이미지 태그(객체) 추출 | sol | minimal | micro_precision=0.851, micro_recall=0.871, micro_f1=0.86, macro_precision=0.915, macro_recall=0.891, macro_f1=0.886 | — |
| IMG-3 · 무기/위협 존재 판별 | sol | minimal | accuracy=0.633, f1=0.776, n_evaluated=30, n_unparsed=0 | — |
| IMG-4 · 성인/NSFW 이미지 판별 | sol | minimal | accuracy=0.933, f1=0.929, n_evaluated=30, n_unparsed=0 | — |
| IMG-5 · 사람 포함 여부 판별 | sol | minimal | accuracy=0.967, f1=0.971, n_evaluated=30, n_unparsed=0 | — |
| IMG-6 · 표 이미지 구조 추출 | sol | minimal | cell_f1=0.889, n_evaluated=30 | — |
| TXT-1 · 문서(PDF) 이해 QA | sol | minimal | anls=0.881, token_f1=0.824, exact_match=0.833, n_evaluated=30, judge_mean=4.467 | — |
| TXT-2 · 표(엑셀) 이해 QA | sol | minimal | accuracy=0.633, token_f1=0.689, n_evaluated=30, judge_mean=4.0 | — |
| TXT-3 · 표 구조 추출 | sol | minimal | cell_f1=0.979, n_evaluated=30 | — |
| TXT-4 · 한국어 독해 QA | sol | minimal | token_f1=0.859, exact_match=0.633, n_evaluated=30, judge_mean=5.0 | — |
| TXT-5 · 텍스트 요약 | sol | minimal | rouge1=0.329, rouge2=0.165, rougeL=0.276, n_evaluated=30, bertscore_f1=0.736, bertscore_n=30 | — |
| TXT-6 · 감정 분석 | sol | minimal | accuracy=0.833, macro_f1=0.829, n_evaluated=30, n_unparsed=0 | — |
| TXT-7 · 키워드 추출 | sol | minimal | precision=0.391, recall=0.432, f1=0.411, n_evaluated=30, macro_precision=0.398, macro_recall=0.446 | — |
| TXT-8 · 비속어/유해성 판별 | sol | minimal | accuracy=0.833, f1=0.737, n_evaluated=30, n_unparsed=0 | — |

> **채점 조건**
> - 한국어 토큰화: **형태소(mecab)** — ROUGE·Token-F1이 형태소 기준이다.
> - 호출 실패: 4개 셀에 실패가 있다(위 '실패' 열). 실패한 샘플은 **채점에서 제외**하므로(0점으로 세지 않음) 그 셀의 점수는 성공한 샘플 기준이다 — 표의 `n_evaluated`가 요청 샘플 수보다 작은 이유다. 실패는 엔드포인트 문제이지 모델 성능이 아니다.
> - judge 실패(응답 잘림·형식 이탈)는 해당 샘플을 평균에서 **제외**하고 위 표에 건수를 표기한다. 중간값으로 메우지 않는다.

### 통계 유의성 (judge 점수, Wilcoxon signed-rank)

| 태스크 | 모델 쌍 | judge 평균 | n(짝) | 판정 |
|---|---|---|---|---|
| IMG-1 · 이미지 캡션 생성 | opus vs sol | 4.30 vs 3.83 | 30 | 유의하지 않음 (p=0.0833) |
| TXT-1 · 문서(PDF) 이해 QA | glm vs opus | 4.62 vs 4.97 | 29 | 유의하지 않음 (p=0.1025) |
| TXT-1 · 문서(PDF) 이해 QA | glm vs sol | 4.50 vs 4.47 | 30 | 유의하지 않음 (p=1.0000) |
| TXT-1 · 문서(PDF) 이해 QA | opus vs sol | 4.97 vs 4.59 | 29 | 유의하지 않음 (p=0.1306) |
| TXT-2 · 표(엑셀) 이해 QA | glm vs opus | 3.50 vs 4.37 | 30 | **유의** (p=0.0394) → opus 우세 |
| TXT-2 · 표(엑셀) 이해 QA | glm vs sol | 3.50 vs 4.00 | 30 | 유의하지 않음 (p=0.1245) |
| TXT-2 · 표(엑셀) 이해 QA | opus vs sol | 4.37 vs 4.00 | 30 | 유의하지 않음 (p=0.1306) |
| TXT-4 · 한국어 독해 QA | glm vs opus | 5.00 vs 5.00 | 30 | 판정 불가(차이 없음/표본 부족) |
| TXT-4 · 한국어 독해 QA | glm vs sol | 5.00 vs 5.00 | 30 | 판정 불가(차이 없음/표본 부족) |
| TXT-4 · 한국어 독해 QA | opus vs sol | 5.00 vs 5.00 | 30 | 판정 불가(차이 없음/표본 부족) |
| TXT-5 · 텍스트 요약 | glm vs opus | 4.48 vs 4.96 | 27 | **유의** (p=0.0103) → opus 우세 |
| TXT-5 · 텍스트 요약 | glm vs sol | 4.48 vs 4.79 | 29 | 유의하지 않음 (p=0.0845) |
| TXT-5 · 텍스트 요약 | opus vs sol | 4.96 vs 4.82 | 28 | 유의하지 않음 (p=0.1573) |

> Wilcoxon signed-rank(양측, α=0.05). **judge 점수에만** 적용한다 — 정량 메트릭은 셀 단위 평균만 저장해(스트리밍 O(1) 설계) 샘플을 짝지을 수 없다. '유의하지 않음'은 두 모델이 같다는 뜻이 아니라 이 표본에서 차이를 확인할 수 없다는 뜻이다.

## 성능: 수행시간·비용 (모델별)

| 모델 | 호출 | 오류 | latency median(ms) | p95(ms) | 입력토큰 | 출력토큰 | 비용(USD) |
|---|---|---|---|---|---|---|---|
| glm | 240 | 0 | 724.2 | 7604.3 | 102954 | 24330 | 0.246797 |
| opus | 420 | 5 | 2527.4 | 9425.9 | 222179 | 55229 | 2.491627 |
| sol | 420 | 0 | 1866.5 | 9889.9 | 161718 | 45195 | 2.164443 |

> 비용은 `config/pricing.yaml` DBU 단가 기반 추정(usd_per_dbu 가정값). 정밀 비용은 `system.ai_gateway.usage` 조인 필요(§10).

## 정성 비교: 모델 간 결과가 갈린 샘플

각 태스크에서 모델들의 출력·판정이 **가장 크게 갈린 샘플**을 골라 나란히 비교한다. 평균 점수로는 안 보이는 모델별 차이를 드러낸다.

### IMG-1 · 이미지 캡션 생성

**샘플 #0** · _모델 간 판정이 갈린 케이스_ · 정답: `['An elegant bathroom features a tub, sink, mirror, and decorations. ', 'An old fashion above ground tub is shown with gold feet.', 'A lovely, vintage-styled ba…`

**질문/지시:** Describe this image in one sentence.

**입력 이미지:**

![IMG-1 sample 0](gallery_IMG-1_s0.png)

| 모델 | 정답? | 출력/판정 |
|---|:--:|---|
| opus | ❌ | `A warmly lit bathroom with golden-toned faux-finish walls features a vintage black and white clawfoot tub with brass fixtures, a white pedestal sink beneath a f…` |
| sol | ❌ | `An elegant vintage bathroom features a clawfoot bathtub, pedestal sink, gold fixtures, framed mirror, and textured beige walls.` |

### IMG-2 · 이미지 태그(객체) 추출

**샘플 #0** · _모델 간 판정이 갈린 케이스_ · 정답: `['boat', 'chair', 'handbag', 'person', 'umbrella']`

**질문/지시:** List every object from the label set below that is visible in this image.

**입력 이미지:**

![IMG-2 sample 0](gallery_IMG-2_s0.png)

| 모델 | 정답? | 출력/판정 |
|---|:--:|---|
| opus | ✅ | `boat, chair, person, umbrella, potted plant` |
| sol | ✅ | `boat, person, umbrella, potted plant` |

### IMG-3 · 무기/위협 존재 판별

**샘플 #2** · _모델 간 판정이 갈린 케이스_ · 정답: `1`

**질문/지시:** Does this image contain a weapon or threat (gun, knife, pistol, rifle, blood, etc.)? Answer exactly yes or no.

**입력 이미지:**

![IMG-3 sample 2](gallery_IMG-3_s2.png)

| 모델 | 정답? | 출력/판정 |
|---|:--:|---|
| opus | ❌ | `No` |
| sol | ✅ | `yes` |

### IMG-4 · 성인/NSFW 이미지 판별

**샘플 #4** (민감 태스크 — 입력 비표시, 판정값만) · _모델 간 판정이 갈린 케이스_ · 정답: `0`

**질문/지시:** You are a content-safety classifier. Does this image contain adult/NSFW (sexually explicit) content? Answer exactly 'nsfw' or 'safe'.

| 모델 | 정답? | 출력/판정 |
|---|:--:|---|
| opus | ❌ | `__ERROR__: FMAPIError: databricks-claude-opus-5 재시도 5회 모두 실패: databricks-claude-opus-5 HTTP 502: {"error_code":"INTERNAL_ERROR","message":"The server received a…` |
| sol | — | `safe` |

### IMG-5 · 사람 포함 여부 판별

**샘플 #14** · _모델 간 판정이 갈린 케이스_ · 정답: `1`

**질문/지시:** Is there a person visible in this image? Answer exactly yes or no.

**입력 이미지:**

![IMG-5 sample 14](gallery_IMG-5_s14.png)

| 모델 | 정답? | 출력/판정 |
|---|:--:|---|
| opus | ❌ | `__ERROR__: FMAPIError: databricks-claude-opus-5 재시도 5회 모두 실패: databricks-claude-opus-5 HTTP 502: {"error_code":"INTERNAL_ERROR","message":"The server received a…` |
| sol | ✅ | `yes` |

### IMG-6 · 표 이미지 구조 추출

**샘플 #0** · _모델 간 판정이 갈린 케이스_ · 정답:

```
<html>
 <head>
  <meta charset="utf-8"/>
  <style>
   table, th, td {
                     border: 1px solid black;
                     font-size: 10px;
      …
```

**질문/지시:** This image contains a table. Extract the table and output it as a valid HTML table using <table>, <tr>, and <td> tags, preserving the row/column structure and cell text exactly as shown. Respond with ONLY the HTML, no explanation.

**입력 이미지:**

![IMG-6 sample 0](gallery_IMG-6_s0.png)

| 모델 | 정답? | 출력/판정 |
|---|:--:|---|
| opus | ✅ | `<table> <tr><td>Clinical parameters</td><td colspan="2">WWOX intensity</td><td>Total</td><td>p value</td></tr> <tr><td></td><td>Low (Negative/Weak)</td><td>High…` |
| sol | ✅ | `<table>   <tr>     <td>Clinical parameters</td>     <td colspan="2">WWOX intensity</td>     <td>Total</td>     <td>p value</td>   </tr>   <tr>     <td></td>    …` |

### TXT-1 · 문서(PDF) 이해 QA

**샘플 #20** · _모델 간 판정이 갈린 케이스_ · 정답: `['5']`

**질문/지시:** What is the age of subject B.L. ?

**입력:**
> Based on the following document text, answer the question.
> 
> Document text:
> SECTION 1 STUDIES ON THE EFFECT OF MILK PROTEIN DILUTION IN CHILDREN AT M. I. T. Studies on the effect of dilution of skim milk protein were commenced on fourteen mentally retarded children. Of these fourteen children, seven were found to be unsatisfactory for study and returned, within two weeks of admission to the M. I. T. Clinical Center, to their respective State schools. Details of experimental design, diet, and analytical procedures were previously described in the 1967 Annual Progress Report and are further described in Section 3 of this report. Details are given in Table 2 for the seven children who were studied during the entire experimental period. Table 2 Age and weight of children and experimental design used for studies of milk protein dilution Subject Age Initial Wt. Final Wt. Days on Days on (yrs. ) ( 1b. ) (1b. ) Baseline Diet Dilution P. K. 61.5 58. 5 18 20 W. M. 33.0 36, 0 26 83 C. R. 33.0 35.0 25 32 R. O'C. 34. 0 36.0 33 32 B. L. 41.0 43.0 29 40 S. D. 35.0 37.0 25 12 J. W. Z2. 0 23.5 40 30 1. Dilution at 20% level. Source: https://www.industrydocuments.ucsf.edu/docs/hldg0227
> 
> Question: Wha…

| 모델 | 정답? | 출력/판정 |
|---|:--:|---|
| glm | ❌ | `4` |
| opus | ❌ | `The age of subject B.L. is not given in the document (the Age column value is missing; 41.0 is the initial weight).` |
| sol | ❌ | `41.0` |

### TXT-2 · 표(엑셀) 이해 QA

**샘플 #1** · _모델 간 판정이 갈린 케이스_ · 정답: `['9']`

**질문/지시:** how many elections had at least a 0.2 percentage of constituency votes?

**입력:**
> Based on the following table, answer the question.
> 
> Table:
> | Election year | # of\nconstituency votes | % of\nconstituency votes | +/– | # of\nparty list votes | % of\nparty list votes | +/– | # of\noverall seats won |
> | --- | --- | --- | --- | --- | --- | --- | --- |
> | 1965 | 587,216 | 1.8 | 1.8 | 664,193 | 2.0 | 2.0 | 0 / 518 |
> | 1969 | 1,189,375 | 3.6 | 1.8 | 1,422,010 | 4.3 | 2.3 | 0 / 518 |
> | 1972 | 194,389 | 0.5 | 3.1 | 207,465 | 0.6 | 3.7 | 0 / 518 |
> | 1976 | 136.023 | 0.4 | 0.1 | 122,661 | 0.3 | 0.3 | 0 / 518 |
> | 1980 |  |  |  | 68,096 | 0.2 | 0.1 | 0 / 497 |
> | 1983 | 57,112 | 0.1 | 0.3 | 91,095 | 0.2 | 0 | 0 / 498 |
> | 1987 | 182,880 | 0.5 | 0.4 | 227,054 | 0.6 | 0.4 | 0 / 497 |
> | 1990 | 190,105 | 0.4 | 0.1 | 145,776 | 0.3 | 0.3 | 0 / 662 |
> | 1998 | 45,043 | 0.1 | 0.3 | 126,571 | 0.3 | 0 | 0 / 669 |
> | 2002 | 103,209 | 0.1 | 0.1 | 215,232 | 0.4 | 0.1 | 0 / 603 |
> | 2005 | 857.777 | 1.8 | 1.6 | 748,568 | 1.6 | 1.2 | 0 / 614 |
> | 2009 | 768,442 | 1.8 | 0 | 635,525 | 1.5 | 0.1 | 0 / 620 |
> | 2013 | 634,842 | 1.5 | 0.3 | 560,660 | 1.3 | 0.2 | 0 / 630 |
> 
> Question: how many elections had at least a 0.2 percentage of constituency votes?
> 
> Output ONLY the answer value(s) copied from the…

| 모델 | 정답? | 출력/판정 |
|---|:--:|---|
| glm | ❌ | `1965, 1969, 1972, 1976, 1987, 1990, 1998, 2002, 2005, 2009, 2013` |
| opus | ✅ | `<thinking>We need elections with % of constituency votes at least 0.2.  1965: 1.8 ✓ 1969: 3.6 ✓ 1972: 0.5 ✓ 1976: 0.4 ✓ 1980: blank ✗ 1983: 0.1 ✗ 1987: 0.5 ✓ 19…` |
| sol | ❌ | `10` |

### TXT-3 · 표 구조 추출

**샘플 #1** · _모델 간 판정이 갈린 케이스_ · 정답:

```
<html>
 <head>
  <meta charset="utf-8"/>
  <style>
   table, th, td {
                     border: 1px solid black;
                     font-size: 10px;
      …
```

**질문/지시:** Given the following table cell contents in row-major order, generate a valid HTML table with proper <table>, <tr>, and <td> tags.

**입력:**

```
Given the following table cell contents in row-major order, generate a valid HTML table with proper <table>, <tr>, and <td> tags.

Table cells:
[] [Group 1] [Group 2] [] [] [] [] [] []
[Cut-off of FibroTest] [Sensitivity] [Specificity] [Positive Predictive Value] [Negative Predictive Value] [Sensitivity] [Specificity] [Positive Predictive Value] [Negative Predictive Value]
[Stage F2F3F4] [Prevalence = 0.24 (40/170)] [] [] [Prevalence = 0.32 (31/97)] [] [] [] []
[0.30] [0.83 (33/40)] [0.78 (101/130)] [0.53 (33/62)] [0.94 (101/108)] [0.71 (22/31)] [0.74 (49/66)] [0.56 (22/39)] [0.84 (49/58)]
[0.70] [0.18 (7/40)] [0.98 (128/130)] [0.78 (7/9)] [0.80 (128/161)] [0.13 (4/31)] [0.98 (65/66)] [0.80 (4/5)] [0.71 (65/92)]
[Stage F3F4] [Prevalence = 0.12 (20/170)] [] [] [Prevalence = 0.16 (16/97)] [] [] [] []
[0.30] [0.95 (19/20)] [0.71 (107/150)] [0.31 (19/62)] [0.99 (107/108)] [0.88 (14/16)] [0.69 (56/81)] [0.36 (14/39)] [0.97 (56/58)]
[0.70] [0.25 (5/20)] [0.97 (146/150)] [0.56 (5/9)] [0.91 (146/161)] [0.25 (4/16)] [0.99 (80/81)] [0.80 (4/5)] [0.87 (80/92)]
[% of patients with FT between 0.30–0.70] [0.32 (54/170)] [] [] [] [0.35 (34/97)] [] [] []

Generate the HTML table structure. Respond…
```

| 모델 | 정답? | 출력/판정 |
|---|:--:|---|
| glm | ✅ | `'''html <table>   <tr>     <td></td>     <td>Group 1</td>     <td>Group 2</td>     <td></td>     <td></td>     <td></td>     <td></td>     <td></td>     <td></t…` |
| opus | ✅ | `<table> <tr><td></td><td>Group 1</td><td>Group 2</td><td></td><td></td><td></td><td></td><td></td><td></td></tr> <tr><td>Cut-off of FibroTest</td><td>Sensitivit…` |
| sol | ✅ | `<table>   <tr>     <td></td>     <td colspan="4">Group 1</td>     <td colspan="4">Group 2</td>   </tr>   <tr>     <td>Cut-off of FibroTest</td>     <td>Sensitiv…` |

### TXT-4 · 한국어 독해 QA

**샘플 #11** · _모델 간 판정이 갈린 케이스_ · 정답: `['교바시']`

**질문/지시:** 기슈지 쾌속의 시발지은 대부분 어느 곳인가?

**입력:**
> 다음 지문을 읽고 질문에 답하세요. 답변은 지문에서 나타나는 정답만 간결하게 기술하세요.
> 
> 지문:
> 한편 기슈지 쾌속(일본어: 紀州路快速 키슈우지카이소쿠)은 오사카 방면에서 와카야마 방면에의 관광객 및 통근객의 증대를 도모하고자 이 때까지 교토 역 ~ 신오사카 역 방면에서 와카야마 방면으로 운행하는 특급 열차에 더하여 1999년 5월부터 신설된 쾌속 열차이다. 오사카 순환선에서 발착하여 덴노지 역에서 교바시, 오사카, 니시쿠조를 경유하여 (다만 대부분은 교바시 시발) 운전되고 있다. 간쿠 쾌속과 같이 223계 전동차 0번대와 2500번대가 운용된다. 히네노 역 이북에서는 전 열차가 8량 편성으로 기본적으로 히네노 역까지는 간쿠 쾌속과 병결 운전을 행하여 히네노 역에서 분할과 병결이 이루어진다. 이전에는 출퇴근 시간대에 1~5호차의 5량이 기슈지 쾌속이었으며, 나머지 시간대에는 그 반대였으나 2008년 3월 15일의 시간표 개정에 따라 8량 편성을 간쿠와 기슈지 쾌속이 4량씩 차지하는 형태로 통일되어 시간대에 따라 기슈지 쾌속과 간쿠 쾌속의 연결 순서가 바뀌는 경우도 있다. 휴일 시간표에서 아침의 상행 열차에는 간쿠 쾌속이 병결되지 않은 채 기세이 본선 고보 역에서 시발하여 와카야마 역에서 8량으로 증결하여 운행하는 특별한 형태도 있다. 오사카 방면으로 가는 열차의 와카야마 역 ~ 히네노 역 간에는 4량 편성으로 운전되기 때문에 시발역인 와카야마 역에서부터 혼잡해지는 경우가 많다.
> 
> 질문: 기슈지 쾌속의 시발지은 대부분 어느 곳인가?
> 
> 답변:

| 모델 | 정답? | 출력/판정 |
|---|:--:|---|
| glm | ✅ | `교바시` |
| opus | ✅ | `교바시(교바시 역)` |
| sol | ❌ | `교바시역` |

### TXT-5 · 텍스트 요약

**샘플 #0** · _모델 간 판정이 갈린 케이스_ · 정답:

```
Cayman Naib, 13, hasn't been heard from since Wednesday .
Police, family, volunteers search for eighth-grader .
```

**질문/지시:** Generate a concise summary of the following document. The summary should capture the key points in 2-3 sentences.

**입력:**
> Generate a concise summary of the following document. The summary should capture the key points in 2-3 sentences.
> 
> Document:
> (CNN)A Pennsylvania community is pulling together to search for an eighth-grade student who has been missing since Wednesday. The search has drawn hundreds of volunteers on foot and online. The parents of Cayman Naib, 13, have been communicating through the Facebook group "Find Cayman" since a day after his disappearance, according to close friend David Binswanger. Newtown Police say Cayman was last seen wearing a gray down winter jacket, black ski pants and hiking boots. He could be in the Radnor-Wayne area, roughly 20 miles from Philadelphia, or may have purchased a train ticket to Philadelphia, according to an alert posted on Facebook. "We think that he got a email from school and was upset by it and left as an impulsive act," Farid Naib, Cayman's father, wrote Thursday on the group page. "We have spoken to his friends and they do not know where he is. Cayman does have his phone, we don't know if he has any cash, he does not have his wallet." The parents said that his phone was out of power at the time. "Cayman left within 30 minutes after he received an e…

| 모델 | 정답? | 출력/판정 |
|---|:--:|---|
| glm | — | `A Pennsylvania community has mobilized hundreds of volunteers both on foot and online to search for 13-year-old Cayman Naib, who went missing after impulsively …` |
| opus | — | `Cayman Naib, a 13-year-old eighth-grader from the Philadelphia suburbs, has been missing since Wednesday, when his parents say he left home impulsively within 3…` |
| sol | — | `Hundreds of volunteers are searching for 13-year-old Cayman Naib, who disappeared from the Radnor-Wayne area after reportedly becoming upset over a school email…` |

### TXT-6 · 감정 분석

**샘플 #20** · _모델 간 판정이 갈린 케이스_ · 정답: `1`

**질문/지시:** Classify the sentiment of the following text as exactly one word: "positive" or "negative".

**입력:**
> Classify the sentiment of the following text as exactly one word: "positive" or "negative".
> 
> Text: 오게두어라! 서리한이 굶주렸다!
> 
> Respond with exactly one word: positive or negative

| 모델 | 정답? | 출력/판정 |
|---|:--:|---|
| glm | ❌ | `negative` |
| opus | ❌ | `<thinking>The text is Korean: "오게두어라! 서리한이 굶주렸다!" which translates to "Let them come! Frostmourne hungers!" This is a famous Warcraft quote from the Lich King (…` |
| sol | ❌ | `negative` |

### TXT-7 · 키워드 추출

**샘플 #0** · _모델 간 판정이 갈린 케이스_ · 정답: `['canonical decomposition apparatus', 'mixed central moment functions', 'moment function', 'nonlinear extrapolation algorithm', 'nonlinear random dependences', …`

**질문/지시:** Extract the key phrases or keywords from the following academic document. List them as a comma-separated list with no additional text.

**입력:**
> Extract the key phrases or keywords from the following academic document. List them as a comma-separated list with no additional text.
> 
> Title and Abstract:
> Nonlinear extrapolation algorithm for realization of a scalar random process
> 
> A method of construction of a nonlinear extrapolation algorithm is proposed. This method makes it possible to take into account any nonlinear random dependences that exist in an investigated process and are described by mixed central moment functions. The method is based on the V. S. Pugachev canonical decomposition apparatus. As an example, the problem of nonlinear extrapolation is solved for a moment function of third order
> 
> Key phrases (comma-separated):

| 모델 | 정답? | 출력/판정 |
|---|:--:|---|
| glm | ✅ | `Nonlinear extrapolation algorithm, scalar random process, nonlinear random dependences, mixed central moment functions, V. S. Pugachev canonical decomposition, …` |
| opus | ✅ | `nonlinear extrapolation, scalar random process, canonical decomposition, V. S. Pugachev canonical expansion, mixed central moment functions, nonlinear random de…` |
| sol | ✅ | `nonlinear extrapolation algorithm, scalar random process, nonlinear random dependences, mixed central moment functions, V. S. Pugachev canonical decomposition, …` |

### TXT-8 · 비속어/유해성 판별

**샘플 #1** · _모델 간 판정이 갈린 케이스_ · 정답: `0`

**질문/지시:** You are a content moderation classifier. Analyze the following text and determine if it contains profanity, hate speech, or toxic language. Respond with exactly one word: "toxic" or "clean".

**입력:**
> You are a content moderation classifier. Analyze the following text and determine if it contains profanity, hate speech, or toxic language. Respond with exactly one word: "toxic" or "clean".
> 
> Text: Hi Explicit, can you block O Fenian for edit-warring on the Giant's Causeway wp. He has made several edits which can only be described as terrorism.
> 
> Respond with exactly one word: toxic or clean

| 모델 | 정답? | 출력/판정 |
|---|:--:|---|
| glm | — | `toxic` |
| opus | — | `clean` |
| sol | — | `toxic` |


## 참고: Reasoning 정책

**Reasoning OFF(minimal) 단일 모드로 고정.** reasoning의 효과가 특정 태스크 성능 개선에만 한정되는 반면 실험 시간을 크게 늘리기 때문(특히 GLM은 full reasoning 시 타임아웃 빈발). 각 모델의 최소 reasoning으로 측정하며, `databricks-claude-opus-5`와 judge는 완전 OFF가 불가해 지원 최소값을 사용한다.

## Fact Sheet (Executive Summary 근거 — 감사용)

```json
{
  "per_task_scores": {
    "IMG-1/minimal": {
      "opus": 0.3312,
      "sol": 0.4294
    },
    "IMG-2/minimal": {
      "opus": 0.8081,
      "sol": 0.8605
    },
    "IMG-3/minimal": {
      "opus": 0.6667,
      "sol": 0.6333
    },
    "IMG-4/minimal": {
      "opus": 0.9286,
      "sol": 0.9333
    },
    "IMG-5/minimal": {
      "opus": 0.931,
      "sol": 0.9667
    },
    "IMG-6/minimal": {
      "opus": 0.8683,
      "sol": 0.8887
    },
    "TXT-1/minimal": {
      "opus": 0.9466,
      "sol": 0.8806,
      "glm": 0.8632
    },
    "TXT-2/minimal": {
      "opus": 0.5333,
      "sol": 0.6333,
      "glm": 0.5333
    },
    "TXT-3/minimal": {
      "opus": 0.9775,
      "sol": 0.9791,
      "glm": 0.9725
    },
    "TXT-4/minimal": {
      "opus": 0.7849,
      "sol": 0.8585,
      "glm": 0.9398
    },
    "TXT-5/minimal": {
      "opus": 0.3252,
      "sol": 0.3286,
      "glm": 0.2853
    },
    "TXT-6/minimal": {
      "opus": 0.8333,
      "sol": 0.8333,
      "glm": 0.8333
    },
    "TXT-7/minimal": {
      "opus": 0.3722,
      "sol": 0.4108,
      "glm": 0.5039
    },
    "TXT-8/minimal": {
      "opus": 0.8667,
      "sol": 0.8333,
      "glm": 0.8667
    }
  },
  "task_winners": {
    "IMG-1/minimal": [
      "sol"
    ],
    "IMG-2/minimal": [
      "sol"
    ],
    "IMG-3/minimal": [
      "opus"
    ],
    "IMG-4/minimal": [
      "sol"
    ],
    "IMG-5/minimal": [
      "sol"
    ],
    "IMG-6/minimal": [
      "sol"
    ],
    "TXT-1/minimal": [
      "opus"
    ],
    "TXT-2/minimal": [
      "sol"
    ],
    "TXT-3/minimal": [
      "sol"
    ],
    "TXT-4/minimal": [
      "glm"
    ],
    "TXT-5/minimal": [
      "sol"
    ],
    "TXT-6/minimal": [
      "glm",
      "opus",
      "sol"
    ],
    "TXT-7/minimal": [
      "glm"
    ],
    "TXT-8/minimal": [
      "glm",
      "opus"
    ]
  },
  "win_counts": {
    "sol": 9,
    "opus": 4,
    "glm": 4
  },
  "n_tied_tasks": 2,
  "excluded_unreliable": [],
  "excluded_low_parse_valid": [],
  "cheapest_model": "glm",
  "fastest_model": "glm",
  "unpriced_models": [],
  "perf": {
    "opus": {
      "n_calls": 420,
      "errors": 5,
      "latency_ms_median": 2527.4,
      "latency_ms_p95": 9425.9,
      "total_usd": 2.491627,
      "in_tokens": 222179,
      "out_tokens": 55229
    },
    "sol": {
      "n_calls": 420,
      "errors": 0,
      "latency_ms_median": 1866.5,
      "latency_ms_p95": 9889.9,
      "total_usd": 2.164443,
      "in_tokens": 161718,
      "out_tokens": 45195
    },
    "glm": {
      "n_calls": 240,
      "errors": 0,
      "latency_ms_median": 724.2,
      "latency_ms_p95": 7604.3,
      "total_usd": 0.246797,
      "in_tokens": 102954,
      "out_tokens": 24330
    }
  }
}
```
