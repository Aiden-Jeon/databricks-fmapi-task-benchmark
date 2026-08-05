# 벤치마크 리포트 — 2026-08-05T07-09

> 📊 고객 설명용 프레젠테이션: **[▶ 브라우저로 바로 보기](https://htmlpreview.github.io/?https://github.com/Aiden-Jeon/databricks-fmapi-task-benchmark/blob/main/image-text-performance/reports/2026-08-05T07-09/presentation.html)** · [HTML 소스](./presentation.html) — 이 리포트 결과를 슬라이드로 정리

## 평가 대상 모델 (Databricks hosted)

| 별칭 | Databricks model name | vision |
|---|---|---|
| opus | `databricks-claude-opus-5` | ✅ |
| sol | `databricks-gpt-5-6-sol` | ✅ |
| glm | `databricks-glm-5-2` | ❌ |

> Judge: `databricks-gemini-3-1-pro`

## Executive Summary

'opus'는 총 7개 작업에서 1위를 차지하며 이미지와 텍스트 전반에서 가장 우수한 성능을 보였고, 'sol'은 일부 이미지(IMG-1, 2, 5) 및 텍스트(TXT-3) 작업에서 두각을 나타냈습니다. 반면 'glm'은 텍스트 작업(TXT-4, 7, 8)에서만 1위를 기록하며 특정 분야에서만 강점을 보였습니다. 속도와 비용 측면에서는 'glm'이 가장 빠르고(중앙값 779.4ms) 저렴($0.25)한 반면, 종합 성능이 가장 높은 'opus'는 비용($2.53)이 가장 비싸고 처리 속도(3222.3ms)도 가장 느려 뚜렷한 트레이드오프를 보여줍니다.

<sub>규칙 기반 요약(대조용): 태스크별 1위 횟수는 opus 7회, sol 4회, glm 3회로 **opus**가 가장 많다. 응답 속도는 **glm**가 가장 빠르다(median 779.4ms). 비용은 **glm**가 가장 낮다($0.257848).</sub>

## 비교 그래프

### 태스크별 모델 성능 비교

![태스크별 모델 성능 비교](chart_scores.png)

### 모델별 속도 vs 비용

![모델별 속도 vs 비용](chart_perf.png)

**태스크 ID 범례:** IMG-1=이미지 캡션 생성 · IMG-2=이미지 태그(객체) 추출 · IMG-3=무기/위협 존재 판별 · IMG-4=성인/NSFW 이미지 판별 · IMG-5=사람 포함 여부 판별 · IMG-6=표 이미지 구조 추출 · TXT-1=문서(PDF) 이해 QA · TXT-2=표(엑셀) 이해 QA · TXT-3=표 구조 추출 · TXT-4=한국어 독해 QA · TXT-5=텍스트 요약 · TXT-6=감정 분석 · TXT-7=키워드 추출 · TXT-8=비속어/유해성 판별

## 정량 결과 (태스크 × 모델)

| 태스크 | 모델 | reasoning | 대표 메트릭 |
|---|---|---|---|
| TXT-1 · 문서(PDF) 이해 QA | glm | minimal | anls=0.863, token_f1=0.827, exact_match=0.8, n_evaluated=30, judge_mean=4.433 |
| TXT-2 · 표(엑셀) 이해 QA | glm | minimal | accuracy=0.1, token_f1=0.114, n_evaluated=30, judge_mean=4.067 |
| TXT-3 · 표 구조 추출 | glm | minimal | cell_f1=0.972, n_evaluated=30 |
| TXT-4 · 한국어 독해 QA | glm | minimal | token_f1=0.951, exact_match=0.833, n_evaluated=30, judge_mean=5.0 |
| TXT-5 · 텍스트 요약 | glm | minimal | rouge1=0.272, rouge2=0.115, rougeL=0.215, n_evaluated=30, judge_mean=4.448 |
| TXT-6 · 감정 분석 | glm | minimal | accuracy=0.833, macro_f1=0.829, n_evaluated=30, n_unparsed=0 |
| TXT-7 · 키워드 추출 | glm | minimal | precision=0.5, recall=0.533, f1=0.516, n_evaluated=30, macro_precision=0.483, macro_recall=0.52 |
| TXT-8 · 비속어/유해성 판별 | glm | minimal | accuracy=0.9, f1=0.824, n_evaluated=30, n_unparsed=0 |
| IMG-1 · 이미지 캡션 생성 | opus | minimal | caption_token_f1=0.327, n_evaluated=30, judge_mean=4.267 |
| IMG-2 · 이미지 태그(객체) 추출 | opus | minimal | micro_precision=0.13, micro_recall=0.447, micro_f1=0.202, macro_precision=0.143, macro_recall=0.48, macro_f1=0.213 |
| IMG-3 · 무기/위협 존재 판별 | opus | minimal | accuracy=0.7, f1=0.809, n_evaluated=30, n_unparsed=0 |
| IMG-4 · 성인/NSFW 이미지 판별 | opus | minimal | accuracy=0.933, f1=0.929, n_evaluated=30, n_unparsed=0 |
| IMG-5 · 사람 포함 여부 판별 | opus | minimal | accuracy=0.933, f1=0.941, n_evaluated=30, n_unparsed=0 |
| IMG-6 · 표 이미지 구조 추출 | opus | minimal | cell_f1=0.841, n_evaluated=30 |
| TXT-1 · 문서(PDF) 이해 QA | opus | minimal | anls=0.913, token_f1=0.893, exact_match=0.867, n_evaluated=30, judge_mean=4.7 |
| TXT-2 · 표(엑셀) 이해 QA | opus | minimal | accuracy=0.1, token_f1=0.167, n_evaluated=30, judge_mean=4.433 |
| TXT-3 · 표 구조 추출 | opus | minimal | cell_f1=0.942, n_evaluated=30 |
| TXT-4 · 한국어 독해 QA | opus | minimal | token_f1=0.837, exact_match=0.6, n_evaluated=30, judge_mean=5.0 |
| TXT-5 · 텍스트 요약 | opus | minimal | rouge1=0.325, rouge2=0.159, rougeL=0.26, n_evaluated=30, judge_mean=4.897 |
| TXT-6 · 감정 분석 | opus | minimal | accuracy=0.867, macro_f1=0.861, n_evaluated=30, n_unparsed=0 |
| TXT-7 · 키워드 추출 | opus | minimal | precision=0.303, recall=0.451, f1=0.362, n_evaluated=30, macro_precision=0.302, macro_recall=0.449 |
| TXT-8 · 비속어/유해성 판별 | opus | minimal | accuracy=0.8, f1=0.667, n_evaluated=30, n_unparsed=0 |
| IMG-1 · 이미지 캡션 생성 | sol | minimal | caption_token_f1=0.444, n_evaluated=30, judge_mean=3.733 |
| IMG-2 · 이미지 태그(객체) 추출 | sol | minimal | micro_precision=0.18, micro_recall=0.4, micro_f1=0.248, macro_precision=0.201, macro_recall=0.431, macro_f1=0.266 |
| IMG-3 · 무기/위협 존재 판별 | sol | minimal | accuracy=0.633, f1=0.776, n_evaluated=30, n_unparsed=0 |
| IMG-4 · 성인/NSFW 이미지 판별 | sol | minimal | accuracy=0.933, f1=0.929, n_evaluated=30, n_unparsed=0 |
| IMG-5 · 사람 포함 여부 판별 | sol | minimal | accuracy=0.967, f1=0.971, n_evaluated=30, n_unparsed=0 |
| IMG-6 · 표 이미지 구조 추출 | sol | minimal | cell_f1=0.786, n_evaluated=30 |
| TXT-1 · 문서(PDF) 이해 QA | sol | minimal | anls=0.881, token_f1=0.824, exact_match=0.833, n_evaluated=30, judge_mean=4.467 |
| TXT-2 · 표(엑셀) 이해 QA | sol | minimal | accuracy=0.0, token_f1=0.189, n_evaluated=30, judge_mean=3.7 |
| TXT-3 · 표 구조 추출 | sol | minimal | cell_f1=0.979, n_evaluated=30 |
| TXT-4 · 한국어 독해 QA | sol | minimal | token_f1=0.881, exact_match=0.667, n_evaluated=30, judge_mean=5.0 |
| TXT-5 · 텍스트 요약 | sol | minimal | rouge1=0.315, rouge2=0.138, rougeL=0.254, n_evaluated=30, judge_mean=4.767 |
| TXT-6 · 감정 분석 | sol | minimal | accuracy=0.833, macro_f1=0.829, n_evaluated=30, n_unparsed=0 |
| TXT-7 · 키워드 추출 | sol | minimal | precision=0.37, recall=0.397, f1=0.383, n_evaluated=30, macro_precision=0.375, macro_recall=0.409 |
| TXT-8 · 비속어/유해성 판별 | sol | minimal | accuracy=0.833, f1=0.737, n_evaluated=30, n_unparsed=0 |

## 성능: 수행시간·비용 (모델별)

| 모델 | 호출 | 오류 | latency median(ms) | p95(ms) | 입력토큰 | 출력토큰 | 비용(USD) |
|---|---|---|---|---|---|---|---|
| glm | 240 | 0 | 779.4 | 7009.5 | 101364 | 26757 | 0.257848 |
| opus | 420 | 4 | 3222.3 | 13139.8 | 209397 | 59329 | 2.530217 |
| sol | 420 | 0 | 2698.9 | 9448.3 | 153498 | 44701 | 2.108523 |

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
| opus | ❌ | `A warmly lit bathroom with golden-textured walls features a vintage black-and-white clawfoot tub with brass fixtures, a white pedestal sink beneath a framed mir…` |
| sol | ❌ | `An elegant beige bathroom features a pedestal sink, framed mirror, and ornate clawfoot bathtub.` |

### IMG-2 · 이미지 태그(객체) 추출

**샘플 #0** · _모델 간 판정이 갈린 케이스_ · 정답: `['boat', 'chair', 'handbag', 'person', 'umbrella']`

**질문/지시:** List the main objects visible in this image as a comma-separated list of simple nouns.

**입력 이미지:**

![IMG-2 sample 0](gallery_IMG-2_s0.png)

| 모델 | 정답? | 출력/판정 |
|---|:--:|---|
| opus | ✅ | `Cruise ships, ocean, sky, pier, palm tree, umbrellas, lounge chairs, people, pool, beach, rocks, railing, planter` |
| sol | ✅ | `ships, pier, ocean, pool, umbrellas, chairs, people, palm tree` |

### IMG-3 · 무기/위협 존재 판별

**샘플 #2** · _모델 간 판정이 갈린 케이스_ · 정답: `1`

**질문/지시:** Does this image contain a weapon or threat (gun, knife, pistol, rifle, blood, etc.)? Answer exactly yes or no.

**입력 이미지:**

![IMG-3 sample 2](gallery_IMG-3_s2.png)

| 모델 | 정답? | 출력/판정 |
|---|:--:|---|
| opus | ✅ | `**Yes**` |
| sol | ✅ | `yes` |

### IMG-4 · 성인/NSFW 이미지 판별

**샘플 #0** (민감 태스크 — 입력 비표시, 판정값만) · _예시 케이스(자동 정답판정 불가 태스크)_ · 정답: `0`

**질문/지시:** You are a content-safety classifier. Does this image contain adult/NSFW (sexually explicit) content? Answer exactly 'nsfw' or 'safe'.

| 모델 | 정답? | 출력/판정 |
|---|:--:|---|
| opus | — | `safe` |
| sol | — | `safe` |

### IMG-5 · 사람 포함 여부 판별

**샘플 #16** · _모델 간 판정이 갈린 케이스_ · 정답: `1`

**질문/지시:** Is there a person visible in this image? Answer exactly yes or no.

**입력 이미지:**

![IMG-5 sample 16](gallery_IMG-5_s16.png)

| 모델 | 정답? | 출력/판정 |
|---|:--:|---|
| opus | ❌ | `The question asks whether a person is visible. There are photographs and framed portraits on the wall showing people, and there appear to be figurines. But actu…` |
| sol | ❌ | `no` |

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

**샘플 #23** · _모델 간 판정이 갈린 케이스_ · 정답: `['Dream Cream', 'Dream cream']`

**질문/지시:** Which is the Sunfeast biscuIt sub brand, placed first at the bottom?

**입력:**
> Based on the following document text, answer the question.
> 
> Document text:
> ITC Limited REPORT AND ACCOUNTS 2013 Sunfeast straddles all segments in the biscuit Sunfeast category and offers high quality products in exciting and innovative formats, which reinforces ITC's commitment to delivering a world-class product experience to the discerning consumer. Snacky snacky sunday Snacky Sunday NICE Quan Delishus Delishus MARIE LIGHT MARIE LICHIST Sunfeast sweet'n Dark Forcosy Dark Fantasy A Dark Fantasy Chaco file Dream Dream Cream Cream Dream Cream 223
> 
> Question: Which is the Sunfeast biscuIt sub brand, placed first at the bottom?
> 
> Answer with only the exact value from the document — no explanation, no full sentence.
> 
> Answer:

| 모델 | 정답? | 출력/판정 |
|---|:--:|---|
| glm | ❌ | `Snacky Sunday` |
| opus | ✅ | `Dream Cream` |
| sol | ❌ | `Dark Fantasy` |

### TXT-2 · 표(엑셀) 이해 QA

**샘플 #0** · _모델 간 판정이 갈린 케이스_ · 정답: `['4']`

**질문/지시:** what's the number of parishes founded in the 1800s?

**입력:**
> Based on the following table, answer the question accurately.
> 
> Table:
> | Parish | Locality | Parish Priest | Founded | Closed |
> | --- | --- | --- | --- | --- |
> | St Mary | Bacup | Fr Frank Thorpe | 1852 | ---- |
> | Our Immaculate Mother & St Anselm | Whitworth | Fr Frank Thorpe | 1860 | ---- |
> | St Joseph | Stacksteads | ---- | 1947 | 2005 |
> | St Joseph & St Peter | Newchurch-In-Rossendale | Fr Philip Boast | 1915 | ---- |
> | The Immaculate Conception | Haslingden | Fr Canon John Mackie | 1854 | ---- |
> | St Veronica (Chapel of Ease) | Helmshore | Served from The Immaculate Conception | 1959 | ---- |
> | St James the Less | Rawtenstall | Fr David Lupton, Rural Dean | 1828 | ---- |
> 
> Question: what's the number of parishes founded in the 1800s?
> 
> Answer:

| 모델 | 정답? | 출력/판정 |
|---|:--:|---|
| glm | ✅ | `Based on the table provided, there are **6** parishes founded in the 1800s:  1. St Mary (1852) 2. Our Immaculate Mother & St Anselm (1860) 3. St Joseph & St Pet…` |
| opus | ✅ | `Looking at the table, the parishes founded in the 1800s are:  1. **St Mary** (Bacup) – 1852 2. **Our Immaculate Mother & St Anselm** (Whitworth) – 1860 3. **The…` |
| sol | ✅ | `4 parishes.` |

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
| glm | — | `A Pennsylvania community has mobilized hundreds of volunteers to search for 13-year-old Cayman Naib, who went missing after impulsively leaving his home due to …` |
| opus | — | `Thirteen-year-old Cayman Naib, an eighth-grader at Pennsylvania's Shipley School, has been missing since Wednesday, when his parents believe he impulsively left…` |
| sol | — | `Hundreds of volunteers are searching for 13-year-old Cayman Naib, who disappeared from his Pennsylvania home after receiving an email about overdue homework. De…` |

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
| opus | ✅ | `positive` |
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
| opus | ✅ | `nonlinear extrapolation, scalar random process, canonical decomposition, V. S. Pugachev canonical decomposition apparatus, mixed central moment functions, nonli…` |
| sol | ✅ | `nonlinear extrapolation algorithm, scalar random process, nonlinear random dependence, mixed central moment functions, V. S. Pugachev canonical decomposition, t…` |

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
| glm | — | `clean` |
| opus | — | `toxic` |
| sol | — | `toxic` |


## 참고: Reasoning 정책

**Reasoning OFF(minimal) 단일 모드로 고정.** reasoning의 효과가 특정 태스크 성능 개선에만 한정되는 반면 실험 시간을 크게 늘리기 때문(특히 GLM은 full reasoning 시 타임아웃 빈발). 각 모델의 최소 reasoning으로 측정하며, `databricks-claude-opus-5`와 judge는 완전 OFF가 불가해 지원 최소값을 사용한다.

## Fact Sheet (Executive Summary 근거 — 감사용)

```json
{
  "per_task_scores": {
    "IMG-1/minimal": {
      "opus": 0.3267,
      "sol": 0.4438
    },
    "IMG-2/minimal": {
      "opus": 0.2016,
      "sol": 0.2482
    },
    "IMG-3/minimal": {
      "opus": 0.7,
      "sol": 0.6333
    },
    "IMG-4/minimal": {
      "opus": 0.9333,
      "sol": 0.9333
    },
    "IMG-5/minimal": {
      "opus": 0.9333,
      "sol": 0.9667
    },
    "TXT-1/minimal": {
      "opus": 0.9132,
      "sol": 0.8806,
      "glm": 0.8632
    },
    "TXT-2/minimal": {
      "opus": 0.1,
      "sol": 0.0,
      "glm": 0.1
    },
    "TXT-3/minimal": {
      "opus": 0.9424,
      "sol": 0.9791,
      "glm": 0.9725
    },
    "TXT-4/minimal": {
      "opus": 0.8371,
      "sol": 0.881,
      "glm": 0.9507
    },
    "TXT-5/minimal": {
      "opus": 0.3249,
      "sol": 0.315,
      "glm": 0.2716
    },
    "TXT-6/minimal": {
      "opus": 0.8667,
      "sol": 0.8333,
      "glm": 0.8333
    },
    "TXT-7/minimal": {
      "opus": 0.3625,
      "sol": 0.383,
      "glm": 0.516
    },
    "TXT-8/minimal": {
      "opus": 0.8,
      "sol": 0.8333,
      "glm": 0.9
    },
    "IMG-6/minimal": {
      "opus": 0.841,
      "sol": 0.7864
    }
  },
  "task_winners": {
    "IMG-1/minimal": "sol",
    "IMG-2/minimal": "sol",
    "IMG-3/minimal": "opus",
    "IMG-4/minimal": "opus",
    "IMG-5/minimal": "sol",
    "TXT-1/minimal": "opus",
    "TXT-2/minimal": "opus",
    "TXT-3/minimal": "sol",
    "TXT-4/minimal": "glm",
    "TXT-5/minimal": "opus",
    "TXT-6/minimal": "opus",
    "TXT-7/minimal": "glm",
    "TXT-8/minimal": "glm",
    "IMG-6/minimal": "opus"
  },
  "win_counts": {
    "sol": 4,
    "opus": 7,
    "glm": 3
  },
  "cheapest_model": "glm",
  "fastest_model": "glm",
  "perf": {
    "opus": {
      "n_calls": 420,
      "errors": 4,
      "latency_ms_median": 3222.3,
      "latency_ms_p95": 13139.8,
      "total_usd": 2.530217,
      "in_tokens": 209397,
      "out_tokens": 59329
    },
    "sol": {
      "n_calls": 420,
      "errors": 0,
      "latency_ms_median": 2698.9,
      "latency_ms_p95": 9448.3,
      "total_usd": 2.108523,
      "in_tokens": 153498,
      "out_tokens": 44701
    },
    "glm": {
      "n_calls": 240,
      "errors": 0,
      "latency_ms_median": 779.4,
      "latency_ms_p95": 7009.5,
      "total_usd": 0.257848,
      "in_tokens": 101364,
      "out_tokens": 26757
    }
  }
}
```
