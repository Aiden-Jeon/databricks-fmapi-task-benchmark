# 벤치마크 리포트 — 2026-07-31T10-58

> 📊 고객 설명용 프레젠테이션: **[▶ 브라우저로 바로 보기](https://htmlpreview.github.io/?https://github.com/Aiden-Jeon/databricks-fmapi-task-benchmark/blob/main/image-text-performance/reports/2026-07-31T10-58/presentation.html)** · [HTML 소스](./presentation.html) — 이 리포트 결과를 슬라이드로 정리

## 평가 대상 모델 (Databricks hosted)

| 별칭 | Databricks model name | vision |
|---|---|---|
| opus | `databricks-claude-opus-5` | ✅ |
| sol | `databricks-gpt-5-6-sol` | ✅ |
| glm | `databricks-glm-5-2` | ❌ |

> Judge: `databricks-gemini-3-1-pro`

## Executive Summary

벤치마크 결과 'sol'은 총 5개 과제(IMG 3개, TXT 2개)에서 1위를 차지해 가장 많은 우위를 점한 반면, 'glm'은 텍스트(TXT) 과제에서만 4개 부문 1위를 기록하며 텍스트 분야에 강점을 보였습니다. 'opus'는 4개 부문에서 1위를 했으나 유일하게 1회의 오류가 발생했고, 'sol'은 TXT-2 과제에서 0점을 기록하는 약점을 드러냈습니다. 속도와 비용 측면에서는 'glm'이 중간 지연시간 972.5ms, 총비용 약 0.22달러로 가장 빠르고 저렴해 효율성이 뛰어납니다. 반대로 'opus'는 중간 지연시간 2641.2ms, 총비용 약 1.83달러로 세 모델 중 가장 느리고 비싸 명확한 트레이드오프를 보여줍니다.

<sub>규칙 기반 요약(대조용): 태스크별 1위 횟수는 sol 5회, opus 4회, glm 4회로 **sol**가 가장 많다. 응답 속도는 **glm**가 가장 빠르다(median 972.5ms). 비용은 **glm**가 가장 낮다($0.222865).</sub>

## 비교 그래프

### 태스크별 모델 성능 비교

![태스크별 모델 성능 비교](chart_scores.png)

### 모델별 속도 vs 비용

![모델별 속도 vs 비용](chart_perf.png)

**태스크 ID 범례:** IMG-1=이미지 캡션 생성 · IMG-2=이미지 태그(객체) 추출 · IMG-3=무기/위협 존재 판별 · IMG-4=성인/NSFW 이미지 판별 · IMG-5=사람 포함 여부 판별 · TXT-1=문서(PDF) 이해 QA · TXT-2=표(엑셀) 이해 QA · TXT-3=표 구조 추출 · TXT-4=한국어 독해 QA · TXT-5=텍스트 요약 · TXT-6=감정 분석 · TXT-7=키워드 추출 · TXT-8=비속어/유해성 판별

## 정량 결과 (태스크 × 모델)

| 태스크 | 모델 | reasoning | 대표 메트릭 |
|---|---|---|---|
| TXT-1 · 문서(PDF) 이해 QA | glm | minimal | token_f1=0.022, exact_match=0.0, n_evaluated=30 |
| TXT-2 · 표(엑셀) 이해 QA | glm | minimal | accuracy=0.2, token_f1=0.302, n_evaluated=10 |
| TXT-3 · 표 구조 추출 | glm | minimal | cell_f1=0.973, n_evaluated=30 |
| TXT-4 · 한국어 독해 QA | glm | minimal | token_f1=0.958, exact_match=0.867, n_evaluated=30 |
| TXT-5 · 텍스트 요약 | glm | minimal | rouge1=0.282, rouge2=0.116, rougeL=0.231, n_evaluated=30 |
| TXT-6 · 감정 분석 | glm | minimal | accuracy=0.833, macro_f1=0.829, n_evaluated=30, n_unparsed=0 |
| TXT-7 · 키워드 추출 | glm | minimal | precision=0.494, recall=0.527, f1=0.51, n_evaluated=30 |
| TXT-8 · 비속어/유해성 판별 | glm | minimal | accuracy=0.867, f1=0.778, n_evaluated=30, n_unparsed=0 |
| IMG-1 · 이미지 캡션 생성 | opus | minimal | caption_token_f1=0.338, n_evaluated=30 |
| IMG-2 · 이미지 태그(객체) 추출 | opus | minimal | micro_precision=0.123, micro_recall=0.482, micro_f1=0.197, macro_precision=0.146 |
| IMG-3 · 무기/위협 존재 판별 | opus | minimal | accuracy=0.7, f1=0.809, n_evaluated=30, n_unparsed=0 |
| IMG-4 · 성인/NSFW 이미지 판별 | opus | minimal | accuracy=0.933, f1=0.929, n_evaluated=30, n_unparsed=0 |
| IMG-5 · 사람 포함 여부 판별 | opus | minimal | accuracy=0.933, f1=0.941, n_evaluated=30, n_unparsed=0 |
| TXT-1 · 문서(PDF) 이해 QA | opus | minimal | token_f1=0.005, exact_match=0.0, n_evaluated=30 |
| TXT-2 · 표(엑셀) 이해 QA | opus | minimal | accuracy=0.1, token_f1=0.192, n_evaluated=10 |
| TXT-3 · 표 구조 추출 | opus | minimal | cell_f1=0.943, n_evaluated=30 |
| TXT-4 · 한국어 독해 QA | opus | minimal | token_f1=0.878, exact_match=0.633, n_evaluated=30 |
| TXT-5 · 텍스트 요약 | opus | minimal | rouge1=0.321, rouge2=0.168, rougeL=0.267, n_evaluated=30 |
| TXT-6 · 감정 분석 | opus | minimal | accuracy=0.867, macro_f1=0.861, n_evaluated=30, n_unparsed=0 |
| TXT-7 · 키워드 추출 | opus | minimal | precision=0.3, recall=0.442, f1=0.357, n_evaluated=30 |
| TXT-8 · 비속어/유해성 판별 | opus | minimal | accuracy=0.8, f1=0.667, n_evaluated=30, n_unparsed=0 |
| IMG-1 · 이미지 캡션 생성 | sol | minimal | caption_token_f1=0.438, n_evaluated=30 |
| IMG-2 · 이미지 태그(객체) 추출 | sol | minimal | micro_precision=0.181, micro_recall=0.388, micro_f1=0.247, macro_precision=0.204 |
| IMG-3 · 무기/위협 존재 판별 | sol | minimal | accuracy=0.633, f1=0.776, n_evaluated=30, n_unparsed=0 |
| IMG-4 · 성인/NSFW 이미지 판별 | sol | minimal | accuracy=0.933, f1=0.929, n_evaluated=30, n_unparsed=0 |
| IMG-5 · 사람 포함 여부 판별 | sol | minimal | accuracy=0.967, f1=0.971, n_evaluated=30, n_unparsed=0 |
| TXT-1 · 문서(PDF) 이해 QA | sol | minimal | token_f1=0.026, exact_match=0.0, n_evaluated=30 |
| TXT-2 · 표(엑셀) 이해 QA | sol | minimal | accuracy=0.0, token_f1=0.265, n_evaluated=10 |
| TXT-3 · 표 구조 추출 | sol | minimal | cell_f1=0.979, n_evaluated=30 |
| TXT-4 · 한국어 독해 QA | sol | minimal | token_f1=0.896, exact_match=0.667, n_evaluated=30 |
| TXT-5 · 텍스트 요약 | sol | minimal | rouge1=0.314, rouge2=0.142, rougeL=0.257, n_evaluated=30 |
| TXT-6 · 감정 분석 | sol | minimal | accuracy=0.833, macro_f1=0.829, n_evaluated=30, n_unparsed=0 |
| TXT-7 · 키워드 추출 | sol | minimal | precision=0.386, recall=0.429, f1=0.407, n_evaluated=30 |
| TXT-8 · 비속어/유해성 판별 | sol | minimal | accuracy=0.8, f1=0.7, n_evaluated=30, n_unparsed=0 |

## 성능: 수행시간·비용 (모델별)

| 모델 | 호출 | 오류 | latency median(ms) | p95(ms) | 입력토큰 | 출력토큰 | 비용(USD) |
|---|---|---|---|---|---|---|---|
| glm | 220 | 0 | 972.5 | 7377.6 | 71850 | 28421 | 0.222865 |
| opus | 370 | 1 | 2641.2 | 7578.4 | 161119 | 40941 | 1.829125 |
| sol | 370 | 0 | 1762.9 | 5314.3 | 119179 | 25442 | 1.359158 |

> 비용은 `config/pricing.yaml` DBU 단가 기반 추정(usd_per_dbu 가정값). 정밀 비용은 `system.ai_gateway.usage` 조인 필요(§10).

## 정성 비교: 모델 간 결과가 갈린 샘플

각 태스크에서 모델들의 출력·판정이 **가장 크게 갈린 샘플**을 골라 나란히 비교한다. 평균 점수로는 안 보이는 모델별 차이를 드러낸다.

### IMG-1 · 이미지 캡션 생성

**샘플 #0** · 정답: `['An elegant bathroom features a tub, sink, mirror, and decorations. ', 'An old fashion above ground tub is shown with gold feet.', 'A lovely, vintage-styled ba…`

**입력 이미지:**

![IMG-1 sample 0](gallery_IMG-1_s0.png)

| 모델 | 출력/판정 |
|---|---|
| opus | `A warmly lit bathroom with faux-finished golden walls features a vintage black-and-white clawfoot tub with brass fixtures, a white pedestal sink beneath a frame…` |
| sol | `An elegant beige bathroom features a pedestal sink, framed mirror, and a freestanding clawfoot bathtub.` |

### IMG-2 · 이미지 태그(객체) 추출

**샘플 #0** · 정답: `{'handbag', 'boat', 'person', 'chair', 'umbrella'}`

**입력 이미지:**

![IMG-2 sample 0](gallery_IMG-2_s0.png)

| 모델 | 출력/판정 |
|---|---|
| opus | `cruise ships, ocean, pier, umbrellas, palm tree, pool, lounge chairs, people, sand, sky, railing, planter` |
| sol | `cruise ships, pier, ocean, pool, umbrellas, lounge chairs, palm tree, people` |

### IMG-3 · 무기/위협 존재 판별

**샘플 #13** · 정답: `0`

**입력 이미지:**

![IMG-3 sample 13](gallery_IMG-3_s13.png)

| 모델 | 출력/판정 |
|---|---|
| opus | `I can see a surveillance image from what appears to be a sandwich shop, with a customer at the counter and an employee behind it. The customer is holding a dark…` |
| sol | `yes` |

### IMG-5 · 사람 포함 여부 판별

**샘플 #23** · 정답: `1`

**입력 이미지:**

![IMG-5 sample 23](gallery_IMG-5_s23.png)

| 모델 | 출력/판정 |
|---|---|
| opus | `no` |
| sol | `yes` |

### TXT-1 · 문서(PDF) 이해 QA

**샘플 #0** · 정답: `['-0.124']`

**입력:**
> Answer the following question:
> 
> Question: What is Thorax size?
> 
> Answer:

| 모델 | 출력/판정 |
|---|---|
| glm | `The question "What is Thorax size?" is quite broad, as the thorax (commonly known as the chest) varies significantly depending on the species, age, sex, and ind…` |
| opus | `Could you clarify what you mean by "Thorax size"?  There are a few different things this could refer to:  **1. The thorax in anatomy (human/animal)** — In human…` |
| sol | `Thorax size refers to the dimensions or circumference of the chest—the body region between the neck and abdomen. It is commonly measured around the widest part …` |

### TXT-2 · 표(엑셀) 이해 QA

**샘플 #0** · 정답: `['Lake Palas Tuzla']`

**입력:**
> Based on the following table, answer the question accurately.
> 
> Table:
> | Name in English | Name in Turkish | Area (km2) | Depth | Location (districts and/or provinces) |
> | --- | --- | --- | --- | --- |
> | Lake Van | Van Gölü | 3755 km2 | 451 m | Van, Bitlis |
> | Lake Tuz | Tuz Gölü | 1500 km2 | 2 m | Aksaray, Ankara, Konya |
> | Lake Beyşehir | Beyşehir Gölü | 656 km2 | 10 m | Beyşehir in Konya, Isparta |
> | Lake Eğirdir | Eğirdir Gölü | 482 km2 |  | Isparta |
> | Lake İznik | İznik Gölü | 308 km2 |  | İznik in Bursa, Yalova |
> | Lake Burdur | Burdur Gölü | 200 km2 |  | Burdur, Isparta |
> | Lake Manyas | Manyas Gölü | 166 km2 |  | Balıkesir |
> | Lake Acıgöl | Acıgöl | 153 km2 |  | Denizli, Afyonkarahisar |
> | Lake Uluabat | Uluabat Gölü | 134 km2 | 1–2 m | Bursa |
> | Lake Çıldır | Çıldır Gölü | 115 km2 |  | Ardahan, Kars |
> | Lake Palas Tuzla | Palas Tuzla Gölü | 106 km2 | 15 m | Palas/Kayseri |
> | Lake Akşehir | Akşehir Gölü | 105 km2 |  | Akşehir in Konya, Afyonkarahisar |
> | Lake Eber | Eber Gölü | 104 km2 |  | Afyonkarahisar |
> | Lake Erçek | Erçek Gölü | 98 km2 |  | Van |
> | Lake Hazar | Hazar Gölü | 86 km2 |  | Elazığ |
> | Lake Bafa | Bafa Gölü | 60 km2 |  | Aydın, Muğla |
> | Lake Köyceğiz | Köy…

| 모델 | 출력/판정 |
|---|---|
| glm | `Lake Palas Tuzla is deeper. According to the table, Lake Palas Tuzla has a depth of 15 m, while Lake Tuz has a depth of 2 m.` |
| opus | `Lake Palas Tuzla is deeper. It has a depth of 15 m, compared with Lake Tuz's depth of only 2 m.` |
| sol | `Lake Palas Tuzla is deeper: **15 m**, compared with **Lake Tuz at 2 m**.` |

### TXT-3 · 표 구조 추출

**샘플 #1** · 정답:

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

| 모델 | 출력/판정 |
|---|---|
| glm | `'''html <table>   <tr>     <td></td>     <td>Group 1</td>     <td>Group 2</td>     <td></td>     <td></td>     <td></td>     <td></td>     <td></td>     <td></t…` |
| opus | `<table> <tr><td></td><td>Group 1</td><td>Group 2</td><td></td><td></td><td></td><td></td><td></td><td></td></tr> <tr><td>Cut-off of FibroTest</td><td>Sensitivit…` |
| sol | `<table>   <tr>     <td></td>     <td colspan="4">Group 1</td>     <td colspan="4">Group 2</td>   </tr>   <tr>     <td>Cut-off of FibroTest</td>     <td>Sensitiv…` |

### TXT-4 · 한국어 독해 QA

**샘플 #11** · 정답: `['교바시']`

**입력:**
> 다음 지문을 읽고 질문에 답하세요. 답변은 지문에서 나타나는 정답만 간결하게 기술하세요.
> 
> 지문:
> 한편 기슈지 쾌속(일본어: 紀州路快速 키슈우지카이소쿠)은 오사카 방면에서 와카야마 방면에의 관광객 및 통근객의 증대를 도모하고자 이 때까지 교토 역 ~ 신오사카 역 방면에서 와카야마 방면으로 운행하는 특급 열차에 더하여 1999년 5월부터 신설된 쾌속 열차이다. 오사카 순환선에서 발착하여 덴노지 역에서 교바시, 오사카, 니시쿠조를 경유하여 (다만 대부분은 교바시 시발) 운전되고 있다. 간쿠 쾌속과 같이 223계 전동차 0번대와 2500번대가 운용된다. 히네노 역 이북에서는 전 열차가 8량 편성으로 기본적으로 히네노 역까지는 간쿠 쾌속과 병결 운전을 행하여 히네노 역에서 분할과 병결이 이루어진다. 이전에는 출퇴근 시간대에 1~5호차의 5량이 기슈지 쾌속이었으며, 나머지 시간대에는 그 반대였으나 2008년 3월 15일의 시간표 개정에 따라 8량 편성을 간쿠와 기슈지 쾌속이 4량씩 차지하는 형태로 통일되어 시간대에 따라 기슈지 쾌속과 간쿠 쾌속의 연결 순서가 바뀌는 경우도 있다. 휴일 시간표에서 아침의 상행 열차에는 간쿠 쾌속이 병결되지 않은 채 기세이 본선 고보 역에서 시발하여 와카야마 역에서 8량으로 증결하여 운행하는 특별한 형태도 있다. 오사카 방면으로 가는 열차의 와카야마 역 ~ 히네노 역 간에는 4량 편성으로 운전되기 때문에 시발역인 와카야마 역에서부터 혼잡해지는 경우가 많다.
> 
> 질문: 기슈지 쾌속의 시발지은 대부분 어느 곳인가?
> 
> 답변:

| 모델 | 출력/판정 |
|---|---|
| glm | `교바시` |
| opus | `교바시(교바시 역)` |
| sol | `교바시역` |

### TXT-5 · 텍스트 요약

**샘플 #0** · 정답:

```
Cayman Naib, 13, hasn't been heard from since Wednesday .
Police, family, volunteers search for eighth-grader .
```

**입력:**
> Generate a concise summary of the following document. The summary should capture the key points in 2-3 sentences.
> 
> Document:
> (CNN)A Pennsylvania community is pulling together to search for an eighth-grade student who has been missing since Wednesday. The search has drawn hundreds of volunteers on foot and online. The parents of Cayman Naib, 13, have been communicating through the Facebook group "Find Cayman" since a day after his disappearance, according to close friend David Binswanger. Newtown Police say Cayman was last seen wearing a gray down winter jacket, black ski pants and hiking boots. He could be in the Radnor-Wayne area, roughly 20 miles from Philadelphia, or may have purchased a train ticket to Philadelphia, according to an alert posted on Facebook. "We think that he got a email from school and was upset by it and left as an impulsive act," Farid Naib, Cayman's father, wrote Thursday on the group page. "We have spoken to his friends and they do not know where he is. Cayman does have his phone, we don't know if he has any cash, he does not have his wallet." The parents said that his phone was out of power at the time. "Cayman left within 30 minutes after he received an e…

| 모델 | 출력/판정 |
|---|---|
| glm | `A Pennsylvania community has mobilized hundreds of volunteers, both on foot and online, to search for 13-year-old Cayman Naib, who went missing after impulsivel…` |
| opus | `Thirteen-year-old Cayman Naib, an eighth-grader at The Shipley School, has been missing since Wednesday in a Pennsylvania community near Philadelphia, having ap…` |
| sol | `Hundreds of volunteers are searching for 13-year-old Cayman Naib, who disappeared from the Radnor-Wayne area after receiving a school email about overdue homewo…` |

### TXT-6 · 감정 분석

**샘플 #20** · 정답: `1`

**입력:**
> Classify the sentiment of the following text as exactly one word: "positive" or "negative".
> 
> Text: 오게두어라! 서리한이 굶주렸다!
> 
> Respond with exactly one word: positive or negative

| 모델 | 출력/판정 |
|---|---|
| glm | `negative` |
| opus | `positive` |
| sol | `negative` |

### TXT-7 · 키워드 추출

**샘플 #0** · 정답: `{'mixed central moment functions', 'nonlinear extrapolation algorithm', 'canonical decomposition apparatus', 'scalar random process', 'nonlinear random dependen…`

**입력:**
> Extract the key phrases or keywords from the following academic document. List them as a comma-separated list with no additional text.
> 
> Title and Abstract:
> Nonlinear extrapolation algorithm for realization of a scalar random process
> 
> A method of construction of a nonlinear extrapolation algorithm is proposed. This method makes it possible to take into account any nonlinear random dependences that exist in an investigated process and are described by mixed central moment functions. The method is based on the V. S. Pugachev canonical decomposition apparatus. As an example, the problem of nonlinear extrapolation is solved for a moment function of third order
> 
> Key phrases (comma-separated):

| 모델 | 출력/판정 |
|---|---|
| glm | `Nonlinear extrapolation algorithm, scalar random process, nonlinear random dependences, mixed central moment functions, V. S. Pugachev canonical decomposition a…` |
| opus | `nonlinear extrapolation, scalar random process, canonical decomposition, V. S. Pugachev canonical expansion, mixed central moment functions, nonlinear random de…` |
| sol | `nonlinear extrapolation algorithm, scalar random process, nonlinear random dependences, mixed central moment functions, V. S. Pugachev canonical decomposition, …` |

### TXT-8 · 비속어/유해성 판별

**샘플 #1** · 정답: `0`

**입력:**
> You are a content moderation classifier. Analyze the following text and determine if it contains profanity, hate speech, or toxic language. Respond with exactly one word: "toxic" or "clean".
> 
> Text: Hi Explicit, can you block O Fenian for edit-warring on the Giant's Causeway wp. He has made several edits which can only be described as terrorism.
> 
> Respond with exactly one word: toxic or clean

| 모델 | 출력/판정 |
|---|---|
| glm | `toxic` |
| opus | `I need to consider what's actually in this text. It's a user asking an admin to block another editor. The word "terrorism" appears, but it's used hyperbolically…` |
| sol | `toxic` |


## 참고: Reasoning 정책

**Reasoning OFF(minimal) 단일 모드로 고정.** reasoning의 효과가 특정 태스크 성능 개선에만 한정되는 반면 실험 시간을 크게 늘리기 때문(특히 GLM은 full reasoning 시 타임아웃 빈발). 각 모델의 최소 reasoning으로 측정하며, `databricks-claude-opus-5`와 judge는 완전 OFF가 불가해 지원 최소값을 사용한다.

## Fact Sheet (Executive Summary 근거 — 감사용)

```json
{
  "per_task_scores": {
    "IMG-1/minimal": {
      "opus": 0.3381,
      "sol": 0.4376
    },
    "IMG-2/minimal": {
      "opus": 0.1966,
      "sol": 0.2472
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
      "opus": 0.0047,
      "sol": 0.0264,
      "glm": 0.0216
    },
    "TXT-2/minimal": {
      "opus": 0.1,
      "sol": 0.0,
      "glm": 0.2
    },
    "TXT-3/minimal": {
      "opus": 0.9432,
      "sol": 0.9791,
      "glm": 0.9733
    },
    "TXT-4/minimal": {
      "opus": 0.8783,
      "sol": 0.8957,
      "glm": 0.9584
    },
    "TXT-5/minimal": {
      "opus": 0.3208,
      "sol": 0.3144,
      "glm": 0.2818
    },
    "TXT-6/minimal": {
      "opus": 0.8667,
      "sol": 0.8333,
      "glm": 0.8333
    },
    "TXT-7/minimal": {
      "opus": 0.3571,
      "sol": 0.4066,
      "glm": 0.5099
    },
    "TXT-8/minimal": {
      "opus": 0.8,
      "sol": 0.8,
      "glm": 0.8667
    }
  },
  "task_winners": {
    "IMG-1/minimal": "sol",
    "IMG-2/minimal": "sol",
    "IMG-3/minimal": "opus",
    "IMG-4/minimal": "opus",
    "IMG-5/minimal": "sol",
    "TXT-1/minimal": "sol",
    "TXT-2/minimal": "glm",
    "TXT-3/minimal": "sol",
    "TXT-4/minimal": "glm",
    "TXT-5/minimal": "opus",
    "TXT-6/minimal": "opus",
    "TXT-7/minimal": "glm",
    "TXT-8/minimal": "glm"
  },
  "win_counts": {
    "sol": 5,
    "opus": 4,
    "glm": 4
  },
  "cheapest_model": "glm",
  "fastest_model": "glm",
  "perf": {
    "opus": {
      "n_calls": 370,
      "errors": 1,
      "latency_ms_median": 2641.2,
      "latency_ms_p95": 7578.4,
      "total_usd": 1.829125,
      "in_tokens": 161119,
      "out_tokens": 40941
    },
    "sol": {
      "n_calls": 370,
      "errors": 0,
      "latency_ms_median": 1762.9,
      "latency_ms_p95": 5314.3,
      "total_usd": 1.359158,
      "in_tokens": 119179,
      "out_tokens": 25442
    },
    "glm": {
      "n_calls": 220,
      "errors": 0,
      "latency_ms_median": 972.5,
      "latency_ms_p95": 7377.6,
      "total_usd": 0.222865,
      "in_tokens": 71850,
      "out_tokens": 28421
    }
  }
}
```
