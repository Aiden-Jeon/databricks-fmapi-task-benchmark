# 벤치마크 리포트 — 2026-08-04T09-54

> 📊 고객 설명용 프레젠테이션: **[▶ 브라우저로 바로 보기](https://htmlpreview.github.io/?https://github.com/Aiden-Jeon/databricks-fmapi-task-benchmark/blob/main/image-text-performance/reports/2026-08-04T09-54/presentation.html)** · [HTML 소스](./presentation.html) — 이 리포트 결과를 슬라이드로 정리

## 평가 대상 모델 (Databricks hosted)

| 별칭 | Databricks model name | vision |
|---|---|---|
| opus | `databricks-claude-opus-5` | ✅ |
| sol | `databricks-gpt-5-6-sol` | ✅ |
| glm | `databricks-glm-5-2` | ❌ |

> Judge: `databricks-gemini-3-1-pro`

## Executive Summary

'opus'는 총 9개 태스크에서 1위를 차지하며 이미지와 텍스트 전반에서 가장 뛰어난 성능을 보였으나, 세 모델 모두 특정 텍스트 태스크(TXT-3, 6, 7, 8)에서 0점을 기록하며 공통적인 취약점을 보였습니다. 'sol'과 'glm'은 각각 3개와 2개의 태스크에서 우승하며 일부 영역에서만 제한적인 강점을 나타냈습니다. 속도와 비용의 트레이드오프 측면에서 'glm'은 가장 빠르고(중간값 1291.8ms) 저렴($0.086)하여 효율성이 높은 반면, 성능이 가장 우수한 'opus'는 응답 속도가 가장 느리고 비용($1.67)도 제일 비쌉니다.

<sub>규칙 기반 요약(대조용): 태스크별 1위 횟수는 opus 9회, sol 3회, glm 2회로 **opus**가 가장 많다. 응답 속도는 **glm**가 가장 빠르다(median 1291.8ms). 비용은 **glm**가 가장 낮다($0.086734).</sub>

## 비교 그래프

### 태스크별 모델 성능 비교

![태스크별 모델 성능 비교](chart_scores.png)

### 모델별 속도 vs 비용

![모델별 속도 vs 비용](chart_perf.png)

**태스크 ID 범례:** IMG-1=이미지 캡션 생성 · IMG-2=이미지 태그(객체) 추출 · IMG-3=무기/위협 존재 판별 · IMG-4=성인/NSFW 이미지 판별 · IMG-5=사람 포함 여부 판별 · IMG-6=표 이미지 구조 추출 · TXT-1=문서(PDF) 이해 QA · TXT-2=표(엑셀) 이해 QA · TXT-3=표 구조 추출 · TXT-4=한국어 독해 QA · TXT-5=텍스트 요약 · TXT-6=감정 분석 · TXT-7=키워드 추출 · TXT-8=비속어/유해성 판별

## 정량 결과 (태스크 × 모델)

| 태스크 | 모델 | reasoning | 대표 메트릭 |
|---|---|---|---|
| TXT-1 · 문서(PDF) 이해 QA | glm | minimal | token_f1=0.004, exact_match=0.0, n_evaluated=30, judge_mean=1.6 |
| TXT-2 · 표(엑셀) 이해 QA | glm | minimal | accuracy=0.3, token_f1=0.421, n_evaluated=10, judge_mean=4.5 |
| TXT-3 · 표 구조 추출 | glm | minimal | cell_f1=0.0, n_evaluated=30 |
| TXT-4 · 한국어 독해 QA | glm | minimal | token_f1=0.947, exact_match=0.833, n_evaluated=30, judge_mean=4.433 |
| TXT-5 · 텍스트 요약 | glm | minimal | rouge1=0.306, rouge2=0.127, rougeL=0.244, n_evaluated=30, judge_mean=4.567 |
| TXT-6 · 감정 분석 | glm | minimal | accuracy=0.0, macro_f1=0.0, n_evaluated=0, n_unparsed=30 |
| TXT-7 · 키워드 추출 | glm | minimal | precision=0.0, recall=0.0, f1=0.0, n_evaluated=30, macro_precision=0.0, macro_recall=0.0 |
| TXT-8 · 비속어/유해성 판별 | glm | minimal | accuracy=0.0, f1=0.0, n_evaluated=0, n_unparsed=30 |
| IMG-1 · 이미지 캡션 생성 | opus | minimal | caption_token_f1=0.337, n_evaluated=30, judge_mean=0.0 |
| IMG-2 · 이미지 태그(객체) 추출 | opus | minimal | micro_precision=0.13, micro_recall=0.494, micro_f1=0.205, macro_precision=0.151, macro_recall=0.511, macro_f1=0.227 |
| IMG-3 · 무기/위협 존재 판별 | opus | minimal | accuracy=0.667, f1=0.792, n_evaluated=30, n_unparsed=0 |
| IMG-4 · 성인/NSFW 이미지 판별 | opus | minimal | accuracy=0.933, f1=0.929, n_evaluated=30, n_unparsed=0 |
| IMG-5 · 사람 포함 여부 판별 | opus | minimal | accuracy=0.967, f1=0.971, n_evaluated=30, n_unparsed=0 |
| IMG-6 · 표 이미지 구조 추출 | opus | minimal | cell_f1=0.895, n_evaluated=30 |
| TXT-1 · 문서(PDF) 이해 QA | opus | minimal | token_f1=0.006, exact_match=0.0, n_evaluated=30, judge_mean=1.733 |
| TXT-2 · 표(엑셀) 이해 QA | opus | minimal | accuracy=0.1, token_f1=0.197, n_evaluated=10, judge_mean=4.2 |
| TXT-3 · 표 구조 추출 | opus | minimal | cell_f1=0.0, n_evaluated=30 |
| TXT-4 · 한국어 독해 QA | opus | minimal | token_f1=0.808, exact_match=0.567, n_evaluated=30, judge_mean=4.133 |
| TXT-5 · 텍스트 요약 | opus | minimal | rouge1=0.319, rouge2=0.163, rougeL=0.261, n_evaluated=30, judge_mean=4.8 |
| TXT-6 · 감정 분석 | opus | minimal | accuracy=0.0, macro_f1=0.0, n_evaluated=0, n_unparsed=30 |
| TXT-7 · 키워드 추출 | opus | minimal | precision=0.0, recall=0.0, f1=0.0, n_evaluated=30, macro_precision=0.0, macro_recall=0.0 |
| TXT-8 · 비속어/유해성 판별 | opus | minimal | accuracy=0.0, f1=0.0, n_evaluated=0, n_unparsed=30 |
| IMG-1 · 이미지 캡션 생성 | sol | minimal | caption_token_f1=0.427, n_evaluated=30, judge_mean=3.0 |
| IMG-2 · 이미지 태그(객체) 추출 | sol | minimal | micro_precision=0.181, micro_recall=0.388, micro_f1=0.247, macro_precision=0.203, macro_recall=0.423, macro_f1=0.268 |
| IMG-3 · 무기/위협 존재 판별 | sol | minimal | accuracy=0.633, f1=0.776, n_evaluated=30, n_unparsed=0 |
| IMG-4 · 성인/NSFW 이미지 판별 | sol | minimal | accuracy=0.933, f1=0.929, n_evaluated=30, n_unparsed=0 |
| IMG-5 · 사람 포함 여부 판별 | sol | minimal | accuracy=0.967, f1=0.971, n_evaluated=30, n_unparsed=0 |
| IMG-6 · 표 이미지 구조 추출 | sol | minimal | cell_f1=0.787, n_evaluated=30 |
| TXT-1 · 문서(PDF) 이해 QA | sol | minimal | token_f1=0.042, exact_match=0.0, n_evaluated=30, judge_mean=1.467 |
| TXT-2 · 표(엑셀) 이해 QA | sol | minimal | accuracy=0.0, token_f1=0.263, n_evaluated=10, judge_mean=3.6 |
| TXT-3 · 표 구조 추출 | sol | minimal | cell_f1=0.0, n_evaluated=30 |
| TXT-4 · 한국어 독해 QA | sol | minimal | token_f1=0.886, exact_match=0.667, n_evaluated=30, judge_mean=4.6 |
| TXT-5 · 텍스트 요약 | sol | minimal | rouge1=0.303, rouge2=0.127, rougeL=0.248, n_evaluated=30, judge_mean=4.733 |
| TXT-6 · 감정 분석 | sol | minimal | accuracy=0.0, macro_f1=0.0, n_evaluated=0, n_unparsed=30 |
| TXT-7 · 키워드 추출 | sol | minimal | precision=0.0, recall=0.0, f1=0.0, n_evaluated=30, macro_precision=0.0, macro_recall=0.0 |
| TXT-8 · 비속어/유해성 판별 | sol | minimal | accuracy=0.0, f1=0.0, n_evaluated=0, n_unparsed=30 |

## 성능: 수행시간·비용 (모델별)

| 모델 | 호출 | 오류 | latency median(ms) | p95(ms) | 입력토큰 | 출력토큰 | 비용(USD) |
|---|---|---|---|---|---|---|---|
| glm | 220 | 120 | 1291.8 | 4216.5 | 47877 | 7574 | 0.086734 |
| opus | 400 | 120 | 2757.3 | 9391.4 | 133907 | 40215 | 1.674914 |
| sol | 400 | 123 | 2150.5 | 8293.5 | 100683 | 23349 | 1.203887 |

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
| opus | ❌ | `A warmly lit bathroom with golden faux-finish walls features a black and white clawfoot slipper tub with brass fixtures, a white pedestal sink beneath a framed …` |
| sol | ❌ | `An elegant bathroom features a pedestal sink, framed mirror, and vintage clawfoot bathtub against textured beige walls.` |

### IMG-2 · 이미지 태그(객체) 추출

**샘플 #0** · _모델 간 판정이 갈린 케이스_ · 정답: `['boat', 'chair', 'handbag', 'person', 'umbrella']`

**질문/지시:** List the main objects visible in this image as a comma-separated list of simple nouns.

**입력 이미지:**

![IMG-2 sample 0](gallery_IMG-2_s0.png)

| 모델 | 정답? | 출력/판정 |
|---|:--:|---|
| opus | ✅ | `Cruise ships, ocean, sky, pier, umbrellas, palm tree, pool, lounge chairs, people, sand, rocks, waves, planter` |
| sol | ✅ | `ships, pier, ocean, pool, umbrellas, chairs, people, palm tree` |

### IMG-3 · 무기/위협 존재 판별

**샘플 #22** · _모델 간 판정이 갈린 케이스_ · 정답: `0`

**질문/지시:** Does this image contain a weapon or threat (gun, knife, pistol, rifle, blood, etc.)? Answer exactly yes or no.

**입력 이미지:**

![IMG-3 sample 22](gallery_IMG-3_s22.png)

| 모델 | 정답? | 출력/판정 |
|---|:--:|---|
| opus | ✅ | `no` |
| sol | ❌ | `yes` |

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
| opus | ❌ | `The question asks whether there is a person visible in the image. Looking at the image, I can see framed photographs and artwork on the wall that depict people …` |
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

**샘플 #0** · _모델 간 판정이 갈린 케이스_ · 정답: `['-0.124']`

**질문/지시:** What is Thorax size?

**입력:**
> Answer the following question:
> 
> Question: What is Thorax size?
> 
> Answer:

| 모델 | 정답? | 출력/판정 |
|---|:--:|---|
| glm | ❌ | `The size of the thorax (commonly known as the chest) varies significantly depending on a person's age, sex, body composition, and height. Because it is a dynami…` |
| opus | ❌ | `Assuming you mean the thorax of an insect (or other arthropod), its "size" isn't a single fixed number — it varies enormously with the species and even between …` |
| sol | ❌ | `Thorax size refers to the dimensions or circumference of the chest—the part of the body between the neck and abdomen. Its normal size varies by age, sex, body s…` |

### TXT-2 · 표(엑셀) 이해 QA

**샘플 #0** · _모델 간 판정이 갈린 케이스_ · 정답: `['Lake Palas Tuzla']`

**질문/지시:** which is deeper, lake tuz or lake palas tuzla?

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

| 모델 | 정답? | 출력/판정 |
|---|:--:|---|
| glm | ✅ | `Based on the table, Lake Palas Tuzla is deeper. Lake Palas Tuzla has a depth of 15 m, while Lake Tuz has a depth of 2 m.` |
| opus | ✅ | `Lake Palas Tuzla is deeper. It has a depth of 15 m, whereas Lake Tuz is only 2 m deep.` |
| sol | ✅ | `Lake Palas Tuzla is deeper at **15 m**, compared with Lake Tuz at **2 m**.` |

### TXT-3 · 표 구조 추출

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

**질문/지시:** Given the following table cell contents in row-major order, generate a valid HTML table with proper <table>, <tr>, and <td> tags.

**입력:**

```
Given the following table cell contents in row-major order, generate a valid HTML table with proper <table>, <tr>, and <td> tags.

Table cells:
[Genotype*] [Perimeter (mm)**] [%N2] [n]
[kin-29(wk61)1, 2)] [1.90 ± 0.15] [73%] [42]
[ctIs40 [pTG96(sur-5::gfp)]dbl-1(+)2] [2.76 ± 0.12] [106%] [42]
[kin-29(wk61); ctIs40 [pTG96(sur-5::gfp)]1] [1.84 ± 0.20] [71%] [51]
[sma-6(e1482) unc-4(e120)3] [1.47 ± 0.13] [57%] [37]
[sma-6(e1482) unc-4(e120);kin-29(wk61)3] [1.48 ± 0.17] [57%] [38]

Generate the HTML table structure. Respond with ONLY the HTML code, no explanation:
```

| 모델 | 정답? | 출력/판정 |
|---|:--:|---|
| glm | ❌ | `__ERROR__: FMAPIError: databricks-glm-5-2 HTTP 403: {"error_code":403,"message":"Source IP address: 49.170.169.150 is blocked by Databricks IP ACL for workspace…` |
| opus | ❌ | `__ERROR__: FMAPIError: databricks-claude-opus-5 HTTP 403: {"error_code":403,"message":"Source IP address: 49.170.169.150 is blocked by Databricks IP ACL for wor…` |
| sol | ❌ | `__ERROR__: FMAPIError: databricks-gpt-5-6-sol HTTP 403: {"error_code":403,"message":"Source IP address: 49.170.169.150 is blocked by Databricks IP ACL for works…` |

### TXT-4 · 한국어 독해 QA

**샘플 #2** · _모델 간 판정이 갈린 케이스_ · 정답: `['8량']`

**질문/지시:** 히네노 역 이북에 증편된 쾌속 열차는 모두 몇 량인가?

**입력:**
> 다음 지문을 읽고 질문에 답하세요. 답변은 지문에서 나타나는 정답만 간결하게 기술하세요.
> 
> 지문:
> 덴노지 역 구내에서는 야마토지 선과 평면 교차하여 단선으로 운전하는 한와 선과의 단락선을 복선화하는 공사가 이루어져 2008년 3월 15일 시간표 개정시부터 사용을 개시하였다. 대낮 시간대의 덴노지 역 ~ 와카야마 역 간의 쾌속과 JR 난바 역 발착의 간쿠 쾌속이 오사카 순환선과 직통 운전하는 간쿠·기슈지 쾌속에 통합되어 같은 시간대의 간쿠·기슈지 쾌속이 매시 3편으로 증발되었다. 이 복선화는 야마토지 선의 시간표 혼란이 큰 영향을 끼치는 것을 막는 효과를 가지고 있다. 또 아침 출근 시간대에 오사카 순환선 내 각역에 정차하는 직통 쾌속도 운전이 개시되었다. 전반적으로 히네노 역 이북에서는 8량 편성의 쾌속 열차가 대폭 증가되었기 때문에 난카이 전기 철도 본선과 수송력에서 차이가 나게 되었다.
> 
> 질문: 히네노 역 이북에 증편된 쾌속 열차는 모두 몇 량인가?
> 
> 답변:

| 모델 | 정답? | 출력/판정 |
|---|:--:|---|
| glm | ✅ | `8량` |
| opus | ✅ | `8량 편성` |
| sol | ✅ | `8량 편성입니다.` |

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
| glm | — | `A Pennsylvania community has rallied with hundreds of volunteers and advanced tracking technology to search for 13-year-old Cayman Naib, who impulsively went mi…` |
| opus | — | `Thirteen-year-old Cayman Naib, an eighth-grader at Pennsylvania's Shipley School, has been missing since Wednesday, when his parents believe he impulsively left…` |
| sol | — | `Hundreds of volunteers are searching Pennsylvania’s Radnor-Wayne area for 13-year-old Cayman Naib, who disappeared after receiving a school email about overdue …` |

### TXT-6 · 감정 분석

**샘플 #0** · _모델 간 판정이 갈린 케이스_ · 정답: `0`

**질문/지시:** Classify the sentiment of the following text as exactly one word: "positive" or "negative".

**입력:**
> Classify the sentiment of the following text as exactly one word: "positive" or "negative".
> 
> Text: it 's slow -- very , very slow . 
> 
> Respond with exactly one word: positive or negative

| 모델 | 정답? | 출력/판정 |
|---|:--:|---|
| glm | ❌ | `__ERROR__: FMAPIError: databricks-glm-5-2 HTTP 403: {"error_code":403,"message":"Source IP address: 49.170.169.150 is blocked by Databricks IP ACL for workspace…` |
| opus | ❌ | `__ERROR__: FMAPIError: databricks-claude-opus-5 HTTP 403: {"error_code":403,"message":"Source IP address: 49.170.169.150 is blocked by Databricks IP ACL for wor…` |
| sol | ❌ | `__ERROR__: FMAPIError: databricks-gpt-5-6-sol HTTP 403: {"error_code":403,"message":"Source IP address: 49.170.169.150 is blocked by Databricks IP ACL for works…` |

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
| glm | ❌ | `__ERROR__: FMAPIError: databricks-glm-5-2 HTTP 403: {"error_code":403,"message":"Source IP address: 49.170.169.150 is blocked by Databricks IP ACL for workspace…` |
| opus | ❌ | `__ERROR__: FMAPIError: databricks-claude-opus-5 HTTP 403: {"error_code":403,"message":"Source IP address: 49.170.169.150 is blocked by Databricks IP ACL for wor…` |
| sol | ❌ | `__ERROR__: FMAPIError: databricks-gpt-5-6-sol HTTP 403: {"error_code":403,"message":"Source IP address: 49.170.169.150 is blocked by Databricks IP ACL for works…` |

### TXT-8 · 비속어/유해성 판별

**샘플 #0** · _모델 간 판정이 갈린 케이스_ · 정답: `0`

**질문/지시:** You are a content moderation classifier. Analyze the following text and determine if it contains profanity, hate speech, or toxic language. Respond with exactly one word: "toxic" or "clean".

**입력:**
> You are a content moderation classifier. Analyze the following text and determine if it contains profanity, hate speech, or toxic language. Respond with exactly one word: "toxic" or "clean".
> 
> Text: You, sir, are my hero. Any chance you remember what page that's on?
> 
> Respond with exactly one word: toxic or clean

| 모델 | 정답? | 출력/판정 |
|---|:--:|---|
| glm | ❌ | `__ERROR__: FMAPIError: databricks-glm-5-2 HTTP 403: {"error_code":403,"message":"Source IP address: 49.170.169.150 is blocked by Databricks IP ACL for workspace…` |
| opus | ❌ | `__ERROR__: FMAPIError: databricks-claude-opus-5 HTTP 403: {"error_code":403,"message":"Source IP address: 49.170.169.150 is blocked by Databricks IP ACL for wor…` |
| sol | ❌ | `__ERROR__: FMAPIError: databricks-gpt-5-6-sol HTTP 403: {"error_code":403,"message":"Source IP address: 49.170.169.150 is blocked by Databricks IP ACL for works…` |


## 참고: Reasoning 정책

**Reasoning OFF(minimal) 단일 모드로 고정.** reasoning의 효과가 특정 태스크 성능 개선에만 한정되는 반면 실험 시간을 크게 늘리기 때문(특히 GLM은 full reasoning 시 타임아웃 빈발). 각 모델의 최소 reasoning으로 측정하며, `databricks-claude-opus-5`와 judge는 완전 OFF가 불가해 지원 최소값을 사용한다.

## Fact Sheet (Executive Summary 근거 — 감사용)

```json
{
  "per_task_scores": {
    "IMG-1/minimal": {
      "opus": 0.3372,
      "sol": 0.4274
    },
    "IMG-2/minimal": {
      "opus": 0.2054,
      "sol": 0.2472
    },
    "IMG-3/minimal": {
      "opus": 0.6667,
      "sol": 0.6333
    },
    "IMG-4/minimal": {
      "opus": 0.9333,
      "sol": 0.9333
    },
    "IMG-5/minimal": {
      "opus": 0.9667,
      "sol": 0.9667
    },
    "IMG-6/minimal": {
      "opus": 0.8948,
      "sol": 0.7871
    },
    "TXT-3/minimal": {
      "opus": 0.0,
      "sol": 0.0,
      "glm": 0.0
    },
    "TXT-6/minimal": {
      "opus": 0.0,
      "sol": 0.0,
      "glm": 0.0
    },
    "TXT-7/minimal": {
      "opus": 0.0,
      "sol": 0.0,
      "glm": 0.0
    },
    "TXT-8/minimal": {
      "opus": 0.0,
      "sol": 0.0,
      "glm": 0.0
    },
    "TXT-1/minimal": {
      "opus": 0.0064,
      "sol": 0.0416,
      "glm": 0.0044
    },
    "TXT-2/minimal": {
      "opus": 0.1,
      "sol": 0.0,
      "glm": 0.3
    },
    "TXT-4/minimal": {
      "opus": 0.8082,
      "sol": 0.8862,
      "glm": 0.9473
    },
    "TXT-5/minimal": {
      "opus": 0.3192,
      "sol": 0.3035,
      "glm": 0.3064
    }
  },
  "task_winners": {
    "IMG-1/minimal": "sol",
    "IMG-2/minimal": "sol",
    "IMG-3/minimal": "opus",
    "IMG-4/minimal": "opus",
    "IMG-5/minimal": "opus",
    "IMG-6/minimal": "opus",
    "TXT-3/minimal": "opus",
    "TXT-6/minimal": "opus",
    "TXT-7/minimal": "opus",
    "TXT-8/minimal": "opus",
    "TXT-1/minimal": "sol",
    "TXT-2/minimal": "glm",
    "TXT-4/minimal": "glm",
    "TXT-5/minimal": "opus"
  },
  "win_counts": {
    "sol": 3,
    "opus": 9,
    "glm": 2
  },
  "cheapest_model": "glm",
  "fastest_model": "glm",
  "perf": {
    "opus": {
      "n_calls": 400,
      "errors": 120,
      "latency_ms_median": 2757.3,
      "latency_ms_p95": 9391.4,
      "total_usd": 1.674914,
      "in_tokens": 133907,
      "out_tokens": 40215
    },
    "sol": {
      "n_calls": 400,
      "errors": 123,
      "latency_ms_median": 2150.5,
      "latency_ms_p95": 8293.5,
      "total_usd": 1.203887,
      "in_tokens": 100683,
      "out_tokens": 23349
    },
    "glm": {
      "n_calls": 220,
      "errors": 120,
      "latency_ms_median": 1291.8,
      "latency_ms_p95": 4216.5,
      "total_usd": 0.086734,
      "in_tokens": 47877,
      "out_tokens": 7574
    }
  }
}
```
