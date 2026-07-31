# 벤치마크 리포트 — 2026-07-31T10-01

> 📊 고객 설명용 프레젠테이션: **[▶ 브라우저로 바로 보기](https://htmlpreview.github.io/?https://github.com/Aiden-Jeon/databricks-fmapi-task-benchmark/blob/main/image-text-performance/reports/2026-07-31T10-01/presentation.html)** · [HTML 소스](./presentation.html) — 이 리포트 결과를 슬라이드로 정리

## 평가 대상 모델 (Databricks hosted)

| 별칭 | Databricks model name | vision |
|---|---|---|
| opus | `databricks-claude-opus-5` | ✅ |
| sol | `databricks-gpt-5-6-sol` | ✅ |
| glm | `databricks-glm-5-2` | ❌ |

> Judge: `databricks-gemini-3-1-pro`

## Reasoning 정책

**Reasoning OFF(minimal) 단일 모드로 고정.** reasoning의 효과가 특정 태스크 성능 개선에만 한정되는 반면 실험 시간을 크게 늘리기 때문(특히 GLM은 full reasoning 시 타임아웃 빈발). 각 모델의 최소 reasoning으로 측정하며, `databricks-claude-opus-5`와 judge는 완전 OFF가 불가해 지원 최소값을 사용한다.

## Executive Summary

제공된 데이터에 따르면 Opus는 총 13개 태스크 중 7개에서 1위를 차지해 전반적인 성능이 가장 우수하지만, 특정 텍스트 작업(TXT-2)에서는 0점을 기록하는 약점도 보였습니다. 반면 GLM은 일부 텍스트 작업(TXT-2, TXT-4, TXT-7)에서 가장 높은 점수를 기록하며 강점을 보였고, Sol은 특정 이미지 작업(IMG-1)과 텍스트 작업(TXT-1, TXT-5)에서 우위를 점했습니다. 비용과 속도 측면에서 GLM은 중간 지연 시간 1025.3ms, 총비용 약 0.083달러로 가장 빠르고 저렴해 효율성이 뛰어납니다. 반면 Opus는 성능이 가장 높은 대신 중간 지연 시간 2750.3ms, 총비용 약 0.697달러로 가장 느리고 비싸, 성능과 비용·속도 간의 명확한 트레이드오프를 보여줍니다.

<sub>규칙 기반 요약(대조용): 태스크별 1위 횟수는 opus 7회, sol 3회, glm 3회로 **opus**가 가장 많다. 응답 속도는 **glm**가 가장 빠르다(median 1025.3ms). 비용은 **glm**가 가장 낮다($0.083319).</sub>

## 비교 그래프

### 태스크별 모델 성능 비교

![태스크별 모델 성능 비교](chart_scores.png)

### 모델별 속도 vs 비용

![모델별 속도 vs 비용](chart_perf.png)

**태스크 ID 범례:** IMG-1=이미지 캡션 생성 · IMG-2=이미지 태그(객체) 추출 · IMG-3=무기/위협 존재 판별 · IMG-4=성인/NSFW 이미지 판별 · IMG-5=사람 포함 여부 판별 · TXT-1=문서(PDF) 이해 QA · TXT-2=표(엑셀) 이해 QA · TXT-3=표 구조 추출 · TXT-4=한국어 독해 QA · TXT-5=텍스트 요약 · TXT-6=감정 분석 · TXT-7=키워드 추출 · TXT-8=비속어/유해성 판별

## 정량 결과 (태스크 × 모델)

| 태스크 | 모델 | reasoning | 대표 메트릭 |
|---|---|---|---|
| TXT-1 · 문서(PDF) 이해 QA | glm | minimal | token_f1=0.001, exact_match=0.0, n_evaluated=10 |
| TXT-2 · 표(엑셀) 이해 QA | glm | minimal | accuracy=0.2, token_f1=0.328, n_evaluated=10 |
| TXT-3 · 표 구조 추출 | glm | minimal | cell_f1=0.964, n_evaluated=10 |
| TXT-4 · 한국어 독해 QA | glm | minimal | token_f1=0.919, exact_match=0.8, n_evaluated=10 |
| TXT-5 · 텍스트 요약 | glm | minimal | rouge1=0.243, rouge2=0.112, rougeL=0.22, n_evaluated=10 |
| TXT-6 · 감정 분석 | glm | minimal | accuracy=1.0, macro_f1=1.0, n_evaluated=10, n_unparsed=0 |
| TXT-7 · 키워드 추출 | glm | minimal | precision=0.536, recall=0.461, f1=0.496, n_evaluated=10 |
| TXT-8 · 비속어/유해성 판별 | glm | minimal | accuracy=0.8, f1=0.5, n_evaluated=10, n_unparsed=0 |
| IMG-1 · 이미지 캡션 생성 | opus | minimal | caption_token_f1=0.294, n_evaluated=10 |
| IMG-2 · 이미지 태그(객체) 추출 | opus | minimal | micro_precision=0.155, micro_recall=0.529, micro_f1=0.24, macro_precision=0.162 |
| IMG-3 · 무기/위협 존재 판별 | opus | minimal | accuracy=0.5, f1=0.667, n_evaluated=10, n_unparsed=0 |
| IMG-4 · 성인/NSFW 이미지 판별 | opus | minimal | accuracy=0.8, f1=0.75, n_evaluated=10, n_unparsed=0 |
| IMG-5 · 사람 포함 여부 판별 | opus | minimal | accuracy=0.9, f1=0.909, n_evaluated=10, n_unparsed=0 |
| TXT-1 · 문서(PDF) 이해 QA | opus | minimal | token_f1=0.002, exact_match=0.0, n_evaluated=10 |
| TXT-2 · 표(엑셀) 이해 QA | opus | minimal | accuracy=0.0, token_f1=0.169, n_evaluated=10 |
| TXT-3 · 표 구조 추출 | opus | minimal | cell_f1=0.966, n_evaluated=10 |
| TXT-4 · 한국어 독해 QA | opus | minimal | token_f1=0.846, exact_match=0.6, n_evaluated=10 |
| TXT-5 · 텍스트 요약 | opus | minimal | rouge1=0.283, rouge2=0.145, rougeL=0.243, n_evaluated=10 |
| TXT-6 · 감정 분석 | opus | minimal | accuracy=1.0, macro_f1=1.0, n_evaluated=10, n_unparsed=0 |
| TXT-7 · 키워드 추출 | opus | minimal | precision=0.374, recall=0.477, f1=0.419, n_evaluated=10 |
| TXT-8 · 비속어/유해성 판별 | opus | minimal | accuracy=0.8, f1=0.5, n_evaluated=10, n_unparsed=0 |
| IMG-1 · 이미지 캡션 생성 | sol | minimal | caption_token_f1=0.379, n_evaluated=10 |
| IMG-2 · 이미지 태그(객체) 추출 | sol | minimal | micro_precision=0.138, micro_recall=0.235, micro_f1=0.174, macro_precision=0.192 |
| IMG-3 · 무기/위협 존재 판별 | sol | minimal | accuracy=0.5, f1=0.667, n_evaluated=10, n_unparsed=0 |
| IMG-4 · 성인/NSFW 이미지 판별 | sol | minimal | accuracy=0.8, f1=0.75, n_evaluated=10, n_unparsed=0 |
| IMG-5 · 사람 포함 여부 판별 | sol | minimal | accuracy=0.9, f1=0.909, n_evaluated=10, n_unparsed=0 |
| TXT-1 · 문서(PDF) 이해 QA | sol | minimal | token_f1=0.013, exact_match=0.0, n_evaluated=10 |
| TXT-2 · 표(엑셀) 이해 QA | sol | minimal | accuracy=0.0, token_f1=0.263, n_evaluated=10 |
| TXT-3 · 표 구조 추출 | sol | minimal | cell_f1=0.957, n_evaluated=10 |
| TXT-4 · 한국어 독해 QA | sol | minimal | token_f1=0.883, exact_match=0.7, n_evaluated=10 |
| TXT-5 · 텍스트 요약 | sol | minimal | rouge1=0.309, rouge2=0.142, rougeL=0.27, n_evaluated=10 |
| TXT-6 · 감정 분석 | sol | minimal | accuracy=1.0, macro_f1=1.0, n_evaluated=10, n_unparsed=0 |
| TXT-7 · 키워드 추출 | sol | minimal | precision=0.419, recall=0.445, f1=0.432, n_evaluated=10 |
| TXT-8 · 비속어/유해성 판별 | sol | minimal | accuracy=0.8, f1=0.5, n_evaluated=10, n_unparsed=0 |

## 성능: 수행시간·비용 (모델별)

| 모델 | 호출 | 오류 | latency median(ms) | p95(ms) | 입력토큰 | 출력토큰 | 비용(USD) |
|---|---|---|---|---|---|---|---|
| glm | 80 | 0 | 1025.3 | 6224.2 | 30528 | 9854 | 0.083319 |
| opus | 130 | 0 | 2750.3 | 8000.2 | 62968 | 15295 | 0.697217 |
| sol | 130 | 0 | 1819.4 | 4469.9 | 45427 | 8781 | 0.490566 |

> 비용은 `config/pricing.yaml` DBU 단가 기반 추정(usd_per_dbu 가정값). 정밀 비용은 `system.ai_gateway.usage` 조인 필요(§10).

## 정성 비교: 모델 간 결과가 갈린 샘플

각 태스크에서 모델들의 출력·판정이 **가장 크게 갈린 샘플**을 골라 나란히 비교한다. 평균 점수로는 안 보이는 모델별 차이를 드러낸다.

### IMG-1 · 이미지 캡션 생성

**샘플 #0** · 정답: `['There are two sinks next to two mirrors.', 'Two very metallic sinks are shown as well as the mirrors above them.', 'A room with stainless steel equipment incl…`

**입력 이미지:**

![IMG-1 sample 0](gallery_IMG-1_s0.png)

| 모델 | 출력/판정 |
|---|---|
| opus | `A cramped ship's washroom features stainless steel sinks and mirrored cabinets along one wall, with exposed pipes and conduits running across the ceiling, a red…` |
| sol | `A narrow industrial washroom features stainless-steel sinks, cabinets, exposed pipes, bright lights, mirrors, and a fire extinguisher.` |

### IMG-2 · 이미지 태그(객체) 추출

**샘플 #0** · 정답: `['giraffe']`

**입력 이미지:**

![IMG-2 sample 0](gallery_IMG-2_s0.png)

| 모델 | 출력/판정 |
|---|---|
| opus | `giraffe, trees, building, roof, fence, rocks, grass, dirt, sky, shelter, logs` |
| sol | `giraffe, trees, grass, rocks, fence, building` |

### IMG-3 · 무기/위협 존재 판별

**샘플 #2** · 정답: `1`

**입력 이미지:**

![IMG-3 sample 2](gallery_IMG-3_s2.png)

| 모델 | 출력/판정 |
|---|---|
| opus | `**Yes**` |
| sol | `yes` |

### TXT-1 · 문서(PDF) 이해 QA

**샘플 #0** · 정답: `['0.28']`

**입력:**
> Answer the following question:
> 
> Question: What is the ‘actual’ value per 1000, during the year 1975?
> 
> Answer:

| 모델 | 출력/판정 |
|---|---|
| glm | `It appears that the data or chart needed to answer this question is missing from your prompt. Please provide the relevant table, graph, or text, and I will glad…` |
| opus | `Based on the reference to a 1975 "actual" value per 1000, the figure is **14.8 per 1,000**.  However, I should note that I don't have the specific document or t…` |
| sol | `Please provide the chart or table containing the “actual” values so I can identify the value for 1975.` |

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
| opus | `Lake Palas Tuzla is deeper. It has a depth of 15 m, compared with Lake Tuz's depth of just 2 m.` |
| sol | `Lake Palas Tuzla is deeper at **15 m**, compared with Lake Tuz at **2 m**.` |

### TXT-3 · 표 구조 추출

**샘플 #4** · 정답:

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
[Study/daya] [Exposure year] [OC] [EC] [Sulfate] [Al] [Cu] [Pb] [Zn]
[A] [2000] [203] [12.8] [330] [2.7] [0.5] [0.17] [0.8]
[B] [2000] [175] [9.7] [505] [0.8] [0.1] [0.1] [0.4]
[C] [2000] [191] [12.6] [308] [3.0] [NA] [0.16] [0.9]
[D] [2000] [244] [15.4] [207] [11.6] [0.7] [0.17] [0.8]
[E] [2000] [205] [10.8] [262] [7.7] [0.2] [0.21] [0.6]
[F] [2000] [356] [27.9] [112] [21.0] [0.3] [0.19] [1.6]
[A1] [2001] [199] [5.5] [256] [1.3] [0.1] [0.09] [0.6]
[A2] [2001] [244] [5.8] [345] [1.4] [0.2] [0.06] [0.3]
[B2] [2001] [279] [17.0] [311] [3.7] [0.9] [0.22] [1.2]
[C1] [2001] [334] [33.6] [232] [8.4] [0.7] [0.39] [1.4]
[C2] [2001] [315] [15.2] [287] [4.1] [0.5] [0.28] [1.2]
[D1] [2001] [308] [40.0] [255] [10.1] [0.9] [0.38] [2.1]
[D2] [2001] [273] [28.5] [273] [4.7] [1.5] [0.28] [1.7]
[E1] [2001] [NA] [NA] [57] [111.5] [4.8] [0.17] [2.4]
[E2] [2001] [NA] [58.0] [89] [68.5] [3.5] [0.17] [2.5]
[F1] [2001] [NA] [56.1] [373] [24.8] [3.8] [0.41] [3.6]
[F2] [2001] [434] [24.3] [247] [22.8] [2.5] [0.24] [2.3]
[G1] [2001] [351] [14.6] [179] [10.0] [0.7] […
```

| 모델 | 출력/판정 |
|---|---|
| glm | `<table>   <tr>     <td>Study/daya</td><td>Exposure year</td><td>OC</td><td>EC</td><td>Sulfate</td><td>Al</td><td>Cu</td><td>Pb</td><td>Zn</td>   </tr>   <tr>   …` |
| opus | `<table> <tr><td>Study/daya</td><td>Exposure year</td><td>OC</td><td>EC</td><td>Sulfate</td><td>Al</td><td>Cu</td><td>Pb</td><td>Zn</td></tr> <tr><td>A</td><td>2…` |
| sol | `<table>   <tr>     <td>Study/daya</td>     <td>Exposure year</td>     <td>OC</td>     <td>EC</td>     <td>Sulfate</td>     <td>Al</td>     <td>Cu</td>     <td>P…` |

### TXT-4 · 한국어 독해 QA

**샘플 #0** · 정답: `['합창교향곡']`

**입력:**
> 다음 지문을 읽고 질문에 답하세요. 답변은 지문에서 나타나는 정답만 간결하게 기술하세요.
> 
> 지문:
> 1839년 바그너는 괴테의 파우스트을 처음 읽고 그 내용에 마음이 끌려 이를 소재로 해서 하나의 교향곡을 쓰려는 뜻을 갖는다. 이 시기 바그너는 1838년에 빛 독촉으로 산전수전을 다 걲은 상황이라 좌절과 실망에 가득했으며 메피스토펠레스를 만나는 파우스트의 심경에 공감했다고 한다. 또한 파리에서 아브네크의 지휘로 파리 음악원 관현악단이 연주하는 베토벤의 교향곡 9번을 듣고 깊은 감명을 받았는데, 이것이 이듬해 1월에 파우스트의 서곡으로 쓰여진 이 작품에 조금이라도 영향을 끼쳤으리라는 것은 의심할 여지가 없다. 여기의 라단조 조성의 경우에도 그의 전기에 적혀 있는 것처럼 단순한 정신적 피로나 실의가 반영된 것이 아니라 베토벤의 합창교향곡 조성의 영향을 받은 것을 볼 수 있다. 그렇게 교향곡 작곡을 1839년부터 40년에 걸쳐 파리에서 착수했으나 1악장을 쓴 뒤에 중단했다. 또한 작품의 완성과 동시에 그는 이 서곡(1악장)을 파리 음악원의 연주회에서 연주할 파트보까지 준비하였으나, 실제로는 이루어지지는 않았다. 결국 초연은 4년 반이 지난 후에 드레스덴에서 연주되었고 재연도 이루어졌지만, 이후에 그대로 방치되고 말았다. 그 사이에 그는 리엔치와 방황하는 네덜란드인을 완성하고 탄호이저에도 착수하는 등 분주한 시간을 보냈는데, 그런 바쁜 생활이 이 곡을 잊게 한 것이 아닌가 하는 의견도 있다.
> 
> 질문: 파우스트 서곡의 라단조 조성이 영향을 받은 베토벤의 곡은?
> 
> 답변:

| 모델 | 출력/판정 |
|---|---|
| glm | `베토벤의 교향곡 9번(합창교향곡)` |
| opus | `베토벤의 교향곡 9번(합창 교향곡)` |
| sol | `교향곡 9번(합창교향곡)` |

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
| glm | `A Pennsylvania community has mobilized hundreds of volunteers, both on foot and online, to search for 13-year-old Cayman Naib, who impulsively went missing afte…` |
| opus | `Thirteen-year-old Cayman Naib, an eighth-grader at Pennsylvania's Shipley School, has been missing since Wednesday, when his parents believe he impulsively left…` |
| sol | `Hundreds of volunteers are searching for 13-year-old Cayman Naib, who disappeared from the Radnor-Wayne area after receiving a school email about overdue homewo…` |

### TXT-7 · 키워드 추출

**샘플 #0** · 정답: `['accelerated simulation', 'general accelerated simulation method', 'non-markovian systems', 'numerical examples', 'steady-state availability']`

**입력:**
> Extract the key phrases or keywords from the following academic document. List them as a comma-separated list with no additional text.
> 
> Title and Abstract:
> Accelerated simulation of the steady-state availability of non-Markovian
> 
> systems A general accelerated simulation method for evaluation of the steady-state availability of non-Markovian systems is proposed. It is applied to the investigation of a class of systems with repair. Numerical examples are given
> 
> Key phrases (comma-separated):

| 모델 | 출력/판정 |
|---|---|
| glm | `Accelerated simulation, steady-state availability, non-Markovian systems, systems with repair, numerical examples` |
| opus | `accelerated simulation, steady-state availability, non-Markovian systems, systems with repair, rare event simulation, variance reduction, importance sampling, r…` |
| sol | `accelerated simulation, steady-state availability, non-Markovian systems, systems with repair, availability evaluation, numerical examples` |

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
| opus | `clean` |
| sol | `toxic` |


## Fact Sheet (Executive Summary 근거 — 감사용)

```json
{
  "per_task_scores": {
    "IMG-1/minimal": {
      "opus": 0.294,
      "sol": 0.3794
    },
    "IMG-2/minimal": {
      "opus": 0.24,
      "sol": 0.1739
    },
    "IMG-3/minimal": {
      "opus": 0.5,
      "sol": 0.5
    },
    "IMG-4/minimal": {
      "opus": 0.8,
      "sol": 0.8
    },
    "IMG-5/minimal": {
      "opus": 0.9,
      "sol": 0.9
    },
    "TXT-1/minimal": {
      "opus": 0.002,
      "sol": 0.0133,
      "glm": 0.001
    },
    "TXT-2/minimal": {
      "opus": 0.0,
      "sol": 0.0,
      "glm": 0.2
    },
    "TXT-3/minimal": {
      "opus": 0.9656,
      "sol": 0.9567,
      "glm": 0.9645
    },
    "TXT-4/minimal": {
      "opus": 0.8461,
      "sol": 0.8833,
      "glm": 0.9193
    },
    "TXT-5/minimal": {
      "opus": 0.2832,
      "sol": 0.3091,
      "glm": 0.2434
    },
    "TXT-6/minimal": {
      "opus": 1.0,
      "sol": 1.0,
      "glm": 1.0
    },
    "TXT-7/minimal": {
      "opus": 0.4192,
      "sol": 0.4318,
      "glm": 0.4958
    },
    "TXT-8/minimal": {
      "opus": 0.8,
      "sol": 0.8,
      "glm": 0.8
    }
  },
  "task_winners": {
    "IMG-1/minimal": "sol",
    "IMG-2/minimal": "opus",
    "IMG-3/minimal": "opus",
    "IMG-4/minimal": "opus",
    "IMG-5/minimal": "opus",
    "TXT-1/minimal": "sol",
    "TXT-2/minimal": "glm",
    "TXT-3/minimal": "opus",
    "TXT-4/minimal": "glm",
    "TXT-5/minimal": "sol",
    "TXT-6/minimal": "opus",
    "TXT-7/minimal": "glm",
    "TXT-8/minimal": "opus"
  },
  "win_counts": {
    "sol": 3,
    "opus": 7,
    "glm": 3
  },
  "cheapest_model": "glm",
  "fastest_model": "glm",
  "perf": {
    "opus": {
      "n_calls": 130,
      "errors": 0,
      "latency_ms_median": 2750.3,
      "latency_ms_p95": 8000.2,
      "total_usd": 0.697217,
      "in_tokens": 62968,
      "out_tokens": 15295
    },
    "sol": {
      "n_calls": 130,
      "errors": 0,
      "latency_ms_median": 1819.4,
      "latency_ms_p95": 4469.9,
      "total_usd": 0.490566,
      "in_tokens": 45427,
      "out_tokens": 8781
    },
    "glm": {
      "n_calls": 80,
      "errors": 0,
      "latency_ms_median": 1025.3,
      "latency_ms_p95": 6224.2,
      "total_usd": 0.083319,
      "in_tokens": 30528,
      "out_tokens": 9854
    }
  }
}
```
