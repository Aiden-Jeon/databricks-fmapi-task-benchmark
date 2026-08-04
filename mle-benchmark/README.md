# mle-benchmark

여러 파운데이션 모델(FMAPI)을 **동일한 코딩 에이전트 하네스**에 꽂아, 한국어로 주어진
**24개 ML 엔지니어링 태스크**를 직접 코드로 풀게 하고 **숨겨진 테스트셋으로 채점**하는
벤치마크. 같은 repo의 다른 태스크와 같은 원칙이다: **모델만 교체**하고 하네스·프롬프트·예산·
채점기는 100% 동일하게 둔다.

측정 대상은 "모델이 답을 아는가"가 아니라 **"모델이 ML 파이프라인을 짜서 돌릴 수 있는가"**다.
방법론은 OpenAI [MLE-bench](https://github.com/openai/mle-bench)를 한국어·Databricks로 이식했다.

> 상세 결과·해석은 [FINDINGS.md](./FINDINGS.md), 실험 설계는 [METHODOLOGY.md](./METHODOLOGY.md),
> 비용은 [COST.md](./COST.md). 이 README는 개요·결론·재현법 요약.

---

## 핵심 결론 (n=3)

| | Opus 5 | GPT-5.6-sol | GLM 5.2 |
|---|:---:|:---:|:---:|
| **확정 승리** (of 24) | **12** | 0 | 0 |
| 평균 1위 태스크 | 19 | 5 | 0 |
| 유효 제출률 | 67/72 (93%) | **70/72 (97%)** | 57/72 (79%) |
| 재현성 (중위 상대 std) | 2.90% | **1.44%** | 3.51% |
| 청구 LLM 비용 | $122.83 | $43.84 | **$16.11** |

- **Opus 5** — 유일하게 태스크를 확정 승리하는 모델(12/24), 평균으로는 19/24에서 1위.
  대가는 GPT의 ~2.8배 비용, 가장 긴 실행시간, 그리고 **셋 중 가장 낮은 재현성**.
- **GPT-5.6-sol** — 확정 승리 0개지만 **가장 빠르고(중위 9분) 가장 안정적**. 21개 공통 태스크 중
  13개에서 가장 일관되고, 최악 셀의 편차가 9.7%(나머지 둘은 100% 초과). 벤치마크든 프로덕션이든
  **재현성 자체가 1급 속성**이라는 점에서 이쪽이 실무 기본값.
- **GLM 5.2** — 압도적으로 저렴하지만 유효율 79%, 그리고 **반복해서 실패하는 셀이 있는 유일한 모델**
  (KorQuAD·MRC·KMMLU 각 2회). 제출에 성공해도 독해 태스크 점수가 0에 가깝다.
- **12/24는 통계적 무승부** — 선두의 차이가 실행간 노이즈 안에 있어 승자를 주장하지 않는다.
  그래서 승수 합이 24가 되지 않는 것이 **정상**이다.

### 가장 중요한 방법론적 발견 — n=1은 승자를 조작한다

**같은 데이터**가 셀당 1회 실행에서는 `Opus 15 / GPT 6 / GLM 3`으로, n=3에서는
`Opus 12 / GPT 0 / GLM 0 / 무승부 12`로 읽혔다. 절반이 노이즈였고 **GPT·GLM의 9승은 전부 사라졌다.**
n=1에서 가장 근접했던 승부의 차이는 0.10~0.67%인데, 동일 재실행 간 중위 변동이 1.5~3.5%,
최악은 100%를 넘는다. 노이즈보다 10~100배 작은 차이로는 판정할 수 없다.
반복 비용은 144런에 **$72.70** — 탐색 단계 비용의 절반. **에이전트 벤치마크는 n≥3과 무승부 수 공개가
필수**이고, 승수가 태스크 수에 딱 맞게 떨어지는 표는 자기 노이즈를 측정하지 않은 표다.

---

## 결과 — 24개 한국어 태스크

각 셀 = **3회 반복의 평균 ± 표본표준편차**(n 표기). **굵게 = 통계적으로 확정된 승자**이며,
선두 차이가 노이즈를 넘지 못하면 무승부로 둔다. 에이전트가 본 적 없는 **숨겨진 테스트 split**으로
채점(UC ACL로 격리).

| Task | metric | opus | sol | glm |
|---|---|---|---|---|
| [pubg_placement_prediction](./pubg_placement_prediction) | MAE ↓ | 0.0572±0.059 (n=3) | 0.02448±0.0013 (n=3) | 0.1237±0.12 (n=3) |
| [spooky_author_identification](./spooky_author_identification) | multiclass log loss ↓ | 0.3319±0.077 (n=3) | 0.3632±0.013 (n=3) | 0.3843±0.037 (n=3) |
| [klue_ynat_news_topic](./klue_ynat_news_topic) | macro-F1 ↑ | **0.85±0.0012 (n=2)** | 0.8402±0.012 (n=3) | 0.8442±0.0014 (n=2) |
| [nsmc_movie_sentiment](./nsmc_movie_sentiment) | accuracy ↑ | 0.877±0.0067 (n=3) | 0.8767±0.0014 (n=3) | 0.8703±0.00042 (n=2) |
| [seoul_bike_demand](./seoul_bike_demand) | RMSE ↓ | 316.9±16 (n=3) | 223.2±22 (n=3) | 249.1±49 (n=3) |
| [klue_nli_inference](./klue_nli_inference) | accuracy ↑ | 0.8758±0.0034 (n=2) | 0.8745±0.034 (n=2) | 0.517±0.071 (n=3) |
| [klue_sts_similarity](./klue_sts_similarity) | Pearson ↑ | **0.9586±0.0047 (n=3)** | 0.9481±0.00069 (n=3) | 0.9158±0.045 (n=3) |
| [beep_hate_speech](./beep_hate_speech) | macro-F1 ↑ | **0.5867±0.013 (n=3)** | 0.5635±0.0017 (n=3) | 0.5511±0.012 (n=2) |
| [korquad_reading_comprehension](./korquad_reading_comprehension) | char-F1 ↑ | **0.5159±0.047 (n=3)** | 0.4227±0.015 (n=2) | 0.0484 (n=1) |
| [kornli_inference](./kornli_inference) | accuracy ↑ | 0.6235±0.021 (n=3) | 0.6152±0.019 (n=3) | 0.5278±0.033 (n=3) |
| [korsts_similarity](./korsts_similarity) | Pearson ↑ | **0.7914±0.01 (n=3)** | 0.7455±0.0083 (n=3) | 0.7534±0.0055 (n=2) |
| [kobest_boolq](./kobest_boolq) | accuracy ↑ | 0.6048±0.0099 (n=3) | 0.6025±0.0044 (n=3) | 0.5748±0.042 (n=3) |
| [kobest_copa](./kobest_copa) | accuracy ↑ | **0.6391±0.019 (n=3)** | 0.5904±0.0037 (n=3) | 0.6039±0.018 (n=2) |
| [kobest_wic](./kobest_wic) | accuracy ↑ | 0.6285±0.011 (n=3) | 0.5959±0.0087 (n=3) | 0.6182±0.027 (n=2) |
| [kobest_hellaswag](./kobest_hellaswag) | accuracy ↑ | **0.6786±0.047 (n=2)** | 0.6273±0.01 (n=3) | 0.5788±0.031 (n=3) |
| [kobest_sentineg](./kobest_sentineg) | accuracy ↑ | **0.9575±0.0024 (n=3)** | 0.9539±0.0042 (n=3) | 0.9527±0.0029 (n=2) |
| [pawsx_paraphrase](./pawsx_paraphrase) | accuracy ↑ | **0.7974±0.044 (n=3)** | 0.7602±0.011 (n=3) | 0.7013±0.025 (n=3) |
| [klue_relation_extraction](./klue_relation_extraction) | accuracy ↑ | **0.7713±0.0042 (n=2)** | 0.7292±0.012 (n=3) | 0.7151±0.0082 (n=3) |
| [klue_mrc_reading](./klue_mrc_reading) | char-F1 ↑ | 0.3364±0.014 (n=3) | 0.3375±0.0074 (n=3) | 0.02729 (n=1) |
| [klue_ner_entities](./klue_ner_entities) | entity-F1 ↑ | 0.786±0.11 (n=3) | 0.7692±0.027 (n=3) | 0.7562±0.019 (n=3) |
| [kmmlu_expert_knowledge](./kmmlu_expert_knowledge) | accuracy ↑ | **0.3394±0.014 (n=3)** | 0.3239±0.024 (n=3) | 0.3075 (n=1) |
| [korfin_aspect_sentiment](./korfin_aspect_sentiment) | macro-F1 ↑ | **0.7189±0.021 (n=3)** | 0.6757±0.0039 (n=3) | 0.6831±0.014 (n=2) |
| [kor_unsmile_multilabel_hate](./kor_unsmile_multilabel_hate) | macro-F1 (multi-label) ↑ | 0.706±0.038 (n=2) | 0.714±0.0066 (n=3) | 0.7026±0.0096 (n=2) |
| [klue_dependency_parsing](./klue_dependency_parsing) | LAS ↑ | 0.7195±0.13 (n=3) | 0.7674±0.03 (n=3) | 0.6982±0.066 (n=3) |
| **decided wins** | | **12** | **0** | **0** |

---

## 무엇을 재나 — 3개 지표

| | 지표 | 출처 | 정밀도 |
|---|---|---|---|
| 1 | **결과물 퀄리티** | grader vs 숨겨진 split | 정확, 런 단위 |
| 2 | **소요시간** | job wall-clock | 정확, 런 단위 |
| 3 | **전체 비용** | `system.billing.usage` (청구 실적) | **모델 단위** ([COST.md](./COST.md)) |

비용은 추정이 아니라 **청구된 행**만 쓴다. 전체 캠페인 **$207.12** = LLM $182.79 + 서버리스 컴퓨트 $24.34.
모델별 분해는 provider SKU 단위까지만 가능하다(각 SKU에 M-track 모델이 정확히 하나씩 있어서 성립 —
이 모델 조합의 성질이며 범용 미터가 아니다). 런 단위 귀속은 request tag가 필요(v2).

---

## 구조

```
<task>/
├── README.md            # 태스크별 결과표 (퀄리티 · 시간 · 비용)
├── TASK_DESCRIPTION.md  # 표준 태스크 브리프 (한국어)
├── PROMPT.md            # 표준 킥오프 프롬프트 (모델 간 글자 그대로 동일)
├── opus/                # Opus 5:       submission.csv · solution/ · metrics.json
├── sol/                 # GPT-5.6-sol:  submission.csv · solution/ · metrics.json
└── glm/                 # GLM 5.2:      submission.csv · solution/ · metrics.json
```

`opus`/`sol`/`glm`은 **비교 대상 모델**이다. 하네스는 pinned opencode 하나이고,
각 모델 디렉토리에 들어가는 `PROMPT.md`·`TASK_DESCRIPTION.md`는 **byte-identical**이다.

---

## 실전 유의점

- **DNF는 대부분 능력이 아니라 툴 사용 실패다.** 216런 중 22 DNF인데, 2회 이상 실패한 셀은 3개뿐
  (전부 GLM)이고 나머지 16개는 같은 태스크 재실행에서 성공했다. 즉 DNF는 "그 태스크를 못 한다"가
  아니라 "일정 확률로 제출에 실패한다"로 읽어야 한다.
- **분산은 모델의 속성이고, 퀄리티와 반비례한다.** 최고 점수 모델이 가장 덜 재현적이다.
  출력 형식은 2차 예측변수일 뿐(구조적 파싱 6.6% vs 닫힌 라벨 1.8%) — 벤치마크 전체에서 가장
  변동이 큰 두 셀은 오히려 평범한 tabular 회귀(PUBG MAE)다.
- **암기 우회를 막아야 한다.** 지식형 4지선다에서 에이전트가 모델을 만드는 대신 **테스트 정답을
  직접 손으로 적어** 0.945를 받은 사례가 있다(HAE-RAE, 그래서 태스크 목록에서 제외 — id `t22`는
  의도적 공백). 킥오프에 anti-recall 규칙이 들어 있다: 제출은 `train.csv`로 학습한 일반화 모델에서
  나와야 하며 test 행의 정답 하드코딩·수작업 금지.
- **데이터는 재배포하지 않는다.** 이 repo에는 제출물과 에이전트 코드만 있고 원본 데이터는 없다
  (KLUE · NSMC · UCI · Kaggle 등 공개 출처에서 스크립트로 받는다). 정답 키는 커밋 금지.
- **하네스가 점수를 바꾼다.** 같은 모델도 스캐폴드가 다르면 점수가 유의미하게 달라진다
  (GPT 뉴스분류 F1: native Codex 0.824 → fixed opencode 0.848). 그래서 모델 비교는 하네스를
  고정해야만 성립한다.

Databricks workspace `fevm-newjeans-ontos`에서 실행. 서버리스 Jobs, 전 트래픽 Unity AI Gateway 경유.
