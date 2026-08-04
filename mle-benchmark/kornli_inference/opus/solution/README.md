# t10_kornli — 자연어 추론 (KorNLI / MultiNLI-ko)

`train.csv`(48,000행)만 사용해 학습하고 `test.csv`(12,000행)의 레이블을
예측합니다. 외부 데이터·사전학습 가중치·인터넷 리소스는 사용하지 않았습니다
(모든 임베딩은 랜덤 초기화 후 `train.csv`로만 학습).

## 실행

```bash
bash solution/run_all.sh      # 작업 루트에서 실행, CPU 4코어 기준 약 35분
```

산출물: `outputs/submission.csv` (`id,label`)

## 구성

### 1. `data.py` — 전처리 / 토큰화
한국어는 교착어라서 어절(공백) 단위 어휘는 매우 희소합니다. 따라서 각 어절을
**하위 단어 집합**(어절 자체 + 문자 3/4-gram, fastText 방식)으로 표현하고
단어 임베딩을 그 평균으로 계산합니다. 굴절형 사이에 통계적 강도를 공유하므로
어휘 희소성 문제가 크게 완화됩니다.
어휘/해시 버킷은 `train.csv`에서만 만듭니다.

또한 NLI에 유용한 명시적 dense 피처 19개(어휘 중첩률, 어간 중첩률, 문자 중첩률,
길이 비/차, 부정어 개수, 가설 전용 단어 비율 등)를 계산합니다.

### 2. `model.py` — 신경망 (from scratch)
Decomposable Attention (Parikh et al. 2016) 구조:

- 하위 단어 합성 임베딩 (dim 160) → 선형 투영 (208)
- `F` MLP 로 전제·가설 토큰을 attend, soft-alignment 계산
- `G` MLP 로 (토큰, 정렬된 상대 토큰) 쌍을 비교
- sum/max pooling 후 dense 피처를 concat → `H` MLP → 3-way softmax

RNN이 없어 CPU에서도 학습이 가능합니다. 임베딩은 sparse gradient + `SparseAdam`,
나머지는 `AdamW` + OneCycle 스케줄로 학습하고 마지막 4 epoch 예측을 평균합니다
(snapshot ensemble).

### 3. `train_linear.py` — 희소 피처 선형 모델
- 가설 1/2-gram tf-idf (hypothesis-only bias)
- 전제 unigram tf-idf
- **unmatched 가설 단어** tf-idf (전제에 없는 가설 단어 → neutral/contradiction 단서)
- 해시된 **cross word-pair** 피처 (전제 단어 × 가설 단어, 2^20 버킷)
- 문자 n-gram tf-idf의 element-wise 곱 / 절대차
- dense 피처와 그 제곱
→ 다항 로지스틱 회귀

### 4. `sibling.py` — 전제 그룹 구조 사전확률
(Multi)NLI는 하나의 전제에 대해 entailment / neutral / contradiction 가설을
각각 작성하게 만든 데이터입니다. `train.csv`에서 측정하면 전제를 공유하는 행들의
레이블이 **97.1%**의 경우 서로 다릅니다. test의 24%(2,891행)는 전제를 train과
공유하므로, 관측된 형제 행의 레이블로부터 추정한 사전확률
`P(y | 형제 레이블들)`(leave-one-out으로 train에서만 추정)을 모델 확률에
로그 공간에서 더해 줍니다. 홀드아웃에서 검증된 순수 학습 기반 신호입니다.

### 5. `blend.py` — 앙상블 + 제출
홀드아웃(StratifiedKFold(10, seed 7)의 fold 0) 정확도를 기준으로 로그 확률
가중치를 coordinate ascent로 탐색하고, 사전확률 가중치도 같은 홀드아웃에서
선택한 뒤 `outputs/submission.csv`를 씁니다.

## 결과 (홀드아웃 4,800행 = StratifiedKFold(10, seed 7) fold 0)

| 모델 | acc |
|---|---|
| 최빈 클래스 | 0.334 |
| 단순 TF-IDF(문장별) + 로지스틱 회귀 | 0.462 |
| 희소 피처 선형 모델 (`train_linear.py`) | 0.558 |
| Decomposable Attention `h1` (dim144) | 0.577 |
| Decomposable Attention `b` (dim128, wdrop .25) | 0.579 |
| Decomposable Attention `c` (dim176/hid240) | 0.571 |
| 4-모델 로그확률 블렌드 | 0.597 |
| **블렌드 + 전제 그룹 사전확률** | **0.631** |

`train_linear.py`의 `--tag a` 결과는 `work/lin_a.log`, 신경망 로그는
`work/nn_*.log`, 블렌드 로그는 `work/blend.log`에 있습니다.

## 관찰

- 신경망은 2 epoch 부근에서 최고 성능에 도달한 뒤 빠르게 과적합합니다
  (48k쌍 대비 파라미터가 많고 사전학습 임베딩이 없기 때문). 그래서
  epoch별 스냅샷을 홀드아웃으로 평가해 greedy 선택합니다.
- CPU 4코어 환경이라 RNN 없는 attention 구조 + sparse gradient 임베딩
  최적화(`SparseAdam`)로 epoch당 약 110초까지 줄였습니다.
- 전제 그룹 사전확률이 단일 기법으로 가장 큰 이득(+3.4%p)을 줍니다.
