# t24_kor_unsmile — 다중 레이블 혐오 표현 분류 (해법)

## 실행

```bash
# 작업 공간 루트(train.csv 가 있는 디렉터리)에서 실행
PYTHONPATH=solution python solution/final.py
# -> outputs/submission.csv, solution/final_report.json
```

`solution/cache/*.npz` 가 없으면 베이스 모델을 자동으로 학습합니다
(4코어 CPU에서 전체 재현 약 40~60분). 캐시가 있으면 스태킹만 수행(약 1분).

빠른 단일 모델 베이스라인: `PYTHONPATH=solution python solution/baseline.py`
(OOF macro F1 ≈ 0.701)

## 환경
`torch`/`transformers`/`lightgbm` 이 없는 환경이라 **scikit-learn 전용**
(TF-IDF + 선형 모델 앙상블)으로 구성했습니다. 외부 데이터·사전학습 가중치는
사용하지 않았고, `train.csv` 만으로 학습합니다.

## 파이프라인

### 1. 전처리 (`common.py`)
- 공백 정규화, 3회 이상 반복 문자를 2회로 축약 (`ㅋㅋㅋㅋㅋ` → `ㅋㅋ`).
- **한글 자모 분해**: `시발` → `ㅅㅣㅂㅏㄹ`. 구어체/변형 표기(`ㅅㅂ`, `시1발`,
  `씨발` 등)가 부분 문자열 특징을 공유하게 되어 은어·우회 표기에 강해집니다.

### 2. 특징 공간 (`zoo.py`)
| 이름 | 구성 |
|---|---|
| `main` | char_wb(2-5) + word(1-2) + 자모 char(2-5) |
| `jamo` | 자모 char(1-6), max 300k |
| `js` | 자모 char(2-4) |
| `cwb` | 원문 char_wb(1-5) |
| `jw` | 자모 char(1-6) + word(1-2) |
| `counts` | char_wb(1-4) 빈도 (NB용) |
| `svd` | `main` 의 TruncatedSVD(350) + L2 정규화 (MLP용) |

모든 벡터라이저는 **train 에만 fit** 합니다 (테스트 누출 없음).

### 3. 베이스 모델 (21개, 각각 5-fold OOF)
레이블별 One-vs-Rest 로지스틱 회귀(C 여러 값, `class_weight` 유/무),
LinearSVC, RidgeClassifier, SGD(modified_huber), ComplementNB,
SVD+MLP. 결정함수는 시그모이드로 [0,1] 로 매핑.

단일 모델 OOF macro F1: 0.60(NB) ~ 0.717(`lr_jamo_c16`).

### 4. 스태킹 (`blend.py`)
레이블 j 마다 21개 베이스 모델의 레이블 j 확률을 입력으로 하는 2단 로지스틱
회귀(C=200, 동일 fold 구조). 단순 확률 평균(0.705)보다 크게 좋습니다.
전체 10개 레이블 확률을 모두 입력으로 주는 변형도 시험했지만 더 나빴습니다.

### 5. 임계값 튜닝 + 제약 후처리 (`common.py`)
- macro F1 은 레이블별 F1 평균이므로 **레이블별 임계값**을 OOF 에서
  coordinate ascent 로 최적화 (희소 레이블은 임계값이 크게 낮아짐).
- 데이터 제약 반영:
  - 모든 학습 행은 레이블이 최소 1개 → 빈 예측이면 상대점수 최대 레이블 부여.
  - `clean` 은 9개 혐오 범주와 **상호배타**(학습 데이터에서 100% 성립)
    → 동시 예측 시 상대점수가 높은 쪽만 남김.

## 결과 (5-fold OOF, 7200행)

| 구성 | OOF macro F1 | 정직한 추정* |
|---|---|---|
| 단일 LogReg (`main`) | 0.700 | — |
| 최고 단일 모델 (`lr_jamo_c16`) | 0.717 | — |
| 21모델 확률 평균 | 0.705 | 0.695 |
| **21모델 스태킹 + 임계값 튜닝** | **0.740** | **0.734** |

\* 임계값을 OOF 절반에서 튜닝하고 나머지 절반에서 평가(8회 반복 평균).
전체 OOF 로 임계값을 맞출 때 생기는 낙관 편향(≈0.006)을 제거한 값이며,
테스트 점수 기대치는 이쪽에 가깝습니다.

## 파일
- `common.py` — 전처리, 자모 분해, 특징 생성, macro F1, 임계값 튜닝/후처리
- `zoo.py` — 특징 공간 + 베이스 모델 정의, OOF/test 예측 캐싱
- `blend.py` — 스태킹, 확률/랭크 평균, 정직한 홀드아웃 평가, greedy 블렌딩 실험
- `final.py` — 최종 제출 생성
- `baseline.py` — 빠른 단일 모델 베이스라인
- `final_report.json` — 최종 사용 모델 목록/임계값/점수
