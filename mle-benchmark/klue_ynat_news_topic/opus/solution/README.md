# t3_ynat — 연합뉴스 제목 토픽 분류 (Macro F1)

## 실행

```bash
bash solution/run_all.sh      # 약 8분, outputs/submission.csv 생성
```

환경 제약: 이 워크스페이스에는 `torch`/`transformers`/`lightgbm`이 없고 사전학습 가중치
다운로드도 금지되어 있으므로, **scikit-learn만으로 구성한 TF-IDF 선형 모델 스태킹 앙상블**을
사용했습니다. 외부 데이터는 일절 사용하지 않았고 `train.csv`만으로 학습합니다.

## 구조

**Level 0 — 12개의 다양한 기반 모델** (`oof.py`의 `build()`)

모두 5-fold StratifiedKFold(seed=42) OOF `decision_function` 행렬 (n×7) 을 만들고,
test 예측은 train 전체로 재학습한 모델에서 얻습니다.

| key | 특징 공간 (TF-IDF, sublinear, min_df=1 기본) | 분류기 | OOF macro-F1 |
|-----|--------------------------------------------|--------|--------------|
| A | word(1,2) + char_wb(1,3) | LinearSVC C=0.2 | 0.84121 |
| B | word(1,2) + char_wb(2,5) | LinearSVC C=0.2 | 0.83534 |
| C | char(1,5) | LinearSVC C=0.2 | 0.82959 |
| E | word(1,2) + char_wb(1,3) | RidgeClassifier α=1 | 0.83858 |
| F | word(1,2) + char_wb(1,4) | ComplementNB α=1 | 0.81701 |
| G | word(1,2) + char_wb(1,4) | SGD modified_huber | 0.81355 |
| H | word(1,2) + char_wb(1,3) | LinearSVC C=0.2, balanced | **0.84164** |
| I | word(1,2) + char_wb(1,4), raw tf | LinearSVC C=0.2 | 0.83975 |
| K | word(1,2) + char_wb(1,4), binary | LinearSVC C=0.3 | 0.83706 |
| M | word(1,2) + char_wb(1,4) → NB log-count-ratio 재가중 | LinearSVC C=0.2 | 0.83828 |
| N | char_wb(2,6), min_df=2 | LinearSVC C=0.3 | 0.84127 |
| P | word(1,2) + char_wb(1,4) | cosine kNN k=40 | 0.77916 |

**Level 1 — 메타 학습기** (`final.py`)

12개 모델의 점수 행렬을 각각 전역 표준화(OOF 통계 기준)한 뒤 concat → 84차원 특징 →
multinomial `LogisticRegression(C=0.1)`.

## 검증 결과

메타 학습기는 OOF 특징 위에서 다시 5-fold nested CV로 정직하게 평가했습니다.

| 구성 | macro-F1 |
|------|----------|
| 단순 baseline: word(1,2)+char_wb(2,5), min_df=2, LinearSVC C=1 | 0.8221 |
| 최고 단일 모델 (H) | 0.8416 |
| 12개 모델 단순 평균 | 0.8388 |
| **12개 모델 스태킹 (최종)** | **0.8498** (seed 7) / 0.8490 (seed 21) |

## 튜닝 과정에서 확인한 점

- **min_df=1 이 min_df=2/3보다 확실히 좋음** (0.833 vs 0.822): 제목이 평균 27자로 매우
  짧아 희귀 n-gram도 버리면 손실이 큼.
- **낮은 C가 유리** (C=1 → 0.822, C=0.2 → 0.840): 고차원 희소 특징에서 강한 정규화 필요.
- **짧은 char n-gram이 유리** (char_wb(1,3) > (2,5) > (2,6)): 한글 음절 단위 1–3-gram이
  형태소 분석기 없이도 충분한 신호를 줌.
- **가중 평균 + 클래스별 bias 튜닝은 실패**: half-split 정직 검증에서 단일 최고 모델 대비
  +0.0006 에 불과(사실상 노이즈). argmax 기반 좌표상승법은 과적합/노이즈가 심함
  (`blend.py`에 검증 코드 보존).
- **반면 LogReg 메타 학습기는 실질적 이득** (+0.0082). 약한 모델(F 0.817, G 0.814,
  P 0.779)도 제거하면 점수가 떨어짐 — 정확도보다 **다양성**이 스태킹에 기여.

## 파일

- `oof.py` — level-0 모델 정의 + OOF/test 점수 행렬 생성 (`NBTransform` 포함)
- `final.py` — level-1 메타 학습기 + 제출 파일 생성
- `run_all.sh` — 전체 재현 스크립트 + 제출 형식 검증
- `stack.py` — 스태킹/bias 튜닝 정직 검증 하네스 (빠른 macro-F1 구현)
- `exp.py`, `exp2.py` — 하이퍼파라미터 탐색 그리드 (탐색 기록)
- `blend.py` — 가중평균+bias 블렌딩 시도 (채택하지 않음, 기록용)
- `predict_baseline.py` — 초기 단일 모델 baseline (0.822)
