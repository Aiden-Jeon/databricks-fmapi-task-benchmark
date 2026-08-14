# Korean UnSmile 다중 레이블 혐오 표현 분류 — Solution

## 접근

GPU/transformer 없이 CPU만으로 동작하는 고전적 NLP 파이프라인.

1. **특징 (TF-IDF, sublinear)** — 3가지를 hstack:
   - `char_wb` (1,5) n-gram, max 300k
   - `word` (1,2) n-gram, max 100k
   - **자모 분해** 음절 (1,4) n-gram: 한글 음절을 초/중/종성으로 분해해 공백 결합.
     한국어 혐오 표현의 철자 변형·난독화에 강건해지는 효과가 컸음 (+0.02 F1).
2. **모델**: 레이블당 LogisticRegression (liblinear, `class_weight='balanced'`, C=0.5) 10개.
3. **임계값**: 5-fold 계층 OOF 확률 위에서 레이블별 임계값을 coordinate ascent로
   튜닝 (constrained macro F1 최대화).
4. **제약**: `clean`과 9개 혐오 레이블은 train 데이터에서 상호 배타적이므로,
   - 혐오 레이블이 하나라도 예측되면 clean=0,
   - 아무것도 예측되지 않으면 P(clean)>=0.5 일 때 clean=1, 아니면 argmax 레이블.
   이 제약이 OOF에서 ~+0.04 macro F1.

## 결과

- 5-fold OOF macro F1: **0.7045** (0.5 고정 임계값 대비 0.6883)
- 80/20 holdout 기준 최고 0.7113

## 파일

- `train.py` — 전체 파이프라인 (특징 구축 → OOF → 임계값 튜닝 → 최종 학습 → 제출 생성)
- `artifacts/model.joblib` — 학습된 모델/벡터라이저/임계값
- `artifacts/cv_score.json` — OOF 점수와 임계값

## 재현

```bash
python solution/train.py
```

`outputs/submission.csv`가 생성됩니다 (1800행, id + 10자리 0/1 labels).
