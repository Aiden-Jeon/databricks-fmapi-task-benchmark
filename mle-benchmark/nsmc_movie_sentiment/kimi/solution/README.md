# 솔루션 개요 (t4_nsmc)

## 방법
- **특징**: TF-IDF 이중 채널 결합
  - 단어 채널: 공백 토큰 기준 unigram+bigram (`token_pattern=\S+`), min_df=2, 40만 특징
  - 글자 채널: `char_wb` 2~6-gram, min_df=3, 60만 특징
- **모델**: LogisticRegression(C=2.0, liblinear) + LinearSVC(C=0.5) 소프트 앙상블
  - SVC 결정함수를 시그모이드로 변환해 확률처럼 사용 후 0.5:0.5 평균, 0.5 임계값
- **검증**: stratified 90/10 홀드아웃 accuracy = **0.86583**
  - 최종 제출은 전체 train(119,996건)으로 재학습 후 test(29,999건) 예측

## 파일
- `solution/solution.py` — 최종 파이프라인 (검증 + 전체 학습 + 제출 생성)
- `solution/experiment.py` — 하이퍼파라미터/특징/앙상블 비교 실험 (선택 근거)

## 재현
```bash
python3 solution/solution.py   # 약 75초 소요, outputs/submission.csv 생성
```

## 규칙 준수
- `train.csv`만 학습에 사용, 외부 데이터/사전학습 가중치/인터넷 미사용
- test 행에 대한 수동 레이블 지정/하드코딩 없음 (일반화 가능한 모델 기반 예측)
