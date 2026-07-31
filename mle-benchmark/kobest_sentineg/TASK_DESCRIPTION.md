# 부정 표현 감성 분석 (KoBEST SentiNeg)

## 배경
부정 표현이 포함된 문장(`sentence`)의 감성을 분류합니다.

## 데이터
- `train.csv` — 컬럼: `id`, `sentence`, `label`
- `test.csv` — `label` 제외
- 레이블: 0(부정) 또는 1(긍정)

## 목표 / 평가 지표
test의 각 행에 대해 `label`을 예측하십시오. **정확도(accuracy)**, 높을수록 좋습니다.

## 제출 형식
`outputs/submission.csv` — 컬럼 `id,label`; test의 모든 id를 정확히 한 번씩.
```
id,label
sentineg_00001,0
```

## 규칙
외부 데이터/인터넷 금지 · train.csv만 사용 · 시간 예산 2시간 · 재현 가능한 코드를 `solution/`에 저장.
