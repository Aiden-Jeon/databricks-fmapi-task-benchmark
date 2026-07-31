# 인과 추론 (KoBEST COPA)

## 배경
전제(`premise`)와 질문 유형(`question`: 원인/결과)이 주어지면 두 대안 중 더 그럴듯한 것을 고릅니다.

## 데이터
- `train.csv` — 컬럼: `id`, `premise`, `question`, `alternative_1`, `alternative_2`, `label`
- `test.csv` — `label` 제외
- 레이블: 0(alternative_1) 또는 1(alternative_2)

## 목표 / 평가 지표
test의 각 행에 대해 `label`을 예측하십시오. **정확도(accuracy)**, 높을수록 좋습니다.

## 제출 형식
`outputs/submission.csv` — 컬럼 `id,label`; test의 모든 id를 정확히 한 번씩.
```
id,label
copa_00001,0
```

## 규칙
외부 데이터/인터넷 금지 · train.csv만 사용 · 시간 예산 2시간 · 재현 가능한 코드를 `solution/`에 저장.
