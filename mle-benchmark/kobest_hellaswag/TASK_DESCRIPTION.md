# 상황 이어짓기 추론 (KoBEST HellaSwag)

## 배경
맥락(`context`) 다음에 가장 자연스럽게 이어질 문장을 4개 후보(`ending_1`~`ending_4`) 중에서 고릅니다.

## 데이터
- `train.csv` — 컬럼: `id`, `context`, `ending_1`, `ending_2`, `ending_3`, `ending_4`, `label`
- `test.csv` — `label` 제외
- 레이블: 0,1,2,3 (ending_1~4)

## 목표 / 평가 지표
test의 각 행에 대해 `label`을 예측하십시오. **정확도(accuracy)**, 높을수록 좋습니다.

## 제출 형식
`outputs/submission.csv` — 컬럼 `id,label`; test의 모든 id를 정확히 한 번씩.
```
id,label
hellaswag_00001,2
```

## 규칙
외부 데이터/인터넷 금지 · train.csv만 사용 · 시간 예산 2시간 · 재현 가능한 코드를 `solution/`에 저장.
