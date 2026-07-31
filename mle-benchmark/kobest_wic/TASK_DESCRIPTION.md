# 문맥 내 동일 의미 판별 (KoBEST WiC)

## 배경
같은 단어(`word`)가 두 문맥(`context_1`,`context_2`)에서 동일한 의미로 쓰였는지 판단합니다.

## 데이터
- `train.csv` — 컬럼: `id`, `word`, `context_1`, `context_2`, `label`
- `test.csv` — `label` 제외
- 레이블: 0(다른 의미) 또는 1(같은 의미)

## 목표 / 평가 지표
test의 각 행에 대해 `label`을 예측하십시오. **정확도(accuracy)**, 높을수록 좋습니다.

## 제출 형식
`outputs/submission.csv` — 컬럼 `id,label`; test의 모든 id를 정확히 한 번씩.
```
id,label
wic_00001,1
```

## 규칙
외부 데이터/인터넷 금지 · train.csv만 사용 · 시간 예산 2시간 · 재현 가능한 코드를 `solution/`에 저장.
