# 문장 의역 판별 (PAWS-X 한국어)

## 배경
두 한국어 문장(`sentence1`,`sentence2`)이 서로 의역(paraphrase) 관계인지 판단합니다.

## 데이터
- `train.csv` — 컬럼: `id`, `sentence1`, `sentence2`, `label`
- `test.csv` — `label` 제외
- 레이블: 0(의역 아님) 또는 1(의역)

## 목표 / 평가 지표
test의 각 행에 대해 `label`을 예측하십시오. **정확도(accuracy)**, 높을수록 좋습니다.

## 제출 형식
`outputs/submission.csv` — 컬럼 `id,label`; test의 모든 id를 정확히 한 번씩.
```
id,label
pawsx_00001,1
```

## 규칙
외부 데이터/인터넷 금지 · train.csv만 사용 · 시간 예산 2시간 · 재현 가능한 코드를 `solution/`에 저장.
