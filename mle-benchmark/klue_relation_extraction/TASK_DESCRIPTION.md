# 관계 추출 (KLUE-RE)

## 배경
문장(`sentence`)과 두 개체(`subject_entity`,`object_entity`)가 주어지면 둘 사이의 관계 유형을 30종 중 하나로 분류합니다. 가능한 레이블은 `train.csv`에서 확인하십시오.

## 데이터
- `train.csv` — 컬럼: `id`, `sentence`, `subject_entity`, `object_entity`, `label`
- `test.csv` — `label` 제외
- 레이블: 관계 레이블 문자열 (예: `no_relation`, `org:top_members/employees`)

## 목표 / 평가 지표
test의 각 행에 대해 `label`을 예측하십시오. **정확도(accuracy)**, 높을수록 좋습니다.

## 제출 형식
`outputs/submission.csv` — 컬럼 `id,label`; test의 모든 id를 정확히 한 번씩.
```
id,label
klue-re-v1_train_00001,no_relation
```

## 규칙
외부 데이터/인터넷 금지 · train.csv만 사용 · 시간 예산 2시간 · 재현 가능한 코드를 `solution/`에 저장.
