# 기계 독해 — 답 없음 포함 (t19_klue_mrc)

## 배경
한국어 지문(`context`)과 질문(`question`)이 주어지면 지문에서 답을 찾습니다.
일부 질문은 지문으로 답할 수 없으며(unanswerable), 이 경우 **빈 문자열**을 답으로 제출합니다.

## 데이터
- `train.csv` — 컬럼: `id`, `context`, `question`, `answer`(답 없으면 빈 값)
- `test.csv` — `answer` 제외

## 목표 / 평가 지표
각 질문의 답 문자열을 예측하십시오(답 없으면 빈 문자열). **문자 단위 F1**, 높을수록 좋습니다.

## 제출 형식
`outputs/submission.csv` — 컬럼 `id,answer`; test의 모든 id를 한 번씩. 답 없는 경우 answer를 빈 값으로.

## 규칙
외부 데이터/인터넷 금지 · train.csv만 사용 · 시간 예산 2시간 · 재현 코드를 `solution/`에 저장.
