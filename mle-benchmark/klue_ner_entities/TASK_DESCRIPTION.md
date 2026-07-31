# 개체명 인식 (t20_klue_ner)

## 배경
한국어 문장(`sentence`)에서 개체명(named entity)을 추출합니다. 개체 유형은
`PS`(인물), `LC`(지역), `OG`(기관), `DT`(날짜), `TI`(시간), `QT`(수량) 6종입니다.

## 데이터
- `train.csv` — 컬럼: `id`, `sentence`, `entities`
- `test.csv` — `entities` 제외
- `entities` 형식: `개체표현:유형` 쌍을 `|`로 구분. 예) `영동고속도로:LC|만종분기점:LC|5km:QT`
  개체가 없으면 빈 문자열.

## 목표 / 평가 지표
각 문장의 개체 목록을 예측하십시오. **개체 단위 micro-F1** (개체표현+유형이 모두
일치해야 정답), 높을수록 좋습니다.

## 제출 형식
`outputs/submission.csv` — 컬럼 `id,entities`; 위 형식대로. test의 모든 id를 한 번씩.
```
id,entities
ner_00001,서울:LC|김철수:PS
```

## 규칙
외부 데이터/인터넷 금지 · train.csv만 사용 · 시간 예산 2시간 · 재현 코드를 `solution/`에 저장.
