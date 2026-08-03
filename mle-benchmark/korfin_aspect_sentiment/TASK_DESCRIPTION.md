# 금융 뉴스 속성 기반 감성 분석 (t23_korfin_asc)

## 배경
한국 금융·경제 뉴스 문장에서 **특정 대상(기업·종목 등)에 대한 감성**을 분류하는
속성 기반 감성 분석 과제입니다 (KorFin-ASC). 같은 문장이라도 대상에 따라 감성이
달라질 수 있으므로, 문장 전체가 아니라 주어진 대상에 국한된 감성을 판별해야 합니다.

## 데이터
- `train.csv` — 컬럼: `id`, `sentence`(문장), `aspect`(감성 판단 대상), `label`(감성)
- `test.csv` — 컬럼: `id`, `sentence`, `aspect` (레이블 없음)
- `sample_submission.csv` — 제출 형식 예시

레이블은 다음 3개 중 하나입니다:
`NEGATIVE`(부정), `NEUTRAL`(중립), `POSITIVE`(긍정)

## 목표
`test.csv`의 각 (문장, 대상) 쌍에 대해, **해당 대상에 대한** 감성을 예측하십시오.

## 평가 지표
**Macro F1** (3개 클래스에 대한 F1 점수의 단순 평균). 높을수록 좋습니다.

## 제출 형식
`outputs/submission.csv` — 헤더 포함, 컬럼 2개:
```
id,label
4208_0,POSITIVE
4209_0,NEGATIVE
```
- `test.csv`의 모든 `id`가 정확히 한 번씩 포함되어야 합니다.
- `label` 값은 `NEGATIVE`, `NEUTRAL`, `POSITIVE` 중 하나여야 합니다 (그 외 값은 무효 처리).

## 규칙
- 외부 데이터 및 사전학습 자료의 추가 다운로드 금지. 인터넷 사용 금지.
- 제공된 `train.csv`만을 학습에 사용하십시오.
- 시간 예산: 2시간. 재현 가능한 코드를 `solution/` 아래에 남기십시오.
