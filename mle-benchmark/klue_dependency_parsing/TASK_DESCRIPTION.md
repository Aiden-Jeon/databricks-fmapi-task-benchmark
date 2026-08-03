# 의존 구문 분석 (t25_klue_dp)

## 배경
한국어 문장의 **의존 구문 구조**를 분석하는 과제입니다 (KLUE-DP). 문장은 어절(토큰)로
나뉘어 주어지며, 각 어절에 대해 (1) 그 어절이 걸리는 지배소(head) 어절의 위치와
(2) 둘 사이의 의존 관계 레이블(deprel)을 예측해야 합니다. 구조적 예측(structured
prediction) 과제로, 단순 분류와 달리 문장 단위의 일관된 트리 구조를 요구합니다.

## 데이터
- `train.csv` — 컬럼: `id`, `sentence`(원문), `tokens`(어절 배열, JSON), `parse`(정답 구문)
- `test.csv` — 컬럼: `id`, `sentence`, `tokens` (정답 없음)
- `sample_submission.csv` — 제출 형식 예시

`tokens`는 어절의 JSON 배열입니다. 예: `["해당", "그림을", "보면", ...]`. 어절 위치는
**1부터** 시작합니다(1 = 첫 어절).

`parse`는 **어절 순서대로** 각 어절의 `head:deprel`을 `|`로 이어 붙인 문자열입니다.
- `head` = 그 어절의 지배소 어절 위치(1-indexed). **루트(root)는 0**.
- `deprel` = 아래 36개 의존 관계 레이블 중 하나.

의존 관계 레이블(deprel):
`NP, NP_AJT, NP_CMP, NP_CNJ, NP_MOD, NP_OBJ, NP_SBJ, VP, VP_AJT, VP_CMP, VP_CNJ,
VP_MOD, VP_OBJ, VP_SBJ, VNP, VNP_AJT, VNP_CMP, VNP_CNJ, VNP_MOD, VNP_OBJ, VNP_SBJ,
AP, AP_AJT, AP_CMP, AP_MOD, DP, IP, X, X_AJT, X_CMP, X_CNJ, X_MOD, X_OBJ, X_SBJ,
L, R`

## 목표
`test.csv`의 각 문장에 대해, `tokens`의 모든 어절에 대한 `head:deprel`을 순서대로
예측하십시오. `parse`의 항목 수는 `tokens`의 어절 수와 같아야 합니다.

## 평가 지표
**LAS (Labeled Attachment Score)** — head와 deprel이 **둘 다** 맞은 어절의 비율
(전체 어절에 대한 micro 평균). 높을수록 좋습니다.

## 제출 형식
`outputs/submission.csv` — 헤더 포함, 컬럼 2개. `parse`에 `,`가 들어가므로 값을
큰따옴표로 감싸십시오:
```
id,parse
dp_00007,"2:NP|3:NP_OBJ|14:VP|5:NP|14:NP_SBJ"
dp_00013,"2:NP_MOD|0:VP"
```
- 각 어절은 `head:deprel` 형식이며 `|`로 구분합니다. `head`는 정수(0=root), `deprel`은
  위 목록의 레이블입니다.
- `test.csv`의 모든 `id`가 정확히 한 번씩 포함되어야 합니다.

## 규칙
- 외부 데이터 및 사전학습 자료의 추가 다운로드 금지. 인터넷 사용 금지.
- 제공된 `train.csv`만을 학습에 사용하십시오.
- 시간 예산: 2시간. 재현 가능한 코드를 `solution/` 아래에 남기십시오.
