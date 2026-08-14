# Solution — KLUE-MRC (답 없음 포함) 문자 단위 F1

## 환경 제약
실행 환경에 GPU, PyTorch/Transformers, 사전학습 가중치, 인터넷 접근이 없었다.
따라서 외부 리소스 없이 동작하는 **고전적(classical) NLP 파이프라인**을
scikit-learn / numpy / pandas 만으로 구현했다.

## 접근
1. **문장 검색 (retrieval)** — 지문을 문장으로 분할하고, 문자 n-gram(2~4)
   TF-IDF 벡터로 질문과의 코사인 유사도를 계산해 상위 k개 문장을 고른다.
   질문의 content word가 문장에 포함되면 보너스를 준다.
2. **후보 스팬 생성** — 상위 문장에서 공백 토큰 기반 n-gram(1~7토큰)과
   한글 음절 run을 후보로 열거한다.
3. **스팬 점수** — 두 가지를 구현했다.
   - `pipeline.py`: Logistic Regression(표준화된 수작업 특징)으로 후보를
     점수화. 특징: 문장유사도, 질문키워드 중첩, 질문유형 일치, 위치, 길이,
     조사/구두점 형태 등.
   - `heuristic.py`: 동일 특징의 가중합을 결정론적으로 계산(학습 불필요,
     매우 빠름). 선택된 스팬의 뒤쪽 조사/구두점/어미를 `trim_answer`로 제거해
     gold 형식에 맞춘다.
4. **무응답(unanswerable) 결정** — 최고 스팬 점수가 임계값 미만이면 빈 문자열.
   임계값은 holdout에서 문자 F1을 최대화하도록 튜닝했다.

## 핵심 발견 / 정직한 한계
검증(holdout)에서, 이 고전적 파이프라인은 **어느 문장에 답이 있는지는 상당히
잘 찾지만(검색 oracle F1≈0.70), 문장 내에서 정확한 스팬을 고르는 정확도는
낮았다**(argmax F1≈0.11~0.16). 사전학습 언어모델 없이는 문맥 이해 기반 스팬
선택이 어렵기 때문이다. 결과적으로 문자 F1을 최대화하는 전략은
**고신뢰(주로 숫자/날짜형) 소수만 답하고 나머지는 빈 문자열**로 두는 것이며,
이는 holdout F1≈0.30 수준(사실상 무응답 비율 + 신뢰 답변 소량)을 준다.
이는 이 벤치마크의 지표 하에서 고전적 방법으로 도달 가능한 정직한 결과이다.

## 재현 방법
```bash
cd solution
python predict.py [THRESHOLD=1.7] [TOP_K=5] [OUT=../outputs/submission.csv]
```
`predict.py`는 `heuristic.py`의 결정론적 모델로 test.csv 전체를 예측해
`outputs/submission.csv`(`id,answer`)를 쓴다. 빈 답은 빈 문자열로 저장된다.

## 파일
- `common.py` — 정규화, 문장 분할, 문자 F1.
- `features.py` — 질문 유형 분류, 키워드 추출.
- `pipeline.py` — 검색 + 후보 생성 + LR 스팬 스코어러.
- `heuristic.py` — 결정론적 스코어링 모델(최종 사용) + trim_answer.
- `tune_heuristic.py` — 가중치/임계값 그리드 탐색.
- `validate.py` — holdout 검증.
- `predict.py` — 제출 생성(최종).
- `train_full.py` — LR 모델 전체 학습 + 제출 생성(대안, 느림).
