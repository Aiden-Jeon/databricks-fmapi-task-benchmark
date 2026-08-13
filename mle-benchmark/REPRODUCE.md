# 재현 가이드 (REPRODUCE)

이 폴더에는 결과물(제출물·코드·점수)뿐 아니라 **벤치마크를 처음부터 다시 돌리는 데
필요한 하네스 전체**가 들어 있다. 원본 데이터와 숨겨진 정답 키만 없다 — 데이터는
공개 출처에서 스크립트로 받고, 정답 키는 로컬에서 재생성해 UC 볼륨에만 올린다
(커밋 금지, `.gitignore`가 막는다).

```
mle-benchmark/
├── config.json          # 워크스페이스 프로필·카탈로그·스키마·레인 정의
├── setup.sh             # 워크스페이스 부트스트랩 (스키마·볼륨·시크릿·러너 업로드)
├── harness/             # 전체 하네스 (준비→실행→채점→집계→리포트)
├── tests/               # 채점기·집계기 단위 테스트
├── queries/             # system.billing.usage 비용 귀속 SQL
├── results/             # 이 캠페인의 원장 (manifest·scores·repeats·variance·cost)
├── snapshots/           # 캠페인별 동결 스냅샷 (시점별 비교용 — TIMELINE.md 참조)
└── <task_slug>/         # 태스크별 결과물 (README·PROMPT·모델별 제출물)
```

## 0. 전제 조건

- Databricks 워크스페이스: 서버리스 Jobs + Unity Catalog + **Unity AI Gateway** +
  pay-per-token 엔드포인트(`databricks-claude-opus-5`, `databricks-gpt-5-6-sol`,
  `databricks-glm-5-2`, 추가 모델은 아래 §6).
- 로컬: Databricks CLI 인증(`databricks auth login --profile <p>`), Python 3.12+,
  `pandas`·`numpy`·`scikit-learn`, `jq`. Kaggle 태스크(t1·t2)는 kaggle CLI 인증 필요.
- `config.json`의 `profile`/`catalog`를 본인 환경으로 수정.

## 1. 데이터 준비 (로컬)

```bash
python harness/fetch_raw25.py     # 공개 출처에서 raw/ 로 다운로드 (재배포 아님)
python harness/prepare.py         # packs/ (spec+train+hidden test) · private/ (정답 키) 생성
```

`prepare.py`가 fresh split을 만든다 — 정답 키가 학습 경로에 노출되지 않게 하는
MLE-bench 방식. `packs/<task>/meta.json`에 metric·방향이 기록되고 이후 모든
스크립트가 이를 참조한다.

## 2. 워크스페이스 부트스트랩 (1회, 멱등)

```bash
./setup.sh
```

스키마·볼륨 생성, PAT를 시크릿(`kmle/pat`)으로 저장, packs/private 업로드,
`runner.py`를 워크스페이스로 임포트. **private 볼륨은 실행 주체가 읽지 못하게
ACL을 분리**할 것 — 유출 방지는 관례가 아니라 권한으로 한다(METHODOLOGY.md).

## 3. 실행

```bash
python harness/submit_matrix.py --mode smoke --lanes M1          # 배선 검증 (10분 캡)
python harness/submit_matrix.py --lanes M1 M2 M3 --mode full     # 본 실행 (2시간 캡)
python harness/poll_matrix.py                                     # 완료 대기
```

M-track 레인 = **pinned opencode**(`runner.py`의 `OPENCODE_PIN`) × 모델 교체.
이 repo의 공식 숫자는 전부 M-track이다. L1~L3(native 하네스)와 L4(Genie Code UI)는
참고 레인. 반복 실행(n≥3)은 같은 명령을 다시 제출하면 된다 — 셀의 반복은 단순히
같은 (task, model)의 유효 런들이다.

## 4. 채점·집계

```bash
python harness/scoreboard.py          # 미채점 런 채점 → results/scores.csv 갱신
python harness/aggregate_repeats.py   # mean±std 보드 + 통계적 승패 판정
python harness/variance_report.py     # 재현성(분산) 리포트
```

채점은 결정적이다: 같은 제출물은 항상 같은 점수. 형식 불량 제출 = DNF.
판정 규칙: 1·2위 평균 차이가 두 표준오차의 max를 넘어야 승리, 아니면 무승부.

## 5. 리포트·repo 반영

```bash
python harness/build_repo.py          # 태스크별 README·제출물·상위 문서 재생성
python harness/snapshot.py            # 현재 캠페인을 snapshots/<id>/ 로 동결
python harness/timeline.py            # snapshots/* → TIMELINE.md (시점별 비교)
```

비용은 추정이 아니라 `queries/cost_attribution.sql`(system.billing.usage 청구 실적)로
집계해 COST.md에 반영한다.

## 6. 새 모델 추가하는 법

1. `harness/runner.py`의 `M_SELECTORS`에 레인 추가 (예: `"M7": "databricks-oss/databricks-kimi-k3"`)
   — OpenAI-호환 모델이면 `databricks-oss` 프로바이더의 `models`에도 등록.
2. `harness/aggregate_repeats.py`의 `LANE`/`NAMES`/`ORDER`에 모델 키 추가.
3. `harness/build_repo.py`의 `MODEL`에 (dir, 표시명, endpoint, list price) 추가.
4. `setup.sh`로 runner 재업로드 → smoke → full n≥3 → 채점·집계·스냅샷.

주의: 표·보드는 데이터에서 생성되지만 **README·FINDINGS의 서사 문단은 3모델
캠페인 기준으로 쓰여 있다** — 새 모델이 들어오면 문단은 손으로 갱신해야 한다.

## 검증(테스트)

```bash
python tests/test_aggregate.py       # 승패 판정기 edge case (packs 불필요)
python tests/test_graders.py         # 채점기 검증 (packs/ 필요, 수 분 소요)
```
