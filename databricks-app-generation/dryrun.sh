#!/usr/bin/env bash
# dryrun.sh — tier1 드라이런 원커맨드 (후보 1개, 배포 제외)
#
#   cd <repo 루트>   # databricks-fmapi-task-benchmark/
#   DATABRICKS_WAREHOUSE_ID=<warehouse-id> ./databricks-app-generation/dryrun.sh [candidate]
#
# 자격: DATABRICKS_HOST/DATABRICKS_TOKEN env가 있으면 사용, 없으면 ucode 자동 해결
# (ai-devtools-prod를 쓰려면 ucode 프로필/env가 해당 워크스페이스를 가리켜야 함).
set -euo pipefail

CANDIDATE="${1:-opus}"
DIR="databricks-app-generation"

[ -f "$DIR/suite.json" ] || { echo "ERROR: repo 루트에서 실행하세요 (BENCHMARK_ROOT 기준)"; exit 1; }
[ -n "${DATABRICKS_WAREHOUSE_ID:-}" ] || {
  echo "ERROR: DATABRICKS_WAREHOUSE_ID 를 설정하세요 (ai-devtools-prod의 serverless SQL warehouse ID)"; exit 1; }

echo "== 0) 설치 =="
uv venv --allow-existing
uv pip install -q -e "$DIR"
uv run playwright install chromium

echo "== 1) tier1 실행: candidate=$CANDIDATE harness=direct-fmapi =="
uv run run-task --tier tier1-gate --candidate "$CANDIDATE" --harness direct-fmapi

echo "== 2) tier1 채점 (배포 제외) =="
uv run grade-task --tier tier1-gate --candidates "$CANDIDATE" --no-deploy

echo "== 결과 =="
echo "  - $DIR/grade_results.json"
echo "  - $DIR/gallery/index.html (브라우저로 열기)"
echo "  - $DIR/tier1-gate/$CANDIDATE/ (산출물·run_meta.json·app_boot.log)"
echo ""
echo "다음 단계: 셀렉터/타이밍 이슈 없으면 --tier all + 3후보로 확장, 배포 검증은 --no-deploy 제거."
