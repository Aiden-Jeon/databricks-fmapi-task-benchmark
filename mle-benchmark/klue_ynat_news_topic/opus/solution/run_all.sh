#!/usr/bin/env bash
# Full reproducible pipeline for t3_ynat (KLUE-YNAT news topic classification).
#
#   bash solution/run_all.sh
#
# Stage 1: build 5-fold OOF + test decision-function matrices for 12 level-0 models
#          (cached as /tmp/opencode/oof_<KEY>.npz)
# Stage 2: fit the level-1 LogisticRegression meta-learner and write
#          outputs/submission.csv
#
# Total runtime ~8 min wall clock on 4 cores. Fully deterministic (seed 42).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore
mkdir -p outputs /tmp/opencode

KEYS=(A B C E F G H I K M N P)

# Stage 1 -- 4 models at a time to match core count.
i=0
for k in "${KEYS[@]}"; do
  python solution/oof.py "$k" &
  i=$((i + 1))
  if [ $((i % 4)) -eq 0 ]; then wait; fi
done
wait

# Stage 2
python solution/final.py

# Sanity check of the submission file
python - <<'PY'
import pandas as pd
s = pd.read_csv("outputs/submission.csv"); t = pd.read_csv("test.csv")
assert list(s.columns) == ["id", "label"]
assert list(s.id) == list(t.id), "id mismatch"
assert s.label.isin(["IT과학","경제","사회","생활문화","세계","스포츠","정치"]).all()
print("submission OK:", s.shape)
PY
