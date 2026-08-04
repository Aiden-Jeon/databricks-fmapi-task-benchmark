#!/usr/bin/env bash
# Full reproducible pipeline (CPU only, no external data / pretrained weights).
# Run from the task root directory:  bash solution/run_all.sh
set -e
cd "$(dirname "$0")/.."

NSH=3

# 1) candidate features for all train questions (random negatives), sharded
for i in $(seq 0 $((NSH-1))); do
  SHARD=$i NSHARD=$NSH python solution/run.py build &
done
wait
NSHARD=$NSH python solution/run.py merge

# 2) first-round ranker
ITERS=400 python solution/run.py train
cp work/model.pkl work/model_v1.pkl

# 3) hard-negative mining with the first-round ranker
for i in $(seq 0 $((NSH-1))); do
  SHARD=$i NSHARD=$NSH python solution/run.py mine &
done
wait

# 4) rebuild training data with hard negatives and refit
for i in $(seq 0 $((NSH-1))); do
  SHARD=$i NSHARD=$NSH NHARD=$NSH python solution/run.py build2 &
done
wait
NSHARD=$NSH python solution/run.py merge
ITERS=500 python solution/run.py train
cp work/model.pkl work/model_v2.pkl

# 5) validation score + submission
python solution/run.py val
python solution/predict_ensemble.py
