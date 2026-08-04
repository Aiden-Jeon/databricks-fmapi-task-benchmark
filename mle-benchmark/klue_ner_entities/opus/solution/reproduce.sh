#!/bin/bash
# Reproduce outputs/submission.csv from train.csv / test.csv.
# Run from the task root directory. ~6 min on 4 CPU cores.
set -e
python solution/run.py --mode full --epochs 10 --kfold 5 --seeds 2 \
    --biases 1 --obias 1 --out outputs/submission.csv
python solution/verify.py outputs/submission.csv
