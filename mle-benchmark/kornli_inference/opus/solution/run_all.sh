#!/usr/bin/env bash
# Reproduce outputs/submission.csv from scratch (train.csv only).
# Run from the task root directory.  CPU-only, ~35 min on 4 cores.
set -euo pipefail
mkdir -p work outputs

# 1) sparse feature-based linear model (hold-out probs + test probs)
python -u solution/train_linear.py --tag a

# 2) neural NLI models: decomposable attention, randomly initialised,
#    trained only on train.csv (10% held out for snapshot / weight tuning).
#    Three configurations with different capacity + regularisation for diversity.
python -u solution/train_nn.py --folds 10 --limit_folds 1 --epochs 7 --seed 1 \
    --dim 144 --dropout 0.3 --wdrop 0.15 --emb_mult 2.0 --tag h1
python -u solution/train_nn.py --folds 10 --limit_folds 1 --epochs 3 --seed 202 \
    --dim 128 --dropout 0.3 --wdrop 0.25 --emb_mult 1.0 --tag b
python -u solution/train_nn.py --folds 10 --limit_folds 1 --epochs 3 --seed 303 \
    --dim 176 --hid 240 --dropout 0.25 --wdrop 0.10 --emb_mult 2.0 --tag c

# 3) blend + premise-sibling structural prior -> submission
python -u solution/blend.py
