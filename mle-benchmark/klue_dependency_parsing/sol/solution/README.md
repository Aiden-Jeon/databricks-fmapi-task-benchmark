# KLUE-DP solution

This solution uses only `train.csv`. It predicts dependency relations with a
linear SVM over token affix and local-context features. A second linear SVM
scores every possible right-headed dependency arc. Interval dynamic
programming selects the highest-scoring projective tree and enforces one root
at the final token, matching structural properties observed throughout the
training set.

## Run

From the task root:

```bash
python solution/train_predict.py
```

The command writes `outputs/submission.csv` and validates IDs, token counts,
labels, head ranges, and root counts before writing. Default paths can be
overridden with `--train`, `--test`, and `--output`.

For the deterministic 80/20 sentence-level holdout evaluation:

```bash
python solution/train_predict.py --validate
```

The final configuration achieved relation accuracy 0.92238, UAS 0.83769, and
LAS 0.79257 on that holdout split (seed 20260801).

## Dependencies

- Python 3.10+
- numpy
- pandas
- scikit-learn
