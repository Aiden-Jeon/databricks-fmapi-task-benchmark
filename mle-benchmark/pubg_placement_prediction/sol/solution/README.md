# PUBG placement solution

Run from the task root:

```bash
python solution/train.py
```

This reads `train.csv` and `test.csv`, trains a match-aware histogram gradient
boosting model, and writes `outputs/submission.csv`. The implementation uses a
fixed random seed. To reproduce the match-level holdout evaluation, run:

```bash
python solution/train.py --cv
```

Dependencies: Python 3, NumPy, pandas, and scikit-learn.
