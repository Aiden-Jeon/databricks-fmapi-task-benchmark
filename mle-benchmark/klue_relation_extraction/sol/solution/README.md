# KLUE-RE solution

The solution uses only `train.csv`. It marks subject/object mentions, combines word
and character TF-IDF with exact entity categorical features, and trains a
deterministic linear SVM. An optional entity-pair majority override is available,
but is disabled by default because it reduced fixed-holdout accuracy.

Run from any directory:

```bash
python solution/train.py
```

Optional fixed holdout diagnostics:

```bash
python solution/train.py --validate-only
```

The prediction command writes `outputs/submission.csv` and validates its columns,
ids, row count, and labels before completing.
