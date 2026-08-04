# Bike Demand Solution

Run from any directory with:

```bash
python /path/to/workspace/solution/train.py
```

The script reads the provided `train.csv`, `test.csv`, and
`sample_submission.csv`, then writes `outputs/submission.csv`.

The model is a deterministic blend of histogram gradient boosting and extra
randomized trees. Features include the provided weather and operating fields,
calendar fields derived from `date`, and cyclic encodings for hour, weekday,
and day of year. Validation used forward-only 73-day holdouts to match the
forecast horizon. Rows marked as non-functioning are set to zero because every
such row in the training data has a zero target.
