# KorFin-ASC solution

The solution uses an ensemble of three character TF-IDF `LinearSVC` models:

- the aspect-marked full sentence;
- a 35-character context window around the aspect;
- the aspect text alone, used only when the test sentence also occurs in the
  training set.

The fixed ensemble weights were selected with 3-fold stratified
cross-validation (Macro F1 approximately 0.666).

Run from any directory with:

```bash
python solution/train_predict.py
```

This reads the CSV files from the workspace root and writes
`outputs/submission.csv`. The script checks row count, ID coverage, ID
uniqueness, and label validity before writing the result.
