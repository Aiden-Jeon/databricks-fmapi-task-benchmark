# KLUE-STS solution

The solution uses only `train.csv`. It combines character/word TF-IDF cosine
similarities, truncated-SVD similarities, character overlap, sequence matching,
length, token, and numeric-agreement features. A histogram gradient boosting
model and a random forest are blended for the final prediction.

Run from any directory with:

```bash
python solution/train_predict.py
```

The default output is `outputs/submission.csv`. Paths can be overridden with
`--data-dir` and `--output`.
