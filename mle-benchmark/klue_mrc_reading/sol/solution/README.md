# Lightweight extractive QA solution

The solution uses only `train.csv` and locally installed Python packages. It ranks
sentences by question overlap, learns to rank exact text spans, and separately
learns whether a question is answerable. All predicted non-empty answers are
copied verbatim from the corresponding context.

Run from the task root:

```bash
python solution/train_predict.py --validate
```

This writes `outputs/submission.csv`. The fixed random seed and document-grouped
validation make the run reproducible.
