# KoBEST COPA solution

This solution uses only `train.csv` and locally available Python packages.
Each row is expanded into two candidate causal statements. Character TF-IDF
features feed two complementary models:

- similarity-weighted retrieval of training candidates with the same question
  direction (`원인` or `결과`)
- logistic regression for global candidate plausibility patterns

The two candidate scores are compared, so the model predicts which alternative
is more plausible rather than directly learning answer position.

Run from any directory with:

```bash
python solution/run.py
```

The script writes `outputs/submission.csv`. It requires Python 3, NumPy,
pandas, and scikit-learn. Hyperparameters were selected with stratified 5-fold
cross-validation (`random_state=42`); the final script fits all training rows.
