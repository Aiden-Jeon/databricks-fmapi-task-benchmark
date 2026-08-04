# KoBEST SentiNeg Solution

The final model is a character 1-5 gram TF-IDF representation followed by a
linear support vector classifier. It was selected using fixed, stratified
5-fold cross-validation on `train.csv` (mean accuracy: 0.9548).

Run from any directory with:

```bash
python solution/train.py
```

This reads the task CSV files from the workspace root and writes
`outputs/submission.csv`. `evaluate.py` reproduces the candidate model
comparison used to select the final pipeline.

Tested with Python 3.12, pandas 1.5.3, and scikit-learn 1.4.2.
