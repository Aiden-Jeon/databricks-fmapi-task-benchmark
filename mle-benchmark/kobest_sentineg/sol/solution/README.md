# KoBEST SentiNeg solution

The solution averages the decision scores of three linear SVM classifiers. Each
classifier uses character TF-IDF features with a slightly different n-gram range
or word-boundary treatment. The fixed ensemble was selected using stratified
5-fold cross-validation on `train.csv`.

Run from any directory with:

```bash
python solution/train.py
```

The script requires Python 3.9+, NumPy, pandas, and scikit-learn. It writes the
submission to `outputs/submission.csv` and validates input IDs and output row
coverage before saving.
