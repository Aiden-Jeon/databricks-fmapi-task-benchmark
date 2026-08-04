# KoBEST WiC solution

The solution uses only the supplied training data. It builds character and word
TF-IDF representations of each context, measures cosine similarity for the full
sentence and several windows around the marked target, and adds word-level LSA
similarities and simple overlap statistics. A regularized logistic regression is
selected with deterministic five-fold cross-validation and then refit on all
training rows.

Run from the task directory:

```bash
python solution/train_predict.py
```

This writes `outputs/submission.csv`. The implementation requires Python 3,
NumPy, pandas, SciPy, and scikit-learn.
