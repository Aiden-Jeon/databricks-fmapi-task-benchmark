# UnSmile solution

The final model is an equal-weight ensemble of two TF-IDF character models:

- character 1-5 grams
- within-word-boundary character 2-5 grams

Each feature set is classified by a class-balanced one-vs-rest linear SVM. Five-fold
out-of-fold predictions determine the F1-optimal positive prevalence for each label.
The two models are then retrained on all training rows, and those prevalences are
transferred to the test ranking. This avoids applying validation thresholds to models
trained on a different amount of data.

Run from any directory with:

```bash
python /path/to/task/solution/train.py
```

The script reads `train.csv` and `test.csv` from the task root and writes
`outputs/submission.csv`. It uses only the provided data and fixed random seeds.

Dependencies: Python 3, NumPy, pandas, SciPy, and scikit-learn 1.4 or newer.
