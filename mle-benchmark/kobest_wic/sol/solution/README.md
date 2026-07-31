# KoBEST WiC solution

The solution uses only the supplied training data. It combines character/word
TF-IDF similarities with nearest labeled context pairs for the same target word,
then trains a deterministic Extra Trees classifier.

Run from any directory with:

```bash
python solution/train_predict.py
```

The script validates the submission schema and writes
`outputs/submission.csv`. It requires Python 3, NumPy, pandas, SciPy, and
scikit-learn.
