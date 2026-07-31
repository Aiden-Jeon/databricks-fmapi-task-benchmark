# KoBEST BoolQ solution

Run from the task root:

```bash
python solution/train_predict.py
```

The script only uses `train.csv` for supervised fitting. It combines character
TF-IDF linear classifiers with lexical paragraph-question matching features and
writes `outputs/submission.csv`. The fixed random seed and deterministic rank
rule make the result reproducible.

Required packages: Python 3.10+, NumPy, pandas, and scikit-learn.
