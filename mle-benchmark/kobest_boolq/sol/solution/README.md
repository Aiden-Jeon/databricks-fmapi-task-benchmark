# KoBEST BoolQ solution

The solution uses only `train.csv`. It augments each question with trainable
paragraph/question relation markers (negation polarity and number mismatch),
then ensembles character and word TF-IDF linear SVM models. The ensemble scale
is estimated with deterministic five-fold out-of-fold predictions.

Run from any directory:

```bash
python solution/train_predict.py
```

The default output is `outputs/submission.csv`. Python 3.10+, NumPy, and
scikit-learn are required.
