# KorNLI solution

Run from any directory with:

```bash
python solution/train_predict.py
```

The script trains only on `train.csv` and writes `outputs/submission.csv` in
the exact row order of `sample_submission.csv`. It uses hypothesis character
TF-IDF features, premise-hypothesis relation features, and a linear SVM.
