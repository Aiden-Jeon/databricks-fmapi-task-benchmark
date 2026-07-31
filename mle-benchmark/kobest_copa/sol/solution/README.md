# KoBEST COPA solution

The solution treats each alternative as a candidate and trains a character
TF-IDF logistic-regression classifier on the two candidates from every training
row. A row's score is the score of `alternative_2` minus that of
`alternative_1`.

Five-fold out-of-fold predictions calibrate the fraction of rows assigned to
label 1. The final model is then fit on all training rows and applies that
fraction to the ranked test scores. No external data or pretrained weights are
used.

Run from any directory with:

```bash
python solution/train_predict.py
```

Required packages: `numpy`, `pandas`, and `scikit-learn`.
