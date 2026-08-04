# KLUE-RE solution

The solution uses only `train.csv`. It combines word and character TF-IDF
features from entity-marked sentences, focused local context, entity names,
and entity identity features. A linear SVM performs the 30-class prediction.

Run from the workspace root:

```bash
python solution/train.py
```

This deterministically writes `outputs/submission.csv`. A fixed stratified
holdout evaluation can be reproduced with:

```bash
python solution/train.py --validate
```

Dependencies: Python 3, pandas, NumPy, SciPy, and scikit-learn.
