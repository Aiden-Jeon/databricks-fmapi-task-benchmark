# BEEP classifier

The solution uses character `(1, 5)`-gram TF-IDF features and a class-balanced
linear SVM. Character features handle spacing variation, slang, and obfuscated
spellings without external tokenizers or pretrained resources. A small fixed
decision-score adjustment improves the under-predicted `offensive` class; all
settings were selected with stratified cross-validation on `train.csv` only.

From the workspace root, reproduce the submission with:

```bash
python solution/train.py
```

To also print the deterministic five-fold validation result before training:

```bash
python solution/train.py --cv
```

The script requires Python 3, pandas, NumPy, and scikit-learn. It writes
`outputs/submission.csv` and validates its columns, IDs, row count, and labels
against `sample_submission.csv`.
