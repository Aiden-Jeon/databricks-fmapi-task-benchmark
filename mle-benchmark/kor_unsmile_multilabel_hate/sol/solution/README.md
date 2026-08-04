# UnSmile solution

The solution uses character-boundary TF-IDF features and ten independent linear SVM classifiers.
Five stratified folds provide out-of-fold decision scores for selecting each label's F1-optimal
threshold. Test decision scores are averaged across the same five models.

Run from the workspace root:

```bash
python solution/train.py
```

The script uses only `train.csv`, `test.csv`, and `sample_submission.csv`, fixes all randomized
splits, validates the input/output schemas, and writes `outputs/submission.csv`.
