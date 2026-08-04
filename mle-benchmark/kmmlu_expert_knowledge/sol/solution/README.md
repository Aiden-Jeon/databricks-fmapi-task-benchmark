# KMMLU solution

The solution trains a word unigram/bigram TF-IDF model on the concatenated
question and answer choices, followed by multinomial logistic regression.
Only `train.csv` labels are used for fitting.

Run from the workspace root:

```bash
python solution/train_predict.py
```

The command writes `outputs/submission.csv` and validates its columns, IDs,
row count, and label range against the provided files.
