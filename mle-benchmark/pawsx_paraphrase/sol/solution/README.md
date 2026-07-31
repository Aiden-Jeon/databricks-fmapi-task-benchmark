# PAWS-X Korean solution

Run from the task root:

```bash
python solution/train.py
```

The script uses deterministic lexical and word-order features with an ensemble of
scikit-learn tree models. It writes `outputs/submission.csv` without requiring
network access, external data, or pretrained weights.
