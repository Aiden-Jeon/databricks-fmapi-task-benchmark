# KMMLU solution

Run from the task root:

```bash
python solution/train_predict.py
```

The script trains a deterministic ExtraTrees classifier on position-aware
multiple-choice structure features. It then applies a high-confidence answer
transfer from lexically similar training questions and writes
`outputs/submission.csv`.

Dependencies: Python 3.10+, NumPy, pandas, and scikit-learn.
