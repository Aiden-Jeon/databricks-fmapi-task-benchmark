# KLUE-NLI solution

Run from the task directory:

```bash
python solution/train_predict.py
```

The script trains only on `train.csv` and writes `outputs/submission.csv`.
It combines word/character TF-IDF features, pairwise lexical features, a linear
SVM, and a group constraint for repeated premises in the dataset.
