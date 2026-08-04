# NSMC solution

The solution is an ensemble of three TF-IDF `LinearSVC` text classifiers:

- raw character 1-5 grams (weight 0.7)
- whitespace-normalized character 2-6 grams (weight 0.2)
- word 1-2 grams (weight 0.1)

All vectorizers and classifiers are fitted only on `train.csv`. Run from any
directory with:

```bash
python solution/train_predict.py
```

The default output is `outputs/submission.csv`. Paths can be changed with
`--train`, `--test`, and `--output`.
