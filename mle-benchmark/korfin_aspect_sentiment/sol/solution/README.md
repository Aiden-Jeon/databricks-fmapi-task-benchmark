# KorFin-ASC solution

The solution uses target-marked character TF-IDF features and a one-vs-rest
NB-SVM classifier. A 25-character context on either side of the aspect is
repeated so that target-local evidence has more weight than unrelated parts of
the sentence. Five stratified fold models and one full-data model are averaged.

Run from any directory with:

```bash
python solution/train_predict.py
```

The script reads the three CSV files from the workspace root by default and
writes `outputs/submission.csv`. Paths can be changed with `--data-dir` and
`--output`.
