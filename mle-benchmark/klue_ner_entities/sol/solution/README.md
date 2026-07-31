# Solution

The submission is produced by a character-level BIO sequence tagger trained only on
`train.csv`. It uses hashed local character features, train-derived gazetteer features,
and Viterbi decoding with learned BIO transition scores.

Run from the task root:

```bash
python solution/train_predict.py
```

This deterministically writes `outputs/submission.csv`. A fixed local holdout can be
checked with:

```bash
python solution/train_predict.py --validate
```
