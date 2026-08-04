# KorQuAD extractive QA solution

This solution trains only on `train.csv`. It learns a hashed sparse linear ranker over
candidate character spans, using question type, sentence relevance, lexical position,
span shape, and Korean answer-boundary features. No external data or pretrained model
is used.

Run from the task root:

```bash
python solution/train_predict.py
```

The command writes `outputs/submission.csv`. A context-grouped holdout check is available
with `python solution/train_predict.py --validate --epochs 1`.
