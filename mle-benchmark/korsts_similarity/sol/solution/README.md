# KorSTS solution

The solution uses multi-resolution word/character TF-IDF cosine similarity,
lexical overlap, length, number agreement, and coarse source-position features.
Two complementary tree regressors are blended with fixed weights. No external data or
pretrained model is used.

Run from the task directory:

```bash
python solution/train.py --validate
```

Omit `--validate` to train only the final models. Both commands write
`outputs/submission.csv` using a fixed random seed.
