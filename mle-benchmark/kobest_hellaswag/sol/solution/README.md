# KoBEST HellaSwag solution

The model learns pairwise preferences between the correct ending and each
incorrect ending. It combines word/character TF-IDF features with lexical
continuity features between the context, its final sentence, and the four
candidate endings. No external data or pretrained weights are used.

Run from the workspace root:

```bash
python solution/train.py
```

This reads `train.csv` and `test.csv` and writes
`outputs/submission.csv`. The run is deterministic. To reproduce the model
selection score before fitting the final model, run:

```bash
python solution/train.py --cv
```

Required packages: Python 3.10+, numpy, pandas, scipy, and scikit-learn.
