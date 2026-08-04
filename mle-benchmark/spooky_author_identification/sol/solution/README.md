# Spooky Author Classifier

The solution trains two independent TF-IDF logistic regression models:

- word unigrams and bigrams for author-specific vocabulary and phrases
- character 3-5 grams for spelling, punctuation, morphology, and style

Their probabilities are combined with a normalized geometric mean. All
features and model parameters are learned only from `train.csv`.

Run from the workspace root:

```bash
python solution/train.py
```

This writes `outputs/submission.csv`. Paths and fitted hyperparameters can be
overridden with command-line options; run `python solution/train.py --help`
for details. The deterministic holdout parameter search can be reproduced with:

```bash
python solution/train.py --validate
```

The selected holdout log loss was 0.395347. The final `C=27` accounts for the
increase from 75% of the rows in holdout fitting to all available training rows.
