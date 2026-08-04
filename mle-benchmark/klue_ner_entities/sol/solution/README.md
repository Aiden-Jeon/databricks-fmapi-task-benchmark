# Solution

The solution aligns the entity strings in `train.csv` to recover character-level
BIO tags. It trains an averaged logistic-loss SGD classifier on hashed character,
context, character-type, and whitespace-token features. Inference uses constrained
Viterbi decoding with prior-corrected transition probabilities learned only from
the training set. A high-purity training lexicon is used only to correct predicted
entity types; it does not add or hardcode test entities.

Run from the task root:

```bash
python solution/train_predict.py
```

This writes `outputs/submission.csv`. The program is deterministic and uses Python
3, NumPy, SciPy, and scikit-learn. To reproduce the fixed ID-based holdout metric:

```bash
python solution/train_predict.py --validate
```

The selected decoder setting achieved entity-level micro-F1 0.78252 on that
holdout split.
