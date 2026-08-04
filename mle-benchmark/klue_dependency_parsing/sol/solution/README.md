# KLUE-DP solution

The solution trains three hashed sparse linear models using only `train.csv`:

- a binary scorer that ranks every permitted `(dependent, rightward head)` arc;
- a multiclass dependency-relation classifier on gold training arcs.
- a root-relation classifier for the sentence-final root token.

The rightward-head and sentence-final-root constraints are learned from the global
training structure and guarantee an acyclic, single-root tree. Token prefixes,
suffixes, neighboring tokens, arc distance, and sentence position are model
features. There are no test-row-specific rules.

Run from the task directory:

```bash
python solution/train_and_predict.py
```

To additionally print deterministic holdout metrics before fitting all data:

```bash
python solution/train_and_predict.py --validate
```

The generated file is `outputs/submission.csv`.
