# KoBEST COPA — solution

Reproduce with:

```bash
cd solution
python run.py ..          # writes ../outputs/submission.csv
python run.py .. --cv     # same + 5x5 repeated stratified CV report
```

Only `numpy / scipy / scikit-learn` are used (no external data, no pretrained
weights, no internet). Runtime: ~10 s for the submission, ~3 min with `--cv`.

## Approach

The task is scored per row, so every model is a **pairwise** linear model:
features are built per alternative and the design matrix is
`f(alternative_2) - f(alternative_1)`; a logistic regression then predicts the
label directly (an intercept absorbs the mild 54/46 position prior in train).

Two complementary families are blended.

### 1. Answer-only models (~0.59 CV each)
TF-IDF of the alternatives (char / char_wb 2–4-grams, word-stem tokens) plus a
small dense block (premise↔alternative cosine/Jaccard overlap, length, negation
cue, question-type interactions). These pick up distractor-construction
artefacts: distractors are recognisable from their surface form alone.

### 2. Association models (~0.60–0.63 CV each) — the main signal
A directed **cause → effect co-occurrence matrix** over sub-word units
(character bigrams, 2-char stems, syllables) is estimated from the *correct*
`(premise, alternative)` pairs of the training split, using the `question`
column to orient each pair (`원인` ⇒ the alternative is the cause, `결과` ⇒ the
premise is the cause). The matrix is turned into PPMI and factorised with
truncated SVD, giving separate "cause" and "effect" embeddings per unit. A
candidate alternative is then scored by

* the normalised raw PPMI mass between its units and the premise units, and
* the cosine between the summed cause-side and effect-side embeddings.

Those scores (difference between the two alternatives) feed a logistic
regression. This is the commonsense-plausibility component: it learns, e.g.,
that units of "지루" go with units of "껐다" in the cause→effect direction.

### Blending
Member decision values are z-normalised, averaged within each family, and the
two families are summed with weight `1 : 2` for (answer-only : association) —
the CV optimum is flat over 1.6–2.5. The sign of the blend is the prediction.

## Fold safety
Every label-dependent statistic (co-occurrence counts, PPMI, SVD basis,
regression weights) is estimated on the training part of each CV split only.
Vectorisers / scalers are unsupervised and fit on all available text
(train + test premises and alternatives), which uses no labels.

## Results (5×5 repeated stratified CV on train.csv)

| model                                  | accuracy |
|----------------------------------------|----------|
| majority class baseline                | 0.543    |
| answer-only family only                | 0.600    |
| best single association model          | 0.625    |
| **final blend (submitted)**            | **0.661 ± 0.004** |
