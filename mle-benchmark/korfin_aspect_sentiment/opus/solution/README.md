# KorFin-ASC — Aspect-Based Sentiment Analysis (macro F1)

## How to reproduce

From the task root directory (the one containing `train.csv`):

```bash
python solution/final.py     # ~11 min, CPU only, writes outputs/submission.csv
```

Dependencies: only `pandas`, `numpy`, `scipy`, `scikit-learn` (all pre-installed).
No internet access, no external data, no pretrained weights.

## Files

| file | purpose |
| --- | --- |
| `common.py` | aspect-aware text views + numeric features |
| `final.py` | final pipeline: CV, blend selection, full-train refit, submission |
| `exp_baseline.py` | round-1 search: which text view / which `C` (kept for the record) |
| `exp2.py` | round-2 search: neighbour encodings, model zoo, greedy blend |

## Approach

No Korean pretrained LM was available offline (no `torch`/`transformers`, no GPU), so the
solution is a strong sparse-feature linear ensemble made aspect-aware by construction.

**1. Aspect-aware text views** (`common.py`). The task is *aspect*-based: the same
sentence can carry different sentiment for different targets, so a plain
bag-of-ngrams over the sentence is inadequate. Five views are generated per row:

* `marked` — every occurrence of the aspect is wrapped in `《…》` markers, so n-grams
  crossing the marker encode "this sentiment word sits next to the target".
* `masked` — the aspect string is replaced by a single placeholder char `㋣`
  (aspect-identity agnostic view; generalises to unseen companies).
* `ctx` / `ctx_s` — ±30 / ±12 character window around each aspect occurrence, aspect masked.
* `clause` — only the clause (split on `.,;` and Korean verb endings) containing the aspect.
* `aspect` — the aspect string alone.

Each view is vectorised with `char_wb` 2–5-gram TF-IDF (Korean has no whitespace
tokenisation to rely on, so character n-grams beat word n-grams here: 0.660 vs 0.616
macro F1 for the `marked` view) plus word 1–2-grams where it helped. Concatenating
all views gives ~460k features and 0.699 OOF macro F1 — far above any single view.

**2. Neighbour / target encodings** (`enc_features`). 8 dense columns computed *only*
from the training rows of the current fold:

* smoothed label distribution of the *other* aspects in the same sentence (+ support count),
* smoothed label distribution of the same aspect elsewhere in the corpus (+ support count).

The exact `(sentence, aspect)` pair being encoded is always subtracted out, so no row
ever sees its own label. This is legitimate at test time: 716 / 1622 distinct test
sentences also occur in train (with *different* aspects — there are **zero** shared
`(sentence, aspect)` pairs), and aspects overlap 55 %. This is the single biggest win:
0.699 → 0.715 OOF.

**3. Model blend.** Rank-1 candidates chosen by 5-fold OOF macro F1:

| model | OOF macro F1 |
| --- | --- |
| LogisticRegression C=1 | 0.7145 |
| LinearSVC C=0.1 (softmax-scaled margins) | 0.7141 |
| HistGradientBoosting on 180-dim SVD + encodings | 0.6814 |
| **blend (4 linear variants + 0.5 × HGB)** | **0.7196** |

The gradient-boosting model is much weaker alone but decorrelated, so a 0.5 weight adds
a little. The blend is selected automatically from OOF scores, then every component is
refit on 100 % of `train.csv` before predicting `test.csv`.

Explicit macro-F1 class-prior re-weighting was evaluated on the OOF predictions and
rejected (+0.0015, i.e. within noise), so `argmax` is used as-is.

## Result

OOF (5-fold stratified) macro F1 = **0.7196**. Predicted test distribution
POSITIVE 676 / NEGATIVE 581 / NEUTRAL 507, close to the train prior — a sanity check
that no class collapsed.
