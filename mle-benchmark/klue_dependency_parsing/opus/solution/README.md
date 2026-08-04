# t25_klue_dp — Korean dependency parsing (KLUE-DP)

## Reproduce

```bash
python solution/train2.py --dev 0 --epochs 16 --lepochs 8 --seeds 1 --minwf 2 \
                          --out outputs/submission.csv
python solution/validate.py          # format / tree-well-formedness check
```

Runtime ≈ 2 min, ≈ 11 GB RAM, CPU-only, numpy + pandas only (no torch/GPU, no internet,
no external data or pretrained weights). Only `train.csv` is used for supervision.

## Approach

### 1. Exploiting the structure of the label space (`solution/explore.py`)

Measured on all 4800 training trees:

| property | value |
|---|---|
| arcs with `head > dependent` (head-final) | **100.0 %** |
| projective trees | **100.0 %** |
| exactly one root | **100.0 %** |
| root is the last eojeol | **100.0 %** |

The last fact is in fact forced: if every head lies to the right, the last word can have
no head, so it must be the root. So the output space is exactly

> projective dependency trees over `1..n` in which every head is to the right of its
> dependent (equivalently: every subtree spans a contiguous range ending at its head).

Restricting the decoder to that space removes a huge amount of the search space for free
and guarantees every prediction is a well-formed single-rooted tree.

### 2. Exact O(n³) decoder (`dp_lib.decode`, `dp_lib.decode2`)

Reversing the word order turns head-final trees into head-initial ones rooted at
position 0, which admits a very compact CKY-style recursion. With `C[i][k]` = best
complete subtree rooted at `i` covering the reversed span `[i,k]`:

```
C[i][i] = 0
C[i][k] = max_{d in (i,k]}  X[i][d] + C[d][k] + arc(d -> i)
X[i][i+1] = valency(i)                                        # i has >= 1 dependent
X[i][d]   = max_{d' in (i,d)} X[i][d'] + C[d'][d-1] + arc(d' -> i) + sib(i, d', d)
```

`C[i][k]` splits off the *last* child `d` of `i`; `X[i][d]` walks the chain of `d`'s
already-attached nearer siblings, which is what makes **adjacent-sibling** (second-order)
and **valency** factors expressible while staying O(n³).

Both decoders were verified to reconstruct all 4800 gold trees exactly from oracle
arc scores (`solution/sanity.py`, and the `decode2` check inline in the transcript).

### 3. Features

No POS tags or morphological analyses are provided, so the morphology is approximated by
**character suffixes of each eojeol** (last 1–4 chars), which in Korean carry the case
markers / verbal endings that determine dependency structure, plus prefixes, a shape
signature (trailing punctuation, length, has-digit, is-latin) and the word form itself.
Word forms occurring `< 2` times in train back off to a suffix pseudo-word (`--minwf`).

* **arc factor** (`arc_feats`, ~55 templates): dependent & head unigrams, their
  conjunctions, distance buckets, sentence-relative position, the immediate left/right
  neighbours of both endpoints, and "in-between" features summarising which kinds of
  words are skipped over.
* **sibling factor** (`sib_feats`, 11 templates) on (head, nearer child, farther child) —
  captures valency/competition, e.g. a verb that already has a nominative child.
* **valency factor** (`fc_feats`, 5 templates) fired once per head that has any dependent.
* **label factor** (`lab_feats`, ~38 templates) on the (dependent, chosen head) pair.

Feature strings are interned into a dict (~17 M features). Crucially, the feature-index
matrices for *all* candidate arcs/triples are cached once (row index computed in closed
form: `C(h,3)+C(c1,2)+c2` for triples), so a training epoch is only numpy gathers + the
DP and takes ~4 s over 4800 sentences.

### 4. Training

Heads: **cost-augmented MIRA** (structured hinge). Inference during training adds +1 to
every non-gold arc, and the step size is the MIRA closed form
`eta = clip((loss - w·(phi(y)-phi(y_hat))) / ||phi(y)-phi(y_hat)||², 0, C)`.
Weights are averaged (lazily, per feature).

Labels: multiclass margin-augmented averaged perceptron over the *decoded* arcs, so the
labeller sees the same head distribution at train and test time.

## Development results (480-sentence held-out split, ~5.5 k tokens)

| model | UAS | LAS |
|---|---|---|
| trivial baseline (head = next word, label by suffix) | 0.664 | – |
| 1st-order averaged perceptron | 0.802 | 0.751 |
| + 2nd-order sibling/valency factors | 0.802 | 0.755 |
| + cost-augmented MIRA training | **0.814** | **0.763** |
| 3-seed weight average | 0.813 | 0.758 |

Cost-augmented MIRA was the decisive change: the plain perceptron reached 99.99 % train
accuracy within 4 epochs and simply memorised, whereas the margin objective keeps
updating and generalises ~1.2 UAS / 1.1 LAS better. The seed ensemble only reproduced the
mean of its members, so the final submission is a single model trained on all 4800
sentences.

Remaining error profile (`solution/diag.py`): distance-1 attachments are 96 % correct
while distance ≥ 3 attachments are 30–47 % correct, and label accuracy given a correct
head saturates at 0.93 (dominant confusions `NP_AJT/NP`, `NP_CNJ/NP`, `VNP/VP`). Both
limits stem from having no morphological analysis of the eojeol — the natural next steps
would be induced word clusters or a proper morpheme segmenter, neither of which fit in
the time budget.

## Files

| file | role |
|---|---|
| `dp_lib.py` | token attributes, feature templates, exact 1st/2nd-order DP decoders, data loading |
| `train2.py` | **final model**: cost-augmented MIRA + label perceptron -> `outputs/submission.csv` |
| `train.py` | earlier first-order perceptron model (kept for reference / ablation) |
| `explore.py` | dataset structure analysis (head-finality, projectivity, root position) |
| `sanity.py` | oracle-decode test + trivial baseline submission |
| `diag.py` | arc-error breakdown by distance / sentence length |
| `lab_exp.py` | fast standalone experiments on the labelling model |
| `validate.py` | submission format + tree well-formedness validator |
