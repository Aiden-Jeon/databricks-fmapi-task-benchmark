# t6_klue_nli — solution

## How to reproduce

```bash
python solution/run.py             # train.csv -> outputs/submission.csv  (~1 min, 4 CPU cores)
python solution/run.py --validate  # held-out evaluation of the identical pipeline
```

Only `train.csv` is used. No internet, no external data, no pretrained weights.
Dependencies: the pre-installed `numpy / scipy / pandas / scikit-learn` (no GPU,
no `torch`/`transformers` available in this environment).

## Files

| file | content |
|---|---|
| `feats.py` | TF-IDF + hand-crafted feature construction |
| `decoder.py` | premise-group structured decoder + empirical label-multiset prior |
| `run.py` | end-to-end training / prediction / validation |

## Approach

**1. Row-level classifier.** Korean text, no morphological analyser available, so
tokens are whitespace-split with a crude particle stripper, combined with
character n-grams. Seven sparse TF-IDF blocks are used: hypothesis (word 1-3,
char_wb 2-5), premise (word 1-2, char_wb 2-4), the *novel* part of the hypothesis
(the tokens that do not appear in the premise — where negation, "only" and new
entities that decide the label live), and the shared part. On top of that, 31
dense features (word/char overlap ratios, length ratios, negation / "only" /
modality counts in premise vs hypothesis, numeral mismatches).
A multinomial logistic regression (C=0.5) yields `p(label | premise, hypothesis)`.

Held-out accuracy of this row-level model alone: **0.590**.

**2. Structured decoding over premise groups (the main win).** KLUE-NLI is
constructed by giving an annotator one premise and asking for three hypotheses —
one *entailment*, one *neutral*, one *contradiction*. Consequences measured on
`train.csv`:

* 8324 unique premises for 19998 rows; 4190 premises carry 3 rows and **90.2 %**
  of those triples contain each label exactly once.
* **4027 of the 5000 test premises also occur in `train.csv`**, i.e. most test
  rows have siblings whose gold label is known.
  Test rows by case: 3084 have 2 known siblings with *distinct* labels, 1588 have
  1 known sibling and 1 unlabeled sibling inside the test set, 159 form a
  3-row all-test group, 168 are other/no-information cases.

So instead of scoring rows independently, each premise group is decoded jointly:

```
argmax_y  sum_i log p(y_i | premise, hyp_i) + w * log Prior(multiset(y_unknown + y_known))
```

`Prior` is the *empirical* distribution of label multisets over training premise
groups of the same size, divided by the number of arrangements of the multiset
(so it is a proper prior over label sequences), Laplace-smoothed. It is estimated
from `train.csv` alone. It captures both the dominant "all three labels distinct"
pattern and the asymmetry of the rarer ones — e.g. a group whose two known labels
are both `entailment` is completed by `contradiction` 152 times but by `neutral`
only 11 times, which a hand-written "distinctness penalty" cannot express
(that case improves 0.762 -> 0.857). `w = 3` compensates for the
under-confidence of the logistic-regression probabilities and was selected on
held-out splits.

## Results (held-out, 5000 rows of `train.csv`)

| variant | held-out acc | re-weighted to test case mix |
|---|---|---|
| row-level logistic regression | 0.590 | — |
| + hard "labels are distinct" penalty | 0.824 | 0.883 |
| + empirical multiset prior (**submitted**) | **0.826** | **0.885** |

The held-out split has a different mix of group cases than `test.csv` (a random
25 % row split leaves fewer rows with two labeled siblings than `test.csv` has),
so the last column re-weights the per-case accuracies by the case counts actually
present in `test.csv`; it is the honest estimate of the submission's score.
The prior-vs-penalty gain (+0.002) reproduced on 3 independent splits.

Per-case held-out accuracy of the submitted pipeline:

| known siblings | unlabeled siblings | n | acc |
|---|---|---|---|
| 2, distinct labels | 0 | 1692 | 0.923 |
| 1 | 0 | 1217 | 0.763 |
| 1 | 1 | 1204 | 0.826 |
| 2, same label | 0 | 63 | 0.857 |
| 0 | 2 | 204 | 0.794 |
| 0 | 0 | 206 | 0.704 |

The 0.923 on the largest case is essentially the structural ceiling: ~7 % of
premise groups genuinely repeat a label, and those cannot be recovered from the
group structure.

## Things that were tried and rejected

* **Gradient boosting on "sibling" features** (similarity / negation difference
  between a row's hypothesis and its labeled siblings): row-level accuracy jumps
  0.59 -> 0.76, but the signal is the same structural one the decoder already
  exploits, and blending it in *lowered* the decoded score (0.883 -> 0.882).
* **Antisymmetric pairwise ranker** (`w·(phi_i − phi_j)` per label pair, to decide
  which of two same-premise hypotheses takes which label). It does beat the
  multiclass margins on that isolated decision (0.861 vs 0.849 over 793 groups)
  but the difference is ~10 groups and it did not survive end-to-end on any
  weighting, so it was dropped.
* C sweep for the logistic regression (0.1 … 2.0) and probability averaging over
  several C: flat within ±0.002 across splits; single C=0.5 kept.
