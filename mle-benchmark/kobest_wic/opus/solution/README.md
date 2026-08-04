# KoBEST WiC — solution

Reproduce with (from the task root):

```bash
python solution/run.py      # ~25 s, writes outputs/submission.csv
```

Only `numpy` / `pandas` / `scikit-learn` are used (no torch/transformers are
installed in this environment, no internet, no external data or pretrained
weights). Everything is fit on `train.csv`; `test.csv` contributes only its
unlabelled texts and its `word` grouping.

## Approach

The prediction is the posterior of a small hierarchical model that combines two
independent sources of evidence.

### 1. Text model (`features.py`)
Logistic regression (C=0.3) on 45 features of the pair:

* **Structural / morphological**: the string glued to the target word in each
  context (Korean particle vs. space vs. derivational suffix), whether those
  match, the neighbouring tokens, the relative position of the target in the
  sentence, sentence lengths.
* **Lexical overlap**: token and character 2/3/4-gram Jaccard and containment
  between the two contexts.
* **Distributional similarity**: cosine similarity in four views — word and
  `char_wb` 2–4-gram TF-IDF, and 180-dimensional LSA (TruncatedSVD) of each.
  The vector spaces are fit unsupervised on all 6 636 contexts.

5-fold OOF accuracy: **0.602**. Sparse pair representations (intersection /
symmetric difference of n-gram sets), element-wise metric learning on the LSA
vectors, target masking, multi-view stacking, HGB/ExtraTrees and blends were all
tried and did not beat this; without pretrained Korean representations the
context signal saturates around 0.60.

### 2. Per-word count model (`count_model.py`)
The benchmark is constructed *per target word*: each word contributes a small
set of pairs (483 of the 775 words have exactly 3) whose positive/negative
counts are close to balanced. The number of positives per word is clearly
**under-dispersed** relative to a binomial — for T=3 words the estimated
distribution over the number of positives m is `[0.06, 0.43, 0.48, 0.04]`
(sd 0.65 vs. 0.87 for a binomial). So the labelled pairs of a word are
informative about its remaining pairs.

* `fit_prior_em` estimates P(m | T) by EM over all words, using the
  hypergeometric likelihood of the observed (k positives out of n labelled
  pairs) — sparse pair-counts back off to a discretised Gaussian whose
  dispersion factor is fitted by pooled ML (it comes out at the binomial value
  for T>=6, i.e. no signal is claimed there).
* Conditional on m all C(T, m) label configurations are exchangeable, so the
  posterior over the unknown labels z of a word is
  `P(z) ~ P_T(k + sum z) / C(T, k + sum z) * prod_i LR_i^{z_i}`
  with `LR_i` the tempered likelihood ratio of the text model. All `2^t`
  configurations are enumerated, giving exact marginals (and a joint constraint
  when a word has several test rows).

### Validation
Words with no rows in `test.csv` are *complete*: their whole original pair set
sits inside `train.csv`. Splitting their rows 80/20 therefore reproduces the
real situation exactly (T observable, prior fit on the retained part). Averaged
over 40 splits:

| model | accuracy |
|---|---|
| count model alone | 0.569 |
| text model alone | 0.607 |
| **combined (text tempering w=1.5)** | **0.625** |

An exhaustive leave-one-out check on the complete words confirms the direction
of the count signal (T=3, 2 labelled pairs: k=0 -> P(y=1)=0.73, k=1 -> 0.54,
k=2 -> 0.18), and the same pattern is visible in the emitted predictions.
