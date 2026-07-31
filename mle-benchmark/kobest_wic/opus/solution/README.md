# KoBEST WiC — solution

## How to run
```bash
cd solution
python run.py          # prints CV scores and writes ../outputs/submission.csv
```
Only `numpy`, `pandas`, `scipy`, `scikit-learn` are required (no GPU, no
pretrained weights, no internet — none were available in this environment).
Runtime ≈ 80 s on 4 CPU cores. Caches the feature matrix in
`content_cache.pkl` (delete it to force a rebuild).

## Files
| file | content |
|---|---|
| `run.py` | final pipeline: CV + full fit + submission |
| `content.py` | builds the pair-level content/similarity feature matrix |
| `feats.py` | text normalisation, Korean particle stripping, TF-IDF/LSA representations |
| `feats2.py` | collocation extraction + PPMI-SVD word embeddings (built from the task corpus only) |
| `prior.py` | per-word label-prior features (fold-honest / leave-one-out) |
| `build.py` | data loading and the context table (each context gets an id) |
| `experiments/` | exploratory CV scripts and their logs (feature sweeps, sense clustering) |

## Method
The environment had **no deep-learning stack and no pretrained Korean LM**, so a
transformer cross-encoder (the usual approach for WiC) was impossible. The
solution is therefore a feature-based linear/GBM blend built on two independent
signal sources.

**1. Content similarity between the two contexts** (`content.py`)
* char-`wb` 2–4-gram TF-IDF cosine, stemmed word-unigram TF-IDF cosine
* LSA (TruncatedSVD-200) cosine of both of the above
* PPMI-SVD word-embedding cosine (embeddings trained on the 6 636 sentences of
  this task only)
* each of the above computed on the whole sentence *and* on a ±20-character
  window around the marked target word
* each similarity additionally **normalised within the target word**
  (z-score and percentile against all context pairs of that word), which makes
  values comparable across words of different topical spread
* surface / collocation features: length and token-count differences, target
  position, particle following the target, first characters of the neighbouring
  eojeol, light-verb markers (하/되/받/…), ellipsis / hanja markers

**2. Per-word label prior** (`prior.py`) — the biggest single gain.
Pairs that share a target word are constructed to contain a **mix** of
same-sense and different-sense examples: among words with exactly 3 pairs,
91 % have both labels present, versus 75 % expected under independence. The
labels of the *other* pairs of the same word are therefore informative and
negatively correlated with the current one (e.g. if the two known pairs of a
3-pair word are both label 1, the remaining one is label 1 only ~25 % of the
time). Features: neighbour counts `n1/n0`, number of known labels `k`, total
number of pairs of the word (known from train+test), an analytic
hyper-geometric conditional probability with an empirically fitted prior over
"#positives per word", and a smoothed empirical target-encoding table.
All of these are computed **leave-one-out** for train rows and only from
training-fold labels inside CV, so the reported CV is honest.

**Model.** Blend (simple average) of three L2 logistic regressions
(C = 0.01/0.02/0.05 on standardised features) and a gradient-boosting
classifier. Evaluated with 3 × 5-fold stratified CV.

## Results (3×5-fold CV accuracy on train)
| model | accuracy | AUC |
|---|---|---|
| majority class | 0.506 | — |
| content features only | 0.599 | 0.643 |
| content + per-word prior, LR C=0.01 | 0.623 | 0.676 |
| **final blend (3×LR + GB)** | **0.626** | **0.676** |

## Things that were tried and did *not* help
* **Constrained sense clustering** per word (signed Ising / correlation
  clustering over all contexts of a word, train labels as must-link /
  cannot-link constraints, `experiments/senseclust.py`): AUC 0.625, i.e. no
  better than the raw similarity it is built from. Reason: *every context in
  the dataset occurs exactly once*, so the label graph is a perfect matching
  with no transitivity to exploit — the only cross-pair link is content
  similarity, which is weak for one-sentence contexts that often share no
  content words at all.
* **Kernel label propagation** across pairs of the same word — anti-correlated
  with the target (AUC 0.43), dropped.
* **Learned diagonal metric** (logistic regression on `[z1*z2, |z1-z2|]` of LSA
  vectors) and **logistic regression on the sparse element-wise minimum of the
  TF-IDF vectors** (learned per-token "sharing" weights): AUC 0.55–0.59, worse
  than plain cosine and redundant with it.
* **id-order / index leakage**: none (ids are shuffled, AUC 0.497).
* Exact per-word label balance: does not hold, only the softer "mixed labels"
  tendency described above.
