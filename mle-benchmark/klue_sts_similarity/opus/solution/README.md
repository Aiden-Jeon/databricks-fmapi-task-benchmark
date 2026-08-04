# t7_klue_sts — Korean sentence similarity (Pearson)

## Result

| model                                    | 5-fold OOF Pearson |
|------------------------------------------|--------------------|
| char TF-IDF cosine (linear rescale)      | 0.837 (in-sample)  |
| Ridge on dense features                  | 0.937              |
| SVR (RBF) on dense features              | 0.936              |
| MLP on dense features                    | 0.939              |
| ExtraTrees on dense features             | 0.950              |
| HistGradientBoosting on dense features   | **0.953**          |
| sparse pair-Ridge (char+word)            | 0.898              |
| sparse pair-Ridge (jamo+word bigram)     | 0.914              |
| stacked HGB (dense + sparse OOF)         | **0.959**          |
| **final NNLS blend**                     | **0.960**          |

`0.960` is the nested-CV number (blend weights refit inside an outer 5-fold
loop), so it is not weight-overfitted.

## Environment constraint

No `torch` / `transformers` / GPU is available in this workspace and downloading
pretrained weights is forbidden, so no pretrained Korean encoder was used.
Everything is built from scratch with numpy / scipy / scikit-learn on the
provided data only.

## Approach

### Feature block A — `feats.py` (473 features)
Fitted on the sentence corpus (train + test sentences, **labels never used**).

* 8 TF-IDF spaces (`char_wb` 2/2-3/2-4/3-5, `char` 1-5, word uni/bi, binary
  word) — for each: cosine, histogram-intersection, Dice.
* LSA (TruncatedSVD 256) on char and word spaces: cosine, L1/L2 distance,
  cosine restricted to the leading 8/16/32/64/128 dims, plus the elementwise
  `|u-v|` and `u*v` interaction vectors (96 dims each).
* Discrete overlap: Jaccard / containment on char 1..5-grams, raw tokens and
  crudely stemmed tokens (Korean particle/ending stripper), IDF-weighted token
  overlap.
* String similarity: `SequenceMatcher` ratio, longest matching block, LCS
  length (3 normalisations).
* Length statistics, digit-set agreement, question-mark agreement, negation-cue
  agreement, first/last token match.
* IDF-weighted greedy soft alignment between tokens in a char-n-gram token
  space (min / mean / hard-match rate).

### Feature block B — `feats2.py` (42 features)
* **Distributional word vectors learned from the provided corpus only**: binary
  sentence–term matrix → term–term co-occurrence → PPMI → TruncatedSVD(200).
  These capture in-domain synonymy that pure lexical overlap misses.
  Features: IDF-weighted centroid cosine, greedy max-alignment (min/mean/max),
  a hybrid alignment taking `max(word-vector sim, char-n-gram sim)`, and the
  residual unmatched IDF mass.
* Jamo (Hangul grapheme) decomposition → char 2-5-gram TF-IDF cosine /
  intersection / Dice and jamo n-gram Jaccard. Robust to inflection.
* Document LSA + KMeans(12) domain proxy: same-cluster flag, cluster
  membership, domain-space cosine.

### Sparse pair models — `sparse_model.py`, `sparse_model2.py`
For each TF-IDF space, the *pair* is represented as
`[A.minimum(B) , |A - B|]` (shared mass and unshared mass per term). A Ridge on
this lets the model learn *which* terms matter when they mismatch (numbers,
negation, content nouns) versus which do not (function words). Alone this is
weaker (0.90 / 0.91) but highly complementary.

### Stacking + blending
`stack_try.py` / `stack_try2.py` append the sparse models' OOF predictions as
two extra columns to the dense feature matrix and refit HGB/SVR on a *different*
fold split. This is the single strongest model (0.959).
`final_blend.py` z-scores every base model's OOF, fits non-negative least
squares weights against the standardised target, rescales the blend back onto
the 0–5 range with a linear fit on OOF, clips to [0, 5], and writes
`outputs/submission.csv`.

## Reproduce

```bash
cd <task root>
python solution/run_all.py          # ~30 min, CPU only
python solution/check_submission.py # format assertions
```

Individual stages:

```bash
python solution/cv.py               # dense block A  -> _cache.npz
python solution/build2.py           # dense block B  -> _cache2.npz
python solution/sparse_model.py     # -> _sparse.npz
python solution/sparse_model2.py    # -> _sparse2.npz
python solution/eval2.py ridge svr hgb hgb2
python solution/eval2.py et
python solution/eval2.py mlp mlp2 hgb_abs
python solution/stack_try.py        # -> _stack.npz
python solution/stack_try2.py       # -> _stack_stackb/c.npz
python solution/final_blend.py      # -> outputs/submission.csv
```

Helper scripts: `baseline.py` (quick cosine baseline), `eval.py` /
`tune_hgb.py` / `blend.py` (exploration on block A only).

## Notes on validity
* No internet, no external data, no pretrained weights.
* Only `train.csv` labels are used for fitting any supervised component.
* Test *sentences* (not labels) are used for fitting the unsupervised
  vectorizers / SVD / KMeans / PPMI embeddings — a transductive but label-free
  step. Removing it changes CV by <0.001; it only improves vocabulary coverage.
* Test predictions are 5-fold bagged averages of models trained on train folds.
