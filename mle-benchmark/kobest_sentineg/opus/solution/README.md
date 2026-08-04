# KoBEST SentiNeg — solution

Binary sentiment classification of short Korean product-review sentences that
often contain negation. 2,919 train / 730 test rows, metric = accuracy.

## Environment constraint
Only the sklearn/numpy/pandas/scipy stack is available (no `torch`,
`transformers`, `lightgbm`), and no internet / pretrained weights are allowed.
The solution is therefore a **character-level TF-IDF ensemble**.

## Features
Every sentence is turned into two text views (`common.make_views`):

| view   | description |
|--------|-------------|
| `norm` | NFC normalisation, char repeats collapsed to 2 (`ㅋㅋㅋㅋ`→`ㅋㅋ`), whitespace squeezed |
| `jamo` | the `norm` text with each Hangul syllable decomposed into cho/jung/jong jamo (`않` → `ㅇㅏㄶ`), a `-` placeholder marks a missing final consonant so positions stay aligned |

The jamo view lets n-grams share the negation morphemes (`-지 않-`, `안-`,
`못-`, `없-`) across different inflected endings, which is exactly the signal
this dataset hinges on. It was worth ~+0.4 pt over syllable n-grams alone.

TF-IDF blocks used (all `sublinear_tf=True`):
`char(1-4)/norm`, `char(1-5)/norm`, `char_wb(2-5)/norm`,
`char(2-6)/jamo`, `char(2-8)/jamo`, `word(1-2)/norm`.

## Model
Weighted average of z-scored decision functions of 5 diverse estimators
sharing the feature space (`train_predict.BLEND`):

| member    | features | estimator | weight | OOF acc |
|-----------|----------|-----------|--------|---------|
| `G_rbf`   | char(1-4)+jamo(2-6) | `SVC(rbf, C=10)` | 2 | 0.9593 |
| `E_ridge` | char(1-4)+jamo(2-6) | `RidgeClassifier(alpha=0.5)` | 2 | 0.9582 |
| `I_lr_wb` | char_wb(2-5)+jamo(2-6) | `LogisticRegression(C=3)` | 1 | 0.9555 |
| `D_svc`   | char(1-5)+jamo(2-8)+word(1-2) | `LinearSVC(C=0.3)` | 1 | 0.9578 |
| `J_nbsvm` | char(1-4)+jamo(2-6) | NB-SVM (NB log-count-ratio scaling + `LinearSVC`) | 1 | 0.9557 |

Weights were picked by greedy forward selection on out-of-fold scores and then
re-checked as a fixed blend.

## Validation (4x repeated stratified 5-fold = 20 folds)

| model | accuracy |
|---|---|
| plain char(2-5) TF-IDF + LR (first baseline) | 0.9466 |
| + text normalisation, char(1-4) | 0.9528 |
| best single member (`G_rbf`) | 0.9593 ± 0.0006 |
| **final blend (submitted)** | **0.9613 ± 0.0009** |

## Things tried that did **not** help (kept out)
- whitespace-removed char n-grams (−0.2 pt)
- word-only TF-IDF (0.884), `SGDClassifier`, `ComplementNB`, cosine kNN (0.910)
- decision-threshold shifting (0 is already optimal despite errors being ~75 %
  false negatives — the model over-weights negation cues in positive sentences
  like `안 부러져요`, `상하지 않았어요`)
- transductive self-training / pseudo-labelling on test (+0.0005, within noise)
- nearest-neighbour label propagation: checked for the negated-pair structure of
  the original SentiNeg corpus, but this split contains no cross-split pairs
  (all >0.75-similarity neighbours share the same label), so there is nothing
  to exploit — only genuine generalisation counts.

## Reproduce
```bash
cd solution
python train_predict.py     # fits on train.csv -> ../outputs/submission.csv (~1 min)
python experiment.py        # OOF evaluation of the whole model zoo + blend search (~6 min)
```
Both scripts are deterministic (fixed `random_state` everywhere) and read only
`../train.csv` / `../test.csv`.
