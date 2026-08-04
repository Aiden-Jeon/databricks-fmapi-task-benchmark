# t23_korfin_asc — Aspect-Based Sentiment Analysis (KorFin-ASC)

Metric: **macro F1** (3 classes). Final out-of-fold CV: **0.7562**
(baseline single TF-IDF model: 0.622).

No pretrained weights, no external data, no internet resources are used: every model is
trained from scratch on `train.csv` only (scikit-learn + LightGBM).

## Run

```bash
python solution/run_all.py            # uses solution/cache if present
python solution/run_all.py --fresh    # full rebuild (~25 min, CPU only)
```
Output: `outputs/submission.csv` (`id,label`, one row per test id).

## Approach

### 1. Aspect-aware text views (`common.py`, `views.py`)
The same sentence carries different sentiment for different aspects, so every view is
built *relative to the aspect*:

| view | description |
|---|---|
| `masked` | aspect string replaced by `@ASP@` |
| `win`, `win15` | ±30 / ±15 character context around every aspect occurrence |
| `text`, `text2` | concatenation of masked + windows + the aspect string |
| `jamo`, `cj_jamo` | Hangul decomposed into jamo (pure-python, catches morphology) |
| `dir` | left context vs. right context kept as *separate* feature blocks (Korean predicates follow the subject) |
| `clause`, `cj` | only the clause(s) containing the aspect (split on `,`/`지만`/`며`/`고`/`면서`/`반면`) |
| `stripped` | heuristic josa (particle) stripping for the word-level view |

Each view is vectorised with `char_wb` (2–6) and word (1–3) TF-IDF.

### 2. Base model bank (`oof.py`, `views.py`) — 33 models, 5-fold OOF
LogisticRegression / LinearSVC / RidgeClassifier / SGD / Complement-&Multinomial-NB
over the views above, plus LightGBM on SVD(250) of the TF-IDF matrix.
Best single model: `lr_multi3` = **0.7126** (multi-view TF-IDF + LogReg).
Greedy blend of base models: 0.678 (`blend.py`).

### 3. Group structure features (`stack2.py`)
44 % of test sentences also occur in `train.csv` with *other* aspects, and aspects repeat
across sentences. Fold-aware (never using the row's own label) target encodings:

* aspect target encoding, exact and normalised, with counts and Laplace smoothing
* fuzzy aspect encoding (aspect is a substring of / contains another aspect)
* sibling encoding: label distribution of the other aspects of the same sentence
* nearest-sibling label distribution + normalised character distance
* hand-crafted numerics: aspect position/count, sentence length, digits, `%`,
  aspect index from the id suffix, transductive sentence group size

### 4. Leak-free sibling predictions (`sibpred.py`, `stack3.py`, `stack4.py`)
Sibling *predictions* (useful when a sibling has no label, e.g. it lives in the test set)
must not be produced by a model that saw the row itself — otherwise the model memorises
the almost identical sentence and the row's label leaks through its sibling.
Naively using the random-fold OOF probabilities inflated CV from 0.7335 to 0.8073.
Instead each **sentence** (train *and* test) is assigned to one of 5 sentence-folds and
predicted by a model trained without any row of that sentence, so train and test rows are
treated identically. The honest gain of the sibling-prediction features is small (+0.002).

### 5. Stack (`final.py`)
LightGBM (4 configs screened, 2 best × 5 seeds averaged) over
[33×3 base probabilities ‖ target-encoding features ‖ numerics ‖ sentence-group-OOF
self prediction], blended with a LogisticRegression stack (weight 0.1).
A per-class multiplier search for macro-F1 is run but only applied if it gains
> 0.004 OOF (it did not, so multipliers stay at 1).

## Validation notes
* All CV numbers use the same `StratifiedKFold(5, seed=42)` split.
* Random-fold CV (rather than group-by-sentence CV) is the right proxy here: at test time
  the model is trained on the full `train.csv`, and 44.5 % of test rows have their
  sentence present in train — almost exactly the 40–44 % rate a random fold split
  reproduces. Group CV would systematically underestimate the test score.
* Score progression: 0.6221 (char TF-IDF + SVC on raw sentence) → 0.6572 (aspect-aware
  views) → 0.6781 (base blend) → 0.7283 (stack + group target encoding) → 0.7354
  (leak-free sibling predictions) → 0.7507 (directional/clause/stripped views) →
  **0.7562** (extended base bank + multi-seed stack).

## Files
`common.py` features · `oof.py`/`views.py` base models · `blend.py` greedy blend ·
`stack2.py` target-encoding stack · `stack3.py` leakage diagnosis · `sibpred.py`
sentence-group-OOF predictions · `stack4.py` leak-free stack · `final.py` final model ·
`make_submission.py` writer · `run_all.py` driver ·
`baseline.py`/`stack.py` early exploratory scripts.
