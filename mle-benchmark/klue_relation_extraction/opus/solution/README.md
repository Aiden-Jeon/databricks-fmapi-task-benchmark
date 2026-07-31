# KLUE-RE (관계 추출) — sklearn-only solution

No GPU, no torch/transformers and no internet were available in this
environment, so the solution is a heavily feature-engineered **sparse linear
model ensemble** (scikit-learn only, `train.csv` as the sole data source).

## Reproduce

```bash
cd solution
python run.py --models svc015:1,svc02:0.7,svc025bal:1.5 --neigh_w 1.0
# -> ../outputs/submission.csv   (~3 min, 4 CPU cores, ~6 GB RAM)
```

## Approach

### 1. Entity-aware text views (`features.py`)
Both entities are always locatable in the sentence, so each row is rendered
into several text views that are separately TF-IDF vectorised:

| view | content |
| --- | --- |
| `tmarked` | full sentence with subject/object replaced by type-tagged markers `Ⓢ<TYPE>` / `Ⓞ<TYPE>` |
| `between` | text between the two entities, direction preserved |
| `xbtw` | `between` tokens prefixed with the entity-type pair → explicit *type × pattern* interactions for the linear model |
| `sctx` / `octx` | ±2 whitespace-token window around each entity (captures Korean postpositions such as `…의`, `…에서`) |
| `left` / `right` | 25-char outer context |
| `subj` / `obj` / `pair` | entity strings, plus the exact `subject‖object` cross string |
| `sentence` | raw sentence |

A coarse rule-based **entity type** (`DAT / NOH / PER / ORG / LOC / POH`) is
derived from suffix gazetteers and regexes, since KLUE-RE labels are strongly
constrained by entity types. 34 dense numeric features (span distance, order,
date/number/latin/hanja flags, suffix flags, …) are appended.

### 2. Relational "neighbour" features (`neighbors.py`)
The single largest gain (**+0.9 % accuracy**). Train/test share many entity
pairs (58 %) and sentences (17 %), so for every row we aggregate the labels of
*other* labelled rows that are related to it, bucketed by relation kind:
`pair`, `rev` (reversed pair), `ssub`, `sobj`, `scross`, `sent`, `gsub`, `gobj`.
Self-rows are always excluded, so the block is leak-free even when fitted on
the rows it describes. Motivation: a reversed pair labelled
`org:top_members/employees` implies `per:employee_of` 75 % of the time.

### 3. Model
`LinearSVC` on the ~1.15 M-dimensional block-weighted sparse matrix,
blended as a weighted sum of z-scored decision functions:

```
1.0 * LinearSVC(C=0.15) + 0.7 * LinearSVC(C=0.2) + 1.5 * LinearSVC(C=0.25, class_weight='balanced')
```

## Measured results (20 % stratified holdout)

| configuration | seed 42 acc | seed 7 acc |
| --- | --- | --- |
| LinearSVC, text features only | 0.7506 | 0.7481 |
| + neighbour features | 0.7612 | — |
| + blend (final) | **0.7629** | **0.7537** |

The final model is refit on all 25,976 training rows.

## Things that were tried and rejected (evidence-based)
- `class_weight='balanced'` alone: better macro-F1 (0.61 vs 0.57) but worse accuracy → only used as a blend member.
- `ComplementNB` / `MultinomialNB` (0.70), cosine-kNN (0.62), `SGDClassifier` (0.71–0.74): too weak, hurt the blend.
- `RidgeClassifier`: did not converge in reasonable time at this dimensionality.
- Per-class bias calibration: gained 0.7 % on the tuning half but *lost* 0.3 % on the held-out half → discarded.
- Single-block ablation: all deltas except `obj_c` (−0.012) were within holdout noise, so no blocks were pruned.
- Transductive pseudo-labelling: only ~5–6 % of test rows share a sentence/reversed pair with another test row, so the expected gain was below measurement noise.

## Files
- `features.py` — text views, entity typing, dense numeric features
- `neighbors.py` — leak-free relational label features
- `model.py` — vectoriser definitions, block weights, `FeatureBuilder`
- `run.py` — final train + predict entry point
- `dev3.py`–`dev8.py` — holdout harnesses used for ablation / model zoo / sweeps
- `*.log` — raw experiment output backing the tables above
