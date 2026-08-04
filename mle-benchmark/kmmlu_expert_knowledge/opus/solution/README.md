# t21_kmmlu — Korean professional-knowledge 4-choice MCQ

Constraint: no internet, no pretrained weights, only `train.csv` (4,800 questions).
Learning actual Korean domain knowledge from 4.8k examples is not feasible, so the
solution targets the *statistically learnable* structure of the item set.

## Reproduce

```bash
cd solution
python run.py            # writes ../outputs/submission.csv  (~40 s, CPU only)
python validate.py       # single 5-fold CV of every component
python validate_repeat.py# 3 x 5-fold CV of the final fixed configuration
```

Dependencies: numpy, pandas, scikit-learn, scipy (all preinstalled).

## Model

Each question is expanded into 4 rows (one per option); a binary model scores
"is this the correct option", scores are z-normalised **within each question** and
the argmax becomes the label.

Features (95, `feats.py` + `pipeline.py`):

1. Option position one-hot — the label prior is strongly non-uniform
   (A 9.1 %, B 30.4 %, C 29.5 %, D 31.1 %), which is the single largest signal.
2. Length / word-count, rank and deviation from the other options.
3. Numeric-option features: all-numeric flag, rank of the numeric value among the
   four options (extreme vs middle value), digit counts, units.
4. Character-n-gram Jaccard overlap between question and option, and among the
   options (centrality / "odd one out").
5. Korean lexical cues (있다/없다/않다/모두/항상/반드시/증가/감소 …), each also
   **interacted with question polarity** (`옳지 않은/아닌/틀린` vs `옳은/맞는`), since
   e.g. an option containing 있다 is much less likely to be the answer of a
   "which is NOT correct" item.
6. Unsupervised LSA (char tf-idf 2–4 grams + 200-dim SVD, fitted on train+test text,
   labels never used) cosine similarity between question and option.
7. Retrieval features: for the 25 most similar training questions, similarity of the
   option to their *correct answers* minus similarity to their *distractors*
   (leave-one-out on train, full train pool for test) — a weak knowledge-transfer proxy.

Model: `HistGradientBoostingClassifier` bagged over 3 hyper-parameter configs x 3 seeds,
blended with small weights on two tf-idf logistic-regression models over the option
text alone (char 2–4 grams: 0.05, whitespace word 1–2 grams: 0.08).

## Measured accuracy (out-of-fold)

| model | CV accuracy |
|---|---|
| always predict D (majority prior) | 0.311 |
| dense features only, bagged GBDT | 0.310 |
| + LSA similarity | 0.319 |
| + retrieval features (final GBDT) | 0.322 – 0.331 (3x5-fold mean **0.3265**) |
| final blend with option-text tf-idf LR | 3x5-fold mean **0.329** |

Notes on rigour:
* A first version leaked the query question into its own retrieval pool at training
  time, which *lowered* validation accuracy to 0.287; `exclude_self=True` fixes it.
* A single 5-fold run suggested 0.334 for large text-blend weights, but repeated CV
  (3 fold seeds) showed that gain was fold noise, so only small, stable weights are used.
* `cv.py`, `exp2.py`, `exp3.py`, `clogit.py` are the exploration scripts (per-option
  binary GBDT/LR, LSA & retrieval ablations, and a listwise conditional-logit model —
  the latter matched but did not beat the GBDT, 0.322).
