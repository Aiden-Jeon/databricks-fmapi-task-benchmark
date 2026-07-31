# t19_klue_mrc — feature-based extractive QA (CPU only)

The environment has no GPU and no `torch`/`transformers`, and downloading pretrained
weights is forbidden, so the solution is a classical (pre-neural) extractive QA
pipeline built on top of `numpy` / `scikit-learn`.

## Approach

1. **Candidate generation** (`qa_lib.gen_candidates`)
   Every whitespace-token n-gram (n <= 3, <= 40 chars) of the context, plus variants
   obtained by stripping trailing Korean particles / punctuation.
   ~600-1000 candidates per context, oracle char-F1 ≈ 0.96 on answerable questions.

2. **Features** (`qa_lib.example_features`, ~90 features per candidate)
   * two question <-> context matching profiles
     - eojeol level: IDF-weighted exact/stem matches of question words
     - character n-gram longest-match profile (robust to Korean inflection)
   * proximity of the candidate to matched question material (windowed sums,
     distance transforms), and how much of the candidate itself is matched by the
     question (answers are usually the *unmatched* part of a relevant sentence)
   * sentence-level relevance: match-profile score, global char-ngram TF-IDF cosine
     similarity between the question and each sentence, rank of the candidate's
     sentence, neighbouring sentence scores
   * answer-type features (digits, dates, units, person/place/organisation suffixes,
     quotes, hangul ratio, IDF of the head word, length, position) and
     question-type x answer-type interactions (누구/언제/얼마/어디/...)

3. **Stage 1 ranker** (`run.train_stage1`)
   `HistGradientBoostingRegressor` trained on *answerable* questions to predict the
   char-level F1 of each candidate against the gold answer (stratified candidate
   sub-sampling). Two folds are used so that stage-2 features can be produced
   out-of-fold. The top-1 candidate by predicted F1 is the extraction output.

4. **Stage 2 answer-vs-empty decision** (`stage2.py`)
   31% of the questions are unanswerable, and a correct empty answer scores 1.0,
   so the decision is an expected-value comparison:
   * a regressor predicts E[char-F1 of the top-1 span]
   * a classifier predicts P(unanswerable) (= expected score of the empty answer)
   Example-level features are the stage-1 score distribution (top-8 scores, margins,
   competition from non-overlapping spans), question/context statistics and the
   question type. An answer is emitted only when `E[F1] > k * P(unanswerable)`,
   with `k` chosen by 5-fold CV on the training set.

## Reproduce

```bash
python run.py all     # candidate features, stage-1 ranker, out-of-fold + test scoring
python stage2.py      # calibration + outputs/submission.csv
```

`cache/` holds intermediate artifacts (IDF, TF-IDF model, sampled feature rows,
stage-1 models, out-of-fold and test records). `dev.py` is a fast subset dev loop.

## Results (honest, out-of-fold on train)

| setting                                   | char-F1 |
|-------------------------------------------|---------|
| always empty (trivial baseline)           | 0.307   |
| always answer with the top-1 span         | 0.187   |
| top-1 span, answerable questions only     | 0.270   |
| oracle over the top-8 ranked spans (ans.) | 0.521   |
| **stage-2 decision rule (`E[F1] - P(unans) > 0.10`)** | **0.328** |

Auxiliary diagnostics: candidate-generation oracle char-F1 0.96, sentence-retrieval
top-1 0.35 / top-3 0.60, unanswerability AUC 0.62. Sentence retrieval from purely
lexical overlap is the main bottleneck; without pretrained language models the
extraction quality stays low, so the pipeline answers only the ~18% of questions
where the expected char-F1 of the extracted span beats the value of an empty answer.

Note: `run.py` originally used the wrong fold's model when producing the
out-of-fold stage-2 features (leaking labels and inflating the calibration);
`rerun_b.py` re-ran that pass with the fix (`NB=3400`, the size that fit the time
budget) and `stage2.py` was then fit on the clean out-of-fold records.
