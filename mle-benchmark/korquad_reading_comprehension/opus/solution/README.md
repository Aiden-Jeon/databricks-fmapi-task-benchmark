# t9_korquad — extractive QA as candidate-span ranking (CPU only)

## Environment constraint
No GPU, no `torch`/`transformers`, no internet, no pretrained weights.
Only `numpy / pandas / scikit-learn` are available, so the solution is a
**classical feature-based extractive reader**: enumerate answer-span
candidates from the context and rank them with gradient boosted trees
trained on `train.csv` only.

## Pipeline
1. **Candidate generation** (`features.py`)
   - context is split into sentences and eojeols (whitespace tokens);
   - a candidate is 1–4 consecutive eojeols, where
     - the *start* may skip leading opening punctuation (`《 ' " 〈 ( …`),
     - the *end* may drop trailing punctuation and Korean particles/endings
       (`의 에 을 를 는 이 가 에서 으로 이다. …`, list induced from the train
       answers' trailing context).
   - ≈840 candidates/context. Oracle quality measured on train:
     **exact recall 86.4 %, oracle char-F1 0.959**.
2. **Features** (58 per candidate)
   - *static / context only* (24): span char & eojeol length, position in
     context/sentence, boundary alignment, number of stripped chars +
     stripped-suffix id, digit/latin/hangul shape flags, year flag, quoted
     flag, neighbouring char classes, "verb-ending" flag, span frequency in
     the context, min/mean word-IDF of covered eojeols, sentence length.
   - *question dependent* (34): question content tokens are stemmed
     (particles stripped, interrogatives dropped) and IDF-weighted (IDF from
     the train contexts). All occurrences are marked on a char array, then via
     cumulative sums we get IDF mass inside the span (answers rarely reuse
     question words), IDF mass in ±20/±50/±120 char windows, number of
     *distinct* question tokens in those windows, char distance to the nearest
     match on the left/right, distance to the rarest / first / last question
     token, sentence-level match score + rank + normalised score, char overlap
     with the question, question type (누구/언제/어디/몇 년/몇/이름/왜/…), and
     **listwise-normalised versions** (value / group max, z-score, rank
     percentile) of the proximity features, which let the tree model compare
     candidates within one context.
3. **Target & training** — regression on the *evaluation metric itself*:
   `y = char-F1(candidate, gold)`. Per question we keep the gold-overlapping
   candidates with F1 ≥ 0.35 (≤8) plus 36–40 sampled negatives
   (≈2.2 M rows × 58). Model: `HistGradientBoostingRegressor`
   (400–500 iters, 63 leaves, lr 0.1).
4. **Inference** — every candidate of the context is scored, `argmax` wins.
5. **Round 2 / ensemble** — hard negatives (top-12 wrongly ranked candidates)
   were mined with the round-1 model and a second model was trained on
   random + hard negatives. The blend `0.6·v1 + 0.4·v2` is used for the
   submission.

## Validation (held-out by document id, 6 % of train docs)
| model | char-F1 | EM |
|---|---|---|
| v1 (random negatives) | 0.5499 | 0.3525 |
| v2 (+hard negatives)  | 0.5380 | 0.2825 |
| **0.6·v1 + 0.4·v2**   | **0.5684** | **0.3617** |

(1200 validation questions; the val set is document-disjoint from training,
matching the train/test split of the task.)

## Reproduce
```bash
bash solution/run_all.sh          # build features, train v1, mine, train v2
python solution/predict_ensemble.py   # -> outputs/submission.csv
```
Individual stages: `python solution/run.py {build,merge,train,val,mine,build2,predict}`
(env: `SHARD`/`NSHARD` for parallel feature building, `ITERS`, `NNEG`, `NVAL`).
Total runtime ≈45 min on 4 CPU cores.
