# KoBEST HellaSwag — solution

`python solution/run.py` → `outputs/submission.csv` (2 min, CPU only, no external data).

## Key observation
The four candidates of a row are **consecutive events of one narrative chain**; the gold
label is the event that comes *immediately* after the context, the distractors are events
further down the same chain. Korean sentences here restate the previous event in a leading
adnominal clause (`오토바이를 멈춘다.` → `멈춘 오토바이의 시동을 끈다.`), so **directional**
similarity between the *beginning* of one sentence and the *end* of another reveals order.

## Features (`feats2.py`, 322 per candidate)
- tf-idf cosine (char_wb 2-4, char 3-5, stemmed word 1-2, plus LSA/SVD of both) between
  candidate views (full / char-fraction prefixes & suffixes / first-k & last-k words) and
  context views (full, last sentence, last two sentences, first sentence, sentence tail).
- 4×4 directional dependency matrices `sim(prefix_i, suffix_j)`; in/out degree, max, diff
  → a candidate whose opening refers to another candidate must come later.
- Structured chain search: over all 4! orderings, score
  `sim(cand_first, ctx_last) + Σ sim(prefix_next, suffix_prev)`; keep the best score per
  "which candidate is first" (`perm_*` features).
- Raw n-gram containment / novelty vs. context, candidate-candidate overlap.
- Korean predicate-stem matching (2-char stems): context's final verb appearing in a
  candidate (esp. in its first two words), and a candidate's verb being referenced by
  other candidates.
- Surface: lengths, word counts, connective markers, position, #context sentences.
Each feature is also expanded with within-group rank and gap-to-group-max views.

## Models (`model.py`)
- `FlatClf('hgb')`: HistGradientBoosting binary correct/incorrect, logits re-normalised
  within each group of 4 (2 normalisation views × 5/3 seeds).
- `CondLogit`: L2-regularised conditional logit (softmax over the 4 candidates), 15 % weight.

## Results (repeated stratified 5-fold CV, 3 seeds)
| model | acc |
|---|---|
| best single heuristic (prefix vs. last context sentence) | 0.540 |
| conditional logit, feature set v1 | 0.640 |
| HGB, feature set v1 | 0.677 |
| HGB, feature set v2 | 0.694 |
| **HGB views + conditional logit blend (submitted)** | **0.700** |

## Tried and rejected
- `aug.py`: synthesising extra rows from the context chain (gold = next context sentence,
  distractors = later events). Valid and 3× more data, but consistently *hurt*
  (0.670 → 0.645 at weight 1.0, 0.654 at weight 0.1) — the synthetic contexts/goldens have
  shifted surface statistics.
- `pairwise.py`: "which event comes first" pair classifier with antisymmetric features
  (6 labelled pairs per row) + Borda aggregation. 0.646 alone, no blend gain → excluded.
- ExtraTrees (0.649) — no blend gain.
