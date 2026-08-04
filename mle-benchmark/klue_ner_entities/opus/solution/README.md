# t20_klue_ner — Korean NER (character-level structured perceptron)

## Environment constraint
The container has **no GPU, no PyTorch and no `transformers`** (only
numpy / scipy / scikit-learn / pandas), and no internet access, so a
pretrained-encoder approach was impossible. The solution is therefore a
classical **feature-based sequence labeler implemented from scratch in numpy**.

## Approach

1. **Label representation.** Each sentence is tagged at **character** level with
   a 13-label BIO scheme (`O` + `B-/I-` × {PS, LC, OG, DT, TI, QT}).
   The `entities` column lists mentions in order of appearance, so gold spans are
   recovered by a left-to-right `str.find` scan (100 % of the 41 103 training
   mentions align exactly).

2. **Model.** Averaged **structured perceptron** with first-order transitions and
   exact **Viterbi** decoding (`solution/ner.py::Perceptron`). Illegal BIO
   transitions (`I-X` not preceded by `B-X`/`I-X`) are masked out, so every
   decoded sequence is well formed. Averaging uses the standard lazy
   timestamp trick so updates stay sparse.

3. **Features** (per character, ~45 active):
   - character uni/bi/tri/4-grams in a ±3 window, plus a skip bigram;
   - character-class n-grams (hangul syllable / jamo / digit / latin / CJK /
     kana / punctuation / space);
   - **jamo decomposition** (onset, coda) of the current and neighbouring
     syllables — the coda of the preceding syllable is a strong cue for Korean
     particles that terminate an entity;
   - whitespace-token features: the token itself, digit-normalised token, word
     shape, prefixes/suffixes, offset inside the token, token length, previous
     and next token;
   - **gazetteer features** from the training mentions: longest dictionary match
     starting at each position, expanded to `B`/`I` roles, with buckets for match
     length, mention frequency, type purity, and whether the match is aligned
     with whitespace-token boundaries.

4. **Out-of-fold gazetteer (the key trick).** A gazetteer built from the whole
   training set is *perfect* on the training set, so a model trained with such
   features learns to blindly trust it and collapses at test time. Training
   features are therefore extracted with **5-fold out-of-fold gazetteers**
   (fold *k*'s features use a gazetteer built from the other folds), while
   inference uses the full-training gazetteer.
   This single change moved dev micro-F1 from **0.739 → 0.832**.

5. **Ensembling.** Several perceptrons trained with different example orderings;
   their averaged weight vectors are averaged again (they share one feature map),
   giving a small additional gain.

6. **O-bias calibration.** Recall was consistently below precision, so a scalar
   penalty subtracted from the `O` emission score is tuned on the dev split to
   maximise entity micro-F1.

## Results (held-out 15 % of train, entity-level micro-F1)

| configuration | P | R | F1 |
|---|---|---|---|
| in-sample gazetteer, 12 epochs | 0.762 | 0.717 | 0.739 |
| + out-of-fold gazetteer | 0.852 | 0.815 | 0.833 |
| + jamo / 4-gram / shape / gaz-boundary features | 0.856 | 0.816 | 0.835 |
| + 2–3 seed weight ensemble | 0.858 | 0.816 | 0.836 |
| + tuned O-bias (=1.0) | 0.856 | 0.818 | **0.836** |

Final submission: full train, 5-fold OOF gazetteer, 10 epochs, 2-seed weight
ensemble, O-bias 1.0 (9 837 predicted mentions, 2.34 per sentence, vs 2.45 in train).

## Reproducing

```bash
bash solution/reproduce.sh            # -> outputs/submission.csv (+ format check)

# validation (85/15 split of train.csv), incl. O-bias sweep
python solution/run.py --mode dev --epochs 10 --kfold 5 --seeds 3 --biases 0,0.5,1,1.5,2,3
```

Runtime: ~7 min on 4 CPU cores, ~2 GB RAM. Everything is deterministic
(fixed numpy seeds, insertion-ordered feature map).

Files:
- `solution/ner.py` — labels, alignment, features, gazetteer, perceptron, Viterbi, metric.
- `solution/run.py` — data loading, out-of-fold encoding, training, evaluation, submission writing.
- `solution/verify.py` — submission format check (ids, columns, entity strings are
  real substrings of the sentence, valid types).
- `solution/reproduce.sh` — one-command reproduction.
