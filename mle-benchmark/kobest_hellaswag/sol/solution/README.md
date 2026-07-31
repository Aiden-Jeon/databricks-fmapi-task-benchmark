# KoBEST HellaSwag solution

Run from any directory with:

```bash
python solution/train_predict.py
```

The script trains a binary candidate ranker on the four endings per question and
writes `outputs/submission.csv`. It combines character and word TF-IDF with
context-to-ending, last-sentence-to-ending, candidate-to-candidate similarity,
and small structural features. The selected logistic-regression configuration
scored 0.5964 mean accuracy in fixed 5-fold stratified validation.

`experiment.py` contains the reproducible cross-validation comparison used to
select the final configuration.
