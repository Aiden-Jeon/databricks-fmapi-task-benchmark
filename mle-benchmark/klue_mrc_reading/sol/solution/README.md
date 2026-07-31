# KLUE-MRC offline solution

Run from the task root:

```bash
python solution/train_predict.py --data-dir . --output outputs/submission.csv
```

The script uses only `train.csv` and locally installed `pandas`, `numpy`, and
`scikit-learn`. It trains a character-overlap sentence retriever, a supervised
candidate-span ranker, and an answerability classifier. Randomized components
use a fixed seed.

Optional grouped holdout validation:

```bash
python solution/train_predict.py --data-dir . --validate
```
