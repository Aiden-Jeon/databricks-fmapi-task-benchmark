# PAWS-X Korean solution

The model uses reproducible, training-only signals:

- character and token n-gram overlap;
- sequence similarity and shared-token order changes;
- number and Latin-name consistency;
- an Extra Trees and Random Forest probability ensemble;
- exact-pair and positive-link graph rules learned from `train.csv`.

Run from any directory with:

```bash
python solution/train.py
```

This writes `outputs/submission.csv`. The fixed random seed is 2026. The required
packages are Python 3, NumPy, pandas, and scikit-learn.
