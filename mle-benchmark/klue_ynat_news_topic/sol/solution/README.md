# YNAT solution

The model combines character 2-6 gram and whitespace-token 1-2 gram TF-IDF
features in a class-balanced linear SVM. It uses only `train.csv`. The default
regularization strength (`C=0.1`) achieved 0.8339 macro F1 on the fixed 20%
stratified holdout.

Run from the task root:

```bash
python solution/train.py
```

This writes `outputs/submission.csv`. To reproduce the fixed stratified
holdout experiment used for selecting the regularization strength:

```bash
python solution/train.py --validate
```
