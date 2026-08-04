"""Train + predict pipeline for KoBEST HellaSwag.

Approach: pairwise scoring.  Each (context, ending_k) pair becomes a feature
row; a classifier learns to rank the correct ending above the others.  At
inference we take argmax over the 4 candidates per example.

Pure scikit-learn (HistGradientBoostingClassifier) -- no internet / no
external weights.

Usage:
    python solution/train.py
writes outputs/submission.csv
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from featurize import (
    build_pairwise_features, labels_to_long, preds_long_to_labels,
)


def load_data():
    train = pd.read_csv(os.path.join(TASK, 'train.csv'))
    test = pd.read_csv(os.path.join(TASK, 'test.csv'))
    return train, test


def make_models():
    """Return list of (name, model_factory) for pairwise scoring."""
    return [
        ('hgb', lambda: HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=600, max_leaf_nodes=31,
            min_samples_leaf=20, l2_regularization=1.0, random_state=42)),
        ('hgb2', lambda: HistGradientBoostingClassifier(
            learning_rate=0.03, max_iter=1000, max_leaf_nodes=15,
            min_samples_leaf=30, l2_regularization=0.5, random_state=7)),
        ('hgb3', lambda: HistGradientBoostingClassifier(
            learning_rate=0.08, max_iter=400, max_leaf_nodes=63,
            min_samples_leaf=15, l2_regularization=2.0, random_state=123)),
        ('rf', lambda: RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_leaf=5,
            n_jobs=-1, random_state=42)),
        ('et', lambda: ExtraTreesClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=4,
            n_jobs=-1, random_state=42)),
        ('lr', lambda: make_pipeline(StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000, random_state=42))),
    ]


def cv_score(train, n_splits=5, seed=42, verbose=True, models=None):
    """Cross-validate by question (group-level) -- stratified on label."""
    X, ids_long, group, tf_char, tf_word, tf_word2, tf_char2 = \
        build_pairwise_features(train, fit=True)
    y_long = labels_to_long(train)
    labels = train['label'].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    if models is None:
        models = make_models()

    # accumulate per-model OOF
    results = {}
    for name, factory in models:
        oof_scores = np.zeros(len(X), dtype=float)
        fold_acc = []
        for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
            tr_rows = np.concatenate([np.arange(4 * i, 4 * i + 4) for i in tr_idx])
            va_rows = np.concatenate([np.arange(4 * i, 4 * i + 4) for i in va_idx])
            Xtr = X.iloc[tr_rows]
            ytr = y_long[tr_rows]
            Xva = X.iloc[va_rows]
            m = factory()
            m.fit(Xtr, ytr)
            oof_scores[va_rows] = m.predict_proba(Xva)[:, 1]
            va_labels = labels[va_idx]
            va_pred = preds_long_to_labels(oof_scores[va_rows])
            acc = accuracy_score(va_labels, va_pred)
            fold_acc.append(acc)
        oof_pred = preds_long_to_labels(oof_scores)
        overall = accuracy_score(labels, oof_pred)
        results[name] = {'oof': oof_scores, 'acc': overall, 'folds': fold_acc}
        if verbose:
            print(f'[{name}] folds={[f"{a:.4f}" for a in fold_acc]} '
                  f'mean={np.mean(fold_acc):.4f} oof={overall:.4f}')
    return results, X, y_long, labels, (tf_char, tf_word, tf_word2, tf_char2)


def ensemble_oof(results, weights):
    names = list(weights.keys())
    oof = np.zeros_like(results[names[0]]['oof'])
    total = 0.0
    for n in names:
        w = weights[n]
        oof += w * results[n]['oof']
        total += w
    return oof / total if total else oof


def search_weights(results, labels):
    """Grid search over a few models' weights to maximize OOF accuracy."""
    names = list(results.keys())
    if len(names) == 1:
        w = {names[0]: 1.0}
        oof = results[names[0]]['oof']
        return w, accuracy_score(labels, preds_long_to_labels(oof))
    import itertools
    # keep grid small for speed; 3^6 = 729 max
    grid = [0.0, 1.0, 2.0]
    best_w = {n: 1.0 for n in names}
    best_acc = -1.0
    for combo in itertools.product(grid, repeat=len(names)):
        if sum(combo) == 0:
            continue
        w = {n: float(c) for n, c in zip(names, combo)}
        oof = ensemble_oof(results, w)
        acc = accuracy_score(labels, preds_long_to_labels(oof))
        if acc > best_acc:
            best_acc = acc
            best_w = w
    return best_w, best_acc


def train_full_and_predict(train, test, weights=None, models=None):
    X, ids_long, group, tf_char, tf_word, tf_word2, tf_char2 = \
        build_pairwise_features(train, fit=True)
    y_long = labels_to_long(train)

    if models is None:
        models = make_models()
    if weights is None:
        weights = {n: 1.0 for n, _ in models}

    Xt, ids_long_t, group_t, _, _, _, _ = build_pairwise_features(
        test, tfidf_char=tf_char, tfidf_word=tf_word, tfidf_word2=tf_word2,
        tfidf_char2=tf_char2, fit=False)

    scores = np.zeros(len(Xt), dtype=float)
    total = 0.0
    for name, factory in models:
        m = factory()
        m.fit(X, y_long)
        s = m.predict_proba(Xt)[:, 1]
        w = weights.get(name, 1.0)
        scores += w * s
        total += w
        print(f'  trained {name} weight={w}')
    scores /= total
    preds = preds_long_to_labels(scores)
    out = pd.DataFrame({'id': test['id'].values, 'label': preds.astype(int)})
    return out


def main():
    train, test = load_data()
    print(f'train={train.shape}  test={test.shape}')

    models = make_models()
    print(f'Running 5-fold CV for {len(models)} models...')
    results, X, y_long, labels, tfidfs = cv_score(
        train, n_splits=5, seed=42, models=models)

    print('Searching ensemble weights...')
    best_w, best_acc = search_weights(results, labels)
    print(f'best ensemble weights={best_w}  oof_acc={best_acc:.4f}')

    print('Training on full data...')
    out = train_full_and_predict(train, test, weights=best_w, models=models)

    assert list(out.columns) == ['id', 'label'], out.columns
    assert len(out) == len(test), (len(out), len(test))
    assert set(out['id']) == set(test['id']), 'id mismatch'
    assert set(out['label'].unique()).issubset({0, 1, 2, 3})

    out_dir = os.path.join(TASK, 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'submission.csv')
    out.to_csv(out_path, index=False)
    print(f'wrote {out_path}  ({len(out)} rows)')
    print('label distribution:')
    print(out['label'].value_counts().sort_index())
    print(f'Final CV OOF accuracy (ensemble): {best_acc:.4f}')


if __name__ == '__main__':
    main()
