"""PUBG winPlacePerc prediction — train + predict pipeline.

Approach:
  1. Engineer features: within-match z-scores, within-team aggregates,
     normalized killPlace (proxy for placement), distance ratios, etc.
  2. Train a 3-seed LightGBM ensemble with MAE (L1) objective, using all
     training data, with iteration count guided by match-grouped validation.
  3. Average predictions across seeds, then average within each (matchId,
     groupId) since teammates share the same placement target. Clip to [0,1].

Reproducible: fixed seeds, no external data, no internet.
"""
import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

from fe import build_features

HERE = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(HERE)
TRAIN_CSV = os.path.join(TASK_DIR, 'train.csv')
TEST_CSV = os.path.join(TASK_DIR, 'test.csv')
SUB_CSV = os.path.join(TASK_DIR, 'outputs', 'submission.csv')


def _match_grouped_split(match_ids, val_frac=0.2, seed=42):
    rng = np.random.RandomState(seed)
    mids = np.array(match_ids)
    rng.shuffle(mids)
    n_val = int(len(mids) * val_frac)
    return set(mids[:n_val])


def train_one(X_tr, y_tr, X_val, y_val, seed, n_estimators=2500, lr=0.05):
    params = dict(
        objective='regression_l1',
        metric='mae',
        learning_rate=lr,
        num_leaves=95,
        min_child_samples=30,
        feature_fraction=0.75,
        bagging_fraction=0.85,
        bagging_freq=5,
        reg_alpha=0.05,
        reg_lambda=0.1,
        n_estimators=n_estimators,
        verbose=-1,
    )
    model = lgb.LGBMRegressor(**params, random_state=seed)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        categorical_feature=['matchType', 'matchTypeClean'],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )
    return model


def main():
    t0 = time.time()
    print("Loading data...")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    print(f"  train={train.shape} test={test.shape}")

    print("Building features...")
    tr, te, y = build_features(train, test)
    print(f"  train_feat={tr.shape} test_feat={te.shape}")

    # Match-grouped validation to find a reasonable n_estimators for full-fit.
    val_matches = _match_grouped_split(train['matchId'].unique(), 0.2, seed=42)
    is_val = train['matchId'].isin(val_matches).values
    X_tr_p, X_val_p = tr[~is_val], tr[is_val]
    y_tr_p, y_val_p = y[~is_val], y[is_val]

    # Calibrate iteration count using a representative seed.
    print("Calibrating iterations on validation...")
    cal_model = train_one(X_tr_p, y_tr_p, X_val_p, y_val_p, seed=42)
    best_iter = cal_model.best_iteration_
    val_pred = cal_model.predict(X_val_p).clip(0, 1)
    print(f"  Val MAE (single seed): {mean_absolute_error(y_val_p, val_pred):.5f}")
    print(f"  best_iteration: {best_iter}")

    # Group averaging on validation gives the big improvement.
    val_groups = train.loc[is_val, ['matchId', 'groupId']].reset_index(drop=True)
    val_df = pd.DataFrame({'pred': val_pred, 'matchId': val_groups['matchId'],
                           'groupId': val_groups['groupId']})
    g_pred = val_df.groupby(['matchId', 'groupId'])['pred'].transform('mean')
    print(f"  Val MAE (group-averaged): {mean_absolute_error(y_val_p, g_pred):.5f}")

    # Full-data training with a small ensemble of seeds. Use a slightly padded
    # iteration count relative to the calibration best_iter for robustness.
    n_full = int(best_iter * 1.05) + 50
    seeds = [42, 7, 2024]
    print(f"Training full-data ensemble with n_estimators={n_full}, seeds={seeds}")
    test_preds = []
    for s in seeds:
        params = dict(
            objective='regression_l1', metric='mae',
            learning_rate=0.05, num_leaves=95, min_child_samples=30,
            feature_fraction=0.75, bagging_fraction=0.85, bagging_freq=5,
            reg_alpha=0.05, reg_lambda=0.1,
            n_estimators=n_full, verbose=-1,
        )
        m = lgb.LGBMRegressor(**params, random_state=s)
        m.fit(tr, y, categorical_feature=['matchType', 'matchTypeClean'])
        test_preds.append(m.predict(te).clip(0, 1))
        print(f"  seed {s} done ({round(time.time()-t0,1)}s)")

    pred = np.mean(test_preds, axis=0)

    # Group-level averaging: teammates share winPlacePerc, so aggregate within
    # (matchId, groupId) to denoise and enforce consistency.
    out = pd.DataFrame({
        'Id': test['Id'].values,
        'pred': pred,
        'matchId': test['matchId'].values,
        'groupId': test['groupId'].values,
    })
    out['pred'] = out.groupby(['matchId', 'groupId'])['pred'].transform('mean')
    out['winPlacePerc'] = out['pred'].clip(0, 1)

    sub = out[['Id', 'winPlacePerc']]
    os.makedirs(os.path.dirname(SUB_CSV), exist_ok=True)
    sub.to_csv(SUB_CSV, index=False)
    print(f"Wrote {SUB_CSV} shape={sub.shape} in {round(time.time()-t0,1)}s")
    print(sub.head())
    print("Range:", sub['winPlacePerc'].min(), sub['winPlacePerc'].max())
    print("Any null:", sub.isnull().sum().sum())


if __name__ == '__main__':
    main()
