"""빠른 실험 루프: fold0/fold1에서 파라미터/피처 변형 MAE 비교."""
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import build_features, encode_categoricals, TARGET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIGS = {
    "A_leaves127": dict(loss="absolute_error", max_iter=800, learning_rate=0.04,
                        max_leaf_nodes=127, min_samples_leaf=20, l2_regularization=1.0,
                        early_stopping=False, random_state=42),
    "B_leaves255": dict(loss="absolute_error", max_iter=1000, learning_rate=0.03,
                        max_leaf_nodes=255, min_samples_leaf=20, l2_regularization=1.0,
                        early_stopping=False, random_state=42),
    "C_l2_10": dict(loss="absolute_error", max_iter=800, learning_rate=0.04,
                    max_leaf_nodes=127, min_samples_leaf=30, l2_regularization=10.0,
                    early_stopping=False, random_state=42),
    "D_squared": dict(loss="squared_error", max_iter=800, learning_rate=0.04,
                      max_leaf_nodes=127, min_samples_leaf=20, l2_regularization=1.0,
                      early_stopping=False, random_state=42),
}


def main():
    which = sys.argv[1:] if len(sys.argv) > 1 else list(CONFIGS)
    train = pd.read_csv(os.path.join(ROOT, "train.csv"))
    X = build_features(train)
    X, cats = encode_categoricals(X)
    y = train[TARGET].values
    groups = train["matchId"].values

    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X, y, groups))

    for name in which:
        params = CONFIGS[name]
        maes = []
        t0 = time.time()
        for fi in [0, 1]:
            tr, va = splits[fi]
            model = HistGradientBoostingRegressor(**params)
            model.fit(X.iloc[tr].values, y[tr])
            pred = np.clip(model.predict(X.iloc[va].values), 0, 1)
            maes.append(mean_absolute_error(y[va], pred))
        print(f"{name:15s} fold0={maes[0]:.5f} fold1={maes[1]:.5f} "
              f"mean={np.mean(maes):.5f}  iter={model.n_iter_}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
