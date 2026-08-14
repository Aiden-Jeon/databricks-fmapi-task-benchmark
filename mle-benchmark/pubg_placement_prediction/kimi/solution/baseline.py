"""빠른 베이스라인: 기본 파생 피처 + HistGBR(absolute_error)로 유효 제출 생성."""
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import build_features, encode_categoricals, TARGET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    t0 = time.time()
    train = pd.read_csv(os.path.join(ROOT, "train.csv"))
    test = pd.read_csv(os.path.join(ROOT, "test.csv"))
    print(f"load: {time.time()-t0:.1f}s train={train.shape} test={test.shape}")

    X = build_features(train)
    Xt = build_features(test)
    X, cats = encode_categoricals(X)
    Xt, _ = encode_categoricals(Xt, cats)
    y = train[TARGET].values

    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=400,
        learning_rate=0.06,
        max_leaf_nodes=63,
        min_samples_leaf=30,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=42,
    )
    t0 = time.time()
    model.fit(X.values, y)
    print(f"fit: {time.time()-t0:.1f}s n_iter={model.n_iter_}")

    pred = model.predict(Xt.values)
    pred = np.clip(pred, 0.0, 1.0)

    sub = pd.DataFrame({"Id": test["Id"], "winPlacePerc": pred})
    assert sub["Id"].nunique() == len(test)
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    sub.to_csv(os.path.join(ROOT, "outputs", "submission.csv"), index=False)
    print("saved outputs/submission.csv", sub.shape, f"total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
