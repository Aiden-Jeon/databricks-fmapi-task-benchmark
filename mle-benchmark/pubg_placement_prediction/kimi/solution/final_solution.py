"""최종 파이프라인: v2 피처 + 시드 앙상블 HistGBR.

- 검증: GroupKFold(matchId)로 MAE 리포트
- 제출: 전체 train으로 시드 앙상블 학습 후 test 예측, outputs/submission.csv 저장
실행: python3 solution/final_solution.py [--skip-cv]
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import encode_categoricals, TARGET
from features_v2 import build_features_v2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# squared_error가 MAE 검증에서 absolute_error보다 일관되게 좋았음
BASE_PARAMS = dict(
    loss="squared_error",
    max_iter=900,
    learning_rate=0.04,
    max_leaf_nodes=127,
    min_samples_leaf=20,
    l2_regularization=1.0,
    early_stopping=False,
)

SEEDS = [42, 7, 2024]


def run_cv(X, y, groups, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    maes = []
    t0 = time.time()
    for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
        model = HistGradientBoostingRegressor(random_state=SEEDS[0], **BASE_PARAMS)
        model.fit(X.iloc[tr].values, y[tr])
        pred = np.clip(model.predict(X.iloc[va].values), 0, 1)
        mae = mean_absolute_error(y[va], pred)
        maes.append(mae)
        print(f"  fold{fold}: MAE={mae:.5f} ({time.time()-t0:.0f}s)")
    print(f"CV MAE: {np.mean(maes):.5f} +- {np.std(maes):.5f}")
    return np.mean(maes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-cv", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    train = pd.read_csv(os.path.join(ROOT, "train.csv"))
    test = pd.read_csv(os.path.join(ROOT, "test.csv"))
    print(f"load: {time.time()-t0:.1f}s")

    X = build_features_v2(train)
    Xt = build_features_v2(test)
    X, cats = encode_categoricals(X)
    Xt, _ = encode_categoricals(Xt, cats)
    assert list(X.columns) == list(Xt.columns)
    y = train[TARGET].values
    groups = train["matchId"].values
    print(f"features: {X.shape[1]} cols, build {time.time()-t0:.1f}s")

    if not args.skip_cv:
        print("GroupKFold CV:")
        run_cv(X, y, groups)

    # 시드 앙상블 최종 학습
    Xv, Xtv = X.values, Xt.values
    preds = []
    for s in SEEDS:
        model = HistGradientBoostingRegressor(random_state=s, **BASE_PARAMS)
        t1 = time.time()
        model.fit(Xv, y)
        p = np.clip(model.predict(Xtv), 0, 1)
        preds.append(p)
        print(f"seed {s}: fit+predict {time.time()-t1:.0f}s")

    pred = np.mean(preds, axis=0)
    pred = np.clip(pred, 0.0, 1.0)

    sub = pd.DataFrame({"Id": test["Id"], "winPlacePerc": pred})
    assert sub["Id"].nunique() == len(test), "Id 중복/누락!"
    assert len(sub) == len(test)
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    out_path = os.path.join(ROOT, "outputs", "submission.csv")
    sub.to_csv(out_path, index=False)
    print(f"saved {out_path} shape={sub.shape} total {time.time()-t0:.0f}s")
    print(sub.head())


if __name__ == "__main__":
    main()
