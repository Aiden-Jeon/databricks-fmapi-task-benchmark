"""HistGradientBoosting 기반 자전거 수요 예측.

- 시간순 fold로 RMSE 검증
- sqrt 타깃 변환 + 두 모델(전체 / functioning만) 앙상블
- 최종: 전체 train으로 재학습 후 test 예측 -> outputs/submission.csv
"""
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, "solution")
from common import TARGET, ID_COL, FEATURES, load_data, add_features, time_folds


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def make_model(seed=0):
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.06,
        max_iter=1200,
        max_leaf_nodes=63,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=60,
        random_state=seed,
    )


def fit_predict(train_df, pred_df, seed=0):
    """sqrt 변환 타깃으로 모델 2개(전체행, functioning만) 학습 후 예측 평균."""
    preds = []
    for subset in ("all", "func"):
        tr = train_df if subset == "all" else train_df[train_df["functioning_day"] == 1]
        model = make_model(seed)
        y = np.sqrt(tr[TARGET].clip(lower=0).to_numpy())
        model.fit(tr[FEATURES], y)
        p = model.predict(pred_df[FEATURES])
        p = np.maximum(p, 0.0) ** 2
        preds.append(p)
    return np.mean(preds, axis=0)


def main():
    df = add_features(load_data())
    train_mask = df["_is_train"] == 1
    train_df = df[train_mask].reset_index(drop=True)
    test_df = df[~train_mask].reset_index(drop=True)

    # ---- 시간순 검증 ----
    scores = []
    for tr_idx, va_idx in time_folds(df):
        tr = df.loc[tr_idx]
        va = df.loc[va_idx]
        pred = fit_predict(tr, va, seed=0)
        # 비운영일은 실제 0이므로, 0 예측 규칙 포함한 RMSE와 포함하지 않은 RMSE 모두 확인
        s_raw = rmse(va[TARGET], pred)
        pred_adj = np.where(va["functioning_day"].to_numpy() == 0, 0.0, pred)
        s_adj = rmse(va[TARGET], pred_adj)
        scores.append((s_raw, s_adj))
        print(f"fold: raw={s_raw:.2f} adj={s_adj:.2f}")
    print(f"CV RMSE raw={np.mean([s[0] for s in scores]):.2f} "
          f"adj={np.mean([s[1] for s in scores]):.2f}")

    # ---- 최종 학습 + 예측 ----
    final_pred = fit_predict(train_df, test_df, seed=0)
    # 비운영일은 수요 0
    final_pred = np.where(test_df["functioning_day"].to_numpy() == 0, 0.0, final_pred)
    final_pred = np.maximum(final_pred, 0.0)

    sub = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET: final_pred})
    # sample_submission 순서와 정확히 일치시키기
    sample = pd.read_csv("sample_submission.csv")
    sub = sample[[ID_COL]].merge(sub, on=ID_COL, how="left")
    assert sub[TARGET].notna().all(), "missing predictions"
    assert len(sub) == len(test_df), "row count mismatch"
    sub.to_csv("outputs/submission.csv", index=False)
    print("saved outputs/submission.csv", sub.shape)
    print(sub.head())


if __name__ == "__main__":
    main()
