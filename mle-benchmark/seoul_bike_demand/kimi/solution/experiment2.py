"""2차 실험: absolute_error 중심 튜닝 + 앙상블 + lag 특성 검증."""
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, "solution")
from common import TARGET, FEATURES, load_data, add_features, time_folds


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def run_cv(model_specs, df, feats, two_model=True, seeds=(0,), tag=""):
    """model_specs: list of (weight, make_model_fn)."""
    fold_scores = []
    for tr_idx, va_idx in time_folds(df):
        tr = df.loc[tr_idx]
        va = df.loc[va_idx]
        weighted = []
        weights = []
        for w, mk in model_specs:
            ps = []
            for seed in seeds:
                if two_model:
                    sub_ps = []
                    for subset in ("all", "func"):
                        d = tr if subset == "all" else tr[tr["functioning_day"] == 1]
                        m = mk(seed)
                        m.fit(d[feats], d[TARGET].clip(lower=0).to_numpy())
                        sub_ps.append(np.maximum(m.predict(va[feats]), 0.0))
                    ps.append(np.mean(sub_ps, axis=0))
                else:
                    m = mk(seed)
                    m.fit(tr[feats], tr[TARGET].clip(lower=0).to_numpy())
                    ps.append(np.maximum(m.predict(va[feats]), 0.0))
            weighted.append(w * np.mean(ps, axis=0))
            weights.append(w)
        pred = np.sum(weighted, axis=0) / np.sum(weights)
        pred = np.where(va["functioning_day"].to_numpy() == 0, 0.0, pred)
        fold_scores.append(rmse(va[TARGET], pred))
    mean = float(np.mean(fold_scores))
    print(f"{tag:28s} CV={mean:8.2f}  folds={['%.1f' % f for f in fold_scores]}", flush=True)
    return mean


def abs_hgb(**kw):
    defaults = dict(
        loss="absolute_error", learning_rate=0.06, max_iter=1500,
        max_leaf_nodes=63, min_samples_leaf=20, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=60,
    )
    defaults.update(kw)
    return lambda seed: HistGradientBoostingRegressor(random_state=seed, **defaults)


def sq_hgb(**kw):
    defaults = dict(
        loss="squared_error", learning_rate=0.06, max_iter=1500,
        max_leaf_nodes=63, min_samples_leaf=20, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=60,
    )
    defaults.update(kw)
    return lambda seed: HistGradientBoostingRegressor(random_state=seed, **defaults)


def add_lag_features(df):
    """train 구간 내에서만 lag 특성 생성 (누출 방지 위해 train 내부 fold에만 사용)."""
    df = df.copy()
    df = df.sort_values(["date", "hour"]).reset_index(drop=True)
    for lag in (24, 48, 168):
        df[f"lag_{lag}"] = df[TARGET].shift(lag)
    df["roll_24"] = df[TARGET].shift(24).rolling(24).mean()
    return df


LAG_FEATS = FEATURES + ["lag_24", "lag_48", "lag_168", "roll_24"]


def main():
    df = add_features(load_data())
    train_only = df[df["_is_train"] == 1].copy()

    # 1) absolute_error 튜닝
    run_cv([(1, abs_hgb())], train_only, FEATURES, tag="abs_base")
    run_cv([(1, abs_hgb(learning_rate=0.04, max_iter=2500))], train_only, FEATURES, tag="abs_lr0.04")
    run_cv([(1, abs_hgb(learning_rate=0.08, max_iter=1200))], train_only, FEATURES, tag="abs_lr0.08")
    run_cv([(1, abs_hgb(max_leaf_nodes=31))], train_only, FEATURES, tag="abs_leaf31")
    run_cv([(1, abs_hgb(max_leaf_nodes=127, min_samples_leaf=15))], train_only, FEATURES, tag="abs_leaf127")
    run_cv([(1, abs_hgb(l2_regularization=0.1))], train_only, FEATURES, tag="abs_l2_0.1")
    run_cv([(1, abs_hgb(l2_regularization=5.0))], train_only, FEATURES, tag="abs_l2_5")
    run_cv([(1, abs_hgb(min_samples_leaf=40))], train_only, FEATURES, tag="abs_minleaf40")
    run_cv([(1, abs_hgb())], train_only, FEATURES, seeds=(0, 1, 2), tag="abs_seeds3")

    # 2) 단일 모델(전체행만)
    run_cv([(1, abs_hgb())], train_only, FEATURES, two_model=False, tag="abs_single")

    # 3) 앙상블: abs + squared
    run_cv([(0.5, abs_hgb()), (0.5, sq_hgb())], train_only, FEATURES, tag="ens_abs_sq")
    run_cv([(0.7, abs_hgb()), (0.3, sq_hgb())], train_only, FEATURES, tag="ens_abs7_sq3")

    # 4) lag 특성 (train 전용; fold 내부에서 lag 재계산은 하지 않고 전체 train에서 shift - fold cut 이전 값만 사용되므로 tr에 대해선 안전,
    #    단 val 구간의 lag는 cut 이전 train 실측을 참조 -> 미래 누출 아님)
    df_lag = add_lag_features(train_only)
    run_cv([(1, abs_hgb())], df_lag.dropna(subset=["lag_168"]), LAG_FEATS, tag="abs_lag")


if __name__ == "__main__":
    main()
