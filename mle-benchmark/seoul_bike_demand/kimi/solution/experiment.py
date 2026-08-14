"""하이퍼파라미터/특성 실험: 시간순 CV로 RMSE 비교."""
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
)

sys.path.insert(0, "solution")
from common import TARGET, FEATURES, load_data, add_features, time_folds


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def cv_eval(make_model, use_sqrt=True, two_model=True, seeds=(0,)):
    df = add_features(load_data())
    folds = time_folds(df)
    fold_scores = []
    for tr_idx, va_idx in time_folds(df):
        tr = df.loc[tr_idx]
        va = df.loc[va_idx]
        preds = []
        for seed in seeds:
            if two_model:
                ps = []
                for subset in ("all", "func"):
                    d = tr if subset == "all" else tr[tr["functioning_day"] == 1]
                    m = make_model(seed)
                    y = d[TARGET].clip(lower=0).to_numpy()
                    if use_sqrt:
                        y = np.sqrt(y)
                    m.fit(d[FEATURES], y)
                    p = np.maximum(m.predict(va[FEATURES]), 0.0)
                    if use_sqrt:
                        p = p ** 2
                    ps.append(p)
                preds.append(np.mean(ps, axis=0))
            else:
                m = make_model(seed)
                y = tr[TARGET].clip(lower=0).to_numpy()
                if use_sqrt:
                    y = np.sqrt(y)
                m.fit(tr[FEATURES], y)
                p = np.maximum(m.predict(va[FEATURES]), 0.0)
                if use_sqrt:
                    p = p ** 2
                preds.append(p)
        pred = np.mean(preds, axis=0)
        pred = np.where(va["functioning_day"].to_numpy() == 0, 0.0, pred)
        fold_scores.append(rmse(va[TARGET], pred))
    return float(np.mean(fold_scores)), fold_scores


def hgb(**kw):
    defaults = dict(
        loss="squared_error", learning_rate=0.06, max_iter=1500,
        max_leaf_nodes=63, min_samples_leaf=20, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=60,
    )
    defaults.update(kw)
    return lambda seed: HistGradientBoostingRegressor(random_state=seed, **defaults)


EXPERIMENTS = {
    "base": (hgb(), True, True),
    "no_sqrt": (hgb(), False, True),
    "no_two_model": (hgb(), True, False),
    "lr0.04": (hgb(learning_rate=0.04, max_iter=2500), True, True),
    "lr0.08": (hgb(learning_rate=0.08, max_iter=1000), True, True),
    "leaf31": (hgb(max_leaf_nodes=31), True, True),
    "leaf127": (hgb(max_leaf_nodes=127, min_samples_leaf=15), True, True),
    "l2_0.1": (hgb(l2_regularization=0.1), True, True),
    "l2_5": (hgb(l2_regularization=5.0), True, True),
    "minleaf40": (hgb(min_samples_leaf=40), True, True),
    "mse_loss_two": (hgb(loss="squared_error"), True, True),
    "abs_loss": (hgb(loss="absolute_error", max_iter=1500), False, True),
    "poisson": (hgb(loss="poisson"), False, True),
    "seeds3": (hgb(), True, True),
}


def main():
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    for name, (mk, use_sqrt, two) in EXPERIMENTS.items():
        if only and name not in only:
            continue
        seeds = (0, 1, 2) if name == "seeds3" else (0,)
        mean, folds = cv_eval(mk, use_sqrt=use_sqrt, two_model=two, seeds=seeds)
        print(f"{name:16s} CV={mean:8.2f}  folds={['%.1f' % f for f in folds]}", flush=True)

    # ExtraTrees / RandomForest (sqrt, two-model) 비교
    for name, mk in [
        ("extratrees", lambda seed: ExtraTreesRegressor(
            n_estimators=300, max_features=0.7, min_samples_leaf=2,
            n_jobs=-1, random_state=seed)),
        ("rf", lambda seed: RandomForestRegressor(
            n_estimators=300, max_features=0.7, min_samples_leaf=2,
            n_jobs=-1, random_state=seed)),
    ]:
        if only and name not in only:
            continue
        mean, folds = cv_eval(mk, use_sqrt=True, two_model=True)
        print(f"{name:16s} CV={mean:8.2f}  folds={['%.1f' % f for f in folds]}", flush=True)


if __name__ == "__main__":
    main()
