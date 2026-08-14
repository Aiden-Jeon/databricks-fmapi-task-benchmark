"""GroupKFold(matchId) CV로 MAE 측정 + 매치 내 후처리 효과 검증."""
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

PARAMS = dict(
    loss="absolute_error",
    max_iter=600,
    learning_rate=0.05,
    max_leaf_nodes=63,
    min_samples_leaf=30,
    l2_regularization=1.0,
    early_stopping=False,
    random_state=42,
)


def match_rank_postprocess(pred: np.ndarray, match_ids: pd.Series, target_max=1.0) -> np.ndarray:
    """매치 내 예측 순위를 유지하며 [0,1] 균등 분위로 재배치하는 후처리는 하지 않음.

    대신 실제 winPlacePerc는 매치 내 '플레이어' 기준이 아니라 팀 기준 순위라
    동점이 많다. 여기서는 단순 클리핑만 적용하고, 대안 후처리는 cv에서 검증.
    """
    return np.clip(pred, 0.0, 1.0)


def main():
    train = pd.read_csv(os.path.join(ROOT, "train.csv"))
    X = build_features(train)
    X, cats = encode_categoricals(X)
    y = train[TARGET].values
    groups = train["matchId"].values

    gkf = GroupKFold(n_splits=5)
    maes = []
    maes_pp = []
    t0 = time.time()
    for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
        model = HistGradientBoostingRegressor(**PARAMS)
        model.fit(X.iloc[tr].values, y[tr])
        pred = model.predict(X.iloc[va].values)
        pred_clip = np.clip(pred, 0, 1)
        mae = mean_absolute_error(y[va], pred_clip)
        maes.append(mae)

        # 후처리 실험: 매치 내 예측값의 rank를 실제 타깃의 매치 내 분포에 맞춤.
        # winPlacePerc = 1 - (place-1)/(maxPlace-1) 형태. 플레이어 단위가 아니라
        # 그룹 단위 순위이므로 여기서는 "매치 내 균등분위 재배치"를 시도.
        va_match = pd.Series(groups[va])
        pred_pp = pred_clip.copy()
        # 매치별로 예측을 순위화해 (n-1)로 나눈 균등 백분위로 치환
        df_va = pd.DataFrame({"m": va_match.values, "p": pred_clip})
        df_va["pp"] = df_va.groupby("m")["p"].rank(method="average")
        cnt = df_va.groupby("m")["p"].transform("size")
        df_va["pp"] = (df_va["pp"] - 1.0) / (cnt - 1.0).clip(lower=1.0)
        pred_pp2 = df_va["pp"].values
        mae_pp = mean_absolute_error(y[va], pred_pp2)
        maes_pp.append(mae_pp)

        # 블렌드: 원래 예측과 균등분위 재배치의 평균
        blend = 0.5 * pred_clip + 0.5 * pred_pp2
        mae_blend = mean_absolute_error(y[va], blend)

        print(f"fold{fold}: MAE={mae:.5f}  rankPP={mae_pp:.5f}  blend={mae_blend:.5f}  "
              f"iter={model.n_iter_}  {time.time()-t0:.0f}s")
        maes_pp.append(mae_blend)

    print(f"\nCV MAE (clip):      {np.mean(maes):.5f} +- {np.std(maes):.5f}")
    print(f"CV MAE (blend pp):  {np.mean(maes_pp):.5f}")


if __name__ == "__main__":
    main()
