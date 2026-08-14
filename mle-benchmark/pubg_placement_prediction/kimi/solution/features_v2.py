"""v2 피처: v1 + '팀 간 상대 순위' 피처 추가.

winPlacePerc는 매치 내 팀(그룹) 순위로 결정되므로, 팀 스탯을 매치 내 다른 팀과
비교한 rank 피처가 가장 직접적인 신호가 된다.
"""
import numpy as np
import pandas as pd

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import build_features as build_v1, TARGET

# 매치 내 팀 순위를 매길 팀 스탯
TEAM_STAT_FOR_RANK = [
    "teamSum_kills", "teamSum_damageDealt", "teamMean_walkDistance",
    "teamMax_damageDealt", "teamMax_kills", "teamMean_totalDistance",
    "teamSum_healsPlusBoosts", "teamMean_killPlace", "teamSize",
]


def build_features_v2(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    X = build_v1(df, is_train)

    df = df.copy()
    df["_gid"] = df["matchId"] + "_" + df["groupId"]

    # 팀 스탯 (팀 내 동일값)
    total_dist = df["walkDistance"] + df["rideDistance"] + df["swimDistance"]
    hb = df["heals"] + df["boosts"]
    team_stats = pd.DataFrame({
        "teamSum_kills": X["teamSum_kills"],
        "teamSum_damageDealt": X["teamSum_damageDealt"],
        "teamMean_walkDistance": X["teamMean_walkDistance"],
        "teamMax_damageDealt": X["teamMax_damageDealt"],
        "teamMax_kills": X["teamMax_kills"],
        "teamMean_totalDistance": total_dist.groupby(df["_gid"]).transform("mean"),
        "teamSum_healsPlusBoosts": hb.groupby(df["_gid"]).transform("sum"),
        "teamMean_killPlace": X["teamMean_killPlace"],
        "teamSize": X["teamSize"],
    }, index=X.index)

    # 팀 단위 테이블로 축소 후 매치 내 rank 계산, 플레이어로 다시 매핑
    team_key = df["_gid"]
    match_of_team = df.groupby("_gid", sort=False)["matchId"].first()
    team_tbl = team_stats.groupby(team_key, sort=False).first()
    team_tbl["_match"] = match_of_team

    rank_cols = {}
    for c in TEAM_STAT_FOR_RANK:
        if c == "teamMean_killPlace":
            # killPlace는 낮을수록 좋으므로 rank 방향 반전
            r = team_tbl.groupby("_match")[c].rank(pct=True, ascending=False)
        else:
            r = team_tbl.groupby("_match")[c].rank(pct=True, ascending=True)
        rank_cols[c] = r
    rank_df = pd.DataFrame(rank_cols)

    mapped = rank_df.reindex(df["_gid"].values)
    mapped.index = X.index
    mapped = mapped.astype(np.float32)
    new_cols = {f"teamRank_{c}": mapped[c] for c in TEAM_STAT_FOR_RANK}
    new_cols["teamKillRank_x_size"] = mapped["teamSum_kills"] * X["teamSize"]

    X = pd.concat([X, pd.DataFrame(new_cols, index=X.index)], axis=1)
    return X
