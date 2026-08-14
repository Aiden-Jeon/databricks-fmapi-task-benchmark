"""v3 피처: v2 + 이상치 정제 + 추가 매치 내 상대 피처.

정제:
- maxPlace > matchPlayerCount 인 매치는 maxPlace를 매치 인원으로 대체
추가:
- player-level 매치 내 rank (kills/damageDealt/totalDistance/walkDistance/health items)
- 팀의 매치 내 '상대 강함' = 팀 스탯 rank들의 조합은 모델이 학습하므로 원시 rank만 추가
"""
import numpy as np
import pandas as pd

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import build_features as build_v1, TARGET, AGG_COLS

TEAM_STAT_FOR_RANK = [
    "teamSum_kills", "teamSum_damageDealt", "teamMean_walkDistance",
    "teamMax_damageDealt", "teamMax_kills", "teamMean_totalDistance",
    "teamSum_healsPlusBoosts", "teamMean_killPlace", "teamSize",
]

# 플레이어 레벨 매치 내 rank를 추가할 컬럼 (높을수록 좋음)
PLAYER_RANK_COLS = [
    "kills", "damageDealt", "walkDistance", "totalDistance",
    "healsPlusBoosts", "weaponsAcquired", "boosts", "heals",
    "killStreaks", "assists", "revives", "longestKill",
]


def build_features_v3(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    df = df.copy()

    # --- 이상치 정제: maxPlace가 매치 실제 인원보다 큰 경우 ---
    match_sizes = df.groupby("matchId", sort=False)["Id"].transform("size")
    bad = df["maxPlace"] > match_sizes
    if bad.any():
        df.loc[bad, "maxPlace"] = match_sizes[bad]
        # numGroups도 최대 maxPlace 이하로 클리핑
        df.loc[bad, "numGroups"] = np.minimum(df.loc[bad, "numGroups"], df.loc[bad, "maxPlace"])

    X = build_v1(df, is_train)

    df["_gid"] = df["matchId"] + "_" + df["groupId"]

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

    team_key = df["_gid"]
    match_of_team = df.groupby("_gid", sort=False)["matchId"].first()
    team_tbl = team_stats.groupby(team_key, sort=False).first()
    team_tbl["_match"] = match_of_team

    rank_cols = {}
    for c in TEAM_STAT_FOR_RANK:
        if c == "teamMean_killPlace":
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

    # --- 플레이어 레벨 매치 내 rank (v1에 일부 있었지만 보강) ---
    pr = {}
    pr_src = pd.DataFrame({
        "kills": df["kills"], "damageDealt": df["damageDealt"],
        "walkDistance": df["walkDistance"], "totalDistance": total_dist,
        "healsPlusBoosts": hb, "weaponsAcquired": df["weaponsAcquired"],
        "boosts": df["boosts"], "heals": df["heals"],
        "killStreaks": df["killStreaks"], "assists": df["assists"],
        "revives": df["revives"], "longestKill": df["longestKill"],
    }, index=X.index)
    grp = pr_src.groupby(df["matchId"], sort=False)
    for c in PLAYER_RANK_COLS:
        pr[f"pRank_{c}"] = grp[c].rank(pct=True).astype(np.float32)
    # 팀 스탯의 매치 내 rank를 개인에게도 (팀 내 상대 기여와 조합되도록)
    pr["teamContrib_kills"] = (df["kills"] / (X["teamSum_kills"] + 1.0)).astype(np.float32)

    new_cols.update(pr)
    X = pd.concat([X, pd.DataFrame(new_cols, index=X.index)], axis=1)
    return X
