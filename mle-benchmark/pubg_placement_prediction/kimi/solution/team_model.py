"""팀-레벨 모델: 팀당 1행으로 축소한 데이터로 winPlacePerc 예측.

플레이어-레벨 모델과 앙상블하기 위한 보완 모델.
팀 피처 = 팀원 스탯의 mean/max/sum + 매치 내 팀 rank + matchType.
"""
import numpy as np
import pandas as pd

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import BASE_NUMERIC, TARGET, _simplify_match_type

NUM_COLS = [c for c in BASE_NUMERIC if c not in ("matchDuration", "maxPlace", "numGroups",
                                                 "rankPoints", "killPoints", "winPoints")]


def build_team_frame(df: pd.DataFrame) -> pd.DataFrame:
    """플레이어 df를 팀(df.index = _gid) 프레임으로 축소."""
    df = df.copy()
    df["_gid"] = df["matchId"] + "_" + df["groupId"]
    df["totalDistance"] = df["walkDistance"] + df["rideDistance"] + df["swimDistance"]
    df["healsPlusBoosts"] = df["heals"] + df["boosts"]
    df["combatScore"] = df["damageDealt"] + 100.0 * df["kills"]

    agg_src = NUM_COLS + ["totalDistance", "healsPlusBoosts", "combatScore"]
    g = df.groupby("_gid", sort=False)

    team = pd.DataFrame(index=g.size().index)
    team["matchId"] = g["matchId"].first()
    team["teamSize"] = g.size()
    for c in agg_src:
        team[f"mean_{c}"] = g[c].mean()
        team[f"max_{c}"] = g[c].max()
        team[f"sum_{c}"] = g[c].sum()
    # 매치 컨텍스트
    team["matchDuration"] = g["matchDuration"].first()
    team["matchPlayerCount"] = g["Id"].transform("size").groupby(team["matchId"]).first() \
        if False else team["matchId"].map(df.groupby("matchId").size())
    team["matchType"] = g["matchType"].first()

    # 매치 내 팀 rank
    for c in ["sum_kills", "sum_damageDealt", "mean_totalDistance", "max_damageDealt",
              "sum_healsPlusBoosts", "mean_walkDistance", "sum_weaponsAcquired", "teamSize",
              "mean_killPlace", "max_kills", "sum_boosts", "sum_heals"]:
        asc = False if c == "mean_killPlace" else True
        team[f"trank_{c}"] = team.groupby("matchId")[c].rank(pct=True, ascending=asc)

    # 파생
    team["killsPerMember"] = team["sum_kills"] / team["teamSize"]
    team["damagePerMember"] = team["sum_damageDealt"] / team["teamSize"]
    team["distPerMember"] = team["sum_totalDistance"] / team["teamSize"]
    team["matchTypeSimple"] = _simplify_match_type(team["matchType"])
    common = {"squad-fpp", "duo-fpp", "squad", "solo-fpp", "duo", "solo"}
    team["matchTypeCat"] = team["matchType"].astype(str).where(
        team["matchType"].astype(str).isin(common), "other")
    return team


def encode_team(team: pd.DataFrame, categories=None):
    cat_cols = ["matchTypeSimple", "matchTypeCat"]
    if categories is None:
        categories = {c: pd.Categorical(team[c]).categories for c in cat_cols}
    team = team.copy()
    for c in cat_cols:
        team[c] = pd.Categorical(team[c], categories=categories[c]).codes.astype(np.int32)
    return team, categories


TEAM_TARGET = TARGET
