"""피처 엔지니어링 모듈 (train/test 공통 적용).

모든 집계 피처는 각 행이 속한 매치(matchId)/그룹(groupId) 내부에서만 계산되며
매치 외부 정보 누수는 없다. train/test 모두 매치 단위로 완전한 로그가 주어지므로
동일한 함수로 처리 가능하다.
"""
import numpy as np
import pandas as pd

# groupId/matchId 해시 충돌 방지를 위해 매치 내 그룹 식별자 사용
TARGET = "winPlacePerc"
ID_COL = "Id"

BASE_NUMERIC = [
    "assists", "boosts", "damageDealt", "DBNOs", "headshotKills", "heals",
    "killPlace", "killPoints", "kills", "killStreaks", "longestKill",
    "matchDuration", "maxPlace", "numGroups", "rankPoints", "revives",
    "rideDistance", "roadKills", "swimDistance", "teamKills",
    "vehicleDestroys", "walkDistance", "weaponsAcquired", "winPoints",
]

# 그룹 집계를 만들 원천 컬럼
AGG_COLS = [
    "kills", "damageDealt", "walkDistance", "rideDistance", "swimDistance",
    "boosts", "heals", "assists", "DBNOs", "revives", "weaponsAcquired",
    "killStreaks", "headshotKills", "longestKill", "killPlace",
]


def _simplify_match_type(s: pd.Series) -> pd.Series:
    """matchType을 squad/duo/solo 계열과 fpp 여부로 단순화."""
    s = s.astype(str)
    main = pd.Series("other", index=s.index)
    main[s.str.contains("squad")] = "squad"
    main[s.str.contains("duo")] = "duo"
    main[s.str.contains("solo")] = "solo"
    main[s.str.contains("crash")] = "crash"
    main[s.str.contains("flare")] = "flare"
    return main


def build_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """원본 df를 받아 모델 입력 피처 X를 반환."""
    df = df.copy()
    # 매치 내 그룹 식별자 (groupId는 해시라 매치 간 중복 가능성 대비)
    df["_gid"] = df["matchId"] + "_" + df["groupId"]

    out = pd.DataFrame(index=df.index)

    # 1) 기본 수치 컬럼
    for c in BASE_NUMERIC:
        out[c] = df[c]

    # 2) 파생 비율/합성 피처
    out["totalDistance"] = df["walkDistance"] + df["rideDistance"] + df["swimDistance"]
    out["walkOverTotal"] = df["walkDistance"] / (out["totalDistance"] + 1.0)
    out["headshotRate"] = df["headshotKills"] / (df["kills"] + 1.0)
    out["damagePerKill"] = df["damageDealt"] / (df["kills"] + 1.0)
    out["killsPerKm"] = df["kills"] / (df["walkDistance"] + 100.0) * 1000.0
    out["healsPlusBoosts"] = df["heals"] + df["boosts"]
    out["combatScore"] = df["damageDealt"] + 100.0 * df["kills"]
    out["itemsPerKm"] = (df["heals"] + df["boosts"]) / (out["totalDistance"] + 100.0) * 1000.0
    out["killPlaceDiff"] = df["killPlace"] - df["kills"]

    # 3) 그룹(팀) 집계 피처: 매치 내 그룹 단위
    g = df.groupby("_gid", sort=False)
    out["teamSize"] = g[BASE_NUMERIC[0]].transform("size")
    for c in AGG_COLS:
        grp_sum = g[c].transform("sum")
        grp_max = g[c].transform("max")
        grp_mean = g[c].transform("mean")
        out[f"teamSum_{c}"] = grp_sum
        out[f"teamMax_{c}"] = grp_max
        out[f"teamMean_{c}"] = grp_mean

    # 팀원 수 보정된 개인 기여 비율
    out["killShareInTeam"] = df["kills"] / (out["teamSum_kills"] + 1.0)
    out["damageShareInTeam"] = df["damageDealt"] / (out["teamSum_damageDealt"] + 1.0)

    # 4) 매치 단위 집계/정규화
    m = df.groupby("matchId", sort=False)
    out["matchPlayerCount"] = m[BASE_NUMERIC[0]].transform("size")
    out["matchKills"] = m["kills"].transform("sum")
    out["killShareInMatch"] = df["kills"] / (out["matchKills"] + 1.0)

    # 매치 내 상대 위치 (rank transform): 높을수록 좋은 지표들
    for c in ["kills", "damageDealt", "walkDistance", "totalDistance", "teamSum_kills",
              "teamSum_damageDealt", "teamSum_walkDistance"]:
        # pct rank within match
        out[f"mRank_{c}"] = out.groupby(df["matchId"])[c].rank(pct=True)

    # killPlace는 매치 내 킬 순위이므로 그 자체로 강력, 정규화 버전 추가
    out["killPlaceNorm"] = df["killPlace"] / (out["matchPlayerCount"] + 1.0)

    # maxPlace / numGroups 스케일 보정 (비정상 maxPlace 대비)
    out["maxPlaceNorm"] = df["maxPlace"] / (out["matchPlayerCount"] + 1.0)
    out["numGroupsNorm"] = df["numGroups"] / (out["matchPlayerCount"] + 1.0)

    # 5) matchType 카테고리 (단순화 + 원본)
    out["matchTypeSimple"] = _simplify_match_type(df["matchType"])
    mt = df["matchType"].astype(str)
    # 희귀 타입은 other로 묶기
    common = {"squad-fpp", "duo-fpp", "squad", "solo-fpp", "duo", "solo"}
    out["matchTypeCat"] = mt.where(mt.isin(common), "other")

    # 6) 플레이어 수 보정된 matchDuration 활용
    out["durationNorm"] = df["matchDuration"] / 2200.0  # 대략적 최대치 스케일

    return out


def encode_categoricals(X: pd.DataFrame, categories=None):
    """카테고리 컬럼을 정수 인코딩. categories가 주어지면 그 매핑 재사용."""
    cat_cols = ["matchTypeSimple", "matchTypeCat"]
    if categories is None:
        categories = {}
        for c in cat_cols:
            cats = pd.Categorical(X[c]).categories
            categories[c] = cats
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=categories[c]).codes.astype(np.int32)
    return X, categories


def get_feature_columns(X: pd.DataFrame):
    return list(X.columns)
