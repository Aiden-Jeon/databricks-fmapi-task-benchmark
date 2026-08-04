#!/usr/bin/env python3
"""Train a match-aware PUBG placement model and create the submission."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


ROOT = Path(__file__).resolve().parents[1]
KEYS = ["matchId", "groupId"]
TARGET = "winPlacePerc"
PLAYER_FEATURES = [
    "assists", "boosts", "damageDealt", "DBNOs", "headshotKills", "heals",
    "killPlace", "killPoints", "kills", "killStreaks", "longestKill",
    "rankPoints", "revives", "rideDistance", "roadKills", "swimDistance",
    "teamKills", "vehicleDestroys", "walkDistance", "weaponsAcquired",
    "winPoints",
]


def build_features(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one feature row and metadata row for every match/group pair."""
    data = rows.copy()
    data["totalDistance"] = data["rideDistance"] + data["walkDistance"] + data["swimDistance"]
    data["healsAndBoosts"] = data["heals"] + data["boosts"]
    data["headshotRate"] = data["headshotKills"] / data["kills"].clip(lower=1)
    data["killPlacePerc"] = data["killPlace"] / data["maxPlace"].clip(lower=1)
    features = PLAYER_FEATURES + ["totalDistance", "healsAndBoosts", "headshotRate", "killPlacePerc"]

    grouped = data.groupby(KEYS, sort=False)
    parts: list[pd.DataFrame] = []
    for statistic in ("mean", "max", "min", "sum"):
        values = grouped[features].agg(statistic).astype("float32")
        values.columns = [f"{column}_{statistic}" for column in values.columns]
        parts.append(values)
        if statistic != "min":
            ranks = values.groupby(level="matchId").rank(pct=True).astype("float32")
            ranks.columns = [f"{column}_matchRank" for column in values.columns]
            parts.append(ranks)

    result = pd.concat(parts, axis=1)
    first = grouped[["matchDuration", "maxPlace", "numGroups"]].first().astype("float32")
    match_type = grouped["matchType"].first()
    extra = first.copy()
    extra["groupSize"] = grouped.size().astype("float32")
    extra["matchSize"] = data.groupby("matchId").size().reindex(result.index, level="matchId").to_numpy(dtype="float32")
    extra["isFpp"] = match_type.str.contains("fpp").astype("float32")
    extra["isSolo"] = match_type.str.contains("solo").astype("float32")
    extra["isDuo"] = match_type.str.contains("duo").astype("float32")
    extra["isSquad"] = match_type.str.contains("squad").astype("float32")
    extra["isEvent"] = match_type.str.contains("flare|crash", regex=True).astype("float32")
    result = pd.concat([result, extra], axis=1)

    metadata = first[["maxPlace", "numGroups"]].copy()
    metadata["matchType"] = match_type
    return result.replace([np.inf, -np.inf], 0).fillna(0), metadata


def make_model(name: str, seed: int):
    if name == "extra":
        return ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            max_features=0.9,
            n_jobs=-1,
            random_state=seed,
        )
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.075,
        max_iter=350,
        max_leaf_nodes=63,
        min_samples_leaf=30,
        l2_regularization=0.5,
        early_stopping=False,
        random_state=seed,
    )


def rank_and_snap(prediction: np.ndarray, metadata: pd.DataFrame, rank_weight: float = 0.75) -> np.ndarray:
    """Enforce team ordering and the discrete placement values used by PUBG."""
    frame = metadata.copy()
    raw = np.clip(prediction, 0, 1)
    frame["prediction"] = raw
    frame["prediction"] = frame.groupby(level="matchId")["prediction"].rank(method="average", pct=True)
    groups = frame["numGroups"].clip(lower=1)
    frame["prediction"] = (frame["prediction"] * groups - 1) / (groups - 1).clip(lower=1)
    frame["prediction"] = rank_weight * frame["prediction"] + (1 - rank_weight) * raw

    max_place = frame["maxPlace"].round().astype(int)
    gap = 1 / (max_place - 1).clip(lower=1)
    snapped = (frame["prediction"] / gap).round() * gap
    snapped[max_place <= 1] = 1.0
    snapped[max_place == 0] = 0.0
    return snapped.clip(0, 1).to_numpy()


def target_by_group(rows: pd.DataFrame, index: pd.Index) -> pd.Series:
    spread = rows.groupby(KEYS)[TARGET].agg(["min", "max"])
    if not np.allclose(spread["min"], spread["max"]):
        raise ValueError("Target is not constant within a group")
    return spread["max"].reindex(index)


def validate(train: pd.DataFrame, features: pd.DataFrame, metadata: pd.DataFrame, model_name: str, seed: int) -> None:
    matches = np.array(sorted(train["matchId"].unique()))
    rng = np.random.default_rng(seed)
    valid_matches = set(rng.choice(matches, size=max(1, len(matches) // 5), replace=False))
    valid_mask = features.index.get_level_values("matchId").isin(valid_matches)
    target = target_by_group(train, features.index)

    model = make_model(model_name, seed)
    model.fit(features.loc[~valid_mask], target.loc[~valid_mask])
    raw = np.clip(model.predict(features.loc[valid_mask]), 0, 1)
    truth = target.loc[valid_mask]
    print(f"validation groups={valid_mask.sum()}, raw_mae={mean_absolute_error(truth, raw):.6f}")
    for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        post = rank_and_snap(raw, metadata.loc[valid_mask], weight)
        print(f"validation rank_weight={weight:.2f}, snapped_mae={mean_absolute_error(truth, post):.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["hist", "extra"], default="hist")
    parser.add_argument("--cv", action="store_true", help="print one match-level holdout score")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    train = pd.read_csv(ROOT / "train.csv")
    train_features, train_meta = build_features(train)
    print(f"training rows={len(train):,}, groups={len(train_features):,}, features={train_features.shape[1]}")
    if args.cv:
        validate(train, train_features, train_meta, args.model, args.seed)
        return

    test = pd.read_csv(ROOT / "test.csv")
    test_features, test_meta = build_features(test)
    target = target_by_group(train, train_features.index)
    model = make_model(args.model, args.seed)
    model.fit(train_features, target)
    prediction = rank_and_snap(model.predict(test_features), test_meta)

    group_prediction = pd.Series(prediction, index=test_features.index, name=TARGET)
    submission = test[["Id", "matchId", "groupId"]].join(group_prediction, on=KEYS)[["Id", TARGET]]
    output = ROOT / "outputs" / "submission.csv"
    output.parent.mkdir(exist_ok=True)
    submission.to_csv(output, index=False)
    print(f"wrote {len(submission):,} predictions to {output}")


if __name__ == "__main__":
    main()
