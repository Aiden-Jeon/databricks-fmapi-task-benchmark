"""
PUBG Finish Placement Prediction (t1_pubg)
Group-level LightGBM regression (MAE) + match-rank post-processing.

Usage:
    python solution/pipeline.py            # full run: CV + fit + submission
"""
import os
import gc
import sys
import time
import numpy as np
import pandas as pd

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

SEED = 42


def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
MATCH_TYPE_MAP = {
    "solo": ("solo", 1), "solo-fpp": ("solo", 1), "normal-solo": ("solo", 1),
    "normal-solo-fpp": ("solo", 1),
    "duo": ("duo", 2), "duo-fpp": ("duo", 2), "normal-duo": ("duo", 2),
    "normal-duo-fpp": ("duo", 2),
    "squad": ("squad", 4), "squad-fpp": ("squad", 4), "normal-squad": ("squad", 4),
    "normal-squad-fpp": ("squad", 4),
    "crashfpp": ("other", 4), "crashtpp": ("other", 4),
    "flarefpp": ("other", 4), "flaretpp": ("other", 4),
}


def add_player_features(df):
    eps = 1e-6
    df["totalDistance"] = df["walkDistance"] + df["rideDistance"] + df["swimDistance"]
    df["healsBoosts"] = df["heals"] + df["boosts"]
    df["killsAssists"] = df["kills"] + df["assists"]
    df["killsAssistsDBNO"] = df["kills"] + df["assists"] + df["DBNOs"]
    df["headshotRate"] = df["headshotKills"] / (df["kills"] + eps)
    df["killStreakRate"] = df["killStreaks"] / (df["kills"] + eps)
    df["damagePerKill"] = df["damageDealt"] / (df["kills"] + eps)
    df["killPlacePerc"] = df["killPlace"] / df["maxPlace"]
    df["killPlaceOverNumGroups"] = df["killPlace"] / df["numGroups"]
    df["walkPerDuration"] = df["walkDistance"] / (df["matchDuration"] + eps)
    df["distPerDuration"] = df["totalDistance"] / (df["matchDuration"] + eps)
    df["itemsPerDistance"] = df["weaponsAcquired"] / (df["totalDistance"] + 1.0)
    df["healsPerDistance"] = df["healsBoosts"] / (df["totalDistance"] + 1.0)
    df["killsPerDistance"] = df["kills"] / (df["totalDistance"] + 1.0)
    df["damagePerDistance"] = df["damageDealt"] / (df["totalDistance"] + 1.0)
    df["killsPerDuration"] = df["kills"] / (df["matchDuration"] + eps)
    df["damagePerDuration"] = df["damageDealt"] / (df["matchDuration"] + eps)
    df["boostsPerWalk"] = df["boosts"] / (df["walkDistance"] + 1.0)
    df["totalItems"] = df["healsBoosts"] + df["weaponsAcquired"]
    df["skill"] = df["headshotKills"] + df["roadKills"]
    df["killsNorm"] = df["kills"] * ((100 - df["numGroups"]) / 100 + 1)
    df["damageNorm"] = df["damageDealt"] * ((100 - df["numGroups"]) / 100 + 1)
    df["maxPlaceNorm"] = df["maxPlace"] / 100.0
    df["numGroupsRatio"] = df["numGroups"] / df["maxPlace"]
    df["walkOverLongest"] = df["walkDistance"] / (df["longestKill"] + 1.0)
    df["zeroWalk"] = (df["walkDistance"] <= 0.0).astype(np.int8)
    df["hasKill"] = (df["kills"] > 0).astype(np.int8)
    df["rankPointsFix"] = df["rankPoints"].replace(-1, 0)
    df["ptsSum"] = df["killPoints"] + df["winPoints"] + df["rankPointsFix"]
    return df


# aggregated over group members
AGG_COLS = [
    "assists", "boosts", "damageDealt", "DBNOs", "headshotKills", "heals",
    "killPlace", "kills", "killStreaks", "longestKill", "revives",
    "rideDistance", "roadKills", "swimDistance", "teamKills", "vehicleDestroys",
    "walkDistance", "weaponsAcquired", "killPoints", "winPoints", "rankPointsFix",
    "totalDistance", "healsBoosts", "killsAssists", "killsAssistsDBNO",
    "headshotRate", "killStreakRate", "damagePerKill", "killPlacePerc",
    "killPlaceOverNumGroups", "walkPerDuration", "distPerDuration",
    "itemsPerDistance", "healsPerDistance", "killsPerDistance",
    "damagePerDistance", "killsPerDuration", "damagePerDuration",
    "boostsPerWalk", "totalItems", "skill", "killsNorm", "damageNorm",
    "walkOverLongest", "zeroWalk", "hasKill", "ptsSum",
]

# match-level constants (same for every row of a match)
MATCH_COLS = ["maxPlace", "numGroups", "matchDuration", "maxPlaceNorm", "numGroupsRatio"]

# columns for which we build a rank-within-match feature (subset -> keep width sane)
RANK_STATS = ["mean", "max", "min"]


def build_group_features(df):
    """Return group-level feature frame (one row per matchId/groupId)."""
    df = add_player_features(df)
    keys = ["matchId", "groupId"]

    for c in AGG_COLS:
        df[c] = df[c].astype(np.float32)

    log("  aggregating group stats ...")
    g = df.groupby(keys, sort=False)
    agg = g[AGG_COLS].agg(["mean", "max", "min", "sum"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.astype(np.float32)

    agg["group_size"] = g.size().astype(np.float32)

    # match constants
    mc = g[MATCH_COLS].first().astype(np.float32)
    agg = agg.join(mc)

    # match type
    mt = g["matchType"].first()
    cat = mt.map(lambda x: MATCH_TYPE_MAP.get(x, ("other", 4))[0])
    exp = mt.map(lambda x: MATCH_TYPE_MAP.get(x, ("other", 4))[1]).astype(np.float32)
    agg["mt_expected_size"] = exp.values
    agg["mt_cat"] = pd.Categorical(cat.values, categories=["solo", "duo", "squad", "other"]).codes
    agg["mt_cat"] = agg["mt_cat"].astype(np.int8)
    agg["is_fpp"] = mt.str.contains("fpp").astype(np.int8).values
    agg["is_normal"] = mt.str.startswith("normal").astype(np.int8).values
    agg["size_vs_expected"] = agg["group_size"] / agg["mt_expected_size"]

    del df, g
    gc.collect()

    agg = agg.reset_index()

    # ---- match aggregates & ranks ----
    rank_src = [f"{c}_{s}" for c in AGG_COLS for s in RANK_STATS]
    log(f"  match ranks over {len(rank_src)} cols ...")
    mg = agg.groupby("matchId", sort=False)

    ranks = mg[rank_src].rank(pct=True, method="average").astype(np.float32)
    ranks.columns = [f"{c}_mrank" for c in rank_src]

    # match mean / max of the group means -> relative position features
    mean_src = [f"{c}_mean" for c in AGG_COLS]
    mmean = mg[mean_src].transform("mean").astype(np.float32)
    mmean.columns = [f"{c}_mmean" for c in mean_src]
    mmax = mg[mean_src].transform("max").astype(np.float32)

    rel = pd.DataFrame(index=agg.index)
    for c in mean_src:
        rel[f"{c}_relmax"] = (agg[c] / (mmax[c] + 1e-6)).astype(np.float32)

    agg["match_groups_obs"] = mg["group_size"].transform("size").astype(np.float32)
    agg["match_players"] = mg["group_size"].transform("sum").astype(np.float32)
    agg["match_size_ratio"] = agg["match_players"] / (agg["maxPlace"] + 1e-6)
    agg["group_size_rank"] = mg["group_size"].rank(pct=True).astype(np.float32)

    out = pd.concat([agg, ranks, mmean, rel], axis=1)
    del agg, ranks, mmean, mmax, rel
    gc.collect()
    out = out.replace([np.inf, -np.inf], np.nan)
    log(f"  group feature frame: {out.shape}")
    return out


# --------------------------------------------------------------------------- #
# Post-processing
# --------------------------------------------------------------------------- #
def postprocess(gdf, pred, mode="rank_grid"):
    """gdf: group frame with matchId, maxPlace, numGroups (actual observed groups).
    Returns adjusted group-level predictions."""
    d = pd.DataFrame({
        "matchId": gdf["matchId"].values,
        "maxPlace": gdf["maxPlace"].values.astype(int),
        "pred": np.clip(pred, 0.0, 1.0),
    })
    if mode == "raw":
        return d["pred"].values

    if mode == "snap":
        gap = 1.0 / np.maximum(d["maxPlace"].values - 1, 1)
        out = np.round(d["pred"].values / gap) * gap
        out = np.clip(out, 0.0, 1.0)
        out[d["maxPlace"].values <= 1] = 0.0
        return out

    # rank based
    d["rank"] = d.groupby("matchId")["pred"].rank(method="first")
    nobs = d.groupby("matchId")["rank"].transform("max")
    mp = d["maxPlace"].values.astype(float)

    if mode == "rank_obs":
        out = (d["rank"].values - 1) / np.maximum(nobs.values - 1, 1)
    else:  # rank_grid: spread ranks over the maxPlace grid
        # place best group at 1.0, worst at 0.0, spacing = 1/(maxPlace-1) steps
        denom = np.maximum(nobs.values - 1, 1)
        frac = (d["rank"].values - 1) / denom
        gap = 1.0 / np.maximum(mp - 1, 1)
        out = np.round(frac / gap) * gap
    out = np.clip(out, 0.0, 1.0)
    out[mp <= 1] = 0.0
    return out


# --------------------------------------------------------------------------- #
def main():
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold

    log("loading data")
    tr = pd.read_csv(os.path.join(ROOT, "train.csv"))
    te = pd.read_csv(os.path.join(ROOT, "test.csv"))
    test_ids = te[["Id", "matchId", "groupId"]].copy()

    log("building train features")
    ytab = tr.groupby(["matchId", "groupId"], sort=False)["winPlacePerc"].first().reset_index()
    Xtr = build_group_features(tr.drop(columns=["winPlacePerc"]))
    Xtr = Xtr.merge(ytab, on=["matchId", "groupId"], how="left")
    del tr
    gc.collect()

    log("building test features")
    Xte = build_group_features(te)
    del te
    gc.collect()

    drop = ["matchId", "groupId", "winPlacePerc"]
    feats = [c for c in Xtr.columns if c not in drop]
    feats = [c for c in feats if c in Xte.columns]
    log(f"n_features = {len(feats)}")

    y = Xtr["winPlacePerc"].values.astype(np.float32)
    w = Xtr["group_size"].values.astype(np.float32)  # weight by #players in group
    groups = Xtr["matchId"].values

    Xtr_m = Xtr[feats].values.astype(np.float32)
    Xte_m = Xte[feats].values.astype(np.float32)

    params = dict(
        objective="mae", metric="mae", learning_rate=0.05,
        num_leaves=255, min_data_in_leaf=20, feature_fraction=0.6,
        bagging_fraction=0.85, bagging_freq=1, lambda_l2=1.0,
        max_bin=255, num_threads=4, verbosity=-1, seed=SEED,
    )
    NFOLD = 5
    ROUNDS = 6000

    gkf = GroupKFold(n_splits=NFOLD)
    oof = np.zeros(len(Xtr), dtype=np.float64)
    test_pred = np.zeros(len(Xte), dtype=np.float64)
    best_iters = []

    for f, (itr, iva) in enumerate(gkf.split(Xtr_m, y, groups)):
        dtr = lgb.Dataset(Xtr_m[itr], y[itr], weight=w[itr], feature_name=feats)
        dva = lgb.Dataset(Xtr_m[iva], y[iva], weight=w[iva], feature_name=feats)
        m = lgb.train(params, dtr, num_boost_round=ROUNDS, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(150, verbose=False),
                                 lgb.log_evaluation(500)])
        oof[iva] = m.predict(Xtr_m[iva], num_iteration=m.best_iteration)
        test_pred += m.predict(Xte_m, num_iteration=m.best_iteration) / NFOLD
        best_iters.append(m.best_iteration)
        log(f"fold {f}: best_iter={m.best_iteration} mae={m.best_score['valid_0']['l1']:.6f}")
        del dtr, dva, m
        gc.collect()

    # ---- evaluate post-processing options on OOF (player-weighted MAE) ----
    log("post-processing comparison (player-weighted group MAE):")
    best_mode, best_mae = "raw", 9e9
    for mode in ["raw", "snap", "rank_obs", "rank_grid"]:
        p = postprocess(Xtr, oof, mode)
        mae = np.average(np.abs(p - y), weights=w)
        log(f"  {mode:10s} -> {mae:.6f}")
        if mae < best_mae:
            best_mae, best_mode = mae, mode
    log(f"selected post-process: {best_mode} (oof MAE {best_mae:.6f})")

    np.save("/tmp/oof.npy", oof)
    np.save("/tmp/testpred.npy", test_pred)
    Xtr[["matchId", "groupId", "group_size", "maxPlace", "winPlacePerc"]].to_pickle("/tmp/xtr_meta.pkl")
    Xte[["matchId", "groupId", "group_size", "maxPlace"]].to_pickle("/tmp/xte_meta.pkl")

    # ---- final submission ----
    final = postprocess(Xte, test_pred, best_mode)
    Xte["winPlacePerc"] = final
    sub = test_ids.merge(Xte[["matchId", "groupId", "winPlacePerc"]],
                         on=["matchId", "groupId"], how="left")
    assert sub["winPlacePerc"].notnull().all()
    sub = sub[["Id", "winPlacePerc"]]

    samp = pd.read_csv(os.path.join(ROOT, "sample_submission.csv"))
    sub = samp[["Id"]].merge(sub, on="Id", how="left")
    assert sub["winPlacePerc"].notnull().all() and len(sub) == len(samp)
    sub.to_csv(os.path.join(OUT, "submission.csv"), index=False)
    log(f"wrote {OUT}/submission.csv  rows={len(sub)}")
    log(f"mean best_iter={np.mean(best_iters):.0f}")


if __name__ == "__main__":
    main()
