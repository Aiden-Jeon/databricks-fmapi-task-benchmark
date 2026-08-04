"""Build and cache group-level features to /tmp/feat_{train,test}.pkl"""
import os, gc, pandas as pd
from pipeline import build_group_features, log

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.environ.get("PUBG_CACHE", "/tmp/pubg_cache")
os.makedirs(CACHE, exist_ok=True)

if __name__ == "__main__":
    log("loading")
    tr = pd.read_csv(os.path.join(ROOT, "train.csv"))
    ytab = tr.groupby(["matchId", "groupId"], sort=False)["winPlacePerc"].first().reset_index()
    Xtr = build_group_features(tr.drop(columns=["winPlacePerc"]))
    Xtr = Xtr.merge(ytab, on=["matchId", "groupId"], how="left")
    Xtr.to_pickle(os.path.join(CACHE, "feat_train.pkl"))
    del tr, Xtr; gc.collect()

    te = pd.read_csv(os.path.join(ROOT, "test.csv"))
    te[["Id", "matchId", "groupId"]].to_pickle(os.path.join(CACHE, "test_ids.pkl"))
    Xte = build_group_features(te)
    Xte.to_pickle(os.path.join(CACHE, "feat_test.pkl"))
    log("cached to " + CACHE)
