"""Train char-level averaged perceptron NER and write submission.

Usage:
  python solution/run.py dev      # 90/10 split, report dev F1 per epoch
  python solution/run.py full     # train on all data, write outputs/submission.csv
"""
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load, bio_to_ents, format_entities, micro_f1  # noqa
from features import build_gaz, gaz_matches, sent_features  # noqa
from tagger import TAGS, TAG2I, Tagger  # noqa

NFOLD = 5
EPOCHS = int(os.environ.get("EPOCHS", 8))
GAZ_MIN_PREC = float(os.environ.get("GAZ_MIN_PREC", 0.2))


def gaz_lens_of(gaz):
    return sorted({len(s) for s in gaz})


def featurize(rows, gaz, with_gold=True):
    gl = gaz_lens_of(gaz)
    out = []
    for r in rows:
        ids = sent_features(r["sentence"], gaz, gl)
        if with_gold:
            gold = np.array([TAG2I[t] for t in r["tags"]], dtype=np.int8)
            out.append((ids, gold))
        else:
            out.append((ids, None))
    return out


def featurize_train_jackknife(rows, seed=7):
    """Gazetteer features for training rows come from out-of-fold dictionaries."""
    rng = np.random.RandomState(seed)
    folds = rng.randint(0, NFOLD, size=len(rows))
    data = [None] * len(rows)
    for k in range(NFOLD):
        sub = [r for r, f in zip(rows, folds) if f != k]
        gaz = build_gaz(sub, min_prec=GAZ_MIN_PREC)
        gl = gaz_lens_of(gaz)
        for i, (r, f) in enumerate(zip(rows, folds)):
            if f != k:
                continue
            ids = sent_features(r["sentence"], gaz, gl)
            gold = np.array([TAG2I[t] for t in r["tags"]], dtype=np.int8)
            data[i] = (ids, gold)
        print("  jackknife fold %d done (%d dict entries)" % (k, len(gaz)), flush=True)
    return data


def decode(tg, feats, rows, W=None, T=None):
    preds = []
    for (ids, _), r in zip(feats, rows):
        path = tg.predict_ids(ids, W, T)
        tags = [TAGS[i] for i in path]
        preds.append(bio_to_ents(r["sentence"], tags))
    return preds


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "dev"
    t0 = time.time()
    rows, ne, nf = load("train.csv")
    print("train sentences %d entities %d align_fail %d" % (len(rows), ne, nf), flush=True)

    if mode == "dev":
        rng = np.random.RandomState(42)
        idx = rng.permutation(len(rows))
        ntr = int(len(rows) * 0.9)
        tr = [rows[i] for i in idx[:ntr]]
        va = [rows[i] for i in idx[ntr:]]
    else:
        tr, va = rows, []

    print("featurizing train (jackknife gaz)...", flush=True)
    trf = featurize_train_jackknife(tr)
    print("  %.1fs" % (time.time() - t0), flush=True)

    gaz_full = build_gaz(tr, min_prec=GAZ_MIN_PREC)
    print("full gaz entries", len(gaz_full), flush=True)

    tg = Tagger()
    dev_eval = None
    if va:
        vaf = featurize(va, gaz_full)
        gold = [r["ents"] for r in va]

        def dev_eval(W, T):
            return micro_f1(gold, decode(tg, vaf, va, W, T))[2]

    tg.train(trf, epochs=EPOCHS, dev_eval=dev_eval)
    print("trained in %.1fs" % (time.time() - t0), flush=True)

    if va:
        preds = decode(tg, vaf, va, tg.Wavg, tg.Tavg)
        print("FINAL dev P/R/F1", micro_f1(gold, preds), flush=True)
        if tg.best:
            print("BEST epoch %d devF1 %.4f" % (tg.best[3], tg.best[0]), flush=True)

    if mode == "full":
        test = load("test.csv", with_labels=False)
        tef = featurize(test, gaz_full, with_gold=False)
        preds = decode(tg, tef, test, tg.Wavg, tg.Tavg)
        os.makedirs("outputs", exist_ok=True)
        pd.DataFrame({"id": [r["id"] for r in test],
                      "entities": [format_entities(p) for p in preds]}).to_csv(
            "outputs/submission.csv", index=False)
        print("wrote outputs/submission.csv", flush=True)
        with open("solution/model.pkl", "wb") as f:
            pickle.dump({"W": tg.Wavg, "T": tg.Tavg}, f)
    print("total %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
