"""KLUE-DP baseline solution (pure sklearn, no internet, no pretrained weights).

Approach
--------
1. Structural facts learned from train.csv (hard constraints for the decoder):
   * head is ALWAYS after the dependent (Korean is head-final, projective);
   * head == 0 (root) occurs EXACTLY once per sentence and ONLY on the LAST token;
   * root deprel is almost always VP or VNP.
2. Per-token deprel classifier using token/suffix/position features.
3. Per-(dep, candidate-head) binary "is this the head?" classifier using
   pair features + predicted deprel; we score every legal candidate (j>i) and
   take argmax per dependent. The last token is forced to 0:<root deprel>.
4. Greedy decoding respecting the head-final + single-root constraints.
"""
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TRAIN = os.path.join(ROOT, "train.csv")
TEST = os.path.join(ROOT, "test.csv")
OUT = os.path.join(ROOT, "outputs", "submission.csv")

SEED = 42
ROOT_DEPRELS = ("VP", "VNP", "NP", "NP_AJT", "AP", "X")  # observed root deprels


def load(path):
    df = pd.read_csv(path)
    df["tokens"] = df["tokens"].apply(json.loads)
    return df


# ---------------------------------------------------------------------------
# Data prep: explode to token-level and pair-level rows
# ---------------------------------------------------------------------------

def build_token_rows(df):
    rows = []
    for sid, toks in zip(df["id"], df["tokens"]):
        feats = F.featurize_tokens(toks)
        for i, f in enumerate(feats):
            rows.append((sid, i + 1, f))
    return rows


def make_root_pf(toks, i, n, deprel):
    """Feature dict for the (dep i -> root) candidate. Consistent across train/test."""
    dep_pos1 = i + 1
    pf = {"dist": 0, "dist1": 0, "dist2": 0, "dist3": 0, "dist4": 0,
          "dist5p": 0, "dist_log": 0.0, "dist_norm": 0.0,
          "dep_tok": toks[i], "head_tok": "<ROOT>",
          "dep_suf1": F.suffix(toks[i], 1), "head_suf1": "<R>",
          "dep_suf2": F.suffix(toks[i], 2), "head_suf2": "<R>",
          "dep_suf3": F.suffix(toks[i], 3), "head_suf3": "<R>",
          "dep_last": toks[i][-1], "head_last": "<R>",
          "dep_first": toks[i][0], "head_first": "<R>",
          "dep_cls": F.char_class(toks[i][-1]), "head_cls": "R",
          "same_suf2": 0, "head_final": 0, "head_da": 0,
          "head_yo": 0, "head_is_last": 1, "head_near_last": 1,
          "dep_is_first": 1 if i == 0 else 0,
          "head_pos_norm": 1.0, "dep_pos_norm": round((i + 1) / n, 2),
          "head_np_end": 0, "head_vp_end": 0, "head_vnp_end": 0,
          "head_has_comma": 0, "head_len5p": 0,
          "dep_len5p": 1 if len(toks[i]) >= 5 else 0,
          "span_len": 0, "span_len0": 1, "span_len1": 0, "span_len2p": 0,
          "span_mid_last": "<NA>", "span_first_last": "<NA>",
          "span_last_last": "<NA>", "span_has_comma": 0, "span_has_conj": 0,
          "dep_left_last": (toks[i - 1][-1] if i - 1 >= 0 else "<BOS>"),
          "head_right_last": "<EOS>",
          "head_pred_region": 1, "deprel": deprel,
          "deprel_coarse": deprel.split("_")[0] if "_" in deprel else deprel,
          "deprel_dist1": f"{deprel}_d0", "deprel_head_is_last": f"{deprel}_hl1"}
    return pf


def build_pair_rows(df, gold_parse=True, deprel_provider=None):
    """Build (dep -> candidate head) rows with label 1/0 for gold head.

    deprel_provider: callable(sid, dep_pos1) -> predicted deprel string used as a
    feature.  When None the pair feature has no deprel.
    """
    rows = []
    labels = []
    for sid, toks in zip(df["id"], df["tokens"]):
        n = len(toks)
        gold = F.parse_string_to_pairs(df.loc[df["id"] == sid, "parse"].iloc[0]) \
            if gold_parse and "parse" in df.columns else None
        gold_head = [g[0] for g in gold] if gold is not None else None
        tfeats = F.featurize_tokens(toks)
        for i in range(n):  # dep index 0..n-1
            dep_pos1 = i + 1
            deprel = None
            if deprel_provider is not None:
                deprel = deprel_provider(sid, dep_pos1)
            # candidate heads: j in (i+1 .. n-1) 1-indexed, plus j==0 only if i==n-1
            cand_heads = list(range(dep_pos1 + 1, n + 1))
            if i == n - 1:
                cand_heads = [0]  # root only for last token
            for h in cand_heads:
                if h == 0:
                    pf = make_root_pf(toks, i, n, deprel if deprel is not None else "VP")
                else:
                    pf = F.pair_features(toks[i], toks[h - 1], dep_pos1, h, n, deprel, toks)
                y = 1 if (gold_head is not None and gold_head[i] == h) else 0
                rows.append((sid, dep_pos1, h, pf))
                labels.append(y)
    return rows, np.array(labels, dtype=np.int8)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_deprel(df):
    trows = build_token_rows(df)
    X = [r[2] for r in trows]
    y = []
    for sid, pos1, f in trows:
        toks = df.loc[df["id"] == sid, "tokens"].iloc[0]
        gold = F.parse_string_to_pairs(df.loc[df["id"] == sid, "parse"].iloc[0])
        y.append(gold[pos1 - 1][1])
    vec = DictVectorizer()
    Xm = vec.fit_transform(X)
    le = LabelEncoder()
    ym = le.fit_transform(y)
    clf = LogisticRegression(
        max_iter=200, C=1.0, solver="liblinear", n_jobs=1, random_state=SEED
    )
    clf.fit(Xm, ym)
    return vec, le, clf


def train_head(df, deprel_provider):
    prows, y = build_pair_rows(df, gold_parse=True, deprel_provider=deprel_provider)
    # subsample negatives heavily (1 positive per dep, many negatives)
    rng = np.random.RandomState(SEED)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    # keep all positives; sample negatives ~ 8x positives (more negatives -> better ranker)
    n_neg = min(len(neg_idx), 8 * len(pos_idx))
    neg_sel = rng.choice(neg_idx, n_neg, replace=False)
    sel = np.sort(np.concatenate([pos_idx, neg_sel]))
    X = [prows[i][3] for i in sel]
    ys = y[sel]
    vec = DictVectorizer()
    Xm = vec.fit_transform(X)
    clf = LogisticRegression(
        max_iter=300, C=1.0, solver="liblinear", n_jobs=1, random_state=SEED,
    )
    clf.fit(Xm, ys)
    return vec, clf


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_deprels(df, vec, le, clf):
    out = {}
    for sid, toks in zip(df["id"], df["tokens"]):
        tfeats = F.featurize_tokens(toks)
        Xm = vec.transform(tfeats)
        pred = le.inverse_transform(clf.predict(Xm))
        out[sid] = list(pred)
    return out


def predict_heads(df, deprel_pred, vec, clf):
    """Return dict sid -> list of head (int, 0=root) per token."""
    out = {}
    for sid, toks in zip(df["id"], df["tokens"]):
        n = len(toks)
        tfeats = F.featurize_tokens(toks)
        heads = [0] * n
        # For each dependent i (0-indexed), score legal candidate heads
        for i in range(n):
            dep_pos1 = i + 1
            d = deprel_pred[sid][i]
            cand_heads = list(range(dep_pos1 + 1, n + 1))
            if i == n - 1:
                cand_heads = [0]
            if not cand_heads:
                # Should not happen, but fall back to last token as root
                heads[i] = 0
                continue
            pf_list = []
            for h in cand_heads:
                if h == 0:
                    pf = make_root_pf(toks, i, n, d)
                else:
                    pf = F.pair_features(toks[i], toks[h - 1], dep_pos1, h, n, d, toks)
                pf_list.append(pf)
            Xp = vec.transform(pf_list)
            scores = clf.predict_proba(Xp)[:, 1]
            best = int(np.argmax(scores))
            heads[i] = cand_heads[best]
        out[sid] = heads
    return out


# ---------------------------------------------------------------------------
# Decode: assemble parse string with structural constraints
# ---------------------------------------------------------------------------

def decode(df, deprel_pred, head_pred):
    """Build final parse strings honoring:
       - last token head=0 (root), deprel from root set (coerce to VP/VNP family)
       - head must be > dep_pos1 (enforced by candidate construction already)
    """
    rows = []
    for sid, toks in zip(df["id"], df["tokens"]):
        n = len(toks)
        d_pred = deprel_pred[sid]
        h_pred = head_pred[sid]
        pairs = []
        for i in range(n):
            h = h_pred[i]
            d = d_pred[i]
            if i == n - 1:
                # force root
                h = 0
                if d not in ROOT_DEPRELS:
                    # coerce: pick most likely root deprel by suffix
                    d = "VP" if toks[i][-1] in ".!?" else "VNP"
            pairs.append((h, d))
        rows.append((sid, F.pairs_to_parse(pairs)))
    return rows


# ---------------------------------------------------------------------------
# Local CV (LAS)
# ---------------------------------------------------------------------------

def las_score(df_true, pairs_pred):
    """pairs_pred: dict sid -> list of (head, deprel). df_true has parse column."""
    correct = 0
    total = 0
    for sid, toks in zip(df_true["id"], df_true["tokens"]):
        gold = F.parse_string_to_pairs(df_true.loc[df_true["id"] == sid, "parse"].iloc[0])
        pred = pairs_pred[sid]
        for g, p in zip(gold, pred):
            total += 1
            if g == p:
                correct += 1
    return correct / max(1, total), correct, total


def run_cv(k=3, seed=SEED):
    df = load(TRAIN)
    rng = np.random.RandomState(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    scores = []
    for f in range(k):
        val_idx = folds[f]
        tr_idx = np.concatenate([folds[g] for g in range(k) if g != f])
        dtr = df.iloc[tr_idx].reset_index(drop=True)
        dval = df.iloc[val_idx].reset_index(drop=True)
        # deprel model
        dvec, dle, dclf = train_deprel(dtr)
        dpr_val = predict_deprels(dval, dvec, dle, dclf)
        dpr_tr = predict_deprels(dtr, dvec, dle, dclf)
        # head model trained with PREDICTED deprels on train (matches inference)
        provider_tr = lambda s, p: dpr_tr[s][p - 1]
        hvec, hclf = train_head(dtr, provider_tr)
        hpr = predict_heads(dval, dpr_val, hvec, hclf)
        pred_pairs = {}
        for sid, toks in zip(dval["id"], dval["tokens"]):
            n = len(toks)
            pp = []
            for i in range(n):
                h = hpr[sid][i]
                d = dpr_val[sid][i]
                if i == n - 1:
                    h = 0
                    if d not in ROOT_DEPRELS:
                        d = "VP" if toks[i][-1] in ".!?" else "VNP"
                pp.append((h, d))
            pred_pairs[sid] = pp
        s, c, t = las_score(dval, pred_pairs)
        scores.append(s)
        print(f"  fold {f}: LAS={s:.4f} ({c}/{t})", flush=True)
    print(f"CV mean LAS = {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
    return scores


def main():
    t0 = time.time()
    print("=== KLUE-DP baseline (sklearn) ===", flush=True)
    df = load(TRAIN)
    print(f"train: {len(df)} sentences", flush=True)

    if os.environ.get("NO_CV") != "1":
        print("Running 3-fold CV...", flush=True)
        run_cv(3)

    print("Training final models on full train...", flush=True)
    dvec, dle, dclf = train_deprel(df)
    dpr_full = predict_deprels(df, dvec, dle, dclf)
    train_dep_acc = np.mean(
        [dpr_full[s][i] == F.parse_string_to_pairs(df.loc[df["id"] == s, "parse"].iloc[0])[i][1]
         for s in df["id"] for i, _ in enumerate(df.loc[df["id"] == s, "tokens"].iloc[0])]
    )
    print(f"train deprel acc = {train_dep_acc:.4f}", flush=True)

    def provider(s, p):
        return dpr_full[s][p - 1]

    hvec, hclf = train_head(df, provider)
    print(f"models trained in {time.time() - t0:.1f}s", flush=True)

    dtest = load(TEST)
    print(f"test: {len(dtest)} sentences", flush=True)
    dpr_test = predict_deprels(dtest, dvec, dle, dclf)
    hpr_test = predict_heads(dtest, dpr_test, hvec, hclf)
    rows = decode(dtest, dpr_test, hpr_test)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out_df = pd.DataFrame(rows, columns=["id", "parse"])
    out_df.to_csv(OUT, index=False, quoting=1)  # csv.QUOTE_ALL = 1
    print(f"wrote {OUT} ({len(out_df)} rows) in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
