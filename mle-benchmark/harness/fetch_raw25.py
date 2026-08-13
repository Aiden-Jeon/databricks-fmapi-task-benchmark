#!/usr/bin/env python
"""Fetch raw data for the 25-task push (t21-t25) into kmle/raw/ via the HF
datasets-server parquet endpoint. Consolidates multi-config datasets (KMMLU,
HAE-RAE) into one parquet, tagging each row with its source config. Downloads
data only to the user's machine — never redistributed (MLE-bench: ship scripts,
not data).
"""
import io
import json
import urllib.request
from pathlib import Path
from urllib.parse import quote

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
RAW.mkdir(parents=True, exist_ok=True)

# repo, split, out-file, config predicate (None = all configs), tag-col
SPECS = [
    ("HAERAE-HUB/KMMLU", "test", "kmmlu.parquet", None, "Category"),
    ("HAERAE-HUB/HAE_RAE_BENCH_1.0", "test", "haerae.parquet", None, "domain"),
    ("klue/klue", "train", "klue_dp.parquet", {"dp"}, None),
    ("amphora/korfin-asc", "train", "korfin_asc.parquet", None, None),
    ("smilegate-ai/kor_unsmile", "train", "kor_unsmile.parquet", None, None),
]


def get(url):
    url = quote(url, safe=":/?=&%#")  # encode literal spaces in config names, keep %2F
    req = urllib.request.Request(url, headers={"User-Agent": "kmle-fetch"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def parquet_urls(repo):
    d = json.loads(get(f"https://datasets-server.huggingface.co/parquet?dataset={repo}"))
    return d.get("parquet_files", [])


def fetch(repo, split, out, cfg_pred, tag_col):
    files = parquet_urls(repo)
    picked = [f for f in files if f["split"] == split
              and (cfg_pred is None or f["config"] in cfg_pred)]
    if not picked:
        print(f"[{out}] NO FILES for split={split} pred={cfg_pred} "
              f"(available: {sorted({(f['config'], f['split']) for f in files})[:6]})")
        return
    frames = []
    for f in picked:
        raw = get(f["url"])
        df = pd.read_parquet(io.BytesIO(raw))
        if tag_col and tag_col not in df.columns:
            df[tag_col] = f["config"]
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    full.to_parquet(RAW / out, index=False)
    print(f"[{out}] {len(full)} rows, {len(picked)} file(s), cols={list(full.columns)[:14]}")


if __name__ == "__main__":
    for repo, split, out, cfg_pred, tag in SPECS:
        try:
            fetch(repo, split, out, cfg_pred, tag)
        except Exception as e:
            print(f"[{out}] FAIL {repo}: {type(e).__name__}: {e}")
