#!/usr/bin/env python
"""Probe candidate HF datasets for the 25-task push: configs/splits, feature
schema, a sample row, and the card license. Read-only — decides what's buildable
and publication-safe before any pack code is written.
"""
import json
import urllib.request

CANDIDATES = [
    ("HAERAE-HUB/KMMLU", None),
    ("HAERAE-HUB/HAE_RAE_BENCH_1.0", None),
    ("e9t/nsmc", None),                     # sanity (known-good)
    ("smilegate-ai/kor_unsmile", None),
    ("klue/klue", "dp"),
    ("kuotient/naver-shopping-review", None),
    ("Blpeng/naver_shopping", None),
]


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "kmle-probe"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def probe(repo, cfg):
    print(f"\n===== {repo}  (cfg={cfg}) =====")
    try:
        card = get(f"https://huggingface.co/api/datasets/{repo}")
        lic = (card.get("cardData") or {}).get("license", "??")
        print(f"license: {lic}   downloads: {card.get('downloads')}")
    except Exception as e:
        print(f"card FAIL: {e}")
    try:
        sp = get(f"https://datasets-server.huggingface.co/splits?dataset={repo}")
        cfgs = sorted({s['config'] for s in sp.get('splits', [])})
        print(f"configs ({len(cfgs)}): {cfgs[:12]}{' …' if len(cfgs) > 12 else ''}")
        splits = [(s['config'], s['split']) for s in sp.get('splits', [])
                  if cfg is None or s['config'] == cfg]
        use_cfg, use_split = (splits[0] if splits else (cfg or "default", "train"))
    except Exception as e:
        print(f"splits FAIL: {e}")
        use_cfg, use_split = cfg or "default", "train"
    try:
        rows = get(f"https://datasets-server.huggingface.co/rows?dataset={repo}"
                   f"&config={use_cfg}&split={use_split}&offset=0&length=2")
        feats = [f["name"] for f in rows["features"]]
        print(f"first split: cfg={use_cfg} split={use_split}")
        print(f"features: {feats}")
        r0 = rows["rows"][0]["row"]
        for k, v in r0.items():
            sv = str(v)
            print(f"   {k}: {sv[:90]}")
    except Exception as e:
        print(f"rows FAIL: {e}")


if __name__ == "__main__":
    for repo, cfg in CANDIDATES:
        probe(repo, cfg)
