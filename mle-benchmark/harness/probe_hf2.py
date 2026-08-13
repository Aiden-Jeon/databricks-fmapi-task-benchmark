#!/usr/bin/env python
"""Round 2: nail licenses (from README when absent from cardData), confirm eval
splits, and pick a clean #5 (korfin-asc / APEACH)."""
import json
import re
import urllib.parse
import urllib.request


def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "kmle-probe"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def readme_license(repo):
    for branch in ("main",):
        try:
            url = f"https://huggingface.co/datasets/{repo}/raw/{branch}/README.md"
            req = urllib.request.Request(url, headers={"User-Agent": "kmle-probe"})
            with urllib.request.urlopen(req, timeout=30) as r:
                head = r.read(1500).decode("utf-8", "replace")
            m = re.search(r"license:\s*(.+)", head)
            return m.group(1).strip() if m else "(* no license: line in header *)"
        except Exception as e:
            return f"README FAIL: {e}"


def splits(repo):
    try:
        sp = get_json(f"https://datasets-server.huggingface.co/splits?dataset={repo}")
        return sorted({(s["config"], s["split"]) for s in sp.get("splits", [])})
    except Exception as e:
        return f"splits FAIL: {e}"


def sample(repo, cfg, split):
    try:
        q = urllib.parse.urlencode({"dataset": repo, "config": cfg,
                                    "split": split, "offset": 0, "length": 2})
        rows = get_json(f"https://datasets-server.huggingface.co/rows?{q}")
        feats = [f["name"] for f in rows["features"]]
        r0 = rows["rows"][0]["row"]
        return feats, {k: str(v)[:70] for k, v in r0.items()}
    except Exception as e:
        return f"rows FAIL: {e}", {}


for repo in ["HAERAE-HUB/KMMLU", "HAERAE-HUB/HAE_RAE_BENCH_1.0",
             "smilegate-ai/kor_unsmile", "amphora/korfin-asc",
             "jason9693/APEACH"]:
    print(f"\n===== {repo} =====")
    print("license(README):", readme_license(repo))
    sp = splits(repo)
    print("splits:", sp[:10] if isinstance(sp, list) else sp)
    if isinstance(sp, list) and sp:
        # prefer a test/eval split
        pick = next((cs for cs in sp if cs[1] in ("test", "validation", "valid")), sp[0])
        feats, row = sample(repo, pick[0], pick[1])
        print(f"sample [{pick}] features:", feats)
        for k, v in row.items():
            print(f"   {k}: {v}")
