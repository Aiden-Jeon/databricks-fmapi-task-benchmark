#!/usr/bin/env python
"""Non-harness (direct-inference) lane: call the model's chat API directly with a
Korean instruction + few-shot + test input → label. No coding agent, no code
execution. The purest Korean-understanding measurement, for classification/NLU
tasks. Not applicable to tabular/regression tasks (pubg, bike).

Usage: direct_infer.py <task> <model_endpoint> [n_test] [k_shot]
Writes outputs/direct/<task>__<model>.csv (submission) and prints accuracy on the
sampled rows. Routes through the same Unity AI Gateway as the agent lanes.
"""
import concurrent.futures as cf
import json
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "config.json").read_text())
WS = "https://fevm-newjeans-ontos.cloud.databricks.com"
URL = f"{WS}/ai-gateway/mlflow/v1/chat/completions"


def token():
    for line in (Path.home() / ".databrickscfg").read_text().splitlines():
        if line.strip().startswith("token") and "kmle" in _scope[0]:
            return line.split("=", 1)[1].strip()
    return _pat[0]


_scope = [""]
_pat = [""]


def load_pat():
    inb = False
    for line in (Path.home() / ".databrickscfg").read_text().splitlines():
        s = line.strip()
        if s.startswith("[kmle-pat]"):
            inb = True
            continue
        if s.startswith("[") and inb:
            break
        if inb and s.startswith("token"):
            _pat[0] = s.split("=", 1)[1].strip()
    return _pat[0]


PAT = load_pat()


def chat(model, content, max_tokens=1536):
    # Budget must clear reasoning models' thinking (GLM 5.2 streams a separate
    # reasoning_content and only writes the label into content afterward);
    # models that answer immediately (Opus/GPT) hit finish_reason=stop early,
    # so the higher cap costs them nothing.
    body = json.dumps({"model": model, "max_tokens": max_tokens,
                       "messages": [{"role": "user", "content": content}]}).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Authorization": f"Bearer {PAT}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
    msg = d["choices"][0]["message"]
    txt = msg.get("content")
    if isinstance(txt, list):  # Claude structured content blocks
        txt = " ".join(b.get("text", "") for b in txt if b.get("type") == "text")
    txt = (txt or "").strip()
    if not txt and msg.get("reasoning_content"):  # truncated before final answer
        txt = str(msg["reasoning_content"]).strip().splitlines()[-1]
    return txt


def build_prompt(spec_title, classes, input_cols, shots, row, is_str):
    lines = [f"과제: {spec_title}", ""]
    opts = ", ".join(str(c) for c in classes)
    lines.append(f"아래 입력을 보고 레이블을 정확히 하나만 출력하세요. 가능한 레이블: {opts}")
    lines.append("레이블만 출력하고 다른 말은 하지 마세요.\n")
    lines.append("예시:")
    for _, ex in shots.iterrows():
        for c in input_cols:
            lines.append(f"{c}: {ex[c]}")
        lines.append(f"레이블: {ex['label']}\n")
    lines.append("이제 다음을 분류하세요:")
    for c in input_cols:
        lines.append(f"{c}: {row[c]}")
    lines.append("레이블:")
    return "\n".join(lines)


def parse_label(out, classes, is_str):
    out = out.strip()
    if is_str:
        # longest label first: "org:founded_by" must win over "org:founded"
        for c in sorted((str(x) for x in classes), key=len, reverse=True):
            if c in out:
                return c
        return out.split()[0] if out.split() else ""
    m = re.search(r"-?\d+", out)
    return int(m.group()) if m else -999


def main():
    task, model = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 150
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    meta = json.loads((ROOT / "packs" / task / "meta.json").read_text())
    classes = meta.get("classes")
    is_str = meta["metric"] == "accuracy_str"
    if not classes:
        print(f"SKIP {task}: not a fixed-class task ({meta['metric']})")
        return
    idc = meta["id_col"]
    tr = pd.read_csv(ROOT / "packs" / task / "train.csv", dtype={idc: str})
    te = pd.read_csv(ROOT / "packs" / task / "test.csv", dtype={idc: str})
    ans = pd.read_csv(ROOT / "private" / task / "answers.csv", dtype={idc: str})
    input_cols = [c for c in te.columns if c != idc]
    # pandas 2.x groupby().apply() drops the grouping column — sample per class
    # explicitly with concat so `label` survives into the few-shot block.
    per = max(1, k // max(1, len(classes)))
    shots = pd.concat([g.sample(min(per, len(g)), random_state=1)
                       for _, g in tr.groupby("label")]).head(k)
    te = te.sample(min(n, len(te)), random_state=1).reset_index(drop=True)

    def infer(row):
        p = build_prompt(meta.get("title_ko", task), classes, input_cols, shots, row, is_str)
        try:
            return row[idc], parse_label(chat(model, p), classes, is_str)
        except Exception as e:
            return row[idc], (str(classes[0]) if is_str else -999)

    preds = {}
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for rid, lab in ex.map(infer, [r for _, r in te.iterrows()]):
            preds[rid] = lab
    sub = pd.DataFrame({idc: list(preds), "label": list(preds.values())})
    outd = ROOT / "results" / "direct"
    outd.mkdir(parents=True, exist_ok=True)
    sub.to_csv(outd / f"{task}__{model}.csv", index=False)

    amap = dict(zip(ans[idc], ans["label"].astype(str)))
    correct = sum(1 for rid, p in preds.items() if str(p) == amap.get(rid))
    acc = correct / len(preds)
    print(f"{task:22s} {model:28s} direct acc={acc:.4f}  (n={len(preds)}, k={k})")


if __name__ == "__main__":
    main()
