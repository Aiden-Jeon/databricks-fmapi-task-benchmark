"""Validate outputs/submission.csv against test.csv and the spec."""
import json, os, sys, collections
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = set("""NP NP_AJT NP_CMP NP_CNJ NP_MOD NP_OBJ NP_SBJ VP VP_AJT VP_CMP VP_CNJ
VP_MOD VP_OBJ VP_SBJ VNP VNP_AJT VNP_CMP VNP_CNJ VNP_MOD VNP_OBJ VNP_SBJ AP AP_AJT
AP_CMP AP_MOD DP IP X X_AJT X_CMP X_CNJ X_MOD X_OBJ X_SBJ L R""".split())

te = pd.read_csv(os.path.join(ROOT, "test.csv"))
ss = pd.read_csv(os.path.join(ROOT, "sample_submission.csv"))
sub = pd.read_csv(os.path.join(ROOT, "outputs", "submission.csv"))

err = []
if list(sub.columns) != list(ss.columns):
    err.append("columns %s != %s" % (list(sub.columns), list(ss.columns)))
if len(sub) != len(te):
    err.append("row count %d != %d" % (len(sub), len(te)))
if sub["id"].duplicated().any():
    err.append("duplicate ids")
if set(sub["id"]) != set(te["id"]):
    err.append("id set mismatch")
if list(sub["id"]) != list(te["id"]):
    err.append("WARN: id order differs from test.csv (usually fine)")
if sub["parse"].isna().any():
    err.append("NaN parse values")

nt = collections.Counter()
for tid, toks, parse in zip(te["id"], te["tokens"], sub.set_index("id").loc[te["id"], "parse"]):
    n = len(json.loads(toks))
    items = str(parse).split("|")
    if len(items) != n:
        err.append("%s: %d items != %d tokens" % (tid, len(items), n))
        continue
    nroot = 0
    for i, x in enumerate(items, start=1):
        if x.count(":") != 1:
            err.append("%s: bad item %r" % (tid, x)); continue
        h, r = x.split(":")
        if not h.lstrip("-").isdigit():
            err.append("%s: non-int head %r" % (tid, h)); continue
        h = int(h)
        if h < 0 or h > n:
            err.append("%s: head %d out of range 0..%d" % (tid, h, n))
        if h == i:
            err.append("%s: self-loop at %d" % (tid, i))
        if h == 0:
            nroot += 1
        if r not in LABELS:
            err.append("%s: unknown deprel %r" % (tid, r))
        nt[r] += 1
    if nroot != 1:
        err.append("%s: %d roots" % (tid, nroot))
    # cycle / connectivity check
    heads = [int(x.split(":")[0]) for x in items]
    for s in range(1, n + 1):
        seen = set()
        c = s
        while c != 0:
            if c in seen:
                err.append("%s: cycle" % tid); break
            seen.add(c); c = heads[c - 1]

print("total tokens:", sum(nt.values()))
print("label distribution:", nt.most_common(10))
if err:
    print("FAILED (%d issues):" % len(err))
    for e in err[:20]:
        print("  -", e)
    sys.exit(1)
print("OK: submission is valid (%d rows, all test ids exactly once, every parse is a "
      "single-rooted acyclic tree with valid heads/deprels)" % len(sub))
