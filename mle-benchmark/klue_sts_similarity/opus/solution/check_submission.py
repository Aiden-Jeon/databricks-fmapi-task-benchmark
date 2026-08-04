import numpy as np, pandas as pd

sub = pd.read_csv("outputs/submission.csv")
te = pd.read_csv("test.csv")
smp = pd.read_csv("sample_submission.csv")

assert list(sub.columns) == list(smp.columns), (sub.columns, smp.columns)
assert len(sub) == len(te), (len(sub), len(te))
assert sub.id.nunique() == len(sub), "duplicate ids"
assert set(sub.id) == set(te.id), "id mismatch vs test.csv"
assert sub.score.notna().all() and np.isfinite(sub.score).all(), "non-finite score"
assert sub.score.std() > 1e-6, "constant prediction"
assert (sub.id.values == te.id.values).all(), "order differs from test.csv (allowed, informational)"
print("OK: columns", list(sub.columns), "rows", len(sub))
print("score  min=%.3f max=%.3f mean=%.3f std=%.3f" % (
    sub.score.min(), sub.score.max(), sub.score.mean(), sub.score.std()))
