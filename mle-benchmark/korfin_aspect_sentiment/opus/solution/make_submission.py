"""Write outputs/submission.csv from a cached probability matrix."""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLASSES, TASK  # noqa

CACHE = os.path.join(TASK, "solution", "cache")
src = sys.argv[1] if len(sys.argv) > 1 else "BLEND"
te = pd.read_csv(f"{TASK}/test.csv")
p = np.load(f"{CACHE}/{src}_test.npy")
assert len(p) == len(te), (len(p), len(te))
lab = np.array(CLASSES)[p.argmax(1)]
os.makedirs(f"{TASK}/outputs", exist_ok=True)
out = pd.DataFrame({"id": te.id, "label": lab})
out.to_csv(f"{TASK}/outputs/submission.csv", index=False)

# validation
ss = pd.read_csv(f"{TASK}/sample_submission.csv")
assert list(out.columns) == list(ss.columns)
assert len(out) == len(ss) and set(out.id) == set(ss.id) and out.id.is_unique
assert out.label.isin(CLASSES).all()
print(f"wrote {len(out)} rows from {src}")
print(out.label.value_counts())
