"""Independent-seed check: baseline vs. final configuration."""
import sys, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from model import FeatureBuilder
from run import make_model, scores

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 7
tr = pd.read_csv("../train.csv")
a, b, ya, yb = train_test_split(
    tr, tr["label"].values, test_size=0.2, random_state=SEED,
    stratify=tr["label"].values)
classes = np.unique(ya)

for tag, use_n, spec in [
    ("baseline(no-neigh, svc02)", False, [("svc02", 1.0)]),
    ("final(neigh, blend)", True, [("svc015", 1.0), ("svc02", 0.7), ("svc025bal", 1.5)]),
]:
    t = time.time()
    fb = FeatureBuilder(use_neighbors=use_n, neigh_w=1.0)
    Xa = fb.fit_transform(a, ya)
    Xb = fb.transform(b)
    total = np.zeros((Xb.shape[0], len(classes)))
    for n, w in spec:
        m = make_model(n)
        m.fit(Xa, ya)
        total += w * scores(m, Xb, classes)
    p = classes[total.argmax(1)]
    print(f"seed={SEED} {tag}: acc={accuracy_score(yb,p):.4f} "
          f"f1m={f1_score(yb,p,average='macro'):.4f} ({time.time()-t:.0f}s)", flush=True)
