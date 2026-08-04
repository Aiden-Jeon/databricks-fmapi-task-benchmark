"""Greedy blend-weight search on the saved holdout score matrices."""
import itertools, json
import numpy as np
from sklearn.metrics import accuracy_score

OUT = "/tmp/opencode/"
yb = np.load(OUT + "yb.npy", allow_pickle=True)
classes = np.load(OUT + "classes.npy", allow_pickle=True)
names = ["svc03", "svc02", "ridge", "sgd", "knn", "knn1"]
S = {}
for n in names:
    try:
        M = np.load(OUT + n + ".npy").astype(np.float64)
        S[n] = (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-9)
    except FileNotFoundError:
        pass


def acc(tot):
    return accuracy_score(yb, classes[tot.argmax(1)])


for n, M in S.items():
    print(n, round(acc(M), 5))

# greedy forward selection with weight grid
best_w, best_a = {}, 0.0
cur = np.zeros_like(next(iter(S.values())))
for step in range(6):
    cand = None
    for n in S:
        for w in [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]:
            a = acc(cur + w * S[n])
            if cand is None or a > cand[0]:
                cand = (a, n, w)
    if cand[0] <= best_a + 1e-6:
        break
    best_a, n, w = cand
    best_w[n] = best_w.get(n, 0.0) + w
    cur = cur + w * S[n]
    print("step", step, n, w, round(best_a, 5), flush=True)

print("BEST", round(best_a, 5), json.dumps(best_w))
with open(OUT + "blend_w.json", "w") as f:
    json.dump(best_w, f)
