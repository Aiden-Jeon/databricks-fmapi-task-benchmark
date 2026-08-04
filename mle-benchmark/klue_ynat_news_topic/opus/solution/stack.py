"""Fast stacking / bias-tuning evaluation on cached OOF matrices."""
import sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

CACHE = "/tmp/opencode"
TASK = "/tmp/kmle/M1_t3_ynat_full_20260804_033458/task"
K = 7


def fast_mf1(pred, yi):
    """pred, yi: int arrays of class indices."""
    cm = np.bincount(yi * K + pred, minlength=K * K).reshape(K, K)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    return float(np.mean(2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)))


def load(keys):
    O, T = [], []
    for k in keys:
        z = np.load(f"{CACHE}/oof_{k}.npz", allow_pickle=True)
        o, t, cls = z["oof"], z["test"], z["classes"]
        m, s = o.mean(), o.std()
        O.append((o - m) / s); T.append((t - m) / s)
    return np.stack(O), np.stack(T), cls


def fit_bias(S, yi, grid=np.arange(-0.5, 0.501, 0.025), rounds=4):
    b = np.zeros(K)
    best = fast_mf1((S + b).argmax(1), yi)
    for _ in range(rounds):
        imp = False
        for j in range(K):
            cand, cbest = b[j], best
            for g in grid:
                b[j] = g
                v = fast_mf1((S + b).argmax(1), yi)
                if v > cbest + 1e-7:
                    cbest, cand = v, g
            b[j] = cand
            if cbest > best + 1e-7:
                best, imp = cbest, True
        if not imp:
            break
    return b, best


if __name__ == "__main__":
    keys = sys.argv[1].split(",")
    tr = pd.read_csv(f"{TASK}/train.csv")
    O, T, cls = load(keys)
    c2i = {c: i for i, c in enumerate(cls)}
    yi = np.array([c2i[v] for v in tr.label.values])
    n = len(yi)

    for k, o in zip(keys, O):
        print(f"  {k}: {fast_mf1(o.argmax(1), yi):.5f}")
    Smean = O.mean(0)
    print(f"  mean({','.join(keys)}): {fast_mf1(Smean.argmax(1), yi):.5f}")

    # ---------- 1) bias-only on simple mean, nested-CV honest estimate ----------
    skf = StratifiedKFold(5, shuffle=True, random_state=7)
    pred = np.empty(n, int)
    for a, b in skf.split(Smean, yi):
        bi, _ = fit_bias(Smean[a], yi[a])
        pred[b] = (Smean[b] + bi).argmax(1)
    print(f"[bias-tuned mean]  honest CV macroF1 = {fast_mf1(pred, yi):.5f}")

    # ---------- 2) LogReg stacking on concatenated scores, nested-CV ----------
    F = np.concatenate(list(O), axis=1)          # n x (M*7)
    FT = np.concatenate(list(T), axis=1)
    for C in [0.1, 0.5, 2.0]:
        pred = np.empty(n, int)
        for a, b in skf.split(F, yi):
            m = LogisticRegression(C=C, max_iter=2000, multi_class="multinomial")
            m.fit(F[a], yi[a])
            pred[b] = m.predict(F[b])
        print(f"[stack LogReg C={C}]  honest CV macroF1 = {fast_mf1(pred, yi):.5f}")

    # ---------- 3) stacking + bias ----------
    for C in [0.5]:
        pred = np.empty(n, int)
        for a, b in skf.split(F, yi):
            m = LogisticRegression(C=C, max_iter=2000, multi_class="multinomial")
            m.fit(F[a], yi[a])
            Pa = np.log(np.clip(m.predict_proba(F[a]), 1e-9, None))
            Pa = (Pa - Pa.mean()) / Pa.std()
            bi, _ = fit_bias(Pa, yi[a])
            Pb = np.log(np.clip(m.predict_proba(F[b]), 1e-9, None))
            Pb = (Pb - Pb.mean()) / Pb.std()
            pred[b] = (Pb + bi).argmax(1)
        print(f"[stack C={C} + bias]  honest CV macroF1 = {fast_mf1(pred, yi):.5f}")
