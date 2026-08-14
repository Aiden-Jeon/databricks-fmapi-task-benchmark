"""Two-view TF-IDF + PyTorch MLP with bagged CV for KoBEST BoolQ."""
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold

from features import clean

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
DEVICE = "cpu"
torch.set_num_threads(4)
RNG = 42


def norm_txt(s):
    return " ".join(clean(s).split())


class MLP(nn.Module):
    def __init__(self, d_in, hidden=256, p_drop=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_one(Xtr, ytr, Xva, hidden=256, p_drop=0.5, epochs=60, lr=1e-3,
              wd=1e-4, bs=64, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    d = Xtr.shape[1]
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    Xva_t = torch.tensor(Xva, dtype=torch.float32)
    model = MLP(d, hidden, p_drop).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss()
    n = len(ytr)
    for ep in range(epochs):
        model.train()
        idx = np.random.permutation(n)
        for i in range(0, n, bs):
            b = idx[i:i + bs]
            opt.zero_grad()
            out = model(Xtr_t[b])
            loss = lossf(out, ytr_t[b])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(Xva_t)).numpy()


def main():
    t0 = time.time()
    train_df = pd.read_csv(ROOT / "train.csv")
    test_df = pd.read_csv(ROOT / "test.csv")
    y = train_df["label"].astype(int).values

    p_tr = train_df.paragraph.map(norm_txt)
    q_tr = train_df.question.map(norm_txt)
    p_te = test_df.paragraph.map(norm_txt)
    q_te = test_df.question.map(norm_txt)

    vw = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95,
                         sublinear_tf=True, max_features=30000)
    vw.fit(pd.concat([p_tr + " " + q_tr]))
    Xw_tr = vw.transform(p_tr + " " + q_tr)
    Xw_te = vw.transform(p_te + " " + q_te)

    vc = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=3,
                         sublinear_tf=True, max_features=30000)
    vc.fit(pd.concat([p_tr, q_tr]))
    Xc_tr = vc.transform(p_tr) + vc.transform(q_tr)
    Xc_te = vc.transform(p_te) + vc.transform(q_te)

    X_tr = hstack([Xw_tr, Xc_tr]).tocsr().toarray()
    X_te = hstack([Xw_te, Xc_te]).tocsr().toarray()
    print("features", X_tr.shape, X_te.shape, f"{time.time()-t0:.0f}s", flush=True)

    skf = StratifiedKFold(5, shuffle=True, random_state=RNG)
    oof = np.zeros(len(y))
    test_p = np.zeros(len(test_df))
    for k, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y)):
        pva = train_one(X_tr[tr_idx], y[tr_idx], X_tr[va_idx], seed=RNG + k)
        oof[va_idx] = pva
        test_p += train_one(X_tr[tr_idx], y[tr_idx], X_te, seed=RNG + 100 + k) / 5
        print(f"fold {k}: acc={accuracy_score(y[va_idx], pva>=0.5):.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    acc = accuracy_score(y, oof >= 0.5)
    ll = log_loss(y, oof)
    print(f"MLP bagged CV: acc={acc:.4f} logloss={ll:.4f}", flush=True)
    np.save(ROOT / "solution" / "oof_mlp.npy", oof)
    np.save(ROOT / "solution" / "testp_mlp.npy", test_p)

    sub = pd.DataFrame({"id": test_df["id"],
                        "label": (test_p >= 0.5).astype(int)})
    sub.to_csv(ROOT / "outputs" / "submission_mlp.csv", index=False)
    print("saved outputs/submission_mlp.csv", sub.label.value_counts().to_dict())


if __name__ == "__main__":
    main()
