"""Train the neural NLI model and dump out-of-fold / test probabilities.

Usage:  python solution/train_nn.py [--folds 5] [--epochs 12] [--seed 1]
Writes work/nn_oof_<tag>.npy and work/nn_test_<tag>.npy
"""
import argparse, os, sys, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import (LABELS, L2I, MAX_L1, MAX_L2, WordTable, WordIndex, encode,
                  pair_features, load)
from model import DecompAttn

torch.set_num_threads(4)


def build(cache="work/nn_cache.npz"):
    tr, te = load()
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return tr, te, {k: z[k] for k in z.files}
    tab = WordTable(min_count=3).fit(np.concatenate(
        [tr.sentence1.values, tr.sentence2.values]))
    widx = WordIndex(tab)
    A1, LA1 = encode(tr.sentence1.values, widx, MAX_L1)
    A2, LA2 = encode(tr.sentence2.values, widx, MAX_L2)
    B1, LB1 = encode(te.sentence1.values, widx, MAX_L1)
    B2, LB2 = encode(te.sentence2.values, widx, MAX_L2)
    widx.finalize()
    d = dict(A1=A1, A2=A2, B1=B1, B2=B2,
             FA=pair_features(tr.sentence1.values, tr.sentence2.values),
             FB=pair_features(te.sentence1.values, te.sentence2.values),
             U=widx.U, N=widx.N, n_units=np.array([tab.n_units]))
    os.makedirs("work", exist_ok=True)
    np.savez_compressed(cache, **d)
    return tr, te, d


def run_fold(d, y, tr_i, va_i, Bidx, epochs, seed, dim, hid, p, lr, bs, log=print,
             emb_mult=2.0, wdrop=0.15):
    torch.manual_seed(seed); np.random.seed(seed)
    U = torch.from_numpy(d["U"]); N = torch.from_numpy(d["N"]).clamp(min=1)
    A1 = torch.from_numpy(d["A1"]); A2 = torch.from_numpy(d["A2"])
    FA = torch.from_numpy(d["FA"])
    B1 = torch.from_numpy(d["B1"]); B2 = torch.from_numpy(d["B2"])
    FB = torch.from_numpy(d["FB"])
    Y = torch.from_numpy(y)
    net = DecompAttn(int(d["n_units"][0]), dim=dim, hid=hid,
                     nfeat=d["FA"].shape[1], p=p, word_drop=wdrop)
    dense = [q for n_, q in net.named_parameters() if n_ != "emb.weight"]
    opt = torch.optim.AdamW(dense, lr=lr, weight_decay=1e-5)
    eopt = torch.optim.SparseAdam([net.emb.weight], lr=lr * emb_mult)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    nsteps = epochs * int(np.ceil(len(tr_i) / bs))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=nsteps,
                                             pct_start=0.15)

    def predict(w1, w2, feat, idx):
        net.eval(); out = []
        with torch.no_grad():
            for s in range(0, len(idx), 512):
                b = idx[s:s + 512]
                x1, x2 = w1[b], w2[b]
                o = net(x1, x2, (x1 > 0).float(), (x2 > 0).float(), feat[b], U, N)
                out.append(torch.softmax(o, -1).numpy())
        net.train()
        return np.concatenate(out)

    snap_va, snap_te, snap_acc = [], [], []
    for ep in range(epochs):
        t0 = time.time()
        perm = np.random.permutation(tr_i)
        tot = 0.0
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]
            x1, x2 = A1[b], A2[b]
            o = net(x1, x2, (x1 > 0).float(), (x2 > 0).float(), FA[b], U, N)
            loss = lossf(o, Y[b])
            opt.zero_grad(); eopt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(dense, 5.0)
            opt.step(); eopt.step(); sch.step()
            for g in eopt.param_groups:
                g["lr"] = opt.param_groups[0]["lr"] * emb_mult
            tot += float(loss.detach()) * len(b)
        msg = f"  ep{ep} loss {tot/len(perm):.4f} {time.time()-t0:.0f}s"
        if len(va_i) and ep >= min(1, epochs - 1):
            pv = predict(A1, A2, FA, va_i)
            acc = (pv.argmax(1) == y[va_i]).mean()
            msg += f" val {acc:.4f}"
            snap_va.append(pv)
            snap_te.append(predict(B1, B2, FB, Bidx))
            snap_acc.append(acc)
        log(msg)
        np.save(f"work/_snap_partial_{seed}.npy", np.array(snap_acc))

    if not len(va_i):
        return np.zeros((0, 3), np.float32), predict(B1, B2, FB, Bidx)

    # greedy snapshot ensembling on the held-out split
    order = np.argsort(snap_acc)[::-1]
    chosen = [order[0]]
    cur = snap_va[order[0]].copy()
    best = snap_acc[order[0]]
    for k in order[1:]:
        cand = cur + snap_va[k]
        a = (cand.argmax(1) == y[va_i]).mean()
        if a >= best - 1e-9:
            cur, best = cand, a
            chosen.append(k)
    log(f"  snapshots {sorted(int(c) for c in chosen)} -> val {best:.4f}")
    va = sum(snap_va[k] for k in chosen) / len(chosen)
    te_ = sum(snap_te[k] for k in chosen) / len(chosen)
    return va, te_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dim", type=int, default=144)
    ap.add_argument("--hid", type=int, default=208)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1.0e-3)
    ap.add_argument("--bs", type=int, default=96)
    ap.add_argument("--tag", default="a")
    ap.add_argument("--limit_folds", type=int, default=0)
    ap.add_argument("--emb_mult", type=float, default=2.0)
    ap.add_argument("--wdrop", type=float, default=0.15)
    a = ap.parse_args()

    tr, te, d = build()
    y = tr.label.map(L2I).values.astype(np.int64)
    Bidx = np.arange(len(te))
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(a.folds, shuffle=True, random_state=7)
    oof = np.zeros((len(tr), 3), dtype=np.float32)
    tp = np.zeros((len(te), 3), dtype=np.float32)
    nf = 0
    for f, (tri, vai) in enumerate(skf.split(y, y)):
        if a.limit_folds and f >= a.limit_folds:
            break
        print(f"fold {f}", flush=True)
        pv, pt = run_fold(d, y, tri, vai, Bidx, a.epochs, a.seed + 100 * f,
                          a.dim, a.hid, a.dropout, a.lr, a.bs,
                          log=lambda m: print(m, flush=True),
                          emb_mult=a.emb_mult, wdrop=a.wdrop)
        oof[vai] = pv / pv.sum(1, keepdims=True)
        tp += pt / pt.sum(1, keepdims=True)
        nf += 1
        print(f"fold {f} acc {(pv.argmax(1)==y[vai]).mean():.4f}", flush=True)
    tp /= nf
    np.save(f"work/nn_oof_{a.tag}.npy", oof)
    np.save(f"work/nn_test_{a.tag}.npy", tp)
    mask = oof.sum(1) > 0
    print("OOF acc", (oof[mask].argmax(1) == y[mask]).mean(), flush=True)


if __name__ == "__main__":
    main()
