"""Character-level span-extraction GRU for KorQuAD.

Trained entirely from scratch on train.csv (no pretrained weights, no
internet). Architecture: bidirectional GRU encoder with start/end span head.

Context is windowed (centered on the answer during training, and centered on a
heuristic candidate during inference) to keep sequence length small enough for
CPU training within the time budget.
"""
import os, sys, re, math, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from collections import Counter

SEED = 42
WIN = 320      # context window length (chars)
QMAX = 80      # max question chars
MAX_LEN = WIN + QMAX + 4

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def build_vocab(train_df, min_count=2):
    cnt = Counter()
    for c, q in zip(train_df['context'].astype(str), train_df['question'].astype(str)):
        cnt.update(c); cnt.update(q)
    specials = ['<pad>', '<cls>', '<sep>', '<unk>']
    vocab = specials + [ch for ch, n in cnt.most_common() if n >= min_count]
    stoi = {ch: i for i, ch in enumerate(vocab)}
    return stoi, vocab

def encode(s, stoi):
    unk = stoi['<unk>']
    return [stoi.get(ch, unk) for ch in str(s)]

def window_around(ctx, anchor_pos, anchor_len, win=WIN):
    """Return (sub_ctx, offset) where sub_ctx is ctx windowed to length <= win
    centered roughly on the anchor (anchor_pos..anchor_pos+anchor_len)."""
    n = len(ctx)
    if n <= win:
        return ctx, 0
    half = (win - anchor_len) // 2
    start = max(0, min(anchor_pos - half, n - win))
    return ctx[start:start + win], start

def make_example_train(ctx, question, answer, stoi):
    """Build training example: window context centered on the answer.

    If answer appears multiple times, pick a random occurrence (so the model
    learns to use question context, not just position). Returns (ids, start, end).
    """
    ctx = str(ctx); q = str(question); a = str(answer)
    # find all occurrences
    positions = []
    start_search = 0
    while True:
        p = ctx.find(a, start_search)
        if p < 0: break
        positions.append(p)
        start_search = p + 1
    if not positions:
        return None
    pos = random.choice(positions)
    sub_ctx, off = window_around(ctx, pos, len(a), WIN)
    # recompute answer position within sub_ctx
    apos = pos - off
    if apos < 0 or apos + len(a) > len(sub_ctx):
        return None
    q2 = q if len(q) <= QMAX else q[:QMAX]
    ids = [stoi['<cls>']] + encode(q2, stoi) + [stoi['<sep>']] + encode(sub_ctx, stoi) + [stoi['<sep>']]
    q_len = 1 + len(q2) + 1
    s = q_len + apos
    e = q_len + apos + len(a) - 1
    if e >= len(ids):
        return None
    return ids, s, e

def make_example_infer(ctx, question, anchor, stoi):
    """Build inference example: window context centered on anchor (a heuristic
    candidate string). Returns (ids, q_len, ctx_len, sub_ctx, offset) or None.
    """
    ctx = str(ctx); q = str(question); anc = str(anchor)
    if len(anc) == 0:
        anc = ctx[:5]
    pos = ctx.find(anc)
    if pos < 0:
        pos = 0
    sub_ctx, off = window_around(ctx, pos, len(anc), WIN)
    q2 = q if len(q) <= QMAX else q[:QMAX]
    ids = [stoi['<cls>']] + encode(q2, stoi) + [stoi['<sep>']] + encode(sub_ctx, stoi) + [stoi['<sep>']]
    q_len = 1 + len(q2) + 1
    ctx_len = len(sub_ctx)
    return ids, q_len, ctx_len, sub_ctx, off

class SpanGRU(nn.Module):
    def __init__(self, vocab_size, d=96, n_layers=1, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d, padding_idx=0)
        self.gru = nn.GRU(d, d * 2, num_layers=n_layers, batch_first=True,
                          bidirectional=True, dropout=dropout if n_layers > 1 else 0.0)
        self.norm = nn.LayerNorm(d * 4)
        self.dropout = nn.Dropout(dropout)
        self.qa = nn.Linear(d * 4, 2)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, ids, mask):
        # mask: True for pad positions (to ignore)
        x = self.emb(ids)
        x, _ = self.gru(x)
        x = self.norm(x)
        x = self.dropout(x)
        return self.qa(x)  # (B, L, 2)

def collate(batch, stoi, pad_id=0):
    """batch: list of (ids, start, end)."""
    max_l = max(len(b[0]) for b in batch)
    ids = torch.full((len(batch), max_l), pad_id, dtype=torch.long)
    mask = torch.ones((len(batch), max_l), dtype=torch.bool)  # True=pad
    starts = torch.zeros(len(batch), dtype=torch.long)
    ends = torch.zeros(len(batch), dtype=torch.long)
    for i, (b_ids, s, e) in enumerate(batch):
        L = len(b_ids)
        ids[i, :L] = torch.tensor(b_ids, dtype=torch.long)
        mask[i, :L] = False
        starts[i] = s; ends[i] = e
    return ids, mask, starts, ends

def find_best_span(start_logits, end_logits, q_len, ctx_len, max_span=30, topk=20):
    """Return (start, end) token positions restricted to the context region."""
    L = start_logits.shape[0]
    cs = q_len
    ce = q_len + ctx_len
    # restrict
    s_full = torch.full_like(start_logits, -1e9)
    e_full = torch.full_like(end_logits, -1e9)
    s_full[cs:ce] = start_logits[cs:ce]
    e_full[cs:ce] = end_logits[cs:ce]
    k = min(topk, ce - cs)
    if k <= 0:
        return cs, cs
    s_top = s_full[cs:ce].topk(k).indices + cs
    e_top = e_full[cs:ce].topk(k).indices + cs
    best = (cs, cs); best_score = -1e18
    for s in s_top.tolist():
        for e in e_top.tolist():
            if e < s: continue
            if e - s + 1 > max_span: continue
            sc = s_full[s].item() + e_full[e].item()
            if sc > best_score:
                best_score = sc; best = (s, e)
    return best

def compute_loss(logits, starts, ends, mask):
    start_logits = logits[:, :, 0]
    end_logits = logits[:, :, 1]
    # Mask out pad positions in softmax by setting to -1e9
    neg = torch.full_like(start_logits, -1e9)
    start_logits = torch.where(mask, neg, start_logits)
    end_logits = torch.where(mask, neg, end_logits)
    ls = F.cross_entropy(start_logits, starts)
    le = F.cross_entropy(end_logits, ends)
    return ls + le

def train_model(train_df, stoi, args, device='cpu', val_df=None, val_interval=400):
    set_seed(args.seed)
    model = SpanGRU(vocab_size=len(stoi), d=args.d, n_layers=args.layers, dropout=args.dropout)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}", flush=True)

    print("Building examples...", flush=True)
    examples = []
    skipped = 0
    for i in range(len(train_df)):
        r = train_df.iloc[i]
        ex = make_example_train(r['context'], r['question'], r['answer'], stoi)
        if ex is None:
            skipped += 1
            continue
        examples.append(ex)
    print(f"Examples: {len(examples)} (skipped {skipped})", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total_steps = args.epochs * (len(examples) // args.bs + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps,
                                                  pct_start=0.1)

    step = 0
    for epoch in range(args.epochs):
        random.shuffle(examples)
        model.train()
        t0 = time.time()
        ep_loss = 0.0; nb = 0
        for bstart in range(0, len(examples), args.bs):
            batch = examples[bstart:bstart + args.bs]
            ids, mask, starts, ends = collate(batch, stoi)
            ids = ids.to(device); mask = mask.to(device)
            starts = starts.to(device); ends = ends.to(device)
            logits = model(ids, mask)
            loss = compute_loss(logits, starts, ends, mask)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            ep_loss += loss.item(); nb += 1; step += 1
            if step % 50 == 0:
                elapsed = time.time() - t0
                done = bstart + len(batch)
                rate = done / max(1e-6, elapsed)
                eta = (len(examples) - done) / max(1e-6, rate)
                print(f"  e{epoch} s{step} loss={loss.item():.4f} {done}/{len(examples)} {rate:.0f} ex/s eta{eta:.0f}s", flush=True)
        avg = ep_loss / max(1, nb)
        print(f"Epoch {epoch}: avg_loss={avg:.4f} time={time.time()-t0:.1f}s", flush=True)
        if val_df is not None and (epoch + 1) % 1 == 0:
            f1 = evaluate_model(model, val_df, stoi, device=device, bs=args.bs)
            print(f"  val F1: {f1:.4f}", flush=True)
            model.train()
    return model

def predict_batch_infer(model, rows, stoi, device='cpu'):
    """Batched inference. rows: list of (ctx, question, gold_or_None).
    Returns list of (pred_string, gold_or_None, f1_or_None)."""
    from heuristic import predict_row
    built = []
    for ctx, q, gold in rows:
        anc = predict_row(ctx, q)
        ex = make_example_infer(ctx, q, anc, stoi)
        if ex is None:
            built.append(None)
        else:
            built.append((ex, anc, gold))
    preds = []
    # group non-None by building a single padded batch
    idxs = [i for i, b in enumerate(built) if b is not None]
    if idxs:
        items = [built[i][0] for i in idxs]  # (ids, q_len, ctx_len, sub_ctx, off)
        max_l = max(len(it[0]) for it in items)
        ids_t = torch.full((len(items), max_l), 0, dtype=torch.long)
        mask_t = torch.ones((len(items), max_l), dtype=torch.bool)
        for j, it in enumerate(items):
            L = len(it[0]); ids_t[j, :L] = torch.tensor(it[0]); mask_t[j, :L] = False
        with torch.no_grad():
            logits = model(ids_t.to(device), mask_t.to(device)).cpu()
        for j, it in enumerate(items):
            ids, q_len, ctx_len, sub_ctx, off = it
            log = logits[j, :len(ids), :]
            s, e = find_best_span(log[:, 0], log[:, 1], q_len, ctx_len)
            ctx_start = s - q_len
            ctx_end = e - q_len + 1
            if 0 <= ctx_start <= ctx_end <= len(sub_ctx):
                pred = sub_ctx[ctx_start:ctx_end]
            else:
                pred = built[idxs[j]][1]  # anc fallback
            preds.append((idxs[j], pred))
    # assemble in order
    out = [None] * len(rows)
    for j, p in preds:
        out[j] = p
    for i, b in enumerate(built):
        if b is None:
            out[i] = b if b is not None else ''  # fallback to anc (b is None here)
            out[i] = built[i][1] if built[i] is not None else ''
    return out

def evaluate_model(model, val_df, stoi, device='cpu', bs=64, max_eval=1000):
    model.eval()
    from common import char_f1
    n = min(len(val_df), max_eval)
    rows = [(str(val_df.iloc[i]['context']), str(val_df.iloc[i]['question']), str(val_df.iloc[i]['answer'])) for i in range(n)]
    # predict in chunks
    scores = []
    for cstart in range(0, len(rows), bs):
        chunk = rows[cstart:cstart + bs]
        preds = predict_batch_infer(model, chunk, stoi, device=device)
        for (ctx, q, gold), pred in zip(chunk, preds):
            scores.append(char_f1(pred, gold))
    return float(np.mean(scores)) if scores else 0.0

def predict_test(model, test_df, stoi, device='cpu', bs=64):
    model.eval()
    rows = [(str(r['context']), str(r['question']), None) for _, r in test_df.iterrows()]
    preds = []
    for cstart in range(0, len(rows), bs):
        chunk = rows[cstart:cstart + bs]
        p = predict_batch_infer(model, chunk, stoi, device=device)
        preds.extend(p)
        if (cstart + bs) % 2000 == 0 or cstart + bs >= len(rows):
            print(f"  predicted {min(cstart + bs, len(rows))}/{len(rows)}", flush=True)
    return preds

class Args:
    def __init__(self):
        self.d = 80
        self.layers = 1
        self.dropout = 0.1
        self.lr = 3e-4
        self.wd = 0.01
        self.bs = 64
        self.epochs = 2
        self.seed = 42

if __name__ == '__main__':
    base = os.path.dirname(os.path.abspath(__file__))
    proj = os.path.dirname(base)
    sys.path.insert(0, base)
    train = pd.read_csv(os.path.join(proj, 'train.csv'))
    test = pd.read_csv(os.path.join(proj, 'test.csv'))
    stoi, vocab = build_vocab(train, min_count=2)
    print("Vocab size:", len(stoi), flush=True)
    args = Args()
    device = 'cpu'
    torch.set_num_threads(4)
    # hold out a small val set for monitoring (article-grouped already by spec, but we just sample)
    val_df = train.iloc[:500]
    tr_df = train.iloc[500:]
    model = train_model(tr_df, stoi, args, device=device, val_df=val_df)
    os.makedirs(os.path.join(base, 'ckpt'), exist_ok=True)
    torch.save({'model': model.state_dict(), 'stoi': stoi, 'args': vars(args)},
               os.path.join(base, 'ckpt', 'model.pt'))
    print("Saved model.", flush=True)
    preds = predict_test(model, test, stoi, device=device)
    out = pd.DataFrame({'id': test['id'], 'answer': preds})
    os.makedirs(os.path.join(proj, 'outputs'), exist_ok=True)
    out.to_csv(os.path.join(proj, 'outputs', 'submission.csv'), index=False)
    print("Wrote", len(out), "rows", flush=True)
