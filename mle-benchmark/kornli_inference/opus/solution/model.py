"""From-scratch neural NLI model (randomly initialised, trained only on train.csv).

Architecture = Decomposable-Attention (Parikh et al. 2016) with
fastText-style compositional word embeddings and a few explicit
lexical-overlap features fed into the classifier.

No pretrained weights are used anywhere.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NEG_INF = -1e9


def mlp(i, h, o, p):
    return nn.Sequential(nn.Dropout(p), nn.Linear(i, h), nn.ReLU(),
                         nn.Dropout(p), nn.Linear(h, o), nn.ReLU())


class DecompAttn(nn.Module):
    def __init__(self, n_units, dim=160, hid=208, nfeat=19, p=0.2, n_cls=3,
                 sparse=True, word_drop=0.15):
        super().__init__()
        self.word_drop = word_drop
        self.emb = nn.Embedding(n_units, dim, padding_idx=0, sparse=sparse)
        nn.init.normal_(self.emb.weight, 0, 0.1)
        with torch.no_grad():
            self.emb.weight[0].zero_()
        self.proj = nn.Linear(dim, hid)
        self.F = mlp(hid, hid, hid, p)
        self.G = mlp(2 * hid, hid, hid, p)
        self.H = nn.Sequential(nn.Dropout(p), nn.Linear(4 * hid + nfeat, hid), nn.ReLU(),
                               nn.Dropout(p), nn.Linear(hid, hid), nn.ReLU(),
                               nn.Linear(hid, n_cls))
        self.fnorm = nn.LayerNorm(nfeat)

    def embed(self, wid, U, N):
        """wid: (B,T) word indices -> (B,T,dim) via mean of sub-word units.

        Words repeat a lot inside a batch, so we only embed the distinct ones.
        """
        flat = wid.reshape(-1)
        uniq, inv = torch.unique(flat, return_inverse=True)
        ue = self.emb(U[uniq]).sum(1) / N[uniq].unsqueeze(-1)
        return ue[inv].view(*wid.shape, -1)

    def wdrop(self, e, m):
        """drop whole words (a strong regulariser for small NLI corpora)"""
        if not self.training or self.word_drop <= 0:
            return e, m
        keep = (torch.rand(m.shape, device=m.device) > self.word_drop).float()
        keep[:, 0] = 1.0                      # never empty a sentence
        m = m * keep
        return e * m.unsqueeze(-1), m

    def forward(self, w1, w2, m1, m2, feat, U, N):
        a, m1 = self.wdrop(self.proj(self.embed(w1, U, N)), m1)
        b, m2 = self.wdrop(self.proj(self.embed(w2, U, N)), m2)
        fa, fb = self.F(a), self.F(b)
        e = torch.bmm(fa, fb.transpose(1, 2))                     # (B,T1,T2)
        e1 = e + (1 - m2).unsqueeze(1) * NEG_INF
        e2 = e + (1 - m1).unsqueeze(2) * NEG_INF
        alpha = torch.bmm(e1.softmax(2), b)                       # aligned b for each a
        beta = torch.bmm(e2.softmax(1).transpose(1, 2), a)        # aligned a for each b
        v1 = self.G(torch.cat([a, alpha], -1)) * m1.unsqueeze(-1)
        v2 = self.G(torch.cat([b, beta], -1)) * m2.unsqueeze(-1)
        s1 = torch.cat([v1.sum(1), v1.max(1).values], -1)
        s2 = torch.cat([v2.sum(1), v2.max(1).values], -1)
        return self.H(torch.cat([s1, s2, self.fnorm(feat)], -1))
