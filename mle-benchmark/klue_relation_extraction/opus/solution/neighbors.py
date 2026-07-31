"""Leak-free relational (neighbour-label) features.

For each row we look at *other* labelled rows that share the sentence or the
entity pair, and encode their labels by relation kind:
  pair   : identical (subject, object)
  rev    : reversed pair (subject<->object)
  ssub   : same sentence, same subject, different object
  sobj   : same sentence, same object, different subject
  scross : same sentence, neighbour subject == my object (or vice versa)
  sent   : same sentence, any other pair
  gsub   : same subject anywhere (global)
  gobj   : same object anywhere (global)
Self rows are always excluded, so the features can be fitted on the very rows
they describe without label leakage.
"""
from collections import defaultdict
import numpy as np
import scipy.sparse as sp

KINDS = ["pair", "rev", "ssub", "sobj", "scross", "sent", "gsub", "gobj"]


class NeighborFeatures:
    def fit(self, df, y):
        self.classes_ = sorted(set(y))
        self.ci = {c: i for i, c in enumerate(self.classes_)}
        self.pair = defaultdict(list)
        self.sent = defaultdict(list)
        self.gsub = defaultdict(list)
        self.gobj = defaultdict(list)
        ids = df["id"].values
        se = df["subject_entity"].values
        oe = df["object_entity"].values
        st = df["sentence"].values
        for i in range(len(df)):
            rec = (ids[i], se[i], oe[i], y[i])
            self.pair[(se[i], oe[i])].append(rec)
            self.sent[st[i]].append(rec)
            self.gsub[se[i]].append(rec)
            self.gobj[oe[i]].append(rec)
        return self

    def transform(self, df):
        n = len(df)
        K = len(KINDS)
        C = len(self.classes_)
        M = np.zeros((n, K * C + K), dtype=np.float32)
        ids = df["id"].values
        se = df["subject_entity"].values
        oe = df["object_entity"].values
        st = df["sentence"].values
        koff = {k: j * C for j, k in enumerate(KINDS)}
        cntoff = K * C

        for i in range(n):
            myid, s, o, sent = ids[i], se[i], oe[i], st[i]
            buckets = defaultdict(list)
            for rec in self.pair.get((s, o), ()):
                if rec[0] != myid:
                    buckets["pair"].append(rec[3])
            for rec in self.pair.get((o, s), ()):
                if rec[0] != myid:
                    buckets["rev"].append(rec[3])
            for rec in self.sent.get(sent, ()):
                if rec[0] == myid:
                    continue
                rs, ro, rl = rec[1], rec[2], rec[3]
                buckets["sent"].append(rl)
                if rs == s and ro != o:
                    buckets["ssub"].append(rl)
                if ro == o and rs != s:
                    buckets["sobj"].append(rl)
                if rs == o or ro == s:
                    buckets["scross"].append(rl)
            for rec in self.gsub.get(s, ()):
                if rec[0] != myid:
                    buckets["gsub"].append(rec[3])
            for rec in self.gobj.get(o, ()):
                if rec[0] != myid:
                    buckets["gobj"].append(rec[3])

            for k, labs in buckets.items():
                if not labs:
                    continue
                base = koff[k]
                w = 1.0 / len(labs)
                for l in labs:
                    M[i, base + self.ci[l]] += w
                M[i, cntoff + KINDS.index(k)] = min(len(labs), 10) / 10.0
        return sp.csr_matrix(M)

    def fit_transform(self, df, y):
        return self.fit(df, y).transform(df)
