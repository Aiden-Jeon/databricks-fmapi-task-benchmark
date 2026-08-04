"""Edit-script text view: describe the transformation s1 -> s2 as a bag of
symbolic edit operations, learned from train.csv only.

Idea: PAWS negatives are produced by *moving/swapping* words while positives
are produced by back-translation (i.e. lexical substitutions with preserved
order).  Encoding the edit script symbolically lets a linear model learn which
substitutions are meaning preserving and which movements matter.
"""
import re

import features as F
from features2 import jac, tri

_DIGIT = re.compile(r"\d")


def edit_tokens(s1, s2):
    a = F.tokenize(F.norm(s1))
    b = F.tokenize(F.norm(s2))
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return "__empty__"
    ta = [tri(t) for t in a]
    tb = [tri(t) for t in b]
    cand = []
    for i in range(na):
        for j in range(nb):
            s = jac(ta[i], tb[j])
            if s > 0.15:
                cand.append((s, i, j))
    cand.sort(key=lambda x: (-x[0], abs(x[1] - x[2])))
    ua = [False] * na
    ub = [False] * nb
    pairs = []
    for s, i, j in cand:
        if ua[i] or ub[j]:
            continue
        ua[i] = ub[j] = True
        pairs.append((i, j, s))
    pairs.sort()

    feats = []
    for i, j, s in pairs:
        x, z = F.stem(a[i]), F.stem(b[j])
        d = j - i
        dbucket = "0" if d == 0 else ("p" if d > 0 else "n") + ("1" if abs(d) == 1 else ("2" if abs(d) == 2 else "3"))
        if s > 0.999:
            if d != 0:
                feats.append("MOVE_" + dbucket)
                feats.append("MOVET_" + x + "_" + dbucket)
        else:
            if x == z:
                feats.append("INFL_" + dbucket)
            else:
                key = "_".join(sorted([x, z]))
                feats.append("SUB_" + key)
                feats.append("SUBD_" + dbucket)
                if _DIGIT.search(x) or _DIGIT.search(z):
                    feats.append("SUBNUM")
    for i in range(na):
        if not ua[i]:
            feats.append("DEL_" + F.stem(a[i]))
            feats.append("DELPOS_%d" % min(4, int(5 * i / na)))
    for j in range(nb):
        if not ub[j]:
            feats.append("INS_" + F.stem(b[j]))
            feats.append("INSPOS_%d" % min(4, int(5 * j / nb)))
    nmove = sum(1 for i, j, s in pairs if s > 0.999 and i != j)
    feats.append("NMOVE_%d" % min(6, nmove))
    nsub = sum(1 for i, j, s in pairs if s <= 0.999)
    feats.append("NSUB_%d" % min(8, nsub))
    feats.append("NUNM_%d" % min(8, (na - len(pairs)) + (nb - len(pairs))))
    return " ".join(feats) if feats else "__identical__"


def build(df, verbose=True):
    s1 = df["sentence1"].tolist()
    s2 = df["sentence2"].tolist()
    out = []
    for i in range(len(s1)):
        out.append(edit_tokens(s1[i], s2[i]))
        if verbose and (i + 1) % 5000 == 0:
            print(f"  edits {i+1}/{len(s1)}", flush=True)
    return out
