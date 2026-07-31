"""Sentence-spread / proximity-binding features."""
import re
import numpy as np
import pandas as pd
from features import sentences, head_stems, NEG_PATTERNS

GENERIC = {"있", "하", "이", "그", "것", "되", "때", "수", "등", "및", "또", "더", "안"}


def qtokens(q):
    toks = [t for t in dict.fromkeys(head_stems(q)) if t not in GENERIC and len(t) >= 2]
    return toks


def occ_positions(tok, text):
    out, s = [], 0
    while True:
        i = text.find(tok, s)
        if i < 0:
            break
        out.append(i)
        s = i + 1
    return out


def build_features3(df):
    rows = []
    for p, q in zip(df.paragraph.values, df.question.values):
        p = str(p)
        q = str(q)
        sents = sentences(p)
        toks = qtokens(q)
        n = max(1, len(toks))
        # sentence membership matrix
        M = np.zeros((len(sents), len(toks)), dtype=bool)
        for j, t in enumerate(toks):
            for i, s in enumerate(sents):
                if t in s:
                    M[i, j] = True
        covered = M.any(axis=0)
        f = {}
        f["f3_cov"] = covered.mean() if len(toks) else 0.0
        per_sent = M.mean(axis=1) if len(sents) else np.array([0.0])
        f["f3_best_sent_cov"] = float(per_sent.max())
        f["f3_spread"] = f["f3_cov"] - f["f3_best_sent_cov"]  # covered but not in one sentence
        # greedy set cover count
        need = covered.copy()
        cnt = 0
        Mm = M.copy()
        while need.any() and cnt < 10:
            gains = (Mm & need).sum(axis=1)
            b = int(np.argmax(gains))
            if gains[b] == 0:
                break
            need &= ~Mm[b]
            cnt += 1
        f["f3_setcover"] = cnt
        f["f3_setcover_norm"] = cnt / n
        # number of sentences that contain >=1 question token
        f["f3_sent_hit_frac"] = float((M.any(axis=1)).mean()) if len(sents) else 0.0
        # two-sentence best coverage
        if len(sents) > 1:
            best2 = 0.0
            for i in range(len(sents)):
                for j2 in range(i + 1, len(sents)):
                    v = (M[i] | M[j2]).mean()
                    if v > best2:
                        best2 = v
            f["f3_best2"] = best2
            f["f3_best2_gain"] = best2 - f["f3_best_sent_cov"]
        else:
            f["f3_best2"] = f["f3_best_sent_cov"]
            f["f3_best2_gain"] = 0.0

        # proximity binding of adjacent question tokens
        pnorm = re.sub(r"\s+", " ", p)
        pos = {t: occ_positions(t, pnorm) for t in toks}
        dists, same_sent, close20 = [], [], []
        sent_of = {}
        acc = 0
        bounds = []
        for s in sents:
            i = pnorm.find(s[:20], acc) if len(s) >= 5 else pnorm.find(s, acc)
            if i < 0:
                i = acc
            bounds.append(i)
            acc = i + max(1, len(s) - 1)
        bounds = np.array(bounds) if bounds else np.array([0])

        def sidx(x):
            return int(np.searchsorted(bounds, x, side="right") - 1)

        for a, b in zip(toks, toks[1:]):
            pa, pb = pos[a], pos[b]
            if not pa or not pb:
                dists.append(1.0)
                same_sent.append(0.0)
                close20.append(0.0)
                continue
            d = min(abs(x - y) for x in pa for y in pb)
            dists.append(min(d, 200) / 200.0)
            same_sent.append(float(any(sidx(x) == sidx(y) for x in pa for y in pb)))
            close20.append(float(d <= 25))
        f["f3_prox_mean"] = float(np.mean(dists)) if dists else 1.0
        f["f3_prox_max"] = float(np.max(dists)) if dists else 1.0
        f["f3_samesent_frac"] = float(np.mean(same_sent)) if same_sent else 0.0
        f["f3_close_frac"] = float(np.mean(close20)) if close20 else 0.0

        # subject-predicate binding: first and last content token
        if len(toks) >= 2:
            a, b = toks[0], toks[-1]
            pa, pb = pos[a], pos[b]
            if pa and pb:
                d = min(abs(x - y) for x in pa for y in pb)
                f["f3_sp_dist"] = min(d, 300) / 300.0
                f["f3_sp_same"] = float(any(sidx(x) == sidx(y) for x in pa for y in pb))
            else:
                f["f3_sp_dist"] = 1.0
                f["f3_sp_same"] = 0.0
        else:
            f["f3_sp_dist"] = 1.0
            f["f3_sp_same"] = 0.0

        # longest contiguous run of question eojeol-stems appearing in order & nearby
        run = best_run = 0
        last = -1
        for t in toks:
            ps_ = pos[t]
            nxt = [x for x in ps_ if x > last]
            if nxt and (last < 0 or nxt[0] - last < 40):
                run += 1
                last = nxt[0]
            else:
                run = 1 if ps_ else 0
                last = ps_[0] if ps_ else -1
            best_run = max(best_run, run)
        f["f3_run"] = best_run
        f["f3_run_norm"] = best_run / n

        # negation in the window around matched tokens
        allpos = [x for t in toks for x in pos[t]]
        if allpos:
            lo, hi = max(0, min(allpos) - 10), min(len(pnorm), max(allpos) + 20)
            win = pnorm[lo:hi]
        else:
            win = ""
        qneg = any(x in q for x in NEG_PATTERNS)
        wneg = any(x in win for x in NEG_PATTERNS)
        f["f3_win_neg"] = float(wneg)
        f["f3_win_neg_xor"] = float(qneg != wneg)
        f["f3_win_len"] = len(win) / 100.0
        rows.append(f)
    return pd.DataFrame(rows).astype(float)
