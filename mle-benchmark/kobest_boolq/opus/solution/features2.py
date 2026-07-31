"""Additional alignment / distributional-similarity features (batched)."""
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from features import sentences, head_stems, NEG_PATTERNS


def _bg(tok):
    return set(tok[i:i + 2] for i in range(max(1, len(tok) - 1)))


def fuzzy_best(tok, pbgs):
    """best char-bigram jaccard of tok against paragraph token bigram-sets"""
    if not pbgs:
        return 0.0
    a = _bg(tok)
    best = 0.0
    for b in pbgs:
        j = len(a & b) / max(1, len(a | b))
        if j > best:
            best = j
            if best > 0.999:
                break
    return best


class LsaSim:
    def __init__(self, n_comp=150):
        self.n_comp = n_comp

    def fit(self, texts):
        corpus = []
        for t in texts:
            corpus.extend(sentences(t))
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3,
                                   sublinear_tf=True, max_features=80000)
        X = self.vec.fit_transform(corpus)
        self.svd = TruncatedSVD(n_components=min(self.n_comp, X.shape[1] - 1), random_state=0)
        self.svd.fit(X)
        return self

    def emb(self, texts):
        return normalize(self.svd.transform(self.vec.transform(texts)))


def build_features2(df, lsa):
    paras = [str(p) for p in df.paragraph.values]
    ques = [str(q) for q in df.question.values]
    sent_lists = [sentences(p) for p in paras]
    flat, offs = [], []
    for s in sent_lists:
        offs.append(len(flat))
        flat.extend(s)
    offs.append(len(flat))
    qv = lsa.emb(ques)
    pv = lsa.emb(paras)
    sv = lsa.emb(flat)

    rows = []
    for k in range(len(df)):
        s0, s1 = offs[k], offs[k + 1]
        sents = sent_lists[k]
        sims = sv[s0:s1] @ qv[k]
        p, q = paras[k], ques[k]
        f = {}
        f["lsa_p"] = float(pv[k] @ qv[k])
        f["lsa_max"] = float(sims.max())
        f["lsa_mean"] = float(sims.mean())
        f["lsa_min"] = float(sims.min())
        f["lsa_top2"] = float(np.sort(sims)[-2]) if len(sims) > 1 else float(sims.max())
        f["lsa_gap"] = f["lsa_max"] - f["lsa_mean"]

        bi = int(np.argmax(sims))
        best = sents[bi]
        qtoks = [t for t in dict.fromkeys(head_stems(q)) if t not in ("있", "하", "이", "그", "것")]
        pbgs = [_bg(t) for t in dict.fromkeys(head_stems(p))]
        bbgs = [_bg(t) for t in dict.fromkeys(head_stems(best))]
        fz = [fuzzy_best(t, pbgs) for t in qtoks]
        fzb = [fuzzy_best(t, bbgs) for t in qtoks]
        f["fuzzy_mean"] = float(np.mean(fz)) if fz else 0.0
        f["fuzzy_min"] = float(np.min(fz)) if fz else 0.0
        f["fuzzy_bmean"] = float(np.mean(fzb)) if fzb else 0.0
        f["fuzzy_exact_frac"] = float(np.mean([x > 0.99 for x in fz])) if fz else 0.0
        f["fuzzy_bad_frac"] = float(np.mean([x < 0.34 for x in fz])) if fz else 0.0

        pred = qtoks[-1] if qtoks else ""
        f["pred_in_p"] = float(pred in p) if pred else 0.0
        f["pred_fuzzy"] = fuzzy_best(pred, pbgs) if pred else 0.0
        subj = qtoks[0] if qtoks else ""
        f["subj_in_p"] = float(subj in p) if subj else 0.0
        f["subj_fuzzy"] = fuzzy_best(subj, pbgs) if subj else 0.0

        qn = any(n in q for n in NEG_PATTERNS)
        bn = any(n in best for n in NEG_PATTERNS)
        f["neg_xor_lsabest"] = float(qn != bn)
        f["lsabest_pos"] = bi / max(1, len(sents) - 1) if len(sents) > 1 else 0.0
        rows.append(f)
    return pd.DataFrame(rows).astype(float)
