"""Second feature block: distributional word vectors (PPMI+SVD learned from the
provided corpus only), jamo-level string similarity, and cluster/topic features."""
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from feats import clean, depunct, stems, tokens, stem, jac

# ------------------------------------------------------------------ jamo
CHO = list("\u3131\u3132\u3134\u3137\u3138\u3139\u3141\u3142\u3143\u3145\u3146\u3147\u3148\u3149\u314a\u314b\u314c\u314d\u314e")
JUNG = list("\u314f\u3150\u3151\u3152\u3153\u3154\u3155\u3156\u3157\u3158\u3159\u315a\u315b\u315c\u315d\u315e\u315f\u3160\u3161\u3162\u3163")
JONG = [""] + list("\u3131\u3132\u3133\u3134\u3135\u3136\u3137\u3139\u313a\u313b\u313c\u313d\u313e\u313f\u3140\u3141\u3142\u3144\u3145\u3146\u3147\u3148\u314a\u314b\u314c\u314d\u314e")


def to_jamo(s):
    out = []
    for ch in s:
        o = ord(ch)
        if 0xAC00 <= o <= 0xD7A3:
            i = o - 0xAC00
            out.append(CHO[i // 588])
            out.append(JUNG[(i % 588) // 28])
            j = JONG[i % 28]
            if j:
                out.append(j)
        else:
            out.append(ch)
    return "".join(out)


class ExtraFeaturizer:
    def __init__(self, wv_dim=200, min_count=3, random_state=0):
        self.wv_dim = wv_dim
        self.min_count = min_count
        self.random_state = random_state

    def fit(self, sentences):
        raw = [clean(s) for s in sentences]
        stemmed = [" ".join(stems(s)) for s in raw]
        jam = [to_jamo(depunct(s)) for s in raw]

        # ---- jamo char-ngram tfidf
        self.jvec = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2,
                                    sublinear_tf=True)
        self.jvec.fit(jam)

        # ---- PPMI word vectors from sentence-level co-occurrence
        cv = CountVectorizer(min_df=self.min_count, binary=True, token_pattern=r"\S+")
        C = cv.fit_transform(stemmed).astype(np.float64)
        self.wvocab = {w: i for i, w in enumerate(cv.get_feature_names_out())}
        M = (C.T @ C).tocoo()
        tot = M.data.sum()
        rowsum = np.asarray(M.sum(axis=1)).ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi = np.log((M.data * tot) / (rowsum[M.row] * rowsum[M.col] + 1e-12) + 1e-12)
        pmi = np.maximum(pmi, 0.0)
        P = sp.coo_matrix((pmi, (M.row, M.col)), shape=M.shape).tocsr()
        P.eliminate_zeros()
        d = min(self.wv_dim, min(P.shape) - 1)
        svd = TruncatedSVD(d, random_state=self.random_state)
        W = svd.fit_transform(P)
        self.W = normalize(W).astype(np.float32)

        # ---- idf for weighting
        wt = TfidfVectorizer(min_df=1, token_pattern=r"\S+").fit(stemmed)
        self.idf = dict(zip(wt.get_feature_names_out(), wt.idf_))
        self.max_idf = float(wt.idf_.max())

        # ---- token-level char ngram space (for morphological token sim)
        self.tokvec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1).fit(raw)

        # ---- document LSA + KMeans clusters (domain proxy)
        self.dvec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3,
                                    sublinear_tf=True).fit(raw)
        self.dsvd = TruncatedSVD(64, random_state=self.random_state).fit(
            normalize(self.dvec.transform(raw)))
        from sklearn.cluster import KMeans
        Z = normalize(self.dsvd.transform(normalize(self.dvec.transform(raw))))
        self.km = KMeans(12, n_init=4, random_state=self.random_state).fit(Z)
        return self

    # ------------------------------------------------------------------
    def _wvec(self, toks):
        idx, w, kept = [], [], []
        for t in toks:
            j = self.wvocab.get(t)
            if j is not None:
                idx.append(j)
                w.append(self.idf.get(t, self.max_idf))
                kept.append(t)
        return idx, np.asarray(w, dtype=np.float32), kept

    def transform(self, s1_list, s2_list, verbose=False):
        S1 = [clean(s) for s in s1_list]
        S2 = [clean(s) for s in s2_list]
        n = len(S1)
        cols, names = [], []

        def add(v, nm):
            cols.append(np.asarray(v, dtype=np.float32).reshape(n, -1))
            k = cols[-1].shape[1]
            names.extend([nm] if k == 1 else [f"{nm}_{i}" for i in range(k)])

        # jamo cosine
        J1 = normalize(self.jvec.transform([to_jamo(depunct(s)) for s in S1]))
        J2 = normalize(self.jvec.transform([to_jamo(depunct(s)) for s in S2]))
        add(np.asarray(J1.multiply(J2).sum(1)).ravel(), "cos_jamo")
        inter = J1.minimum(J2)
        add(np.asarray(inter.sum(1)).ravel(), "min_jamo")
        add(2 * np.asarray(inter.sum(1)).ravel() /
            np.maximum(np.asarray(J1.sum(1)).ravel() + np.asarray(J2.sum(1)).ravel(), 1e-9), "dice_jamo")
        for k in (2, 3, 4):
            v = [jac({a[i:i+k] for i in range(max(0, len(a)-k+1))},
                     {b[i:i+k] for i in range(max(0, len(b)-k+1))})
                 for a, b in zip((to_jamo(depunct(s)).replace(" ", "") for s in S1),
                                 (to_jamo(depunct(s)).replace(" ", "") for s in S2))]
            add(np.array(v), f"jamojac{k}")

        # document cluster / LSA-domain features
        D1 = normalize(self.dsvd.transform(normalize(self.dvec.transform(S1))))
        D2 = normalize(self.dsvd.transform(normalize(self.dvec.transform(S2))))
        c1, c2 = self.km.predict(D1), self.km.predict(D2)
        add((c1 == c2).astype(np.float32), "same_cluster")
        K = self.km.n_clusters
        oh = np.zeros((n, K), np.float32)
        for i in range(n):
            oh[i, c1[i]] += 0.5
            oh[i, c2[i]] += 0.5
        add(oh, "cluster")
        add((D1 * D2).sum(1), "dom_cos")

        # word-vector features
        WV = self.W
        rows = np.zeros((n, 14), np.float32)
        for i in range(n):
            t1, t2 = stems(S1[i]), stems(S2[i])
            i1, w1, k1 = self._wvec(t1)
            i2, w2, k2 = self._wvec(t2)
            r = []
            if len(i1) and len(i2):
                A, B = WV[i1], WV[i2]
                # idf-weighted centroid cosine
                ca = (A * w1[:, None]).sum(0); cb = (B * w2[:, None]).sum(0)
                na, nb = np.linalg.norm(ca), np.linalg.norm(cb)
                r.append(float(ca @ cb / (na * nb + 1e-9)))
                ma, mb = A.mean(0), B.mean(0)
                r.append(float(ma @ mb / (np.linalg.norm(ma) * np.linalg.norm(mb) + 1e-9)))
                S = A @ B.T
                # greedy alignment (word-vector only)
                m1, m2 = S.max(1), S.max(0)
                a1 = float(m1 @ w1 / w1.sum()); a2 = float(m2 @ w2 / w2.sum())
                r += [min(a1, a2), (a1 + a2) / 2, max(a1, a2),
                      float(m1.mean()), float(m2.mean()), float(min(m1.mean(), m2.mean()))]
                # hybrid: max(word-vector sim, char-ngram token sim)
                Ta = normalize(self.tokvec.transform(k1))
                Tb = normalize(self.tokvec.transform(k2))
                Sc = (Ta @ Tb.T).toarray()
                H = np.maximum(S, Sc)
                h1, h2 = H.max(1), H.max(0)
                b1 = float(h1 @ w1 / w1.sum()); b2 = float(h2 @ w2 / w2.sum())
                r += [min(b1, b2), (b1 + b2) / 2]
                # unmatched idf mass (soft)
                r.append(float(((1 - h1) * w1).sum() / w1.sum()))
                r.append(float(((1 - h2) * w2).sum() / w2.sum()))
                r.append(float(max(((1 - h1) * w1).sum() / w1.sum(),
                                   ((1 - h2) * w2).sum() / w2.sum())))
                r.append(1.0)
            else:
                r = [0.0] * 13 + [0.0]
            rows[i] = r
            if verbose and i % 2000 == 0:
                print("  wv feats", i, "/", n, flush=True)
        add(rows, "wv")

        # coverage of tokens by embedding vocab
        cov = []
        for i in range(n):
            t1, t2 = stems(S1[i]), stems(S2[i])
            c1_ = sum(t in self.wvocab for t in t1) / max(len(t1), 1)
            c2_ = sum(t in self.wvocab for t in t2) / max(len(t2), 1)
            cov.append([min(c1_, c2_), (c1_ + c2_) / 2])
        add(np.array(cov), "wvcov")

        X = np.nan_to_num(np.hstack(cols).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        self.feature_names_ = names
        return X
