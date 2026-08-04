import copy
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score


DATA = "/tmp/kmle/M3_t23_korfin_asc_full_20260804_033756/task"
RANDOM_STATE = 42

POS_WORDS = [
    "상승", "급등", "증가", "성장", "수혜", "호조", "개선", "흑자", "신고가",
    "강세", "급증", "양호", "우상향", "돌파", "안착", "효과", "기대", "긍정",
    "장려", "유지", "견조", "상향", "플러스", "승리", "성공", "혁신", "선두",
    "우위", "경쟁력", "이익", "급등", "폭등", "반등", "회복", "호황", "래리",
    "사상최고", "최대", "최고", "상승세", "증가세", "개선세", "수요증가",
    "팽창", "확대", "확장", "도약", "약진", "선방", "반등", "반격",
]
NEG_WORDS = [
    "하락", "급락", "감소", "부진", "타격", "악화", "적자", "하향", "손실",
    "약세", "급감", "저조", "우하향", "하회", "마이너스", "패배", "실패",
    "위기", "리스크", "부도", "우려", "하향", "약화", "하루", "폭락",
    "급감", "감소세", "하락세", "악화세", "수요감소", "축소", "위축",
    "침체", "부진", "적자", "손실", "누적", "미수", "부실", "우려",
    "하회", "미달", "감원", "퇴출", "중단", "중지", "연착", "정체",
    "사상최저", "최저", "최악", "약전", "하락장", "폭락",
]
# negation cues
NEG_CUES = ["않", "못", "없", "아니", "미", "안 "]


def aspect_window(sentence, aspect, before=40, after=40):
    idx = sentence.find(aspect)
    if idx == -1:
        return sentence, "", "", -1
    s = max(0, idx - before)
    e = min(len(sentence), idx + len(aspect) + after)
    left = sentence[s:idx]
    right = sentence[idx + len(aspect):e]
    return sentence[s:e], left, right, idx


def lexicon_features(df):
    feats = []
    for _, r in df.iterrows():
        s = str(r["sentence"])
        ctx, left, right, _ = aspect_window(s, str(r["aspect"]))
        pos_c = sum(s.count(p) for p in POS_WORDS)
        neg_c = sum(s.count(n) for n in NEG_WORDS)
        pos_ctx = sum(ctx.count(p) for p in POS_WORDS)
        neg_ctx = sum(ctx.count(n) for n in NEG_WORDS)
        pos_l = sum(left.count(p) for p in POS_WORDS)
        neg_l = sum(left.count(n) for n in NEG_WORDS)
        pos_r = sum(right.count(p) for p in POS_WORDS)
        neg_r = sum(right.count(n) for n in NEG_WORDS)
        neg_cue = sum(s.count(c) for c in NEG_CUES)
        feats.append([pos_c, neg_c, pos_ctx, neg_ctx, pos_l, neg_l, pos_r, neg_r,
                      pos_c - neg_c, pos_ctx - neg_ctx, pos_l - neg_l,
                      pos_r - neg_r, neg_cue])
    return np.array(feats, dtype=float)


def add_features(df):
    df = df.copy()
    df["sentence"] = df["sentence"].astype(str)
    df["aspect"] = df["aspect"].astype(str)
    ctxs, lefts, rights = [], [], []
    for _, r in df.iterrows():
        c, l, rr, _ = aspect_window(r["sentence"], r["aspect"])
        ctxs.append(c); lefts.append(l); rights.append(rr)
    df["ctx"] = ctxs
    df["left"] = lefts
    df["right"] = rights
    df["combo"] = df["ctx"] + " [ASP] " + df["aspect"] + " [ASP] " + df["sentence"]
    df["left_asp"] = df["left"] + " [ASP] " + df["aspect"]
    df["asp_right"] = df["aspect"] + " [ASP] " + df["right"]
    df["left_only"] = df["left"]
    df["right_only"] = df["right"]
    return df


def build_mat(train_df, test_df, col, analyzer="char_wb", ngram=(1, 5), min_df=2):
    vec = TfidfVectorizer(
        analyzer=analyzer, ngram_range=ngram, min_df=min_df, max_df=0.95,
        sublinear_tf=True, lowercase=False,
    )
    Xtr = vec.fit_transform(train_df[col])
    Xte = vec.transform(test_df[col])
    return Xtr, Xte


def run_cv(Xtr, Xte, y, model, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y))
    te_prob = np.zeros((Xte.shape[0], len(np.unique(y))))
    for tr, va in skf.split(Xtr, y):
        m = copy.deepcopy(model)
        m.fit(Xtr[tr], y[tr])
        oof[va] = m.predict(Xtr[va])
        te_prob += m.predict_proba(Xte) / n_splits
    score = f1_score(y, oof, average="macro")
    return oof, te_prob, score


def main():
    train = pd.read_csv(f"{DATA}/train.csv")
    test = pd.read_csv(f"{DATA}/test.csv")
    train = add_features(train)
    test = add_features(test)

    le = LabelEncoder()
    y = le.fit_transform(train["label"])

    feats = [
        ("combo", "char_wb", (1, 6), 2),
        ("combo", "word", (1, 2), 2),
        ("left_asp", "char_wb", (1, 4), 2),
        ("asp_right", "char_wb", (1, 4), 2),
        ("left_only", "char_wb", (1, 4), 1),
        ("right_only", "char_wb", (1, 4), 1),
        ("ctx", "char_wb", (1, 4), 1),
        ("sentence", "char_wb", (1, 4), 1),
    ]
    Xtr_parts, Xte_parts = [], []
    for col, an, ng, md in feats:
        a, b = build_mat(train, test, col, an, ng, md)
        Xtr_parts.append(a); Xte_parts.append(b)
    Xtr = hstack(Xtr_parts).tocsr()
    Xte = hstack(Xte_parts).tocsr()
    print("TFIDF Xtr:", Xtr.shape)

    # add lexicon features
    lx_tr = lexicon_features(train)
    lx_te = lexicon_features(test)
    from scipy.sparse import csr_matrix
    Xtr = hstack([Xtr, csr_matrix(lx_tr)]).tocsr()
    Xte = hstack([Xte, csr_matrix(lx_te)]).tocsr()
    print("Full Xtr:", Xtr.shape, "Xte:", Xte.shape)

    models = {
        "lr_c1": LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced",
                                     n_jobs=-1, random_state=RANDOM_STATE),
        "lr_c15": LogisticRegression(C=1.5, max_iter=3000, class_weight="balanced",
                                     n_jobs=-1, random_state=RANDOM_STATE),
        "lr_c05": LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced",
                                     n_jobs=-1, random_state=RANDOM_STATE),
        "lr_c2": LogisticRegression(C=2.0, max_iter=3000, class_weight="balanced",
                                    n_jobs=-1, random_state=RANDOM_STATE),
    }
    results = {}
    te_probs = {}
    for name, model in models.items():
        oof, te_p, sc = run_cv(Xtr, Xte, y, model)
        results[name] = sc
        te_probs[name] = te_p
        print(f"{name}: {sc:.4f}")

    # weight by score
    weights = np.array([results[n] for n in models])
    w = weights / weights.sum()
    te_avg = sum(te_probs[n] * ww for n, ww in zip(models, w))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_ens = np.zeros(len(y))
    for tr, va in skf.split(Xtr, y):
        fp = np.zeros((len(va), len(np.unique(y))))
        for i, (n, m) in enumerate(models.items()):
            mm = copy.deepcopy(m)
            mm.fit(Xtr[tr], y[tr])
            fp += mm.predict_proba(Xtr[va]) * w[i]
        oof_ens[va] = fp.argmax(1)
    ens_sc = f1_score(y, oof_ens, average="macro")
    print(f"ENSEMBLE: {ens_sc:.4f}")

    # pick best
    best = max(results, key=results.get)
    print("best single:", best, results[best])
    if ens_sc > results[best]:
        print("using ensemble")
        final = te_avg.argmax(1)
    else:
        print("using", best)
        final = te_probs[best].argmax(1)

    labels = le.inverse_transform(final)
    out = pd.DataFrame({"id": test["id"], "label": labels})
    out.to_csv(f"{DATA}/outputs/submission.csv", index=False)
    print("Saved:", out.shape)
    print(out["label"].value_counts())


if __name__ == "__main__":
    main()
