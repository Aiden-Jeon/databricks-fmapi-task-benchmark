import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC


ENDINGS = [f"ending_{i}" for i in range(1, 5)]


def normalize(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def handcrafted(contexts, endings, positions):
    rows = []
    cues = ["후", "뒤", "다음", "마지막", "완성", "시작", "먼저", "다시", "계속", "결국", "그러", "그래서", "이윽고"]
    for context, ending, position in zip(contexts, endings, positions):
        context = normalize(context)
        ending = normalize(ending)
        last = re.split(r"[.!?]", context.rstrip(".!?"))[-1]
        c_words = set(re.findall(r"[가-힣A-Za-z0-9]+", context))
        e_words = set(re.findall(r"[가-힣A-Za-z0-9]+", ending))
        c_bigrams = {context[i:i + 2] for i in range(len(context) - 1)}
        l_bigrams = {last[i:i + 2] for i in range(len(last) - 1)}
        e_bigrams = {ending[i:i + 2] for i in range(len(ending) - 1)}
        rows.append([
            len(ending) / 100,
            len(context) / 500,
            ending.count(".") / 3,
            len(c_words & e_words) / max(1, len(e_words)),
            len(c_bigrams & e_bigrams) / max(1, len(e_bigrams)),
            len(l_bigrams & e_bigrams) / max(1, len(e_bigrams)),
            float(ending.split(" ", 1)[0] in context),
            *[float(cue in ending) for cue in cues],
            *[float(position == i) for i in range(4)],
        ])
    rows = np.asarray(rows)
    lengths = rows[:, 0].reshape(-1, 4)
    relative_length = ((lengths - lengths.mean(axis=1, keepdims=True)) /
                       (lengths.std(axis=1, keepdims=True) + 1e-6)).reshape(-1, 1)
    length_rank = np.argsort(np.argsort(lengths, axis=1), axis=1).reshape(-1, 1) / 3
    return sparse.csr_matrix(np.hstack([rows, relative_length, length_rank]))


def similarities(vectorizer, contexts, endings):
    ending_x = vectorizer.transform(endings)
    context_x = vectorizer.transform(contexts)
    lasts = [re.split(r"[.!?]", normalize(text).rstrip(".!?"))[-1] for text in contexts]
    last_x = vectorizer.transform(lasts)
    whole = np.asarray(context_x.multiply(ending_x).sum(axis=1))
    last = np.asarray(last_x.multiply(ending_x).sum(axis=1))
    pairwise = []
    for start in range(0, len(endings), 4):
        matrix = (ending_x[start:start + 4] @ ending_x[start:start + 4].T).toarray()
        matrix[np.arange(4), np.arange(4)] = np.nan
        pairwise.extend(zip(np.nanmean(matrix, axis=1), np.nanmax(matrix, axis=1)))
    return sparse.csr_matrix(np.hstack([whole, last, np.asarray(pairwise)]))


def flatten(df):
    contexts, endings, positions, targets = [], [], [], []
    has_label = "label" in df
    for row in df.itertuples(index=False):
        for pos, column in enumerate(ENDINGS):
            contexts.append(row.context)
            endings.append(getattr(row, column))
            positions.append(pos)
            if has_label:
                targets.append(int(row.label == pos))
    return contexts, endings, np.asarray(positions), np.asarray(targets)


def evaluate(train):
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    scores = {"hand": [], "c0.5": [], "c1": [], "c2": [], "c4": [], "svc": []}
    labels = train.label.to_numpy()
    for fold, (fit_idx, val_idx) in enumerate(splitter.split(train, labels), 1):
        fit = train.iloc[fit_idx]
        val = train.iloc[val_idx]
        fc, fe, fp, fy = flatten(fit)
        vc, ve, vp, _ = flatten(val)

        char = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, max_features=120000, sublinear_tf=True)
        word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=80000, sublinear_tf=True)
        xc = char.fit_transform(fe)
        vc_x = char.transform(ve)
        xw = word.fit_transform(fe)
        vw_x = word.transform(ve)
        xh = handcrafted(fc, fe, fp)
        vh = handcrafted(vc, ve, vp)
        xs = similarities(char, fc, fe)
        vs = similarities(char, vc, ve)
        xws = similarities(word, fc, fe)
        vws = similarities(word, vc, ve)

        weighted_x = sparse.hstack([xc, xw * 0.3, xh, xs, xws])
        weighted_v = sparse.hstack([vc_x, vw_x * 0.3, vh, vs, vws])
        configurations = [("hand", sparse.hstack([xh, xs, xws]), sparse.hstack([vh, vs, vws]), 2.0)]
        configurations += [(f"c{c:g}", weighted_x, weighted_v, c) for c in [0.5, 1.0, 2.0, 4.0]]
        for name, xfit, xval, c in configurations:
            model = LogisticRegression(C=c, max_iter=1000, class_weight="balanced", solver="liblinear", random_state=2026)
            model.fit(xfit, fy)
            pred = model.decision_function(xval).reshape(-1, 4).argmax(axis=1)
            scores[name].append(accuracy_score(labels[val_idx], pred))
        svc = LinearSVC(C=0.3, class_weight="balanced", random_state=2026)
        svc.fit(sparse.hstack([xc, xw * 0.3, xh, xs]), fy)
        pred = svc.decision_function(sparse.hstack([vc_x, vw_x * 0.3, vh, vs])).reshape(-1, 4).argmax(axis=1)
        scores["svc"].append(accuracy_score(labels[val_idx], pred))
        print(f"fold {fold}: " + " ".join(f"{key}={values[-1]:.4f}" for key, values in scores.items()))
    for key, values in scores.items():
        print(f"{key}: {np.mean(values):.4f} +/- {np.std(values):.4f}")


if __name__ == "__main__":
    evaluate(pd.read_csv("train.csv"))
