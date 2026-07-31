"""KoBEST HellaSwag solution.

Idea: the correct ending continues the narrative, so it tends to share
character-level surface form with the LAST sentence of the context. We compute
TF-IDF char(1,2) cosine similarity between each ending and the last sentence of
the context, then pick the highest-similarity ending per row.

CV accuracy ~54% (vs 25% random). Only sklearn/pandas/numpy used; no external
data or pretrained weights.
"""
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold

TASK = "/tmp/kmle/M3_t15_kobest_hellaswag_full_20260731_043738/task"
RNG = 42


def split_sentences(s):
    parts = re.split(r'(?<=[\.!?])\s+', s.strip())
    return [p for p in parts if p]


def last_sentence(s):
    sp = split_sentences(s)
    return sp[-1] if sp else s


def score_endings(df, vec):
    """Return (n,4) cosine-similarity of each ending with the last sentence."""
    refs = [last_sentence(c) for c in df['context']]
    R = vec.transform(refs)
    S = np.zeros((len(df), 4))
    for i in range(4):
        E = vec.transform(df[f'ending_{i+1}'])
        S[:, i] = cosine_similarity(R, E).diagonal()
    return S


def fit_vec(df):
    refs = [last_sentence(c) for c in df['context']]
    ends = [e for i in range(1, 5) for e in df[f'ending_{i}']]
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 2), min_df=1)
    vec.fit(refs + ends + df['context'].tolist())
    return vec


def main():
    tr = pd.read_csv(f"{TASK}/train.csv")
    te = pd.read_csv(f"{TASK}/test.csv")
    y = tr['label'].values

    # Cross-validation estimate
    skf = StratifiedKFold(5, shuffle=True, random_state=RNG)
    rows = np.arange(len(tr))
    accs = []
    for trr, var in skf.split(rows, y):
        vec = fit_vec(tr.iloc[trr])
        S = score_endings(tr.iloc[var], vec)
        accs.append((S.argmax(1) == y[var]).mean())
    print(f"CV accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f}  folds={[f'{a:.3f}' for a in accs]}")

    # Fit on all training data, predict test
    vec = fit_vec(tr)
    S = score_endings(te, vec)
    pred = S.argmax(1)

    out = pd.DataFrame({'id': te['id'], 'label': pred})
    out.to_csv(f"{TASK}/outputs/submission.csv", index=False)
    print("saved", out.shape, "label dist:", out['label'].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
