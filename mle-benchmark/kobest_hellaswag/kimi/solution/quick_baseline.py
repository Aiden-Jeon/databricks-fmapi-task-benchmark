"""빠른 유효 baseline: context와 각 ending의 TF-IDF 코사인 유사도가 최대인 후보 선택."""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tr = pd.read_csv("train.csv")
te = pd.read_csv("test.csv")

END = ["ending_1", "ending_2", "ending_3", "ending_4"]


def predict(df, vec):
    ctx = vec.transform(df["context"])
    sims = []
    for c in END:
        e = vec.transform(df[c])
        sims.append(cosine_similarity(ctx, e).diagonal())
    import numpy as np
    return np.argmax(np.vstack(sims).T, axis=1)


corpus = pd.concat([tr["context"], te["context"]] + [tr[c] for c in END] + [te[c] for c in END])
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=80000)
vec.fit(corpus)

pred = predict(te, vec)
sub = pd.DataFrame({"id": te["id"], "label": pred})
sub.to_csv("outputs/submission.csv", index=False)
print(sub["label"].value_counts(normalize=True))
print("saved", sub.shape)
