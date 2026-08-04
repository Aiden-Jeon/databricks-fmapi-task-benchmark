"""Quick safe baseline submission: TF-IDF(word+char) -> LinearSVC."""
import pandas as pd
from sklearn.pipeline import make_pipeline, make_union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

TASK = "/tmp/kmle/M1_t3_ynat_full_20260804_033458/task"
tr = pd.read_csv(f"{TASK}/train.csv")
te = pd.read_csv(f"{TASK}/test.csv")

pipe = make_pipeline(
    make_union(
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
    ),
    LinearSVC(C=1, dual=True),
)
pipe.fit(tr.title.values, tr.label.values)
pred = pipe.predict(te.title.values)
pd.DataFrame({"id": te.id, "label": pred}).to_csv(f"{TASK}/outputs/submission.csv", index=False)
print("wrote", len(pred))
