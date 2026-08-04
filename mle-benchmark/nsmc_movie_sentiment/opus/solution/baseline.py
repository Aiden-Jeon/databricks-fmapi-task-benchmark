"""Quick baseline: char+word TF-IDF -> LogisticRegression. Writes outputs/submission.csv."""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_union
from sklearn.linear_model import LogisticRegression

tr = pd.read_csv('train.csv')
te = pd.read_csv('test.csv')
Xtr_txt = tr.document.astype(str).values
Xte_txt = te.document.astype(str).values

feats = make_union(
    TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 4), min_df=3, sublinear_tf=True),
    TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2, sublinear_tf=True),
)
Xtr = feats.fit_transform(Xtr_txt)
Xte = feats.transform(Xte_txt)
print('features', Xtr.shape)

clf = LogisticRegression(C=4, max_iter=2000, solver='liblinear')
clf.fit(Xtr, tr.label.values)
pred = clf.predict(Xte)
pd.DataFrame({'id': te.id.values, 'label': pred.astype(int)}).to_csv('outputs/submission.csv', index=False)
print('done', pd.Series(pred).value_counts().to_dict())
