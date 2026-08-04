"""Quick TF-IDF cosine baseline for KLUE-STS."""
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.stats import pearsonr

tr = pd.read_csv('train.csv')
te = pd.read_csv('test.csv')
corpus = pd.concat([tr.sentence1, tr.sentence2, te.sentence1, te.sentence2]).tolist()

vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2, sublinear_tf=True)
vec.fit(corpus)


def cos(a, b):
    A = normalize(vec.transform(a)); B = normalize(vec.transform(b))
    return np.asarray(A.multiply(B).sum(axis=1)).ravel()


tr_c = cos(tr.sentence1, tr.sentence2)
te_c = cos(te.sentence1, te.sentence2)
print('train pearson (raw cosine):', pearsonr(tr_c, tr.score)[0])

# linear map cosine -> score range
a, b = np.polyfit(tr_c, tr.score, 1)
pred = np.clip(a * te_c + b, 0, 5)
import os
os.makedirs('outputs', exist_ok=True)
pd.DataFrame({'id': te.id, 'score': pred}).to_csv('outputs/submission.csv', index=False)
print('written', pred.min(), pred.max(), pred.mean())
