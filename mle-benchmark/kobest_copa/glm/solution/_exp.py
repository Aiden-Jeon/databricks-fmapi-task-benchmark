import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline, FeatureUnion
import re, sys

RANDOM_STATE = 0

tr = pd.read_csv('train.csv')
tr['question'] = tr['question'].str.strip()

# Approach: represent row as combined text and classify label directly
tr['text'] = (tr['premise'].astype(str) + ' [Q] ' + tr['question'].astype(str)
              + ' [A1] ' + tr['alternative_1'].astype(str)
              + ' [A2] ' + tr['alternative_2'].astype(str))

word_vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2)
char_vec = TfidfVectorizer(sublinear_tf=True, analyzer='char_wb', ngram_range=(2, 4), min_df=2)
union = FeatureUnion([('w', word_vec), ('c', char_vec)])
pipe = Pipeline([('f', union), ('c', LogisticRegression(C=4.0, max_iter=2000, random_state=RANDOM_STATE))])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scores = cross_val_score(pipe, tr['text'], tr['label'], cv=cv, scoring='accuracy')
print('CV acc:', scores, 'mean', scores.mean())
