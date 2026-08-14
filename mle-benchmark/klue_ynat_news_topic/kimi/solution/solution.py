# -*- coding: utf-8 -*-
"""YNAT (KLUE Topic Classification) - Korean news title topic classifier.

Approach: TF-IDF feature union (char 2-5, char_wb 3-5, word 1-2 grams)
+ Logistic Regression (C=2.0).
Validation Macro F1 (90/10 stratified split): ~0.826.
"""
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

LABELS = ['IT과학', '경제', '사회', '생활문화', '세계', '스포츠', '정치']

FEATS = [
    dict(analyzer='char',    ngram_range=(2, 5), min_df=3, max_features=400000, sublinear_tf=True),
    dict(analyzer='char_wb', ngram_range=(3, 5), min_df=3, max_features=400000, sublinear_tf=True),
    dict(analyzer='word',    ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True,
         token_pattern=r'(?u)\b\w\w+\b'),
]

def main():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    train['title'] = train['title'].fillna('')
    test['title'] = test['title'].fillna('')

    vecs = [TfidfVectorizer(**kw) for kw in FEATS]
    X_train = hstack([v.fit_transform(train['title']) for v in vecs])
    X_test = hstack([v.transform(test['title']) for v in vecs])

    clf = LogisticRegression(C=2.0, max_iter=3000, n_jobs=-1)
    clf.fit(X_train, train['label'])
    test_pred = clf.predict(X_test)

    sub = pd.DataFrame({'id': test['id'], 'label': test_pred})
    assert sub['id'].is_unique and len(sub) == len(test)
    assert sub['label'].isin(LABELS).all()
    sub.to_csv('outputs/submission.csv', index=False)
    print('Saved outputs/submission.csv')
    print(sub['label'].value_counts().to_dict())

if __name__ == '__main__':
    main()
