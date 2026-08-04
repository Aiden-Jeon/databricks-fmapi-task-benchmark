"""Models for the 4-way choice task: conditional logit (softmax over the 4
candidates of a group) + gradient boosting on within-group normalised features."""
import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class CondLogit:
    """Multinomial conditional logit: P(choice=i) ∝ exp(w·x_i)."""

    def __init__(self, C=1.0, maxiter=400):
        self.C = C
        self.maxiter = maxiter

    def fit(self, X, y, sw=None):
        n, k, d = X.shape
        Y = np.zeros((n, k)); Y[np.arange(n), y] = 1.0
        w_ = np.ones(n) if sw is None else np.asarray(sw, float)
        w_ = w_ / w_.mean()
        lam = 1.0 / (self.C * n)

        def f(w):
            z = X @ w
            p = _softmax(z)
            nll = -(w_ * np.log(np.clip(p[np.arange(n), y], 1e-12, None))).mean()
            g = (((p - Y) * w_[:, None])[:, :, None] * X).sum((0, 1)) / n
            return nll + lam * 0.5 * (w @ w), g + lam * w

        w0 = np.zeros(d)
        r = minimize(f, w0, jac=True, method='L-BFGS-B',
                     options=dict(maxiter=self.maxiter))
        self.w = r.x
        return self

    def decision(self, X):
        return X @ self.w

    def predict_proba(self, X):
        return _softmax(self.decision(X))


class FlatClf:
    """Binary correct/incorrect classifier on flattened candidates; scores are
    re-normalised inside each group."""

    def __init__(self, kind='hgb', **kw):
        self.kind = kind
        self.kw = kw

    def _new(self):
        if self.kind == 'hgb':
            p = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
                     min_samples_leaf=25, l2_regularization=1.0,
                     random_state=0, early_stopping=False)
            p.update(self.kw)
            return HistGradientBoostingClassifier(**p)
        if self.kind == 'et':
            p = dict(n_estimators=500, min_samples_leaf=5, max_features='sqrt',
                     n_jobs=-1, random_state=0)
            p.update(self.kw)
            return ExtraTreesClassifier(**p)
        p = dict(C=0.3, max_iter=3000)
        p.update(self.kw)
        return LogisticRegression(**p)

    def fit(self, X, y, sw=None):
        n, k, d = X.shape
        Xf = X.reshape(n * k, d)
        yf = np.zeros(n * k); yf[np.arange(n) * k + y] = 1
        swf = None if sw is None else np.repeat(np.asarray(sw, float), k)
        self.m = self._new().fit(Xf, yf, sample_weight=swf)
        return self

    def decision(self, X):
        n, k, d = X.shape
        p = self.m.predict_proba(X.reshape(n * k, d))[:, 1].reshape(n, k)
        return np.log(np.clip(p, 1e-9, 1)) - np.log(np.clip(1 - p, 1e-9, 1))

    def predict_proba(self, X):
        return _softmax(self.decision(X))
