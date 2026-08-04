"""Conditional (listwise softmax) logistic regression for k-way MCQ."""
import numpy as np
from scipy import sparse as sp
from scipy.optimize import minimize


class CondLogit:
    def __init__(self, C=1.0, k=4, max_iter=300, fit_intercept=False):
        self.C = C; self.k = k; self.max_iter = max_iter

    def fit(self, X, y):
        """X: (n*k, d) sparse or dense, rows grouped by question in order.
        y: (n,) index of correct option in 0..k-1"""
        k = self.k
        n = X.shape[0] // k
        d = X.shape[1]
        Xc = X.tocsr() if sp.issparse(X) else np.asarray(X, dtype=np.float64)
        lam = 1.0 / (self.C * n)
        onehot = np.zeros((n, k)); onehot[np.arange(n), y] = 1.0

        def f(w):
            s = (Xc @ w).reshape(n, k)
            m = s.max(1, keepdims=True)
            e = np.exp(s - m)
            Z = e.sum(1, keepdims=True)
            logp = s - m - np.log(Z)
            loss = -logp[np.arange(n), y].mean()
            P = e / Z
            G = (P - onehot) / n
            grad = Xc.T @ G.ravel()
            grad = np.asarray(grad).ravel()
            loss += 0.5 * lam * (w @ w)
            grad += lam * w
            return loss, grad

        w0 = np.zeros(d)
        res = minimize(f, w0, jac=True, method="L-BFGS-B",
                       options={"maxiter": self.max_iter, "maxfun": self.max_iter * 2})
        self.w = res.x
        self.loss_ = res.fun
        return self

    def decision(self, X):
        Xc = X.tocsr() if sp.issparse(X) else X
        return np.asarray(Xc @ self.w).ravel()

    def predict_proba_groups(self, X):
        s = self.decision(X).reshape(-1, self.k)
        s = s - s.max(1, keepdims=True)
        e = np.exp(s)
        return e / e.sum(1, keepdims=True)
