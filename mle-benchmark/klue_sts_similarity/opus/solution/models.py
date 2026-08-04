import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVR


def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))


def svr(C=8.0, gamma="scale", eps=0.1):
    return make_pipeline(
        QuantileTransformer(output_distribution="normal", n_quantiles=1000, random_state=0),
        SVR(C=C, gamma=gamma, epsilon=eps, cache_size=800),
    )


def krr(alpha=0.3, gamma=None):
    return make_pipeline(
        QuantileTransformer(output_distribution="normal", n_quantiles=1000, random_state=0),
        KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma),
    )


def hgb(seed=0, lr=0.05, leaves=31, it=600, l2=1.0, mf=0.6):
    return HistGradientBoostingRegressor(
        learning_rate=lr, max_iter=it, max_leaf_nodes=leaves, min_samples_leaf=20,
        l2_regularization=l2, max_features=mf, early_stopping=False,
        random_state=seed,
    )


def et(seed=0):
    return ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, max_features=0.35,
                               n_jobs=-1, random_state=seed)


def rf(seed=0):
    return RandomForestRegressor(n_estimators=400, min_samples_leaf=2, max_features=0.3,
                                 n_jobs=-1, random_state=seed)


def hgb_abs(seed=0):
    return HistGradientBoostingRegressor(
        loss="absolute_error", learning_rate=0.04, max_iter=1200, max_leaf_nodes=31,
        min_samples_leaf=20, l2_regularization=2.0, max_features=0.4,
        early_stopping=False, random_state=seed)


def mlp(seed=0, hidden=(256, 64), alpha=1e-3):
    from sklearn.neural_network import MLPRegressor
    return make_pipeline(
        QuantileTransformer(output_distribution="normal", n_quantiles=1000, random_state=0),
        MLPRegressor(hidden_layer_sizes=hidden, alpha=alpha, learning_rate_init=1e-3,
                     batch_size=128, max_iter=300, early_stopping=True,
                     n_iter_no_change=15, validation_fraction=0.1, random_state=seed),
    )


ZOO = {"ridge": ridge, "svr": svr, "krr": krr, "hgb": hgb, "et": et, "rf": rf,
       "hgb_abs": hgb_abs, "mlp": mlp}
