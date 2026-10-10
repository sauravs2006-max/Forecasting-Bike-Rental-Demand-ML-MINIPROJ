"""Model zoo mirroring the paper: Linear, GLM, GBM, PCR, SVR, RF, tree, + stacking."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from features import CATEGORICAL


def make_preprocessor(X, onehot):
    if not onehot:
        return "passthrough"
    cat = [c for c in CATEGORICAL if c in X.columns]
    num = [c for c in X.columns if c not in cat]
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ("num", StandardScaler(), num),
    ])


def log_target(reg):
    """Train on log1p(count) so squared error == squared *logarithmic* error."""
    return TransformedTargetRegressor(regressor=reg, func=np.log1p, inverse_func=np.expm1)


def get_models(X, fast=False, seed=0):
    """Return {name: (pipeline, param_grid)}. In fast mode only the first grid value is used."""
    R = "reg__regressor__"
    n_rf = 100 if fast else 300
    models = {
        "Linear": (Pipeline([("pre", make_preprocessor(X, True)),
                             ("reg", log_target(LinearRegression()))]), {}),
        "Poisson GLM": (Pipeline([("pre", make_preprocessor(X, True)),
                                  ("reg", PoissonRegressor(max_iter=1000))]),
                        {"reg__alpha": [1e-3, 1e-4, 1e-2, 1e-1]}),
        "PCR": (Pipeline([("pre", make_preprocessor(X, True)),
                          ("reg", log_target(Pipeline([("pca", PCA()), ("lr", LinearRegression())])))]),
                {R + "pca__n_components": [30, 20, 10]}),
        "SVR (RBF)": (Pipeline([("pre", make_preprocessor(X, True)),
                                ("reg", log_target(SVR(kernel="rbf")))]),
                      {R + "C": [10, 1, 30], R + "epsilon": [0.1]}),
        "GBM (Hist)": (Pipeline([("pre", "passthrough"),
                                 ("reg", log_target(HistGradientBoostingRegressor(random_state=seed)))]),
                       {R + "learning_rate": [0.05, 0.1], R + "max_iter": [400, 200, 800],
                        R + "max_leaf_nodes": [31, 15]}),
        "Random Forest": (Pipeline([("pre", "passthrough"),
                                    ("reg", log_target(RandomForestRegressor(
                                        n_estimators=n_rf, n_jobs=-1, random_state=seed)))]),
                          {R + "min_samples_leaf": [2, 1, 4], R + "max_features": [0.7, 0.5, 1.0]}),
        "Decision Tree": (Pipeline([("pre", "passthrough"),
                                    ("reg", log_target(DecisionTreeRegressor(random_state=seed)))]),
                          {R + "max_depth": [12, 8, 16, None], R + "min_samples_leaf": [5, 2, 10]}),
    }
    if fast:
        models = {n: (p, {k: v[:1] for k, v in g.items()}) for n, (p, g) in models.items()}
    return models


class LogStack(BaseEstimator, RegressorMixin):
    """Stacking in log space: out-of-fold base predictions -> linear meta-model."""

    def __init__(self, estimators, cv=5, random_state=0):
        self.estimators = estimators
        self.cv = cv
        self.random_state = random_state

    def fit(self, X, y):
        kf = KFold(self.cv, shuffle=True, random_state=self.random_state)
        oof = np.column_stack([
            np.log1p(np.clip(cross_val_predict(clone(e), X, y, cv=kf), 0, None))
            for _, e in self.estimators])
        self.final_ = LinearRegression().fit(oof, np.log1p(np.asarray(y, dtype=float)))
        self.fitted_ = [clone(e).fit(X, y) for _, e in self.estimators]
        return self

    def predict(self, X):
        P = np.column_stack([np.log1p(np.clip(m.predict(X), 0, None)) for m in self.fitted_])
        return np.expm1(self.final_.predict(P))
