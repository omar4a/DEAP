from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable

from sklearn.decomposition import PCA
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import LinearSVC, SVC, SVR

from .riemann import RiemannianTangentSpace

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator_cls: Any
    default_params: Dict[str, Any]
    param_grid: Dict[str, Iterable[Any]]
    needs_scaler: bool
    task: str


MODEL_SPECS = {
    "logreg": ModelSpec(
        name="logreg",
        estimator_cls=LogisticRegression,
        default_params={
            "solver": "lbfgs",
            "max_iter": 2000,
        },
        param_grid={
            "clf__C": [0.1, 1.0, 10.0],
        },
        needs_scaler=True,
        task="classification",
    ),
    "linear_svm": ModelSpec(
        name="linear_svm",
        estimator_cls=LinearSVC,
        default_params={},
        param_grid={
            "clf__C": [0.1, 1.0, 10.0],
        },
        needs_scaler=True,
        task="classification",
    ),
    "rbf_svm": ModelSpec(
        name="rbf_svm",
        estimator_cls=SVC,
        default_params={
            "kernel": "rbf",
            "probability": True,
        },
        param_grid={
            "clf__C": [0.1, 1.0, 10.0, 100.0, 300.0],
            "clf__gamma": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
            "clf__class_weight": [None, "balanced"],
        },
        needs_scaler=True,
        task="classification",
    ),
    "hgb": ModelSpec(
        name="hgb",
        estimator_cls=HistGradientBoostingClassifier,
        default_params={"random_state": 42},
        param_grid={
            "clf__max_depth": [None, 3, 5],
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_iter": [200, 400],
        },
        needs_scaler=False,
        task="classification",
    ),
    "rf": ModelSpec(
        name="rf",
        estimator_cls=RandomForestClassifier,
        default_params={
            "n_estimators": 300,
            "class_weight": "balanced_subsample",
            "random_state": 42,
        },
        param_grid={
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [None, 10, 20],
        },
        needs_scaler=False,
        task="classification",
    ),
    "mlp": ModelSpec(
        name="mlp",
        estimator_cls=MLPClassifier,
        default_params={
            "hidden_layer_sizes": (128, 64),
            "max_iter": 500,
        },
        param_grid={
            "clf__hidden_layer_sizes": [(128, 64), (256, 128)],
            "clf__alpha": [0.0001, 0.001],
        },
        needs_scaler=True,
        task="classification",
    ),
    "ridge": ModelSpec(
        name="ridge",
        estimator_cls=Ridge,
        default_params={},
        param_grid={
            "clf__alpha": [0.1, 1.0, 10.0],
        },
        needs_scaler=True,
        task="regression",
    ),
    "svr": ModelSpec(
        name="svr",
        estimator_cls=SVR,
        default_params={"kernel": "rbf"},
        param_grid={
            "clf__C": [0.1, 1.0, 10.0],
            "clf__gamma": [0.001, 0.01, 0.1],
        },
        needs_scaler=True,
        task="regression",
    ),
    "rf_reg": ModelSpec(
        name="rf_reg",
        estimator_cls=RandomForestRegressor,
        default_params={"n_estimators": 300, "random_state": 42},
        param_grid={
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [None, 10, 20],
        },
        needs_scaler=False,
        task="regression",
    ),
    "mlp_reg": ModelSpec(
        name="mlp_reg",
        estimator_cls=MLPRegressor,
        default_params={
            "hidden_layer_sizes": (128, 64),
            "max_iter": 500,
        },
        param_grid={
            "clf__hidden_layer_sizes": [(128, 64), (256, 128)],
            "clf__alpha": [0.0001, 0.001],
        },
        needs_scaler=True,
        task="regression",
    ),
}

if XGBClassifier is not None:
    MODEL_SPECS["xgb"] = ModelSpec(
        name="xgb",
        estimator_cls=XGBClassifier,
        default_params={
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        },
        param_grid={
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [3, 5],
            "clf__learning_rate": [0.05, 0.1],
            "clf__subsample": [0.8, 1.0],
            "clf__colsample_bytree": [0.8, 1.0],
        },
        needs_scaler=False,
        task="classification",
    )

if LGBMClassifier is not None:
    MODEL_SPECS["lgbm"] = ModelSpec(
        name="lgbm",
        estimator_cls=LGBMClassifier,
        default_params={"objective": "binary", "n_estimators": 200, "verbose": -1},
        param_grid={
            "clf__n_estimators": [200, 400],
            "clf__learning_rate": [0.05, 0.1],
            "clf__num_leaves": [31, 63],
        },
        needs_scaler=False,
        task="classification",
    )

if CatBoostClassifier is not None:
    MODEL_SPECS["catboost"] = ModelSpec(
        name="catboost",
        estimator_cls=CatBoostClassifier,
        default_params={
            "iterations": 200,
            "verbose": False,
            "random_state": 42,
        },
        param_grid={
            "clf__iterations": [200, 400],
            "clf__learning_rate": [0.05, 0.1],
            "clf__depth": [4, 6, 8],
        },
        needs_scaler=False,
        task="classification",
    )


def build_model(
    name: str,
    params: Dict[str, Any] | None = None,
    pca_variance: float | None = None,
    feature_selection: str = "none",
    mi_k: int | None = None,
    l1_c: float | None = None,
    l1_alpha: float | None = None,
    scaler: str = "standard",
    riemann_cov_dim: int | None = None,
    riemann_epsilon: float = 1e-6,
) -> Pipeline:
    if name not in MODEL_SPECS:
        raise ValueError(f"Unknown model: {name}")
    spec = MODEL_SPECS[name]
    cfg = dict(spec.default_params)
    if params:
        cfg.update(params)
    estimator = spec.estimator_cls(**cfg)

    steps = []
    if riemann_cov_dim is not None:
        steps.append(("riemann", RiemannianTangentSpace(riemann_cov_dim, riemann_epsilon)))
    if spec.needs_scaler or pca_variance is not None or feature_selection in ("mi", "l1"):
        scaler_step = RobustScaler() if scaler == "robust" else StandardScaler()
        steps.append(("scaler", scaler_step))
    if feature_selection == "mi":
        if mi_k is None:
            raise ValueError("mi_k must be set when using MI feature selection.")
        score_fn = mutual_info_classif if spec.task == "classification" else mutual_info_regression
        steps.append(("select", SelectKBest(score_func=score_fn, k=mi_k)))
    if feature_selection == "l1":
        if spec.task == "classification":
            selector_c = 1.0 if l1_c is None else l1_c
            selector = LogisticRegression(
                penalty="l1",
                solver="liblinear",
                C=selector_c,
                max_iter=2000,
            )
        else:
            selector_alpha = 0.001 if l1_alpha is None else l1_alpha
            selector = Lasso(alpha=selector_alpha, max_iter=2000)
        steps.append(("select", SelectFromModel(selector)))
    if pca_variance is not None:
        steps.append(("pca", PCA(n_components=pca_variance, svd_solver="full")))
    steps.append(("clf", estimator))
    return Pipeline(steps)


def default_grid(name: str) -> Dict[str, Iterable[Any]]:
    if name not in MODEL_SPECS:
        raise ValueError(f"Unknown model: {name}")
    return MODEL_SPECS[name].param_grid


def model_task(name: str) -> str:
    if name not in MODEL_SPECS:
        raise ValueError(f"Unknown model: {name}")
    return MODEL_SPECS[name].task
