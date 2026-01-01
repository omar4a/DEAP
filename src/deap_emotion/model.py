from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def build_classifier(model_name: str, random_state: int) -> Pipeline:
    if model_name == "logreg":
        clf = LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            random_state=random_state,
            max_iter=2000,
        )
    elif model_name == "linear_svm":
        clf = LinearSVC(class_weight="balanced", random_state=random_state)
    else:
        raise ValueError(f"Unknown classifier: {model_name}")

    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
