from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    r2_score,
    explained_variance_score,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred)
    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": cm.tolist(),
    }

    labels = np.unique(y_true)
    if len(labels) == 2 and cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics.update(
            {
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tpr": float(tp / (tp + fn)) if (tp + fn) else 0.0,
                "tnr": float(tn / (tn + fp)) if (tn + fp) else 0.0,
                "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
                "fnr": float(fn / (fn + tp)) if (fn + tp) else 0.0,
                "ppv": float(tp / (tp + fp)) if (tp + fp) else 0.0,
                "npv": float(tn / (tn + fn)) if (tn + fn) else 0.0,
            }
        )

    if y_score is not None:
        try:
            if y_score.ndim == 1:
                scores = y_score
            elif y_score.ndim == 2 and y_score.shape[1] == 2:
                scores = y_score[:, 1]
            else:
                scores = y_score
            if len(labels) == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
                metrics["pr_auc"] = float(average_precision_score(y_true, scores))
            else:
                metrics["roc_auc"] = float(roc_auc_score(y_true, scores, multi_class="ovr"))
                metrics["pr_auc"] = float(
                    average_precision_score(y_true, scores, average="macro")
                )
        except Exception:
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None

        if y_score.ndim == 2:
            try:
                metrics["log_loss"] = float(log_loss(y_true, y_score))
            except Exception:
                metrics["log_loss"] = None

        if len(labels) == 2 and y_score.ndim <= 2:
            try:
                prob_scores = scores if scores.ndim == 1 else scores[:, 1]
                metrics["brier"] = float(brier_score_loss(y_true, prob_scores))
            except Exception:
                metrics["brier"] = None

    return metrics


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "explained_variance": float(explained_variance_score(y_true, y_pred)),
    }
    try:
        metrics["pearson_r"] = float(pearsonr(y_true, y_pred)[0])
    except Exception:
        metrics["pearson_r"] = None
    try:
        metrics["spearman_r"] = float(spearmanr(y_true, y_pred)[0])
    except Exception:
        metrics["spearman_r"] = None
    return metrics
