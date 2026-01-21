import numpy as np

from deap_emotion.metrics import classification_metrics, regression_metrics


def test_classification_metrics_binary():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_score = np.array([0.1, 0.9, 0.7, 0.8])
    metrics = classification_metrics(y_true, y_pred, y_score)
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert "roc_auc" in metrics


def test_regression_metrics():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    metrics = regression_metrics(y_true, y_pred)
    assert "rmse" in metrics
    assert "r2" in metrics
