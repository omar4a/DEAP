import numpy as np

from deap_emotion.labels import build_labels


def test_binary_labels():
    labels = np.array([[4.9, 5.1, 5.0, 7.0]])
    result = build_labels(labels, "valence", "binary", threshold=5.0, bins=(3.0, 5.0, 7.0))
    assert result.tolist() == [0]


def test_quartile_labels():
    labels = np.array([[2.0, 4.0, 6.0, 8.0], [4.0, 5.0, 7.5, 8.5]])
    result = build_labels(labels, "dominance", "quartile", threshold=5.0, bins=(3.0, 5.0, 7.0))
    assert result.tolist() == [2, 3]


def test_three_class_labels():
    labels = np.array([[2.0, 4.0, 6.5, 8.0], [7.0, 6.0, 8.0, 9.0]])
    result = build_labels(labels, "valence", "three_class", threshold=5.0, bins=(3.0, 6.0))
    assert result.tolist() == [0, 2]


def test_subject_median_labels():
    labels = np.array([[2.0, 4.0, 6.0, 8.0], [7.0, 6.0, 8.0, 9.0]])
    result = build_labels(labels, "valence", "subject_median", threshold=5.0, bins=(3.0, 6.0))
    assert result.tolist() == [0, 1]


def test_subject_tertile_labels():
    labels = np.array([[2.0, 4.0, 6.0, 8.0], [7.0, 6.0, 8.0, 9.0]])
    result = build_labels(labels, "valence", "subject_tertile", threshold=5.0, bins=(3.0, 6.0))
    assert result.tolist() == [0, 2]
