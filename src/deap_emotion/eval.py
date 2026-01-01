from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .config import Config
from .data import SubjectData, load_dataset
from .features import extract_features
from .labels import build_labels
from .model import build_classifier
from .preprocess import baseline_correct
from .splits import cross_subject_split, subject_dependent_split


@dataclass
class MetricSummary:
    accuracy_mean: float
    accuracy_std: float
    f1_mean: float
    f1_std: float


def _cache_key(config: Config, subject_id: int) -> str:
    payload = {
        "subject_id": subject_id,
        "eeg_channels": config.eeg_channels,
        "sfreq": config.sfreq,
        "baseline": config.baseline_seconds,
        "bands": config.bands,
        "feature_mode": config.feature_mode,
        "window": config.window_seconds,
        "step": config.step_seconds,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest


def _load_or_compute_features(
    subject: SubjectData, config: Config
) -> tuple[np.ndarray, np.ndarray]:
    cache_dir = config.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"features_{_cache_key(config, subject.subject_id)}.npz"

    if config.use_cache and cache_path.exists():
        data = np.load(cache_path)
        return data["features"], data["groups"]

    eeg = baseline_correct(subject.eeg, config.baseline_samples())
    features, groups = extract_features(
        eeg,
        config.sfreq,
        config.band_edges(),
        config.feature_mode,
        window_samples=config.window_samples(),
        step_samples=config.step_samples(),
    )

    if config.use_cache:
        np.savez_compressed(cache_path, features=features, groups=groups)

    return features, groups


def _evaluate(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    split_iter: Iterable[tuple[np.ndarray, np.ndarray]],
    classifier: str,
    random_state: int,
) -> MetricSummary:
    accuracies: List[float] = []
    f1s: List[float] = []

    for train_idx, test_idx in split_iter:
        if len(np.unique(labels[train_idx])) < 2:
            continue
        model = build_classifier(classifier, random_state)
        model.fit(features[train_idx], labels[train_idx])
        preds = model.predict(features[test_idx])
        accuracies.append(accuracy_score(labels[test_idx], preds))
        f1s.append(f1_score(labels[test_idx], preds, average="macro"))

    if not accuracies:
        raise ValueError("No valid folds with at least two classes.")

    return MetricSummary(
        accuracy_mean=float(np.mean(accuracies)),
        accuracy_std=float(np.std(accuracies)),
        f1_mean=float(np.mean(f1s)),
        f1_std=float(np.std(f1s)),
    )


def run_subject_dependent(
    config: Config,
    subjects: Iterable[int] | None = None,
    trial_ids: Iterable[int] | None = None,
) -> Dict[str, MetricSummary]:
    dataset = load_dataset(config.data_dir, config.eeg_channels, subjects, trial_ids)
    dimension_metrics: Dict[str, List[MetricSummary]] = {
        "valence": [],
        "arousal": [],
        "dominance": [],
    }

    for subject in dataset:
        features, groups = _load_or_compute_features(subject, config)
        split_list = list(subject_dependent_split(groups, config.subject_folds))
        for dimension in dimension_metrics.keys():
            trial_labels = build_labels(
                subject.labels,
                dimension=dimension,
                mode=config.label_mode,
                threshold=config.label_threshold,
                bins=config.label_bins,
            )
            sample_labels = trial_labels[groups]
            try:
                metrics = _evaluate(
                    features,
                    sample_labels,
                    groups,
                    split_list,
                    classifier=config.classifier,
                    random_state=config.random_state,
                )
            except ValueError:
                continue
            dimension_metrics[dimension].append(metrics)

    results = {}
    for dimension, metrics in dimension_metrics.items():
        if not metrics:
            raise ValueError(f"No valid folds for {dimension}.")
        accuracy_mean = float(np.mean([m.accuracy_mean for m in metrics]))
        accuracy_std = float(np.mean([m.accuracy_std for m in metrics]))
        f1_mean = float(np.mean([m.f1_mean for m in metrics]))
        f1_std = float(np.mean([m.f1_std for m in metrics]))
        results[dimension] = MetricSummary(
            accuracy_mean=accuracy_mean,
            accuracy_std=accuracy_std,
            f1_mean=f1_mean,
            f1_std=f1_std,
        )
    return results


def run_cross_subject(
    config: Config,
    subjects: Iterable[int] | None = None,
    trial_ids: Iterable[int] | None = None,
) -> Dict[str, MetricSummary]:
    dataset = load_dataset(config.data_dir, config.eeg_channels, subjects, trial_ids)

    all_features = []
    all_groups = []
    all_labels = {key: [] for key in ("valence", "arousal", "dominance")}

    for subject in dataset:
        features, groups = _load_or_compute_features(subject, config)
        all_features.append(features)
        all_groups.append(np.full(len(features), subject.subject_id, dtype=int))
        for dimension in all_labels.keys():
            trial_labels = build_labels(
                subject.labels,
                dimension=dimension,
                mode=config.label_mode,
                threshold=config.label_threshold,
                bins=config.label_bins,
            )
            all_labels[dimension].append(trial_labels[groups])

    features = np.concatenate(all_features, axis=0)
    subject_groups = np.concatenate(all_groups, axis=0)
    split_list = list(cross_subject_split(subject_groups))

    results: Dict[str, MetricSummary] = {}
    for dimension, label_list in all_labels.items():
        labels = np.concatenate(label_list, axis=0)
        metrics = _evaluate(
            features,
            labels,
            subject_groups,
            split_list,
            classifier=config.classifier,
            random_state=config.random_state,
        )
        results[dimension] = metrics
    return results


def metrics_to_dict(metrics: Dict[str, MetricSummary]) -> Dict[str, dict]:
    return {key: asdict(value) for key, value in metrics.items()}
