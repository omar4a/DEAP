from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, ParameterGrid

from .config import Config
from .data import SubjectData, load_dataset
from .features import extract_features
from .labels import build_labels
from .metrics import classification_metrics, regression_metrics
from .model_registry import build_model, default_grid, model_task
from .preprocess import baseline_correct, preprocess_eeg


@dataclass(frozen=True)
class ExperimentSettings:
    task: str = "classification"
    dimensions: Tuple[str, ...] = ("valence", "arousal")
    outer_repeats: int = 5
    train_ratio: float = 0.8
    inner_folds: int = 4
    scoring: str = "f1_macro"
    use_cache: bool = True
    progress: bool = False
    progress_path: Path | None = None
    include_cross_subject: bool = False
    search_strategy: str = "grid"
    search_trials: int | None = None
    search_random_state: int = 42
    checkpoint_path: Path | None = None
    resume: bool = True


def _cache_key(config: Config, feature_tag: str) -> str:
    feature_set = "none" if config.feature_set is None else "-".join(config.feature_set)
    bands_tag = "-".join(f"{name}{low}-{high}" for name, low, high in config.bands)
    # Include preprocessing params in cache key
    preprocess_tag = (
        f"filt{int(config.apply_filtering)}_bp{config.bandpass_low}-{config.bandpass_high}_"
        f"notch{config.notch_freq}"
    )
    return (
        f"{feature_tag}_dir{config.data_dir.name}_{preprocess_tag}_"
        f"w{config.window_seconds}_s{config.step_seconds}_"
        f"{config.feature_mode}_{feature_set}_bands{bands_tag}_psd{config.psd_min_hz}-"
        f"{config.psd_max_hz}_{config.psd_bin_width}_norm{int(config.per_subject_normalize)}"
        f"_riemann{config.riemann_epsilon}"
    )


def _load_or_compute_features(
    subject: SubjectData,
    config: Config,
    feature_tag: str,
) -> Tuple[np.ndarray, np.ndarray]:
    cache_dir = config.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"features_{subject.subject_id}_{_cache_key(config, feature_tag)}.npz"

    if config.use_cache and cache_path.exists():
        data = np.load(cache_path)
        return data["features"], data["groups"]

    # Apply preprocessing based on Paper 13 methodology
    if config.apply_filtering:
        eeg = preprocess_eeg(
            subject.eeg,
            fs=config.sfreq,
            bandpass_low=config.bandpass_low,
            bandpass_high=config.bandpass_high,
            bandpass_order=config.bandpass_order,
            notch_freq=config.notch_freq,
            baseline_samples=config.baseline_samples(),
        )
    else:
        eeg = baseline_correct(subject.eeg, config.baseline_samples())
    features, groups = extract_features(
        eeg,
        config.sfreq,
        config.band_edges(),
        config.feature_mode,
        window_samples=config.window_samples(),
        step_samples=config.step_samples(),
        feature_set=config.feature_set,
        psd_bands=config.psd_band_edges(),
        asymmetry_pairs=config.asymmetry_indices(),
        riemann_epsilon=config.riemann_epsilon,
    )
    if config.per_subject_normalize:
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std[std < 1e-8] = 1.0
        features = (features - mean) / std

    if config.use_cache:
        np.savez_compressed(cache_path, features=features, groups=groups)

    return features, groups


def _metric_value(metrics: Dict[str, Any], scoring: str) -> float:
    if scoring.startswith("neg_"):
        base_metric = scoring.replace("neg_", "")
        value = metrics.get(base_metric)
        if value is None:
            raise ValueError(f"Scoring metric {base_metric} not available.")
        return -float(value)

    value = metrics.get(scoring)
    if value is None:
        raise ValueError(f"Scoring metric {scoring} not available.")
    return float(value)


def _log_progress(message: str, enabled: bool, progress_path: Path | None = None) -> None:
    if not enabled:
        return
    timestamp = time.strftime("%H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    if progress_path is not None:
        try:
            progress_path.write_text(line + "\n", encoding="utf-8")
        except OSError:
            pass


def _trial_vote_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    y_score: np.ndarray | None = None,
    aggregation: str = "majority",
) -> Dict[str, float]:
    trial_ids = np.unique(groups)
    trial_true = []
    trial_pred = []
    trial_scores: list[np.ndarray | float] = []
    for trial_id in trial_ids:
        idx = groups == trial_id
        if not np.any(idx):
            continue
        trial_true.append(int(y_true[idx][0]))
        if aggregation == "mean_prob" and y_score is not None:
            scores = y_score[idx]
            if scores.ndim == 1:
                mean_score = float(np.mean(scores))
                pred = int(mean_score >= 0.0)
                trial_scores.append(mean_score)
            else:
                mean_score = np.mean(scores, axis=0)
                pred = int(np.argmax(mean_score))
                trial_scores.append(mean_score)
            trial_pred.append(pred)
        else:
            counts = np.bincount(y_pred[idx].astype(int))
            majority = int(np.argmax(counts)) if counts.size else 0
            trial_pred.append(majority)
    if not trial_true:
        return {"trial_accuracy": 0.0, "trial_f1_macro": 0.0}
    trial_true_arr = np.asarray(trial_true)
    trial_pred_arr = np.asarray(trial_pred)
    score_arr: np.ndarray | None = None
    if trial_scores:
        score_arr = np.vstack(trial_scores) if isinstance(trial_scores[0], np.ndarray) else np.asarray(trial_scores)
    metrics = classification_metrics(trial_true_arr, trial_pred_arr, score_arr)
    return {f"trial_{key}": float(value) if isinstance(value, float) else value for key, value in metrics.items()}


def _resolve_grid(
    model_name: str, grid_overrides: Dict[str, Dict[str, Iterable[Any]]] | None
) -> Dict[str, Iterable[Any]]:
    if grid_overrides and model_name in grid_overrides:
        return grid_overrides[model_name]
    return default_grid(model_name)


def _softmax(scores: np.ndarray) -> np.ndarray:
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    denom = np.sum(exp_scores, axis=1, keepdims=True) + 1e-8
    return exp_scores / denom


def _predict_proba(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)
        return probs if probs.ndim == 2 else np.column_stack([1.0 - probs, probs])
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x)
        if scores.ndim == 1:
            probs_pos = 1.0 / (1.0 + np.exp(-scores))
            return np.column_stack([1.0 - probs_pos, probs_pos])
        return _softmax(scores)
    preds = model.predict(x)
    return np.column_stack([1.0 - preds, preds])


def _load_checkpoint(
    path: Path | None,
) -> Dict[Tuple[str, str, str, int, int], Dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    results: Dict[Tuple[str, str, str, int, int], Dict[str, Any]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                key = (
                    row.get("feature_tag", ""),
                    row.get("model", ""),
                    row.get("dimension", ""),
                    int(row.get("subject_id", "0")),
                    int(row.get("split", "0")),
                )
            except ValueError:
                continue
            try:
                metrics = json.loads(row.get("metrics_json", "{}"))
            except json.JSONDecodeError:
                metrics = {}
            try:
                best_params = json.loads(row.get("best_params_json", "{}"))
            except json.JSONDecodeError:
                best_params = {}
            try:
                grid_summary = json.loads(row.get("grid_summary_json", "{}"))
            except json.JSONDecodeError:
                grid_summary = {}
            try:
                train_trials = json.loads(row.get("train_trials_json", "[]"))
            except json.JSONDecodeError:
                train_trials = []
            try:
                test_trials = json.loads(row.get("test_trials_json", "[]"))
            except json.JSONDecodeError:
                test_trials = []
            results[key] = {
                "metrics": metrics,
                "best_params": best_params,
                "grid_summary": grid_summary,
                "train_trials": train_trials,
                "test_trials": test_trials,
            }
    return results


def _append_checkpoint(
    path: Path | None,
    feature_tag: str,
    model_name: str,
    dimension: str,
    subject_id: int,
    split: int,
    train_trials: List[int],
    test_trials: List[int],
    metrics: Dict[str, Any],
    best_params: Dict[str, Any],
    grid_summary: Dict[str, Any],
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "feature_tag",
            "model",
            "dimension",
            "subject_id",
            "split",
            "train_trials_json",
            "test_trials_json",
            "metrics_json",
            "best_params_json",
            "grid_summary_json",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "feature_tag": feature_tag,
                "model": model_name,
                "dimension": dimension,
                "subject_id": subject_id,
                "split": split,
                "train_trials_json": json.dumps(train_trials),
                "test_trials_json": json.dumps(test_trials),
                "metrics_json": json.dumps(metrics),
                "best_params_json": json.dumps(best_params),
                "grid_summary_json": json.dumps(grid_summary),
            }
        )


def _write_checkpoint_summary(
    path: Path | None,
    feature_tag: str,
    model_name: str,
    dimension: str,
    total_subjects: int,
    total_splits: int,
    subject_trial_accs: Dict[int, List[float]],
    subject_window_accs: Dict[int, List[float]],
    last_subject_id: int,
    last_split: int,
    last_metrics: Dict[str, Any],
) -> None:
    if path is None:
        return
    summary_path = path.with_name("summary_live.txt")
    completed = sum(len(values) for values in subject_trial_accs.values())
    subject_means = [
        float(np.mean(values)) for values in subject_trial_accs.values() if values
    ]
    overall_mean = float(np.mean(subject_means)) if subject_means else 0.0
    overall_std = float(np.std(subject_means)) if subject_means else 0.0
    lines = []
    lines.append("Live Checkpoint Summary")
    lines.append(f"feature_tag: {feature_tag}")
    lines.append(f"model: {model_name}")
    lines.append(f"dimension: {dimension}")
    lines.append(
        f"completed_splits: {completed} / {total_subjects * total_splits}"
    )
    lines.append(
        f"last_update: subject {last_subject_id}, split {last_split + 1}/{total_splits}"
    )
    lines.append(
        f"last_trial_accuracy: {last_metrics.get('trial_accuracy', 0.0):.4f}"
    )
    if "trial_balanced_accuracy" in last_metrics:
        lines.append(
            f"last_trial_balanced_accuracy: {last_metrics.get('trial_balanced_accuracy', 0.0):.4f}"
        )
    if "trial_f1_macro" in last_metrics:
        lines.append(
            f"last_trial_f1_macro: {last_metrics.get('trial_f1_macro', 0.0):.4f}"
        )
    if "trial_roc_auc" in last_metrics and last_metrics.get("trial_roc_auc") is not None:
        lines.append(
            f"last_trial_roc_auc: {last_metrics.get('trial_roc_auc', 0.0):.4f}"
        )
    lines.append(
        f"last_window_accuracy: {last_metrics.get('accuracy', 0.0):.4f}"
    )
    lines.append(f"overall_mean_trial_accuracy: {overall_mean:.4f}")
    lines.append(f"overall_std_trial_accuracy: {overall_std:.4f}")
    lines.append("")
    lines.append("Per-subject mean trial accuracy (so far):")
    for subject_id in sorted(subject_trial_accs):
        values = subject_trial_accs[subject_id]
        if not values:
            continue
        mean_trial = float(np.mean(values))
        mean_window = float(np.mean(subject_window_accs.get(subject_id, [])))
        lines.append(
            f"  subject {subject_id}: trial={mean_trial:.4f} window={mean_window:.4f} splits={len(values)}"
        )
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def _predict_with_scores(model, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray | None]:
    y_pred = model.predict(x)
    y_score = None
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(x)
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(x)
    return y_pred, y_score


def _evaluate_fold(
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    test_groups: np.ndarray,
    task: str,
    trial_aggregation: str,
) -> Dict[str, Any]:
    model.fit(x_train, y_train)
    if task == "classification":
        y_pred, y_score = _predict_with_scores(model, x_test)
        metrics = classification_metrics(y_test, y_pred, y_score)
        metrics.update(
            _trial_vote_metrics(
                y_test,
                y_pred,
                test_groups,
                y_score,
                trial_aggregation,
            )
        )
        return metrics
    y_pred = model.predict(x_test)
    return regression_metrics(y_test, y_pred)


def _evaluate_soft_vote(
    base_models: Iterable[str],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    test_groups: np.ndarray,
    task: str,
    grid_overrides: Dict[str, Dict[str, Iterable[Any]]] | None,
    feature_selection: str,
    mi_k: int | None,
    pca_variance: float | None,
    settings: ExperimentSettings,
    config: Config,
    groups_train: np.ndarray,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    if task != "classification":
        raise ValueError("soft_vote supports classification only.")
    prob_list = []
    best_params_map: Dict[str, Any] = {}
    grid_summary_map: Dict[str, Any] = {}
    for base_model in base_models:
        grid = _resolve_grid(base_model, grid_overrides)
        best_params, grid_info = _grid_search(
            base_model,
            grid,
            x_train,
            y_train,
            groups_train,
            task,
            settings.inner_folds,
            settings.scoring,
            pca_variance,
            feature_selection,
            mi_k,
            settings.search_strategy,
            settings.search_trials,
            settings.search_random_state,
            config.l1_c,
            config.l1_alpha,
            config.scaler,
            config.covariance_feature_dim()
            if config.feature_set and "riemann" in config.feature_set
            else None,
            config.riemann_epsilon,
            config.trial_aggregation,
        )
        model = build_model(
            base_model,
            params={k.replace("clf__", ""): v for k, v in best_params.items()},
            pca_variance=pca_variance,
            feature_selection=feature_selection,
            mi_k=mi_k,
            l1_c=config.l1_c,
            l1_alpha=config.l1_alpha,
            scaler=config.scaler,
            riemann_cov_dim=config.covariance_feature_dim()
            if config.feature_set and "riemann" in config.feature_set
            else None,
            riemann_epsilon=config.riemann_epsilon,
        )
        model.fit(x_train, y_train)
        prob_list.append(_predict_proba(model, x_test))
        best_params_map[base_model] = best_params
        grid_summary_map[base_model] = grid_info
    avg_probs = np.mean(prob_list, axis=0)
    y_pred = np.argmax(avg_probs, axis=1)
    metrics = classification_metrics(y_test, y_pred, avg_probs)
    metrics.update(
        _trial_vote_metrics(
            y_test,
            y_pred,
            test_groups,
            avg_probs,
            config.trial_aggregation,
        )
    )
    return metrics, best_params_map, grid_summary_map


def _grid_search(
    model_name: str,
    grid: Dict[str, Iterable[Any]],
    x_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    task: str,
    inner_folds: int,
    scoring: str,
    pca_variance: float | None,
    feature_selection: str,
    mi_k: int | None,
    search_strategy: str,
    search_trials: int | None,
    search_random_state: int,
    l1_c: float | None,
    l1_alpha: float | None,
    scaler: str,
    riemann_cov_dim: int | None,
    riemann_epsilon: float,
    trial_aggregation: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    best_params = {}
    best_score = -np.inf
    scores: Dict[str, Any] = {}

    unique_groups = len(np.unique(groups))
    if unique_groups < 2 or not grid:
        return best_params, {"scores": scores, "best_score": best_score}

    splitter = GroupKFold(n_splits=min(inner_folds, unique_groups))
    rng = np.random.RandomState(search_random_state)

    def _sample_value(space: Any) -> Any:
        if isinstance(space, dict) and "dist" in space:
            dist = space.get("dist")
            if dist == "loguniform":
                low = float(space.get("low", 1e-4))
                high = float(space.get("high", 1.0))
                return float(np.exp(rng.uniform(np.log(low), np.log(high))))
            if dist == "uniform":
                low = float(space.get("low", 0.0))
                high = float(space.get("high", 1.0))
                return float(rng.uniform(low, high))
            if dist == "int":
                low = int(space.get("low", 0))
                high = int(space.get("high", low))
                return int(rng.randint(low, high + 1))
            if dist == "choice":
                values = space.get("values", [])
                return values[int(rng.randint(0, len(values)))] if values else None
        if isinstance(space, (list, tuple, np.ndarray)):
            if not space:
                return None
            return space[int(rng.randint(0, len(space)))]
        return space

    def _random_param_sets(target: int) -> List[Dict[str, Any]]:
        param_sets: List[Dict[str, Any]] = []
        seen = set()
        attempts = 0
        keys = list(grid.keys())
        while len(param_sets) < target and attempts < target * 10:
            params = {key: _sample_value(grid[key]) for key in keys}
            key = json.dumps(params, sort_keys=True)
            if key in seen:
                attempts += 1
                continue
            seen.add(key)
            param_sets.append(params)
            attempts += 1
        return param_sets

    has_distributions = any(isinstance(value, dict) and "dist" in value for value in grid.values())
    if search_strategy == "random" and search_trials is not None:
        if has_distributions:
            param_grid = _random_param_sets(search_trials)
        else:
            param_grid = list(ParameterGrid(grid))
            if len(param_grid) > search_trials:
                indices = rng.choice(len(param_grid), size=search_trials, replace=False)
                param_grid = [param_grid[idx] for idx in indices]
    else:
        param_grid = list(ParameterGrid(grid))
    for param_set in param_grid:
        fold_scores = []
        for train_idx, val_idx in splitter.split(x_train, y_train, groups):
            if task == "classification" and len(np.unique(y_train[train_idx])) < 2:
                continue
            model = build_model(
                model_name,
                params={k.replace("clf__", ""): v for k, v in param_set.items()},
                pca_variance=pca_variance,
                feature_selection=feature_selection,
                mi_k=mi_k,
                l1_c=l1_c,
                l1_alpha=l1_alpha,
                scaler=scaler,
                riemann_cov_dim=riemann_cov_dim,
                riemann_epsilon=riemann_epsilon,
            )
            metrics = _evaluate_fold(
                model,
                x_train[train_idx],
                y_train[train_idx],
                x_train[val_idx],
                y_train[val_idx],
                groups[val_idx],
                task,
                trial_aggregation,
            )
            fold_scores.append(_metric_value(metrics, scoring))
        avg_score = float(np.mean(fold_scores)) if fold_scores else float("-inf")
        scores[str(param_set)] = avg_score
        if avg_score > best_score:
            best_score = avg_score
            best_params = param_set

    return best_params, {"scores": scores, "best_score": best_score}


def _prepare_subject_features(
    subject: SubjectData,
    config: Config,
    feature_tag: str,
) -> Tuple[np.ndarray, np.ndarray]:
    features, trial_groups = _load_or_compute_features(subject, config, feature_tag)
    return features, trial_groups


def _repeat_trial_splits(
    trial_groups: np.ndarray,
    outer_repeats: int,
    train_ratio: float,
    rng: np.random.RandomState,
) -> Iterable[Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    trial_ids = np.unique(trial_groups)
    n_train = max(1, int(round(len(trial_ids) * train_ratio)))
    for repeat in range(outer_repeats):
        perm = rng.permutation(trial_ids)
        train_trials = perm[:n_train]
        test_trials = perm[n_train:]
        train_idx = np.where(np.isin(trial_groups, train_trials))[0]
        test_idx = np.where(np.isin(trial_groups, test_trials))[0]
        yield repeat, train_trials, test_trials, train_idx, test_idx


def _subject_dependent(
    dataset: List[SubjectData],
    config: Config,
    settings: ExperimentSettings,
    model_name: str,
    grid: Dict[str, Iterable[Any]],
    feature_tag: str,
    grid_overrides: Dict[str, Dict[str, Iterable[Any]]] | None = None,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {"mode": "subject_dependent", "model": model_name, "dimensions": {}}
    checkpoint_rows = _load_checkpoint(settings.checkpoint_path) if settings.resume else {}
    total_subjects = len(dataset)
    total_splits = settings.outer_repeats

    for dimension in settings.dimensions:
        dimension_results = []
        subject_trial_accs: Dict[int, List[float]] = {}
        subject_trial_bal_accs: Dict[int, List[float]] = {}
        subject_trial_f1s: Dict[int, List[float]] = {}
        subject_trial_roc: Dict[int, List[float]] = {}
        subject_window_accs: Dict[int, List[float]] = {}
        for subject in dataset:
            features, trial_groups = _prepare_subject_features(subject, config, feature_tag)
            base_labels = None
            if config.label_mode != "subject_mean":
                base_labels = build_labels(
                    subject.labels,
                    dimension=dimension,
                    mode=config.label_mode,
                    threshold=config.label_threshold,
                    bins=config.label_bins,
                )

            rng = np.random.RandomState(config.random_state + subject.subject_id)
            for repeat, train_trials, test_trials, train_idx, test_idx in _repeat_trial_splits(
                trial_groups, settings.outer_repeats, settings.train_ratio, rng
            ):
                key = (feature_tag, model_name, dimension, subject.subject_id, repeat)
                if key in checkpoint_rows:
                    row = checkpoint_rows[key]
                    metrics = row["metrics"]
                    dimension_results.append(
                        {
                            "subject_id": subject.subject_id,
                            "split": repeat,
                            "train_trials": row.get("train_trials", []),
                            "test_trials": row.get("test_trials", []),
                            "metrics": metrics,
                            "best_params": row.get("best_params", {}),
                            "grid_summary": row.get("grid_summary", {}),
                        }
                    )
                    subject_trial_accs.setdefault(subject.subject_id, []).append(
                        float(metrics.get("trial_accuracy", 0.0))
                    )
                    subject_trial_bal_accs.setdefault(subject.subject_id, []).append(
                        float(metrics.get("trial_balanced_accuracy", 0.0))
                    )
                    subject_trial_f1s.setdefault(subject.subject_id, []).append(
                        float(metrics.get("trial_f1_macro", 0.0))
                    )
                    if metrics.get("trial_roc_auc") is not None:
                        subject_trial_roc.setdefault(subject.subject_id, []).append(
                            float(metrics.get("trial_roc_auc", 0.0))
                        )
                    subject_window_accs.setdefault(subject.subject_id, []).append(
                        float(metrics.get("accuracy", 0.0))
                    )
                    _log_progress(
                        f"{feature_tag} model={model_name} dim={dimension} subject={subject.subject_id} "
                        f"split {repeat + 1}/{settings.outer_repeats} skipped (checkpoint)",
                        settings.progress,
                        settings.progress_path,
                    )
                    continue
                _log_progress(
                    f"{feature_tag} model={model_name} dim={dimension} subject={subject.subject_id} "
                    f"split {repeat + 1}/{settings.outer_repeats}",
                    settings.progress,
                    settings.progress_path,
                )
                if base_labels is None:
                    trial_labels = build_labels(
                        subject.labels,
                        dimension=dimension,
                        mode=config.label_mode,
                        threshold=config.label_threshold,
                        bins=config.label_bins,
                        train_trial_ids=train_trials,
                    )
                else:
                    trial_labels = base_labels
                y = trial_labels[trial_groups]

                if len(np.unique(y[train_idx])) < 2 and settings.task == "classification":
                    continue

                if config.feature_selection == "pca":
                    pca_variance = config.pca_variance
                    feature_selection = "none"
                    mi_k = None
                elif config.feature_selection == "mi":
                    pca_variance = None
                    feature_selection = "mi"
                    mi_k = config.mi_k
                elif config.feature_selection == "l1":
                    pca_variance = None
                    feature_selection = "l1"
                    mi_k = None
                else:
                    pca_variance = None
                    feature_selection = "none"
                    mi_k = None

                if model_name == "soft_vote":
                    metrics, best_params, grid_info = _evaluate_soft_vote(
                        base_models=("rbf_svm", "logreg", "hgb"),
                        x_train=features[train_idx],
                        y_train=y[train_idx],
                        x_test=features[test_idx],
                        y_test=y[test_idx],
                        test_groups=trial_groups[test_idx],
                        task=settings.task,
                        grid_overrides=grid_overrides,
                        feature_selection=feature_selection,
                        mi_k=mi_k,
                        pca_variance=pca_variance,
                        settings=settings,
                        config=config,
                        groups_train=trial_groups[train_idx],
                    )
                else:
                    best_params, grid_info = _grid_search(
                        model_name,
                        grid,
                        features[train_idx],
                        y[train_idx],
                        trial_groups[train_idx],
                        settings.task,
                        settings.inner_folds,
                        settings.scoring,
                        pca_variance,
                        feature_selection,
                        mi_k,
                        settings.search_strategy,
                        settings.search_trials,
                        settings.search_random_state,
                        config.l1_c,
                        config.l1_alpha,
                        config.scaler,
                        config.covariance_feature_dim()
                        if config.feature_set and "riemann" in config.feature_set
                        else None,
                        config.riemann_epsilon,
                        config.trial_aggregation,
                    )
                    model = build_model(
                        model_name,
                        params={k.replace("clf__", ""): v for k, v in best_params.items()},
                        pca_variance=pca_variance,
                        feature_selection=feature_selection,
                        mi_k=mi_k,
                        l1_c=config.l1_c,
                        l1_alpha=config.l1_alpha,
                        scaler=config.scaler,
                        riemann_cov_dim=config.covariance_feature_dim()
                        if config.feature_set and "riemann" in config.feature_set
                        else None,
                        riemann_epsilon=config.riemann_epsilon,
                    )
                    metrics = _evaluate_fold(
                        model,
                        features[train_idx],
                        y[train_idx],
                        features[test_idx],
                        y[test_idx],
                        trial_groups[test_idx],
                        settings.task,
                        config.trial_aggregation,
                    )
                subject_trial_accs.setdefault(subject.subject_id, []).append(
                    float(metrics.get("trial_accuracy", 0.0))
                )
                subject_trial_bal_accs.setdefault(subject.subject_id, []).append(
                    float(metrics.get("trial_balanced_accuracy", 0.0))
                )
                subject_trial_f1s.setdefault(subject.subject_id, []).append(
                    float(metrics.get("trial_f1_macro", 0.0))
                )
                if metrics.get("trial_roc_auc") is not None:
                    subject_trial_roc.setdefault(subject.subject_id, []).append(
                        float(metrics.get("trial_roc_auc", 0.0))
                    )
                subject_window_accs.setdefault(subject.subject_id, []).append(
                    float(metrics.get("accuracy", 0.0))
                )
                dimension_results.append(
                    {
                        "subject_id": subject.subject_id,
                        "split": repeat,
                        "train_trials": train_trials.tolist(),
                        "test_trials": test_trials.tolist(),
                        "metrics": metrics,
                        "best_params": best_params,
                        "grid_summary": grid_info,
                    }
                )
                _append_checkpoint(
                    settings.checkpoint_path,
                    feature_tag,
                    model_name,
                    dimension,
                    subject.subject_id,
                    repeat,
                    train_trials.tolist(),
                    test_trials.tolist(),
                    metrics,
                    best_params,
                    grid_info,
                )
                _write_checkpoint_summary(
                    settings.checkpoint_path,
                    feature_tag,
                    model_name,
                    dimension,
                    total_subjects,
                    total_splits,
                    subject_trial_accs,
                    subject_window_accs,
                    subject.subject_id,
                    repeat,
                    metrics,
                )
                _log_progress(
                    f"{feature_tag} model={model_name} dim={dimension} subject={subject.subject_id} "
                    f"split {repeat + 1}/{settings.outer_repeats} "
                    f"trial_acc={metrics.get('trial_accuracy', 0.0):.3f}",
                    settings.progress,
                    settings.progress_path,
                )

        results["dimensions"][dimension] = {
            "folds": dimension_results,
            "per_subject_accuracy": {
                str(subject_id): {
                    "mean_trial_accuracy": float(np.mean(trial_accs)) if trial_accs else 0.0,
                    "mean_trial_balanced_accuracy": float(
                        np.mean(subject_trial_bal_accs.get(subject_id, []))
                    )
                    if subject_trial_bal_accs.get(subject_id)
                    else 0.0,
                    "mean_trial_f1_macro": float(
                        np.mean(subject_trial_f1s.get(subject_id, []))
                    )
                    if subject_trial_f1s.get(subject_id)
                    else 0.0,
                    "mean_trial_roc_auc": float(
                        np.mean(subject_trial_roc.get(subject_id, []))
                    )
                    if subject_trial_roc.get(subject_id)
                    else None,
                    "mean_window_accuracy": float(np.mean(subject_window_accs.get(subject_id, [])))
                    if subject_window_accs.get(subject_id)
                    else 0.0,
                    "splits": len(trial_accs),
                }
                for subject_id, trial_accs in subject_trial_accs.items()
            },
        }
        subject_means = [
            entry["mean_trial_accuracy"]
            for entry in results["dimensions"][dimension]["per_subject_accuracy"].values()
        ]
        results["dimensions"][dimension]["overall_mean_trial_accuracy"] = (
            float(np.mean(subject_means)) if subject_means else 0.0
        )
        results["dimensions"][dimension]["overall_std_trial_accuracy"] = (
            float(np.std(subject_means)) if subject_means else 0.0
        )
        bal_means = [
            entry["mean_trial_balanced_accuracy"]
            for entry in results["dimensions"][dimension]["per_subject_accuracy"].values()
        ]
        results["dimensions"][dimension]["overall_mean_trial_balanced_accuracy"] = (
            float(np.mean(bal_means)) if bal_means else 0.0
        )
        results["dimensions"][dimension]["overall_std_trial_balanced_accuracy"] = (
            float(np.std(bal_means)) if bal_means else 0.0
        )
        f1_means = [
            entry["mean_trial_f1_macro"]
            for entry in results["dimensions"][dimension]["per_subject_accuracy"].values()
        ]
        results["dimensions"][dimension]["overall_mean_trial_f1_macro"] = (
            float(np.mean(f1_means)) if f1_means else 0.0
        )
        results["dimensions"][dimension]["overall_std_trial_f1_macro"] = (
            float(np.std(f1_means)) if f1_means else 0.0
        )
        roc_means = [
            entry["mean_trial_roc_auc"]
            for entry in results["dimensions"][dimension]["per_subject_accuracy"].values()
            if entry.get("mean_trial_roc_auc") is not None
        ]
        results["dimensions"][dimension]["overall_mean_trial_roc_auc"] = (
            float(np.mean(roc_means)) if roc_means else None
        )
        results["dimensions"][dimension]["overall_std_trial_roc_auc"] = (
            float(np.std(roc_means)) if roc_means else None
        )

    return results


def _cross_subject(
    dataset: List[SubjectData],
    config: Config,
    settings: ExperimentSettings,
    model_name: str,
    grid: Dict[str, Iterable[Any]],
    feature_tag: str,
) -> Dict[str, Any]:
    all_features = []
    all_subject_groups = []
    label_map: Dict[str, List[np.ndarray]] = {d: [] for d in settings.dimensions}

    for subject in dataset:
        features, trial_groups = _prepare_subject_features(subject, config, feature_tag)
        all_features.append(features)
        all_subject_groups.append(np.full(len(features), subject.subject_id, dtype=int))
        for dimension in settings.dimensions:
            labels = build_labels(
                subject.labels,
                dimension=dimension,
                mode=config.label_mode,
                threshold=config.label_threshold,
                bins=config.label_bins,
            )
            label_map[dimension].append(labels[trial_groups])

    features = np.concatenate(all_features, axis=0)
    subject_groups = np.concatenate(all_subject_groups, axis=0)
    splitter = GroupKFold(n_splits=min(settings.outer_repeats, len(np.unique(subject_groups))))

    results: Dict[str, Any] = {"mode": "cross_subject", "model": model_name, "dimensions": {}}
    for dimension in settings.dimensions:
        y = np.concatenate(label_map[dimension], axis=0)
        folds = []
        for fold_idx, (train_idx, test_idx) in enumerate(
            splitter.split(features, y, subject_groups)
        ):
            if len(np.unique(y[train_idx])) < 2 and settings.task == "classification":
                continue
            if config.feature_selection == "pca":
                pca_variance = config.pca_variance
                feature_selection = "none"
                mi_k = None
            elif config.feature_selection == "mi":
                pca_variance = None
                feature_selection = "mi"
                mi_k = config.mi_k
            elif config.feature_selection == "l1":
                pca_variance = None
                feature_selection = "l1"
                mi_k = None
            else:
                pca_variance = None
                feature_selection = "none"
                mi_k = None
            best_params, grid_info = _grid_search(
                model_name,
                grid,
                features[train_idx],
                y[train_idx],
                subject_groups[train_idx],
                settings.task,
                settings.inner_folds,
                settings.scoring,
                pca_variance,
                feature_selection,
                mi_k,
                settings.search_strategy,
                settings.search_trials,
                settings.search_random_state,
                config.l1_c,
                config.l1_alpha,
                config.scaler,
                config.covariance_feature_dim()
                if config.feature_set and "riemann" in config.feature_set
                else None,
                config.riemann_epsilon,
                config.trial_aggregation,
            )
            model = build_model(
                model_name,
                params={k.replace("clf__", ""): v for k, v in best_params.items()},
                pca_variance=pca_variance,
                feature_selection=feature_selection,
                mi_k=mi_k,
                l1_c=config.l1_c,
                l1_alpha=config.l1_alpha,
                scaler=config.scaler,
                riemann_cov_dim=config.covariance_feature_dim()
                if config.feature_set and "riemann" in config.feature_set
                else None,
                riemann_epsilon=config.riemann_epsilon,
            )
            metrics = _evaluate_fold(
                model,
                features[train_idx],
                y[train_idx],
                features[test_idx],
                y[test_idx],
                subject_groups[test_idx],
                settings.task,
                config.trial_aggregation,
            )
            test_subjects = sorted(set(subject_groups[test_idx].tolist()))
            folds.append(
                {
                    "fold": fold_idx,
                    "test_subjects": test_subjects,
                    "metrics": metrics,
                    "best_params": best_params,
                    "grid_summary": grid_info,
                }
            )
        results["dimensions"][dimension] = {"folds": folds}

    return results


def run_experiments(
    config: Config,
    settings: ExperimentSettings,
    model_names: Iterable[str],
    grid_overrides: Dict[str, Dict[str, Iterable[Any]]] | None = None,
    subjects: Iterable[int] | None = None,
    trial_ids: Iterable[int] | None = None,
    feature_tag: str = "default",
) -> Dict[str, Any]:
    dataset = load_dataset(config.data_dir, config.eeg_channels, subjects, trial_ids)

    def serialize(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, tuple):
            return [serialize(item) for item in obj]
        if isinstance(obj, dict):
            return {key: serialize(value) for key, value in obj.items()}
        return obj

    results: Dict[str, Any] = {
        "config": serialize(config.__dict__),
        "settings": serialize(settings.__dict__),
        "models": {},
    }

    for model_name in model_names:
        if model_name == "soft_vote":
            if settings.task != "classification":
                continue
            grid = {}
        else:
            if model_task(model_name) != settings.task:
                continue
            grid = (
                grid_overrides.get(model_name, default_grid(model_name))
                if grid_overrides
                else default_grid(model_name)
            )
        model_results = {
            "subject_dependent": _subject_dependent(
                dataset,
                config,
                settings,
                model_name,
                grid,
                feature_tag,
                grid_overrides=grid_overrides,
            ),
        }
        if settings.include_cross_subject:
            model_results["cross_subject"] = _cross_subject(
                dataset, config, settings, model_name, grid, feature_tag
            )
        results["models"][model_name] = model_results

    return results


def write_report(report: Dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "experiment_report.json"
    path.write_text(json.dumps(report, indent=2))
    return path


def write_summary_csv(report: Dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.csv"

    rows = []
    for model_name, model_data in report.get("models", {}).items():
        for mode in ("subject_dependent", "cross_subject"):
            mode_data = model_data.get(mode, {})
            for dimension, payload in mode_data.get("dimensions", {}).items():
                for fold in payload.get("folds", []):
                    row = {
                        "model": model_name,
                        "mode": mode,
                        "dimension": dimension,
                        "fold": fold.get("fold"),
                        "best_params": json.dumps(fold.get("best_params", {})),
                    }
                    row.update(fold.get("metrics", {}))
                    rows.append(row)

    if not rows:
        return path

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_summary_txt(report: Dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.txt"

    config = report.get("config", {})
    settings = report.get("settings", {})

    lines = []
    lines.append("DEAP Experiment Summary")
    lines.append("")
    lines.append("Config")
    lines.append(f"  data_dir: {config.get('data_dir')}")
    lines.append(f"  eeg_channels: {len(config.get('eeg_channels', []))}")
    lines.append(f"  bands: {config.get('bands')}")
    lines.append(f"  window_seconds: {config.get('window_seconds')}")
    lines.append(f"  step_seconds: {config.get('step_seconds')}")
    lines.append(f"  feature_mode: {config.get('feature_mode')}")
    lines.append(f"  feature_set: {config.get('feature_set')}")
    lines.append(f"  feature_selection: {config.get('feature_selection')}")
    lines.append(f"  mi_k: {config.get('mi_k')}")
    lines.append(f"  l1_c: {config.get('l1_c')}")
    lines.append(f"  l1_alpha: {config.get('l1_alpha')}")
    lines.append(f"  scaler: {config.get('scaler')}")
    lines.append(f"  trial_aggregation: {config.get('trial_aggregation')}")
    lines.append(f"  riemann_epsilon: {config.get('riemann_epsilon')}")
    lines.append(f"  pca_variance: {config.get('pca_variance')}")
    lines.append(f"  label_mode: {config.get('label_mode')}")
    lines.append(f"  label_threshold: {config.get('label_threshold')}")
    lines.append(f"  label_bins: {config.get('label_bins')}")
    lines.append("")
    lines.append("Splits")
    lines.append(f"  outer_repeats: {settings.get('outer_repeats')}")
    lines.append(f"  train_ratio: {settings.get('train_ratio')}")
    lines.append(f"  inner_folds: {settings.get('inner_folds')}")
    lines.append(f"  scoring: {settings.get('scoring')}")
    lines.append(f"  search_strategy: {settings.get('search_strategy')}")
    lines.append(f"  search_trials: {settings.get('search_trials')}")
    lines.append("")

    for model_name, model_data in report.get("models", {}).items():
        lines.append(f"Model: {model_name}")
        subject_data = model_data.get("subject_dependent", {})
        for dimension, payload in subject_data.get("dimensions", {}).items():
            lines.append(f"  Dimension: {dimension}")
            mean_acc = payload.get("overall_mean_trial_accuracy")
            std_acc = payload.get("overall_std_trial_accuracy")
            mean_bal = payload.get("overall_mean_trial_balanced_accuracy")
            std_bal = payload.get("overall_std_trial_balanced_accuracy")
            mean_f1 = payload.get("overall_mean_trial_f1_macro")
            std_f1 = payload.get("overall_std_trial_f1_macro")
            mean_roc = payload.get("overall_mean_trial_roc_auc")
            std_roc = payload.get("overall_std_trial_roc_auc")
            if mean_acc is not None:
                lines.append(f"    overall_mean_trial_accuracy: {mean_acc:.4f}")
            if std_acc is not None:
                lines.append(f"    overall_std_trial_accuracy: {std_acc:.4f}")
            if mean_bal is not None:
                lines.append(f"    overall_mean_trial_balanced_accuracy: {mean_bal:.4f}")
            if std_bal is not None:
                lines.append(f"    overall_std_trial_balanced_accuracy: {std_bal:.4f}")
            if mean_f1 is not None:
                lines.append(f"    overall_mean_trial_f1_macro: {mean_f1:.4f}")
            if std_f1 is not None:
                lines.append(f"    overall_std_trial_f1_macro: {std_f1:.4f}")
            if mean_roc is not None:
                lines.append(f"    overall_mean_trial_roc_auc: {mean_roc:.4f}")
            if std_roc is not None:
                lines.append(f"    overall_std_trial_roc_auc: {std_roc:.4f}")
            folds = payload.get("folds", [])
            if folds:
                lines.append(f"    folds: {len(folds)}")
                param_counts: Dict[str, int] = {}
                for fold in folds:
                    params = fold.get("best_params", {})
                    params_str = json.dumps(params, sort_keys=True)
                    param_counts[params_str] = param_counts.get(params_str, 0) + 1
                if param_counts:
                    lines.append("    top_best_params:")
                    for params_str, count in sorted(
                        param_counts.items(), key=lambda item: item[1], reverse=True
                    )[:5]:
                        lines.append(f"      {params_str} (n={count})")
            per_subject = payload.get("per_subject_accuracy", {})
            if per_subject:
                lines.append("    per_subject_mean_trial_accuracy:")
                for subject_id in sorted(per_subject, key=lambda x: int(x)):
                    entry = per_subject[subject_id]
                    mean_trial = entry.get("mean_trial_accuracy", 0.0)
                    splits = entry.get("splits", 0)
                    lines.append(f"      subject {subject_id}: {mean_trial:.4f} (splits={splits})")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path
