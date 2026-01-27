"""DEAP Leaky Max-Accuracy Experiment (XGBoost).

WARNING: This experiment intentionally uses DATA LEAKAGE.
All windows are pooled across subjects/videos and split randomly.
Use only for inflated upper-bound numbers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from deap_emotion.config import Config
from deap_emotion.data import load_dataset
from deap_emotion.features import extract_features
from deap_emotion.labels import build_labels
from deap_emotion.preprocess import baseline_correct, apply_bandpass, apply_notch

try:
    from xgboost import XGBClassifier
except Exception as exc:
    raise RuntimeError("xgboost is required for this experiment.") from exc

try:
    from scipy.stats import randint, uniform, loguniform
except Exception:
    randint = None
    uniform = None
    loguniform = None


def _split_feature_blocks(
    features: np.ndarray, n_channels: int, n_bands: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split concatenated [bandpower_log | de] into 3D tensors."""
    n_band_features = n_channels * n_bands
    bandpower_log = features[:, :n_band_features].reshape(-1, n_channels, n_bands)
    de_features = features[:, n_band_features:].reshape(-1, n_channels, n_bands)
    return bandpower_log, de_features


def _asymmetry_diff(
    features: np.ndarray, asym_pairs: tuple[tuple[int, int], ...]
) -> np.ndarray:
    if not asym_pairs:
        return np.empty((features.shape[0], 0), dtype=float)
    left_idx = [p[0] for p in asym_pairs]
    right_idx = [p[1] for p in asym_pairs]
    diff = features[:, left_idx, :] - features[:, right_idx, :]
    return diff.reshape(features.shape[0], -1)


def _feature_cache_key(config: Config) -> str:
    payload = json.dumps(
        {
            "sfreq": config.sfreq,
            "baseline_seconds": config.baseline_seconds,
            "bandpass_low": config.bandpass_low,
            "bandpass_high": config.bandpass_high,
            "bandpass_order": config.bandpass_order,
            "notch_freq": config.notch_freq,
            "window_seconds": config.window_seconds,
            "step_seconds": config.step_seconds,
            "bands": config.bands,
            "feature_mode": "bandpower_log",
            "feature_set": ("bandpower", "de"),
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def build_feature_matrix(
    eeg: np.ndarray, config: Config
) -> tuple[np.ndarray, np.ndarray]:
    """Extract DE + log bandpower features and add asymmetry for both."""
    # Preprocessing
    if config.apply_filtering:
        eeg = apply_bandpass(
            eeg,
            config.bandpass_low,
            config.bandpass_high,
            config.sfreq,
            config.bandpass_order,
        )
        if config.notch_freq:
            eeg = apply_notch(eeg, config.notch_freq, config.sfreq)

    # Baseline correction, then drop baseline segment
    baseline_samples = config.baseline_samples()
    eeg = baseline_correct(eeg, baseline_samples)
    if baseline_samples > 0 and eeg.shape[-1] > baseline_samples:
        eeg = eeg[:, :, baseline_samples:]

    # Extract bandpower (log) + DE per window
    features, trial_groups = extract_features(
        eeg,
        config.sfreq,
        config.band_edges(),
        "bandpower_log",
        window_samples=config.window_samples(),
        step_samples=config.step_samples(),
        feature_set=("bandpower", "de"),
    )

    n_channels = len(config.eeg_channels)
    n_bands = len(config.bands)
    bandpower_log, de_features = _split_feature_blocks(features, n_channels, n_bands)

    # Asymmetry (diff only) for DE and log bandpower
    asym_pairs = config.asymmetry_indices()
    asym_de = _asymmetry_diff(de_features, asym_pairs)
    asym_bp = _asymmetry_diff(bandpower_log, asym_pairs)

    X = np.concatenate(
        [
            bandpower_log.reshape(features.shape[0], -1),
            de_features.reshape(features.shape[0], -1),
            asym_de,
            asym_bp,
        ],
        axis=1,
    )
    return X, trial_groups


def load_or_compute_subject_features(
    subject_id: int,
    eeg: np.ndarray,
    config: Config,
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _feature_cache_key(config)
    cache_path = cache_dir / f"features_s{subject_id:02d}_{key}.npz"
    if cache_path.exists():
        data = np.load(cache_path)
        return data["features"], data["groups"]

    features, groups = build_feature_matrix(eeg, config)
    np.savez_compressed(cache_path, features=features, groups=groups)
    return features, groups


def build_search(
    n_iter: int, random_state: int
) -> RandomizedSearchCV:
    """Build RandomizedSearchCV with XGBoost + StandardScaler."""
    base = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", base),
        ]
    )

    if randint is None or uniform is None or loguniform is None:
        raise RuntimeError("scipy.stats is required for randomized search.")

    param_dist = {
        "clf__n_estimators": randint(500, 1501),
        "clf__max_depth": randint(3, 9),
        "clf__learning_rate": loguniform(0.01, 0.1),
        "clf__subsample": uniform(0.7, 0.3),
        "clf__colsample_bytree": uniform(0.6, 0.4),
        "clf__reg_lambda": loguniform(1.0, 10.0),
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="accuracy",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )
    return search


def evaluate_best(
    X: np.ndarray,
    y: np.ndarray,
    best_params: dict,
    random_state: int,
) -> list[float]:
    """Evaluate best params with 5-fold Stratified CV (still leaky)."""
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        **{k.replace("clf__", ""): v for k, v in best_params.items() if k.startswith("clf__")},
    )
    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", model)])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    return scores.tolist()


def run_experiment(
    config: Config,
    subjects: list[int] | None,
    n_iter: int,
    output_dir: Path,
    random_state: int,
    progress: bool,
) -> dict:
    dataset = load_dataset(config.data_dir, config.eeg_channels, subjects=subjects)
    feature_cache_dir = output_dir / "feature_cache"

    all_features = []
    all_labels_valence = []
    all_labels_arousal = []

    for idx, subj in enumerate(dataset, start=1):
        if progress:
            print(f"[Subject {subj.subject_id}] ({idx}/{len(dataset)}) extracting...")

        X, trial_groups = load_or_compute_subject_features(
            subj.subject_id,
            subj.eeg,
            config,
            feature_cache_dir,
        )

        labels_v = build_labels(
            subj.labels,
            dimension="valence",
            mode=config.label_mode,
            threshold=config.label_threshold,
            bins=config.label_bins,
        )
        labels_a = build_labels(
            subj.labels,
            dimension="arousal",
            mode=config.label_mode,
            threshold=config.label_threshold,
            bins=config.label_bins,
        )

        if len(labels_v) != len(X):
            labels_v = labels_v[trial_groups]
            labels_a = labels_a[trial_groups]

        all_features.append(X)
        all_labels_valence.append(labels_v)
        all_labels_arousal.append(labels_a)

    X = np.concatenate(all_features, axis=0)
    y_valence = np.concatenate(all_labels_valence, axis=0)
    y_arousal = np.concatenate(all_labels_arousal, axis=0)

    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for dim, y in [("valence", y_valence), ("arousal", y_arousal)]:
        if progress:
            print(f"\n[{dim}] RandomizedSearchCV (leaky pooled windows)...")

        search = build_search(n_iter=n_iter, random_state=random_state)
        search.fit(X, y)

        best_params = search.best_params_
        best_score = float(search.best_score_)
        scores = evaluate_best(X, y, best_params, random_state)

        results[dim] = {
            "best_score_cv": best_score,
            "best_params": best_params,
            "fold_accs": scores,
            "mean_acc": float(np.mean(scores)),
            "std_acc": float(np.std(scores)),
        }

        if progress:
            print(
                f"  best CV acc: {best_score*100:.2f}% | "
                f"re-eval mean: {np.mean(scores)*100:.2f}%"
            )

    meta = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_subjects": int(len(dataset)),
        "config": {
            "window_seconds": config.window_seconds,
            "step_seconds": config.step_seconds,
            "label_threshold": config.label_threshold,
            "bands": config.bands,
            "feature_set": ["de", "log_bandpower", "asym_de", "asym_logpower"],
            "asym_pairs": config.asymmetry_pairs,
        },
    }

    summary = {
        "meta": meta,
        "results": results,
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with (output_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("LEAKY MAX-ACCURACY EXPERIMENT (XGBoost)\n")
        f.write("WARNING: WINDOWS ARE POOLED ACROSS SUBJECTS/VIDEOS (LEAKAGE)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Samples: {meta['n_samples']} | Features: {meta['n_features']}\n")
        f.write(f"Window: {config.window_seconds}s | Step: {config.step_seconds}s\n")
        f.write(f"Label threshold: {config.label_threshold}\n")
        f.write(f"Bands: {config.bands}\n")
        f.write("Features: DE + log bandpower + asymmetry (DE & logpower)\n\n")
        for dim in ["valence", "arousal"]:
            r = results[dim]
            f.write(f"{dim}:\n")
            f.write(f"  best CV acc (search): {r['best_score_cv']*100:.2f}%\n")
            f.write(f"  re-eval mean: {r['mean_acc']*100:.2f}% ± {r['std_acc']*100:.2f}%\n")
            f.write(f"  fold accs: {[f'{a*100:.2f}%' for a in r['fold_accs']]}\n")
            f.write(f"  best params: {r['best_params']}\n\n")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leaky max-accuracy DEAP experiment (XGBoost, pooled windows)."
    )
    parser.add_argument("--subjects", nargs="*", type=int, default=None)
    parser.add_argument("--n-iter", type=int, default=40)
    parser.add_argument(
        "--label-threshold", type=float, default=5.0, help="Binary threshold for valence/arousal"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/leaky_xgb_maxacc"))
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    config = Config(
        data_dir=Path("archive"),
        eeg_channels=tuple(range(32)),
        sfreq=128,
        baseline_seconds=3.0,
        bandpass_low=4.0,
        bandpass_high=45.0,
        bandpass_order=4,
        notch_freq=50.0,
        apply_filtering=True,
        window_seconds=3.0,
        step_seconds=1.5,
        feature_mode="bandpower_log",
        feature_set=("bandpower", "de"),
        label_mode="binary",
        label_threshold=args.label_threshold,
        feature_selection="none",
        use_cache=not args.no_cache,
    )

    print("!" * 70)
    print("LEAKY MAX-ACCURACY EXPERIMENT (XGBoost)")
    print("WARNING: POOLED WINDOWS, DATA LEAKAGE INTENTIONAL")
    print("!" * 70)
    print(f"Window: {config.window_seconds}s, Step: {config.step_seconds}s (50% overlap)")
    print(f"Label threshold: {config.label_threshold}")
    print(f"Features: DE + log bandpower + asymmetry (DE & logpower)")
    print("Model: XGBoost + StandardScaler, RandomizedSearchCV\n")

    summary = run_experiment(
        config=config,
        subjects=args.subjects,
        n_iter=args.n_iter,
        output_dir=args.output_dir,
        random_state=args.random_state,
        progress=args.progress,
    )

    print("\n" + "=" * 60)
    print("FINAL RESULTS (LEAKY)")
    print("=" * 60)
    for dim in ["valence", "arousal"]:
        r = summary["results"][dim]
        print(
            f"{dim}: {r['mean_acc']*100:.2f}% ± {r['std_acc']*100:.2f}% "
            f"(best search CV: {r['best_score_cv']*100:.2f}%)"
        )
    print(f"\nResults written to {args.output_dir}")


if __name__ == "__main__":
    main()
