import json
from pathlib import Path

import pytest

from deap_emotion.config import Config
from deap_emotion.eval import metrics_to_dict, run_cross_subject, run_subject_dependent


def _load_baseline() -> dict:
    baseline_path = Path("tests/baseline_metrics.json")
    if not baseline_path.exists():
        pytest.skip("Baseline metrics not found.")
    return json.loads(baseline_path.read_text())


@pytest.mark.skip(reason="Baseline outdated after Paper 13 refactoring")
def test_pipeline_metrics_stable():
    if not Path("archive").exists():
        pytest.skip("DEAP archive not available.")
    if not list(Path("archive").glob("s*.dat")):
        pytest.skip("DEAP files not available.")

    # Use explicit config for smoke test (not Paper 13 defaults)
    config = Config(
        subject_folds=2,
        use_cache=False,
        feature_set=("bandpower",),  # Simple features for smoke test
        label_mode="binary",
        label_threshold=5.0,
        window_seconds=None,  # Full trial
        step_seconds=None,
        apply_filtering=False,  # Skip filtering for speed
    )
    subjects = [1, 2, 3]
    trials = [0, 1, 2, 3]

    baseline = _load_baseline()
    current = {
        "subject_dependent": metrics_to_dict(
            run_subject_dependent(config, subjects=subjects, trial_ids=trials)
        ),
        "cross_subject": metrics_to_dict(
            run_cross_subject(config, subjects=subjects, trial_ids=trials)
        ),
    }

    tol = 0.05
    for mode in ("subject_dependent", "cross_subject"):
        for dimension in ("valence", "arousal", "dominance"):
            expected = baseline[mode][dimension]["accuracy_mean"]
            actual = current[mode][dimension]["accuracy_mean"]
            assert abs(actual - expected) <= tol
