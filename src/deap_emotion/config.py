from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Tuple


@dataclass(frozen=True)
class Config:
    data_dir: Path = Path("archive")
    eeg_channels: Tuple[int, ...] = tuple(range(32))
    sfreq: int = 128
    baseline_seconds: float = 3.0
    bands: Tuple[Tuple[str, float, float], ...] = (
        ("delta", 1.0, 4.0),
        ("theta", 4.0, 8.0),
        ("alpha", 8.0, 13.0),
        ("beta", 13.0, 30.0),
        ("gamma", 30.0, 45.0),
    )
    feature_mode: str = "bandpower_log"
    window_seconds: float | None = None
    step_seconds: float | None = None
    label_mode: str = "binary"
    label_threshold: float = 5.0
    label_bins: Tuple[float, ...] = (3.0, 5.0, 7.0)
    classifier: str = "logreg"
    random_state: int = 42
    subject_folds: int = 5
    use_cache: bool = True
    cache_dir: Path = Path("cache")

    def baseline_samples(self) -> int:
        return int(self.baseline_seconds * self.sfreq)

    def window_samples(self) -> int | None:
        if self.window_seconds is None:
            return None
        return int(self.window_seconds * self.sfreq)

    def step_samples(self) -> int | None:
        if self.step_seconds is None:
            return None
        return int(self.step_seconds * self.sfreq)

    def band_edges(self) -> Iterable[Tuple[str, float, float]]:
        return self.bands
