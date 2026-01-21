from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple


@dataclass(frozen=True)
class Config:
    data_dir: Path = Path("archive")
    eeg_channels: Tuple[int, ...] = tuple(range(32))
    sfreq: int = 128
    baseline_seconds: float = 3.0
    # Preprocessing options (from Paper 13)
    bandpass_low: float = 4.0  # Bandpass filter low cutoff (Hz)
    bandpass_high: float = 45.0  # Bandpass filter high cutoff (Hz)
    bandpass_order: int = 4  # Butterworth filter order
    notch_freq: float | None = 50.0  # Notch filter frequency (None to disable)
    apply_filtering: bool = True  # Whether to apply bandpass/notch filters
    channel_names: Tuple[str, ...] = (
        "Fp1",
        "AF3",
        "F7",
        "F3",
        "FC1",
        "FC5",
        "T7",
        "C3",
        "CP1",
        "CP5",
        "P7",
        "P3",
        "Pz",
        "PO3",
        "O1",
        "Oz",
        "O2",
        "PO4",
        "P4",
        "P8",
        "CP6",
        "CP2",
        "C4",
        "T8",
        "FC6",
        "FC2",
        "F4",
        "F8",
        "AF4",
        "Fp2",
        "Fz",
        "Cz",
    )
    asymmetry_pairs: Tuple[Tuple[str, str], ...] = (
        ("Fp1", "Fp2"),
        ("F3", "F4"),
        ("F7", "F8"),
        ("C3", "C4"),
        ("T7", "T8"),
        ("P3", "P4"),
        ("O1", "O2"),
    )
    bands: Tuple[Tuple[str, float, float], ...] = (
        ("theta", 4.0, 8.0),
        ("alpha", 8.0, 13.0),
        ("beta", 13.0, 30.0),
        ("gamma", 30.0, 45.0),
    )
    feature_mode: str = "bandpower"
    feature_set: Tuple[str, ...] | None = (
        "de_histogram",  # Paper 13: histogram-based DE
        "hfd",           # Paper 13: Higuchi's Fractal Dimension
    )
    psd_min_hz: float = 1.0
    psd_max_hz: float = 45.0
    psd_bin_width: float = 2.0
    pca_variance: float | None = None
    feature_selection: str = "mi"
    mi_k: int | None = 200
    l1_c: float = 0.5
    l1_alpha: float = 0.001
    scaler: str = "standard"
    trial_aggregation: str = "majority"
    riemann_epsilon: float = 1e-6
    per_subject_normalize: bool = False
    window_seconds: float | None = 3.0  # Paper 13: 3s segments
    step_seconds: float | None = 3.0   # Paper 13: no overlap
    label_mode: str = "binary"          # Paper 13: binary classification
    label_threshold: float = 4.5        # Paper 13: threshold 4.5
    label_bins: Tuple[float, ...] = (3.0, 6.0)
    classifier: str = "rbf_svm"
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

    def psd_band_edges(self) -> Tuple[Tuple[str, float, float], ...]:
        edges = []
        start = self.psd_min_hz
        while start < self.psd_max_hz:
            end = min(start + self.psd_bin_width, self.psd_max_hz)
            label = f"{start:.1f}-{end:.1f}"
            edges.append((label, start, end))
            start = end
        return tuple(edges)

    def asymmetry_indices(self) -> Tuple[Tuple[int, int], ...]:
        name_to_abs = {name: idx for idx, name in enumerate(self.channel_names)}
        abs_to_rel = {abs_idx: rel for rel, abs_idx in enumerate(self.eeg_channels)}
        pairs = []
        for left, right in self.asymmetry_pairs:
            if left not in name_to_abs or right not in name_to_abs:
                continue
            left_abs = name_to_abs[left]
            right_abs = name_to_abs[right]
            if left_abs in abs_to_rel and right_abs in abs_to_rel:
                pairs.append((abs_to_rel[left_abs], abs_to_rel[right_abs]))
        return tuple(pairs)

    def covariance_feature_dim(self) -> int:
        n_channels = len(self.eeg_channels)
        return n_channels * (n_channels + 1) // 2
