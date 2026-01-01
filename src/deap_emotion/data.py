from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np


@dataclass
class SubjectData:
    subject_id: int
    eeg: np.ndarray
    labels: np.ndarray


def _load_dat_file(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def list_subject_files(data_dir: Path) -> List[Path]:
    return sorted(data_dir.glob("s*.dat"))


def load_subject(
    path: Path,
    eeg_channels: Sequence[int],
    trial_ids: Sequence[int] | None = None,
) -> SubjectData:
    payload = _load_dat_file(path)
    data = payload["data"]
    labels = payload["labels"]

    eeg = data[:, eeg_channels, :]
    if trial_ids is not None:
        eeg = eeg[trial_ids]
        labels = labels[trial_ids]

    subject_id = int(path.stem.lstrip("s"))
    return SubjectData(subject_id=subject_id, eeg=eeg, labels=labels)


def load_dataset(
    data_dir: Path,
    eeg_channels: Sequence[int],
    subjects: Iterable[int] | None = None,
    trial_ids: Sequence[int] | None = None,
) -> List[SubjectData]:
    subject_set = None if subjects is None else set(subjects)
    results: List[SubjectData] = []
    for path in list_subject_files(data_dir):
        subject_id = int(path.stem.lstrip("s"))
        if subject_set is not None and subject_id not in subject_set:
            continue
        results.append(load_subject(path, eeg_channels, trial_ids))
    return results
