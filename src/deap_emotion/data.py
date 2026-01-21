from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from scipy.io import loadmat


@dataclass
class SubjectData:
    subject_id: int
    eeg: np.ndarray
    labels: np.ndarray


def _load_dat_file(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def _load_mat_file(path: Path) -> dict:
    try:
        payload = loadmat(path)
        data = payload.get("data")
        if data is None:
            data = payload.get("data_clean")
        if data is None:
            raise ValueError(f"No data array found in {path}")
        labels = payload.get("labels")
        if labels is None:
            raise ValueError(f"No labels array found in {path}")
        return {"data": np.asarray(data), "labels": np.asarray(labels)}
    except NotImplementedError:
        import h5py

        with h5py.File(path, "r") as handle:
            data = handle.get("data")
            if data is None:
                data = handle.get("data_clean")
            if data is None:
                raise ValueError(f"No data array found in {path}")
            labels = handle.get("labels")
            if labels is None:
                raise ValueError(f"No labels array found in {path}")

            data_arr = np.array(data)
            labels_arr = np.array(labels)
            data_arr = np.transpose(data_arr, tuple(range(data_arr.ndim - 1, -1, -1)))
            labels_arr = np.transpose(labels_arr, tuple(range(labels_arr.ndim - 1, -1, -1)))
            return {"data": data_arr, "labels": labels_arr}


def _load_npz_file(path: Path) -> dict:
    payload = np.load(path)
    if "data" not in payload or "labels" not in payload:
        raise ValueError(f"Missing data or labels in {path}")
    return {"data": payload["data"], "labels": payload["labels"]}


def list_subject_files(data_dir: Path) -> List[Path]:
    candidates = []
    for pattern in ("s*.mat", "s*.npz", "s*.dat"):
        candidates.extend(sorted(data_dir.glob(pattern)))

    priority = {".mat": 0, ".npz": 1, ".dat": 2}
    by_subject: dict[str, Path] = {}
    for path in candidates:
        key = path.stem
        ext = path.suffix.lower()
        if key not in by_subject or priority[ext] < priority[by_subject[key].suffix.lower()]:
            by_subject[key] = path
    return sorted(by_subject.values(), key=lambda p: int(p.stem.lstrip("s")))


def load_subject(
    path: Path,
    eeg_channels: Sequence[int],
    trial_ids: Sequence[int] | None = None,
) -> SubjectData:
    if path.suffix.lower() == ".mat":
        payload = _load_mat_file(path)
    elif path.suffix.lower() == ".npz":
        payload = _load_npz_file(path)
    else:
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
