from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut


def subject_dependent_split(groups: np.ndarray, n_splits: int):
    splitter = GroupKFold(n_splits=n_splits)
    dummy = np.zeros(len(groups))
    return splitter.split(dummy, groups=groups)


def cross_subject_split(groups: np.ndarray):
    splitter = LeaveOneGroupOut()
    dummy = np.zeros(len(groups))
    return splitter.split(dummy, groups=groups)
