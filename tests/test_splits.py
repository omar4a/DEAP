import numpy as np

from deap_emotion.splits import subject_dependent_split


def test_group_splits_no_leakage():
    groups = np.array([0, 0, 1, 1, 2, 2])
    splits = list(subject_dependent_split(groups, n_splits=3))
    for train_idx, test_idx in splits:
        train_groups = set(groups[train_idx].tolist())
        test_groups = set(groups[test_idx].tolist())
        assert train_groups.isdisjoint(test_groups)
