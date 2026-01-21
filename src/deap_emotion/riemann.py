from __future__ import annotations

from typing import Tuple

import numpy as np


def _infer_channels(cov_dim: int) -> int:
    # Solve n(n+1)/2 = cov_dim
    n = int((np.sqrt(1 + 8 * cov_dim) - 1) / 2)
    if n * (n + 1) // 2 != cov_dim:
        raise ValueError(f"Invalid covariance feature length: {cov_dim}")
    return n


def _vector_to_sym(vec: np.ndarray, n_channels: int) -> np.ndarray:
    tri = np.triu_indices(n_channels)
    mat = np.zeros((n_channels, n_channels), dtype=float)
    mat[tri] = vec
    mat = mat + np.triu(mat, 1).T
    return mat


def _sym_to_vector(mat: np.ndarray) -> np.ndarray:
    tri = np.triu_indices(mat.shape[0])
    return mat[tri]


def _sym_logm(mat: np.ndarray, epsilon: float) -> np.ndarray:
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, epsilon, None)
    log_vals = np.log(vals)
    return (vecs * log_vals) @ vecs.T


def _sym_expm(mat: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(mat)
    exp_vals = np.exp(vals)
    return (vecs * exp_vals) @ vecs.T


def _invsqrtm(mat: np.ndarray, epsilon: float) -> np.ndarray:
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, epsilon, None)
    inv_sqrt = 1.0 / np.sqrt(vals)
    return (vecs * inv_sqrt) @ vecs.T


def _tangent_vector(mat: np.ndarray) -> np.ndarray:
    n = mat.shape[0]
    tri = np.triu_indices(n)
    vec = mat[tri].astype(float)
    off_diag = tri[0] != tri[1]
    vec[off_diag] *= np.sqrt(2.0)
    return vec


class RiemannianTangentSpace:
    def __init__(self, cov_dim: int, epsilon: float = 1e-6) -> None:
        self.cov_dim = cov_dim
        self.epsilon = epsilon
        self.n_channels = _infer_channels(cov_dim)
        self._invsqrt_ref: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> "RiemannianTangentSpace":
        covs = self._extract_covs(x)
        logm_sum = np.zeros_like(covs[0])
        for cov in covs:
            logm_sum += _sym_logm(cov, self.epsilon)
        mean_log = logm_sum / max(len(covs), 1)
        ref = _sym_expm(mean_log)
        self._invsqrt_ref = _invsqrtm(ref, self.epsilon)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self._invsqrt_ref is None:
            raise ValueError("RiemannianTangentSpace is not fitted.")
        covs = self._extract_covs(x)
        tangent = []
        for cov in covs:
            aligned = self._invsqrt_ref @ cov @ self._invsqrt_ref
            logm = _sym_logm(aligned, self.epsilon)
            tangent.append(_tangent_vector(logm))
        tangent_arr = np.stack(tangent, axis=0)
        if x.shape[1] > self.cov_dim:
            rest = x[:, self.cov_dim :]
            return np.concatenate([tangent_arr, rest], axis=1)
        return tangent_arr

    def fit_transform(self, x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        return self.fit(x, y).transform(x)

    def _extract_covs(self, x: np.ndarray) -> Tuple[np.ndarray, ...]:
        cov_part = x[:, : self.cov_dim]
        covs = []
        for vec in cov_part:
            cov = _vector_to_sym(vec, self.n_channels)
            cov = cov + np.eye(self.n_channels) * self.epsilon
            covs.append(cov)
        return tuple(covs)
