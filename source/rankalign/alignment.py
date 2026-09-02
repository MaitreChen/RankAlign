from __future__ import annotations

import numpy as np


def euclidean_align_subject(x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    centered = x - x.mean(axis=-1, keepdims=True)
    covariance = np.einsum("nct,ndt->ncd", centered, centered) / max(x.shape[-1] - 1, 1)
    reference = covariance.mean(axis=0)
    reference = (reference + reference.T) / 2
    ridge = eps * np.trace(reference) / reference.shape[0] + 1e-8
    values, vectors = np.linalg.eigh(reference + ridge * np.eye(reference.shape[0]))
    transform = (vectors * (1 / np.sqrt(np.maximum(values, 1e-8)))) @ vectors.T
    return np.einsum("cd,ndt->nct", transform, x).astype(np.float32)


def euclidean_align(x: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float32)
    for subject in np.unique(subjects):
        indices = np.flatnonzero(subjects == subject)
        out[indices] = euclidean_align_subject(x[indices])
    return out
