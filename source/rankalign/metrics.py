from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def subject_rank_predictions(y_true: np.ndarray, scores: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    predictions = np.zeros(len(scores), dtype=np.int64)
    for subject in np.unique(subjects):
        indices = np.flatnonzero(subjects == subject)
        positives = int(y_true[indices].sum())
        if positives:
            predictions[indices[np.argsort(scores[indices])[-positives:]]] = 1
    return predictions


def top_k_predictions(scores: np.ndarray, subjects: np.ndarray, top_k: int = 4) -> np.ndarray:
    predictions = np.zeros(len(scores), dtype=np.int64)
    for subject in np.unique(subjects):
        indices = np.flatnonzero(subjects == subject)
        if len(indices) < top_k:
            raise ValueError(f"Subject {subject} has only {len(indices)} trials")
        predictions[indices[np.argsort(scores[indices])[-top_k:]]] = 1
    return predictions


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray, subjects: np.ndarray) -> dict[str, float]:
    predictions = subject_rank_predictions(y_true, scores, subjects)
    return {
        "rank_bacc": float(balanced_accuracy_score(y_true, predictions)),
        "auc": float(roc_auc_score(y_true, scores)),
    }
