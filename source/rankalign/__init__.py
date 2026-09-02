"""Core components for RankAlign-EEG."""

from .metrics import evaluate_scores, subject_rank_predictions

__all__ = ["evaluate_scores", "subject_rank_predictions"]
