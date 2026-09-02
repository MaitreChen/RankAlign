from __future__ import annotations

import math

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))


def de_welch_features(x: np.ndarray, sample_rate: int = 250) -> np.ndarray:
    """SEED-style DE plus absolute/relative Welch bandpower."""
    de_blocks = []
    for low, high in BANDS:
        sos = butter(1, (low, high), btype="bandpass", fs=sample_rate, output="sos")
        filtered = sosfiltfilt(sos, x, axis=-1)
        variance = np.var(filtered, axis=-1)
        de_blocks.append(0.5 * np.log(2 * np.pi * math.e * np.maximum(variance, 1e-12)))
    de = np.stack(de_blocks, axis=-1).reshape(len(x), -1)

    frequencies, psd = welch(x, fs=sample_rate, nperseg=500, noverlap=250, axis=-1)
    total_mask = (frequencies >= 1) & (frequencies <= 45)
    total = np.maximum(np.trapz(psd[..., total_mask], frequencies[total_mask], axis=-1), 1e-12)
    absolute, relative = [], []
    for low, high in BANDS:
        mask = (frequencies >= low) & (frequencies < high)
        power = np.maximum(np.trapz(psd[..., mask], frequencies[mask], axis=-1), 1e-12)
        absolute.append(np.log(power))
        relative.append(np.log(power / total))
    spectral = np.concatenate((np.stack(absolute, axis=-1), np.stack(relative, axis=-1)), axis=-1)
    return np.concatenate((de, spectral.reshape(len(x), -1)), axis=1).astype(np.float32)


def subject_normalize(x: np.ndarray, subjects: np.ndarray, mode: str) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float32)
    for subject in np.unique(subjects):
        indices = np.flatnonzero(subjects == subject)
        block = x[indices]
        if mode == "minmax":
            low, high = block.min(0, keepdims=True), block.max(0, keepdims=True)
            out[indices] = (block - low) / np.maximum(high - low, 1e-6)
        elif mode == "zscore":
            mean, std = block.mean(0, keepdims=True), block.std(0, keepdims=True)
            out[indices] = (block - mean) / np.maximum(std, 1e-6)
        else:
            raise ValueError(f"Unknown subject normalization: {mode}")
    return out


def aggregate_trials(features: np.ndarray, data, segments_per_trial: int = 5):
    rows, labels, subjects, diagnoses, trial_ids = [], [], [], [], []
    for subject in np.unique(data.subjects):
        subject_indices = np.flatnonzero(data.subjects == subject)
        for label, offset in ((0, 0), (1, 20)):
            for trial in range(4):
                wanted = set(range(offset + trial * segments_per_trial, offset + (trial + 1) * segments_per_trial))
                indices = [i for i in subject_indices if data.segment_ids[i] in wanted]
                if len(indices) != segments_per_trial:
                    raise ValueError(f"{subject}: expected {segments_per_trial} segments for label={label}, trial={trial}")
                rows.append(features[indices].mean(axis=0))
                labels.append(label)
                subjects.append(subject)
                diagnoses.append(data.diagnosis[indices[0]])
                trial_ids.append(f"{subject}_v{label}_{trial}")
    return np.vstack(rows), np.asarray(labels), np.asarray(subjects), np.asarray(diagnoses), np.asarray(trial_ids)
