from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat


@dataclass(frozen=True)
class EEGData:
    x: np.ndarray
    y: np.ndarray
    subjects: np.ndarray
    diagnosis: np.ndarray
    segment_ids: np.ndarray


def _mat_array(path: Path, key: str) -> np.ndarray:
    try:
        with h5py.File(path, "r") as handle:
            return np.asarray(handle[key])
    except OSError:
        return np.asarray(loadmat(path)[key])


def _segments(array: np.ndarray, n_channels: int, points: int) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(f"Expected 2-D EEG, got {array.shape}")
    if array.shape[0] != n_channels and array.shape[1] == n_channels:
        array = array.T
    if array.shape[0] != n_channels:
        raise ValueError(f"Expected {n_channels} channels, got {array.shape}")
    usable = array.shape[1] - array.shape[1] % points
    if usable == 0:
        raise ValueError(f"Signal in {array.shape} is shorter than one segment")
    return array[:, :usable].reshape(n_channels, -1, points).transpose(1, 0, 2).astype(np.float32)


def load_training_data(data_root: Path, n_channels: int = 30, sample_rate: int = 250) -> EEGData:
    points = sample_rate * 10
    xs, ys, subjects, diagnoses, segment_ids = [], [], [], [], []
    folders = (("DEP", "DEP"), ("HC", "HC"))
    for folder, diagnosis in folders:
        for path in sorted((Path(data_root) / "train" / folder).glob("*timedata.mat")):
            subject = path.stem.replace("timedata", "")
            neutral = _segments(_mat_array(path, "EEG_data_neu"), n_channels, points)
            positive = _segments(_mat_array(path, "EEG_data_pos"), n_channels, points)
            x = np.concatenate((neutral, positive))
            y = np.r_[np.zeros(len(neutral), dtype=np.int64), np.ones(len(positive), dtype=np.int64)]
            xs.append(x)
            ys.append(y)
            subjects.extend([subject] * len(x))
            diagnoses.extend([diagnosis] * len(x))
            segment_ids.extend(range(len(x)))
    if not xs:
        raise FileNotFoundError(f"No *timedata.mat files under {Path(data_root) / 'train'}")
    return EEGData(
        np.concatenate(xs),
        np.concatenate(ys),
        np.asarray(subjects),
        np.asarray(diagnoses),
        np.asarray(segment_ids, dtype=np.int64),
    )
