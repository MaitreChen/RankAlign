from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rankalign.data import load_training_data
from rankalign.config import load_config
from rankalign.features import aggregate_trials, de_welch_features, subject_normalize
from rankalign.metrics import evaluate_scores


def train_branch(x, y, subjects, c_value, folds, seed):
    oof = np.zeros(len(y), dtype=np.float32)
    splitter = GroupKFold(n_splits=folds)
    for fold, (train, valid) in enumerate(splitter.split(x, y, subjects), 1):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c_value, max_iter=3000, class_weight="balanced", random_state=seed + fold),
        )
        model.fit(x[train], y[train])
        oof[valid] = model.predict_proba(x[valid])[:, 1]
    return oof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/rankalign.json"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/spectral"))
    args = parser.parse_args()

    config = load_config(args.config)
    data = load_training_data(args.data_root)
    segment_features = de_welch_features(data.x)
    x, y, subjects, diagnosis, video_ids = aggregate_trials(segment_features, data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    branches = (("minmax", config["spectral"]["minmax_c"]), ("zscore", config["spectral"]["zscore_c"]))
    for mode, c_value in branches:
        normalized = subject_normalize(x, subjects, mode)
        oof = train_branch(normalized, y, subjects, c_value, config["folds"], config["seed"])
        frame = pd.DataFrame({
            "video_id": video_ids,
            "subject_id": subjects,
            "diagnosis": diagnosis,
            "y_true": y,
            "positive_probability": oof,
        })
        frame.to_csv(args.output_dir / f"{mode}_oof.csv", index=False)
        print(mode, evaluate_scores(y, oof, subjects))


if __name__ == "__main__":
    main()
