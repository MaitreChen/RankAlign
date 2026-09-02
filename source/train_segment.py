from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rankalign.alignment import euclidean_align
from rankalign.config import load_config
from rankalign.data import EEGData, load_training_data
from rankalign.features import de_welch_features
from rankalign.metrics import evaluate_scores
from rankalign.model import EEGNetLite


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def standardize_fold(x_train: np.ndarray, x_valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=(0, 2), keepdims=True)
    std = np.maximum(x_train.std(axis=(0, 2), keepdims=True), 1e-6)
    return ((x_train - mean) / std).astype(np.float32), ((x_valid - mean) / std).astype(np.float32)


def loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensors = TensorDataset(torch.from_numpy(x[:, None]), torch.from_numpy(y.astype(np.int64)))
    return DataLoader(tensors, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def predict(model: nn.Module, data_loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    probabilities = []
    with torch.no_grad():
        for x_batch, _ in data_loader:
            logits = model(x_batch.to(device))
            probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probabilities).astype(np.float32)


def train_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    settings: dict,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    set_seed(seed)
    x_train, x_valid = standardize_fold(x_train, x_valid)
    train_loader = loader(x_train, y_train, settings["batch_size"], True)
    valid_loader = loader(x_valid, y_valid, settings["batch_size"], False)
    model = EEGNetLite(dropout=settings["dropout"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    criterion = nn.CrossEntropyLoss()
    best_auc, best_probabilities, stale_epochs = -np.inf, None, 0
    for _ in range(settings["epochs"]):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_batch.to(device)), y_batch.to(device))
            loss.backward()
            optimizer.step()
        probabilities = predict(model, valid_loader, device)
        auc = roc_auc_score(y_valid, probabilities)
        if auc > best_auc + 1e-6:
            best_auc, best_probabilities, stale_epochs = auc, probabilities.copy(), 0
        else:
            stale_epochs += 1
            if stale_epochs >= settings["patience"]:
                break
    if best_probabilities is None:
        raise RuntimeError("Training did not produce validation probabilities")
    return best_probabilities


def boundary_indices(scores: np.ndarray, subjects: np.ndarray, window: int) -> np.ndarray:
    selected = []
    for subject in np.unique(subjects):
        indices = np.flatnonzero(subjects == subject)
        order = np.argsort(scores[indices])
        center = len(indices) // 2
        selected.extend(indices[order[max(0, center - window):min(len(indices), center + window)]])
    return np.asarray(sorted(selected), dtype=np.int64)


def boundary_matrix(scores: np.ndarray, subjects: np.ndarray, spectral: np.ndarray, indices: np.ndarray) -> np.ndarray:
    rows = []
    for index in indices:
        subject_indices = np.flatnonzero(subjects == subjects[index])
        subject_scores = scores[subject_indices]
        ranks = np.empty(len(subject_indices), dtype=np.float32)
        ranks[np.argsort(subject_scores)] = np.linspace(0, 1, len(subject_indices), dtype=np.float32)
        local_index = int(np.flatnonzero(subject_indices == index)[0])
        sorted_scores = np.sort(subject_scores)
        center = len(sorted_scores) // 2
        margin = sorted_scores[center] - sorted_scores[center - 1]
        row = [scores[index], ranks[local_index], scores[index] - np.median(subject_scores), margin]
        row.extend(spectral[index])
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def refine_boundaries(data: EEGData, scores: np.ndarray, window: int, strength: float) -> np.ndarray:
    spectral = de_welch_features(data.x).reshape(len(data.x), 30, -1)
    spectral = np.concatenate((spectral.mean(axis=1), spectral.std(axis=1)), axis=1).astype(np.float32)
    indices = boundary_indices(scores, data.subjects, window)
    matrix = boundary_matrix(scores, data.subjects, spectral, indices)
    labels, groups = data.y[indices], data.subjects[indices]
    expert_oof = np.zeros(len(indices), dtype=np.float32)
    for train, valid in LeaveOneGroupOut().split(matrix, labels, groups):
        expert = ExtraTreesClassifier(
            n_estimators=600,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        expert.fit(matrix[train], labels[train])
        expert_oof[valid] = expert.predict_proba(matrix[valid])[:, 1]
    delta = (expert_oof - expert_oof.mean()) / max(float(expert_oof.std()), 1e-8)
    refined = scores.copy()
    refined[indices] += float(strength) * delta
    return refined.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the RankAlign segment-level OOF branch.")
    parser.add_argument("--config", type=Path, default=Path("configs/rankalign.json"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/segment_oof.csv"))
    parser.add_argument("--no-boundary-refinement", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    settings = config["segment"]
    data = load_training_data(args.data_root, config["n_channels"], config["sample_rate"])
    aligned = euclidean_align(data.x, data.subjects)
    splitter = list(GroupKFold(n_splits=config["folds"]).split(aligned, data.y, data.subjects))
    seed_predictions = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for seed in settings["seeds"]:
        oof = np.zeros(len(data.y), dtype=np.float32)
        for fold, (train, valid) in enumerate(splitter, 1):
            print(f"seed={seed} fold={fold}/{config['folds']} device={device}")
            oof[valid] = train_fold(
                aligned[train], data.y[train], aligned[valid], data.y[valid], settings, seed + fold, device
            )
        seed_predictions.append(oof)
    scores = np.mean(seed_predictions, axis=0).astype(np.float32)
    if not args.no_boundary_refinement:
        scores = refine_boundaries(
            data, scores, settings["boundary_window"], settings["boundary_strength"]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "subject_id": data.subjects,
            "diagnosis": data.diagnosis,
            "segment_id": data.segment_ids,
            "y_true": data.y,
            "positive_probability": scores,
        }
    ).to_csv(args.output, index=False)
    print(evaluate_scores(data.y, scores, data.subjects))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
