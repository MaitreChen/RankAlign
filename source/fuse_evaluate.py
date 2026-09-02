from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rankalign.metrics import evaluate_scores
from rankalign.data import load_training_data

KEYS = ["video_id", "subject_id", "diagnosis", "y_true"]


def load_branch(path: Path, name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = set(KEYS + ["positive_probability"]) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame.duplicated(KEYS).any():
        raise ValueError(f"{path} contains duplicate trial rows")
    return frame[KEYS + ["positive_probability"]].rename(columns={"positive_probability": name})


def load_segment_branch(path: Path, data_root: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "video_id" in frame:
        return load_branch(path, "segment")
    if "positive_probability" not in frame:
        raise ValueError(f"{path} has no positive_probability column")
    data = load_training_data(data_root)
    if len(frame) != len(data.y):
        raise ValueError(f"Segment OOF has {len(frame)} rows; expected {len(data.y)}")
    scores = frame.positive_probability.to_numpy(dtype=np.float32)
    rows = []
    for subject in np.unique(data.subjects):
        subject_indices = np.flatnonzero(data.subjects == subject)
        for label, offset in ((0, 0), (1, 20)):
            for trial in range(4):
                wanted = set(range(offset + trial * 5, offset + (trial + 1) * 5))
                indices = [i for i in subject_indices if data.segment_ids[i] in wanted]
                rows.append({
                    "video_id": f"{subject}_v{label}_{trial}",
                    "subject_id": subject,
                    "diagnosis": data.diagnosis[indices[0]],
                    "y_true": label,
                    "segment": float(scores[indices].mean()),
                })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment-oof", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, help="Required when segment OOF contains 2400 segment rows")
    parser.add_argument("--minmax-oof", type=Path, required=True)
    parser.add_argument("--zscore-oof", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/rankalign_oof.csv"))
    args = parser.parse_args()

    segment_frame = pd.read_csv(args.segment_oof, nrows=1)
    if "video_id" not in segment_frame and args.data_root is None:
        parser.error("--data-root is required for a segment-level OOF file")
    merged = load_segment_branch(args.segment_oof, args.data_root)
    merged = merged.merge(load_branch(args.minmax_oof, "minmax"), on=KEYS, validate="one_to_one")
    merged = merged.merge(load_branch(args.zscore_oof, "zscore"), on=KEYS, validate="one_to_one")
    merged["positive_probability"] = 0.600 * merged.segment + 0.305 * merged.minmax + 0.095 * merged.zscore
    metrics = evaluate_scores(
        merged.y_true.to_numpy(), merged.positive_probability.to_numpy(), merged.subject_id.astype(str).to_numpy()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged[KEYS + ["positive_probability"]].to_csv(args.output, index=False)
    print(f"rank_bacc={metrics['rank_bacc']:.4f} auc={metrics['auc']:.4f}")


if __name__ == "__main__":
    main()
