from __future__ import annotations

import json
from pathlib import Path


def load_config(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    weights = config["fusion"]
    if set(weights) != {"segment", "spectral_minmax", "spectral_zscore"}:
        raise ValueError("fusion must define segment, spectral_minmax, and spectral_zscore")
    if abs(sum(float(value) for value in weights.values()) - 1.0) > 1e-8:
        raise ValueError("fusion weights must sum to one")
    return config
