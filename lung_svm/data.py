from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from . import DISEASES


def load_labels(csv_path: str | Path) -> pd.DataFrame:
    """Load the NIH-style labels CSV and add one binary column per disease."""
    df = pd.read_csv(csv_path)
    required = {"Image Index", "Finding Labels", "Patient ID"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")
    df = df.dropna(subset=["Image Index", "Finding Labels", "Patient ID"]).copy()
    label_sets = df["Finding Labels"].astype(str).map(
        lambda value: {item.strip() for item in value.split("|") if item.strip()}
    )
    unknown = sorted(set().union(*label_sets) - set(DISEASES) - {"No Finding"})
    if unknown:
        raise ValueError(f"Unexpected label(s): {', '.join(unknown)}")
    for disease in DISEASES:
        df[disease] = label_sets.map(lambda labels: int(disease in labels)).astype(np.uint8)
    df["No_Finding"] = (df[list(DISEASES)].sum(axis=1) == 0).astype(np.uint8)
    return df.reset_index(drop=True)


def patient_splits(df: pd.DataFrame, seed: int = 42) -> dict[str, np.ndarray]:
    """Create approximately 70/15/15 image splits without sharing patients."""
    outer = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, held_idx = next(outer.split(df, groups=df["Patient ID"]))
    held = df.iloc[held_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_local, test_local = next(inner.split(held, groups=held["Patient ID"]))
    result = {
        "train": train_idx,
        "validation": held_idx[val_local],
        "test": held_idx[test_local],
    }
    groups = {key: set(df.iloc[idx]["Patient ID"]) for key, idx in result.items()}
    if any(groups[a] & groups[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise RuntimeError("Patient leakage detected in split")
    return result


def resolve_image(image_root: str | Path, filename: str) -> Path:
    """Find image under root, including when archives unpack into nested folders."""
    root = Path(image_root)
    direct = root / filename
    if direct.is_file():
        return direct
    matches = list(root.rglob(filename))
    return matches[0] if matches else direct
