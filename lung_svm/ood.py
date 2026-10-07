from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def fit_ood_reference(train_features: np.ndarray, validation_features: np.ndarray,
                      quantile: float = 0.99, max_reference: int = 2048,
                      seed: int = 42) -> tuple[np.ndarray, float, np.ndarray]:
    """Calibrate a nearest-neighbor distance gate using patient-held-out X-rays."""
    rng = np.random.default_rng(seed)
    if len(train_features) > max_reference:
        selected = np.sort(rng.choice(len(train_features), max_reference, replace=False))
        reference = np.asarray(train_features[selected], dtype=np.float32)
    else:
        reference = np.asarray(train_features, dtype=np.float32)
    neighbors = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1).fit(reference)
    validation_distances = neighbors.kneighbors(validation_features, return_distance=True)[0][:, 0]
    threshold = float(np.quantile(validation_distances, quantile))
    return reference, threshold, validation_distances


def ood_distances(reference: np.ndarray, features: np.ndarray) -> np.ndarray:
    neighbors = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1).fit(reference)
    return neighbors.kneighbors(features, return_distance=True)[0][:, 0]
