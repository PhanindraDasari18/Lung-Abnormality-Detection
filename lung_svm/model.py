from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

from . import DISEASES


def fit_preprocessor(x_train: np.ndarray, components: int = 256):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train).astype(np.float32, copy=False)
    n_components = min(components, x_scaled.shape[0] - 1, x_scaled.shape[1])
    if n_components < 1:
        raise ValueError("Not enough training images to fit PCA")
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
    return scaler, pca, pca.fit_transform(x_scaled).astype(np.float32, copy=False)


def fit_classifiers(x: np.ndarray, y: np.ndarray) -> list[LinearSVC | None]:
    """Fit one binary LinearSVC per class; absent/single-valued labels are skipped."""
    models = []
    for column, name in enumerate(DISEASES):
        target = y[:, column]
        if np.unique(target).size < 2:
            constant = int(target[0])
            print(f"{name}: training split contains only {constant}; using constant prediction")
            models.append(constant)
            continue
        model = LinearSVC(C=1.0, class_weight="balanced", max_iter=20000, random_state=42)
        model.fit(x, target)
        models.append(model)
    return models


def decision_scores(models: list, x: np.ndarray) -> np.ndarray:
    scores = np.zeros((len(x), len(DISEASES)), dtype=np.float32)
    for i, model in enumerate(models):
        if isinstance(model, (int, np.integer)):
            scores[:, i] = 1.0 if model == 1 else -1.0
        elif model is not None:
            scores[:, i] = model.decision_function(x)
    return scores


def tune_thresholds(y_true: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Choose per-class F1 thresholds on validation data only."""
    thresholds = np.zeros(len(DISEASES), dtype=np.float32)
    for i in range(len(DISEASES)):
        candidates = np.unique(np.quantile(scores[:, i], np.linspace(0, 1, 101)))
        best_f1, best_threshold = -1.0, 0.0
        for threshold in candidates:
            value = f1_score(y_true[:, i], scores[:, i] >= threshold, zero_division=0)
            if value > best_f1:
                best_f1, best_threshold = value, float(threshold)
        thresholds[i] = best_threshold
    return thresholds
