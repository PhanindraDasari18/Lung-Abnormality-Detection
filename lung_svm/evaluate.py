from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("artifacts/.mplconfig").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from . import DISEASES


def evaluate(y_true: np.ndarray, scores: np.ndarray, thresholds: np.ndarray,
             output_dir: str | Path) -> pd.DataFrame:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predicted = scores >= thresholds[None, :]
    rows, curve_data = [], []
    for i, disease in enumerate(DISEASES):
        truth = y_true[:, i]
        try:
            class_auc = float(roc_auc_score(truth, scores[:, i])) if np.unique(truth).size == 2 else np.nan
        except ValueError:
            class_auc = np.nan
        precision = precision_score(truth, predicted[:, i], zero_division=0)
        recall = recall_score(truth, predicted[:, i], zero_division=0)
        f1 = f1_score(truth, predicted[:, i], zero_division=0)
        rows.append({"disease": disease, "support": int(truth.sum()), "roc_auc": class_auc,
                     "precision": precision, "recall": recall, "f1": f1,
                     "threshold": float(thresholds[i])})
        if np.unique(truth).size == 2:
            fpr, tpr, _ = roc_curve(truth, scores[:, i])
            curve_data.append((disease, fpr, tpr, class_auc))
        tn, fp, fn, tp = confusion_matrix(truth, predicted[:, i], labels=[0, 1]).ravel()
        rows[-1].update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    table = pd.DataFrame(rows)
    numeric = ["roc_auc", "precision", "recall", "f1"]
    macro = {key: table[key].mean() for key in numeric}
    table.to_csv(output / "per_class_metrics.csv", index=False)
    pd.DataFrame([macro]).to_csv(output / "macro_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 7))
    for name, fpr, tpr, score in curve_data:
        ax.plot(fpr, tpr, label=f"{name} (AUC={score:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="Per-class ROC curves")
    ax.legend(fontsize="small", ncol=2)
    fig.tight_layout(); fig.savefig(output / "roc_curves.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    plot_data = table.sort_values("roc_auc", na_position="first")
    ax.barh(plot_data["disease"], plot_data["roc_auc"], color="#2878a5")
    ax.set(xlim=(0, 1), xlabel="ROC-AUC", title="ROC-AUC by abnormality")
    fig.tight_layout(); fig.savefig(output / "class_auc.png", dpi=160); plt.close(fig)
    return table
