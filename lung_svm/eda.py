from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("artifacts/.mplconfig").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from . import DISEASES


def make_eda(df: pd.DataFrame, paths: list[Path], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    counts = df[list(DISEASES) + ["No_Finding"]].sum().sort_values()
    counts.rename_axis("label").rename("images").to_csv(output / "class_distribution.csv")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(counts.index, counts.values, color="#2878a5")
    ax.set(xlabel="Number of images", title="Label distribution (multi-label counts)")
    fig.tight_layout(); fig.savefig(output / "class_distribution.png", dpi=160); plt.close(fig)

    examples: list[tuple[str, Path]] = []
    labels = df["Finding Labels"].astype(str).tolist()
    priorities = ["No Finding", "Cardiomegaly", "Effusion", "Pneumothorax", "Infiltration"]
    for label in priorities:
        for i, value in enumerate(labels):
            label_set = set(value.split("|"))
            if (label == "No Finding" and label_set == {label}) or label in label_set:
                examples.append((label, paths[i]))
                break
    if examples:
        fig, axes = plt.subplots(1, len(examples), figsize=(3.2 * len(examples), 3.8))
        axes = np.atleast_1d(axes)
        for ax, (label, path) in zip(axes, examples):
            with Image.open(path) as source:
                ax.imshow(source.convert("L"), cmap="gray")
            ax.set_title(label.replace("_", " "), fontsize=9)
            ax.axis("off")
        fig.suptitle("Representative chest X-rays")
        fig.tight_layout(); fig.savefig(output / "sample_xrays.png", dpi=160); plt.close(fig)
