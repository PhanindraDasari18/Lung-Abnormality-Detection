from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile
from skimage.feature import hog
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


def image_hog(path: str | Path, size: int = 96) -> np.ndarray:
    """Read a chest X-ray as grayscale and extract a normalized HOG vector."""
    with Image.open(path) as source:
        image = source.convert("L").resize((size, size), Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
    return hog(
        pixels, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
        block_norm="L2-Hys", feature_vector=True,
    ).astype(np.float32)


def extract_many(paths: list[Path], cache_path: str | Path, size: int = 96) -> tuple[np.ndarray, np.ndarray]:
    """Load or build a feature cache; missing/unreadable images are skipped."""
    cache_path = Path(cache_path)
    index_path = cache_path.with_suffix(".images.txt")
    rows_path = cache_path.with_suffix(".indices.npy")
    meta_path = cache_path.with_suffix(".meta.txt")
    metadata = f"size={size};orientations=9;pixels_per_cell=8;cells_per_block=2;block_norm=L2-Hys"
    expected = [str(p.resolve()) for p in paths]
    if cache_path.exists() and index_path.exists():
        cached_paths = index_path.read_text(encoding="utf-8").splitlines()
        if rows_path.exists() and meta_path.exists() and meta_path.read_text(encoding="utf-8") == metadata:
            cached_rows = np.load(rows_path)
            if cached_paths == [expected[i] for i in cached_rows]:
                return np.load(cache_path, mmap_mode="r"), cached_rows
    features: list[np.ndarray] = []
    valid_indices: list[int] = []
    def extract(item: tuple[int, Path]):
        i, path = item
        try:
            return i, image_hog(path, size=size), None
        except (OSError, ValueError) as exc:
            return i, None, exc

    workers = max(1, min(8, (os.cpu_count() or 2) - 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, feature, error in tqdm(
            pool.map(extract, enumerate(paths)), total=len(paths),
            desc=f"Extracting HOG ({workers} workers)", unit="image",
        ):
            if error is not None:
                print(f"Skipping unreadable image {paths[i]}: {error}")
                continue
            features.append(feature)
            valid_indices.append(i)
    if not features:
        raise ValueError("No readable X-ray images were found. Check --images and filenames.")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, np.stack(features))
    valid_paths = [expected[i] for i in valid_indices]
    index_path.write_text("\n".join(valid_paths), encoding="utf-8")
    np.save(rows_path, np.asarray(valid_indices, dtype=np.int64))
    meta_path.write_text(metadata, encoding="utf-8")
    return np.load(cache_path, mmap_mode="r"), np.asarray(valid_indices, dtype=np.int64)
