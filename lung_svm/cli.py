from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from . import DISEASES
from .data import load_labels, patient_splits, resolve_image
from .eda import make_eda
from .evaluate import evaluate
from .features import extract_many, image_hog
from .model import decision_scores, fit_classifiers, fit_preprocessor, tune_thresholds
from .ood import fit_ood_reference, ood_distances


def train(args):
    df = load_labels(args.labels)
    paths = [resolve_image(args.images, name) for name in df["Image Index"].astype(str)]
    features, valid_idx = extract_many(paths, args.cache, args.image_size)
    df = df.iloc[valid_idx].reset_index(drop=True)
    valid_paths = [paths[i] for i in valid_idx]
    x = np.asarray(features, dtype=np.float32)
    splits = patient_splits(df, args.seed)
    y = df[list(DISEASES)].to_numpy(dtype=np.uint8)
    make_eda(df, valid_paths, args.output)
    print(f"Usable images: {len(df):,}; patients: {df['Patient ID'].nunique():,}")
    for name, idx in splits.items():
        print(f"{name:10s}: {len(idx):5,d} images, {df.iloc[idx]['Patient ID'].nunique():4,d} patients")
    scaler, pca, x_train = fit_preprocessor(x[splits["train"]], args.pca_components)
    x_val = pca.transform(scaler.transform(x[splits["validation"]])).astype(np.float32)
    x_test = pca.transform(scaler.transform(x[splits["test"]])).astype(np.float32)
    reference, ood_threshold, val_ood = fit_ood_reference(x_train, x_val, seed=args.seed)
    test_ood = ood_distances(reference, x_test)
    print(f"OOD distance gate (99th percentile of validation X-rays): {ood_threshold:.3f}; "
          f"held-out X-rays rejected: {(test_ood > ood_threshold).mean():.1%}")
    models = fit_classifiers(x_train, y[splits["train"]])
    val_scores = decision_scores(models, x_val)
    thresholds = tune_thresholds(y[splits["validation"]], val_scores)
    test_scores = decision_scores(models, x_test)
    table = evaluate(y[splits["test"]], test_scores, thresholds, args.output)
    print("\nTest metrics by class:")
    print(table[["disease", "support", "roc_auc", "precision", "recall", "f1"]].to_string(index=False))
    print("\nMacro averages:")
    print(table[["roc_auc", "precision", "recall", "f1"]].mean().to_string())
    bundle = {"models": models, "scaler": scaler, "pca": pca, "thresholds": thresholds,
              "diseases": DISEASES, "image_size": args.image_size,
              "ood_reference": reference, "ood_threshold": ood_threshold}
    Path(args.output).mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, Path(args.output) / "lung_svm_model.joblib", compress=3)
    print(f"\nSaved model and reports under {Path(args.output).resolve()}")


def predict(args):
    bundle = joblib.load(args.model)
    feature = image_hog(args.image, bundle["image_size"])[None, :]
    x = bundle["pca"].transform(bundle["scaler"].transform(feature))
    if "ood_reference" in bundle:
        distance = float(ood_distances(bundle["ood_reference"], x)[0])
        if distance > bundle["ood_threshold"]:
            print(f"Prediction: rejected (image is outside the training X-ray range; distance {distance:.3f})")
            print("Please upload a chest X-ray similar to the images used to train this educational model.")
            return
    scores = decision_scores(bundle["models"], x)[0]
    selected = [(name, float(score)) for name, score, threshold in zip(
        bundle["diseases"], scores, bundle["thresholds"]) if score >= threshold]
    print(f"Image: {Path(args.image).name}")
    if not selected:
        print("Prediction: No Finding")
    else:
        print("Predicted abnormalities:")
        for name, score in sorted(selected, key=lambda item: item[1], reverse=True):
            print(f"  {name}: decision score {score:.3f}")
    print("Note: educational model output; not a medical diagnosis.")


def main():
    parser = argparse.ArgumentParser(description="HOG + PCA + one-vs-rest SVM chest X-ray classifier")
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train", help="Train and evaluate using patient-level splits")
    fit.add_argument("--labels", type=Path, default=Path("labels.csv"))
    fit.add_argument("--images", type=Path, default=Path("images"))
    fit.add_argument("--cache", type=Path, default=Path("artifacts/hog_features.npy"))
    fit.add_argument("--output", type=Path, default=Path("artifacts"))
    fit.add_argument("--image-size", type=int, default=96)
    fit.add_argument("--pca-components", type=int, default=256)
    fit.add_argument("--seed", type=int, default=42)
    fit.set_defaults(func=train)
    pred = sub.add_parser("predict", help="Predict abnormalities for one X-ray")
    pred.add_argument("image", type=Path)
    pred.add_argument("--model", type=Path, default=Path("artifacts/lung_svm_model.joblib"))
    pred.set_defaults(func=predict)
    args = parser.parse_args()
    if args.command == "train" and not args.labels.is_file():
        parser.error(f"Labels CSV not found: {args.labels}")
    if args.command == "train" and not args.images.is_dir():
        parser.error(f"Image directory not found: {args.images}")
    if args.command == "predict" and not args.image.is_file():
        parser.error(f"Image not found: {args.image}")
    args.func(args)


if __name__ == "__main__":
    main()
