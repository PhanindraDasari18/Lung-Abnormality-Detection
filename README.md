# Lung Abnormality Classification Using SVM

An educational multi-label chest X-ray classification project. It extracts HOG image features, standardizes them, reduces them with PCA, and trains one balanced RBF-kernel SVM per abnormality. Validation data is used to select a separate F1 threshold for each class. The final test set is held out by patient. The nonlinear model modestly improves held-out ROC-AUC over the linear baseline, but its macro F1 remains low and it is not suitable for medical use.

The uploaded dataset has been trained and evaluated. See [PROJECT_REPORT.md](PROJECT_REPORT.md) for the measured results and [artifacts/](artifacts/) for the model, metrics, and figures.

The interface uses a responsive teal and navy theme. It rejects images beyond the 90th-percentile distance calibrated on validation chest X-rays instead of always forcing a disease prediction. This stricter setting rejected 15.4% of the held-out test X-rays in this run and still cannot prove an accepted upload is a chest X-ray. For GitHub upload and public hosting steps, see [GITHUB_DEPLOYMENT.md](GITHUB_DEPLOYMENT.md).

## Live app

[Open LungLens](https://lung-abnormality-detection-ytsckck5z9b6nqpyyzbsfy.streamlit.app/)

The supplied data is arranged at the project root:

```text
labels.csv
images/
├── 00000001_000.png
└── ...
```

Images may also be nested below `images`; filenames are searched recursively. The dataset currently contains 14,999 PNG files. The assignment and teacher-guide PDFs mentioned in the original notes were not present in the workspace, so this implementation follows the detailed requirements pasted into the chat.

## Setup

Python 3.10 or newer is recommended.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train and evaluate

```powershell
python -m lung_svm.cli train --classifier rbf
```

The default classifier is an RBF SVM (`C=1`, `gamma=scale`); training takes longer than the linear baseline. The pipeline uses 96×96 grayscale images, HOG (9 orientations, 8×8 pixel cells, 2×2 cell blocks), 256 PCA components (or fewer when sample count requires it), and 70/15/15 patient-level train/validation/test splits. Use `--classifier linear` to train the older linear baseline. Set paths or resource options as needed:

```powershell
python -m lung_svm.cli train --labels D:\dataset\labels.csv --images D:\dataset\images --cache artifacts\hog_features.npy --output artifacts --pca-components 128
```

Training caches HOG features to avoid recomputing them on later runs. The cache is rebuilt if its image list or HOG configuration differs. Unreadable images are reported and skipped. Output files:

```text
artifacts/
├── class_distribution.csv
├── class_distribution.png
├── sample_xrays.png
├── lung_svm_model.joblib
├── per_class_metrics.csv
├── macro_metrics.csv
├── roc_curves.png
└── class_auc.png
```

`per_class_metrics.csv` also includes positive support, thresholds, and binary confusion counts. A class whose test data contains only one target value has an undefined AUC, recorded as blank/NaN. Macro metrics are unweighted averages across classes.

## Predict one image

```powershell
python -m lung_svm.cli predict path\to\chest_xray.png
```

The CLI rejects an image whose HOG/PCA representation is too far from the training X-rays; otherwise, it prints abnormalities above their validation-selected thresholds, or `No Finding` when none pass. SVM decision scores are margins, not calibrated probabilities.

## Run the optional demo

```powershell
streamlit run app.py
```

Train the model first; the app expects `artifacts/lung_svm_model.joblib`.

## Method and limitations

- `Finding Labels` is split on `|`; the 14 abnormalities are independent binary targets. `No Finding` is derived when all 14 labels are zero, not trained as a separate disease.
- All images for a patient stay within one split to reduce patient leakage. Group splitting is reproducible, but not label-stratified; inspect supports in the generated report, especially for rare findings.
- `class_weight="balanced"` addresses training imbalance. Per-label score cutoffs are tuned on validation data only, then evaluated on untouched test data. RBF SVM margins are not calibrated probabilities.
- An out-of-distribution distance gate rejects images beyond the 90th percentile of patient-held-out validation X-ray distances. It rejected 15.4% of held-out test X-rays in the current run and still needs external images to measure rejection of chairs, cars, and other objects. Unusual X-rays may also be rejected.
- HOG may miss subtle image patterns, dataset labels may be noisy, and `No Finding` does not guarantee that a patient is healthy. This system is not suitable for clinical decisions.

## Project layout

```text
lung_svm/       data loading, image features, model, EDA, metrics, and CLI
app.py          optional Streamlit prediction interface
labels.csv      supplied image-level labels
images/         supplied PNG chest X-rays
artifacts/      trained model, metrics, and plots
```
