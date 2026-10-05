# Lung Abnormality Classification Using HOG and SVM

## Abstract

This project implements a multi-label chest X-ray classification baseline using Histogram of Oriented Gradients (HOG), feature standardization, Principal Component Analysis (PCA), and one-vs-rest linear Support Vector Machines. It predicts 14 abnormality labels. The supplied 14,999 images were divided by patient into training, validation, and test subsets. Class-specific decision thresholds were tuned on validation data. On the held-out test data, macro ROC-AUC was **0.6402** and macro F1 was **0.1308**. These results show modest discrimination and low thresholded classification performance; the model is an educational baseline and not a medical diagnostic system.

## 1. Objective

Given a chest X-ray, predict any present abnormalities from Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural Thickening, Pneumonia, and Pneumothorax. Because an image may have more than one finding, each abnormality is treated as an independent binary target. `No Finding` is reported when no abnormality passes its decision threshold.

## 2. Dataset

The project folder contains `labels.csv` and **14,999 PNG images**. The CSV contains 14,999 image records and 3,923 unique patient IDs. All listed images were found and read successfully during feature extraction. The per-label counts below are multi-label image counts, so their sum is greater than the image count.

| Label | Images |
|---|---:|
| No Finding | 8,753 |
| Infiltration | 2,260 |
| Effusion | 1,382 |
| Atelectasis | 1,350 |
| Pneumothorax | 670 |
| Nodule | 666 |
| Consolidation | 557 |
| Mass | 470 |
| Pleural Thickening | 468 |
| Cardiomegaly | 396 |
| Fibrosis | 379 |
| Emphysema | 318 |
| Edema | 196 |
| Pneumonia | 184 |
| Hernia | 46 |

`No Finding` is derived when all 14 encoded abnormalities equal zero. It is not trained as a separate classifier target.

## 3. Method

1. Read each image as grayscale, resize it to 96×96, and scale pixel values to [0, 1].
2. Extract HOG features using 9 orientations, 8×8 pixel cells, 2×2 cell blocks, and L2-Hys normalization.
3. Standardize features and fit PCA with up to 256 components using training data only.
4. Fit one `LinearSVC` per abnormality with `class_weight="balanced"` and `max_iter=20000`.
5. Select each abnormality's decision threshold by validation-set F1. These margins are not calibrated probabilities.
6. Evaluate once on the held-out test patients using ROC-AUC, precision, recall, and F1.

The patient-group split produced:

| Split | Images | Patients |
|---|---:|---:|
| Train | 10,699 | 2,746 |
| Validation | 1,926 | 588 |
| Test | 2,374 | 589 |

Patient groups are disjoint between all splits. Group splitting is not label-stratified, so exact percentages vary and rare labels can have small test support.

## 4. Test results

| Abnormality | Test positives | ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Atelectasis | 210 | 0.6987 | 0.1936 | 0.3762 | 0.2557 |
| Cardiomegaly | 61 | 0.7974 | 0.1897 | 0.1803 | 0.1849 |
| Consolidation | 94 | 0.6499 | 0.0725 | 0.2979 | 0.1167 |
| Edema | 35 | 0.7187 | 0.0575 | 0.1429 | 0.0820 |
| Effusion | 232 | 0.7139 | 0.1925 | 0.5302 | 0.2824 |
| Emphysema | 54 | 0.7126 | 0.0393 | 0.7963 | 0.0750 |
| Fibrosis | 68 | 0.6755 | 0.1231 | 0.1176 | 0.1203 |
| Hernia | 7 | 0.5020 | 0.0000 | 0.0000 | 0.0000 |
| Infiltration | 401 | 0.6376 | 0.2630 | 0.2893 | 0.2755 |
| Mass | 75 | 0.5406 | 0.0461 | 0.1333 | 0.0685 |
| Nodule | 107 | 0.5576 | 0.0484 | 0.3364 | 0.0846 |
| Pleural Thickening | 86 | 0.5855 | 0.0627 | 0.2674 | 0.1015 |
| Pneumonia | 43 | 0.5252 | 0.0179 | 0.1628 | 0.0323 |
| Pneumothorax | 125 | 0.6475 | 0.0942 | 0.3920 | 0.1519 |
| **Macro average** | — | **0.6402** | **0.1000** | **0.2873** | **0.1308** |

Macro averages weight all 14 labels equally. Hernia has only seven positive test examples, so its score is particularly uncertain. The high Emphysema recall comes with very low precision, illustrating the false-positive cost of per-class thresholding. The result should not be described as strong diagnostic performance.

## 5. Generated figures and files

- [Class distribution](artifacts/class_distribution.png)
- [Representative X-rays](artifacts/sample_xrays.png)
- [ROC curves](artifacts/roc_curves.png)
- [AUC by class](artifacts/class_auc.png)
- [Per-class metrics and confusion counts](artifacts/per_class_metrics.csv)
- [Macro metrics](artifacts/macro_metrics.csv)
- [Serialized model for prediction](artifacts/lung_svm_model.joblib)

## 6. Limitations and responsible use

HOG is a hand-crafted representation and may miss subtle disease patterns. The label distribution is severely imbalanced, and labels can be noisy. Validation-based thresholds can favor recall at the expense of precision. The test set contains few examples of rare conditions. The model has not been clinically validated, and `No Finding` does not guarantee a healthy patient. Do not use its output to make healthcare decisions.

## 7. Reproducibility

The run used Python 3.14.4, NumPy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, scikit-image 0.26.0, and Pillow 12.3.0, with random seed 42, 96×96 image preprocessing, up to 256 PCA components, and patient-level splits. Run `python -m lung_svm.cli train` to regenerate the cached features, model, metrics, and plots. Run `python -m lung_svm.cli predict path\to\image.png` to classify a new image. See [README.md](README.md) for environment setup and the optional Streamlit interface.

The separate assignment and teacher-guide PDFs mentioned in the original project notes were not present in the workspace at the time of implementation; this report follows the requirements pasted into the conversation.
