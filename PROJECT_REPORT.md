# Lung Abnormality Classification Using HOG and SVM

## Abstract

This project implements multi-label chest X-ray classification using Histogram of Oriented Gradients (HOG), feature standardization, Principal Component Analysis (PCA), and one-vs-rest RBF-kernel Support Vector Machines. It predicts 14 abnormality labels. The supplied 14,999 images were divided by patient into training, validation, and test subsets. Class-specific decision thresholds were tuned on validation data. On the held-out test data, macro ROC-AUC was **0.6688** and macro F1 was **0.1342**. The nonlinear model improves ROC-AUC over the earlier linear baseline (0.6402), but macro F1 remains very low and false positives are common. This is an educational baseline, not a medical diagnostic system.

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
4. Fit one RBF-kernel `SVC` per abnormality with `C=1`, `gamma="scale"`, and `class_weight="balanced"`.
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
| Atelectasis | 210 | 0.6991 | 0.2267 | 0.1619 | 0.1889 |
| Cardiomegaly | 61 | 0.8788 | 0.1901 | 0.4426 | 0.2660 |
| Consolidation | 94 | 0.6366 | 0.0591 | 0.3723 | 0.1020 |
| Edema | 35 | 0.7673 | 0.0526 | 0.0857 | 0.0652 |
| Effusion | 232 | 0.7480 | 0.2565 | 0.4267 | 0.3204 |
| Emphysema | 54 | 0.7214 | 0.0309 | 0.0926 | 0.0463 |
| Fibrosis | 68 | 0.6851 | 0.1091 | 0.1765 | 0.1348 |
| Hernia | 7 | 0.6478 | 0.0000 | 0.0000 | 0.0000 |
| Infiltration | 401 | 0.6251 | 0.2577 | 0.4190 | 0.3191 |
| Mass | 75 | 0.5564 | 0.0337 | 0.1733 | 0.0564 |
| Nodule | 107 | 0.5380 | 0.0586 | 0.3645 | 0.1009 |
| Pleural Thickening | 86 | 0.6196 | 0.0813 | 0.1163 | 0.0957 |
| Pneumonia | 43 | 0.5444 | 0.0313 | 0.0465 | 0.0374 |
| Pneumothorax | 125 | 0.6950 | 0.0808 | 0.7120 | 0.1452 |
| **Macro average** | — | **0.6688** | **0.1049** | **0.2564** | **0.1342** |

Macro averages weight all 14 labels equally. On this split, Cardiomegaly ROC-AUC rises from 0.7974 to 0.8788 and F1 from 0.1849 to 0.2660 compared with the linear baseline. The aggregate F1 gain is small (0.1308 to 0.1342), and macro precision is only 0.1049, so false positives remain common. Hernia has only seven positive test cases and no true positives at the selected threshold. These results are not strong diagnostic performance.

## 5. Generated figures and files

- [Class distribution](artifacts/class_distribution.png)
- [Representative X-rays](artifacts/sample_xrays.png)
- [ROC curves](artifacts/roc_curves.png)
- [AUC by class](artifacts/class_auc.png)
- [Per-class metrics and confusion counts](artifacts/per_class_metrics.csv)
- [Macro metrics](artifacts/macro_metrics.csv)
- [Serialized model for prediction](artifacts/lung_svm_model.joblib)

## 6. Limitations and responsible use

HOG is a hand-crafted representation and may miss subtle disease patterns. The label distribution is severely imbalanced, and labels can be noisy. Validation-based thresholds can favor recall at the expense of precision. The test set contains few examples of rare conditions. RBF SVM margins are not probabilities. The model has not been clinically validated, and `No Finding` does not guarantee a healthy patient. Do not use its output to make healthcare decisions.

## 7. Reproducibility

The run used Python 3.14.4, NumPy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, scikit-image 0.26.0, and Pillow 12.3.0, with random seed 42, 96×96 image preprocessing, up to 256 PCA components, RBF SVMs, and patient-level splits. Run `python -m lung_svm.cli train --classifier rbf` to regenerate the cached features, model, metrics, and plots. Run `python -m lung_svm.cli predict path\to\image.png` to classify a new image. See [README.md](README.md) for environment setup and the optional Streamlit interface.

The separate assignment and teacher-guide PDFs mentioned in the original project notes were not present in the workspace at the time of implementation; this report follows the requirements pasted into the conversation.
