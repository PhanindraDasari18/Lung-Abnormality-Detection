# Lung Abnormality Classification Using SVM

> Replace bracketed fields with measured values after training. Do not report example or invented scores as results.

## Abstract

This project implements a multi-label chest X-ray classifier using Histogram of Oriented Gradients (HOG), Principal Component Analysis (PCA), and one-vs-rest linear Support Vector Machines. It predicts 14 abnormalities. A patient-level split is used to reduce leakage between training and evaluation. The model is an educational experiment and is not a diagnostic device.

## 1. Introduction

Describe chest radiography, the classification task, and the project objective. Explain that multiple findings may be present in a single image.

## 2. Dataset

Dataset source: [fill in]. Images available: [count]. Unique patients: [count]. Labels CSV: [filename/version]. Describe image views, label provenance, missing files, and any filtering. The project derives `No Finding` when all 14 abnormality targets are absent.

## 3. Methodology

Images are converted to grayscale and resized to 96×96. HOG uses 9 orientations, 8×8 pixel cells, and 2×2 cell blocks. Features are standardized and reduced with PCA, fitted on training data only. Fourteen balanced one-vs-rest `LinearSVC` classifiers are fitted. Per-class decision thresholds are selected by F1 on validation data.

Data is split approximately 70/15/15 with `Patient ID` groups so that one patient's images do not cross subsets. The split is not label-stratified; report actual image, patient, and positive-label counts for each subset.

## 4. Implementation

Environment: [Python and package versions]. Hardware: [CPU/GPU, RAM]. Preprocessing and model settings: [record command and any changed settings]. Training duration: [measured].

## 5. Results

Insert `artifacts/per_class_metrics.csv`, `artifacts/macro_metrics.csv`, and `artifacts/roc_curves.png` after a completed run. Report test metrics only from the held-out test subset. AUC is undefined for classes with only one target value in test data.

| Metric | Test macro average |
|---|---:|
| ROC-AUC | [measured] |
| Precision | [measured] |
| Recall | [measured] |
| F1 | [measured] |

Discuss rare class support, threshold effects, false positives, false negatives, and the gap between validation and test performance.

## 6. Limitations and ethics

HOG may not represent subtle pathology; the labels can be noisy and imbalanced; patient-level random group splits may vary in rare-label support. This model is a learning project, not clinically validated, and must not guide diagnosis or treatment.

## 7. Conclusion

Summarize the implemented pipeline and measured findings after running the experiment. Suggest future work such as better label-aware group partitioning and learned image representations, with a new patient-held-out evaluation.
