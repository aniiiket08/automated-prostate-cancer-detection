<div align="center">

# Prostate Cancer Detection

**Texture features and gradient boosting on histopathology images, compressed for on-device inference.**

![Python](https://img.shields.io/badge/python-3.10-3b4252?style=flat-square&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/xgboost-classifier-3b4252?style=flat-square)
![TensorFlow](https://img.shields.io/badge/tensorflow-keras-3b4252?style=flat-square&logo=tensorflow&logoColor=white)
![TFLite](https://img.shields.io/badge/tflite-edge%20deployment-3b4252?style=flat-square)

[Overview](#-overview) · [Pipeline](#-pipeline) · [Results](#-results) · [Approach](#-approach) · [Dataset](#-dataset) · [Structure](#-project-structure) · [Reference](#-reference)

</div>

<br>

## 🔬 Overview

Reading prostate biopsy slides is slow and subjective, and the same tissue can receive different grades from different pathologists. Deep learning is the obvious fix, but it usually needs large annotated datasets, which histopathology rarely has.

This project asks a narrower question: on a modest dataset, can handcrafted texture features with an ensemble classifier hold up against deep CNNs, and can the best model be shrunk enough to run on a phone?

On SICAPv2, a boosted tree model on GLCM and LBP features outperformed every deep network tested, at a fraction of the compute.

|  |  |
|---|---|
| **Task** | Binary classification, cancerous vs. non-cancerous tissue |
| **Dataset** | SICAPv2, 18,783 patches |
| **Best model** | XGBoost on GLCM + LBP features |
| **Accuracy / ROC-AUC** | 0.8437 / 0.9243 |
| **Baselines** | MobileNetV2, MobileNetV3, EfficientNetB0 |
| **Deployment** | TFLite with quantization and pruning |

<br>

## 🧭 Pipeline

<p align="center">
  <img src="assets/methodology_pipeline.png" alt="Methodology pipeline: dataset, preprocessing, feature extraction, model, evaluation, prediction" width="860">
</p>

<br>

## 📊 Results

### Handcrafted features with XGBoost

Trained on 256 x 256 tissue patches.

| Feature set | Accuracy | ROC-AUC | Sensitivity | Specificity | F1 (macro) |
|---|:---:|:---:|:---:|:---:|:---:|
| 🥇 **GLCM + LBP** | **0.8437** | **0.9243** | 0.8462 | 0.8418 | 0.8420 |
| GLCM + HOG | 0.6984 | 0.7607 | 0.6130 | 0.7630 | 0.6880 |
| HOG + LBP | 0.6912 | 0.7416 | 0.5690 | 0.7830 | 0.6910 |

Each descriptor alone (LBP, GLCM, HOG) landed between 0.67 and 0.71 accuracy. Combining GLCM and LBP gave the jump, since one captures pixel-level texture statistics and the other captures cell-boundary and micro-structure detail. HOG added mostly noise, because prostate tissue lacks the consistent edge orientation it is designed for.

The deep baselines (MobileNetV2, MobileNetV3, EfficientNetB0) were trained on the same splits. At this dataset size they trailed the XGBoost model and overfit more readily.

For context, 84.37% accuracy edges out the 83.04% DenseNet121 baseline reported by Rao and RaviSankar (2024) on the same dataset.

<p align="center">
  <img src="assets/roc_curves.png" alt="ROC curves comparing XGBoost feature combinations against deep CNNs" width="640">
  &nbsp;
  <img src="assets/confusion_matrix.png" alt="Confusion matrix for the XGBoost GLCM+LBP model" width="360">
</p>

### 📱 Edge deployment

The best deep model was converted to TFLite and compressed with dynamic-range, float16, and full INT8 quantization, plus pruning.

| Variant | ROC-AUC |
|---|:---:|
| Float TFLite | 0.9458 |
| Pruned Keras | 0.9458 |
| Float16 quantized TFLite | 0.9456 |
| Dynamic-range quantized TFLite | 0.9374 |
| Pruned + dynamic-range quantized TFLite | 0.9360 |

Compression costs under one point of AUC in exchange for a smaller model and lower on-device latency.

<br>

## 🔧 Approach

**1. Preprocessing**
Whole-slide images are cut into non-overlapping 256 x 256 patches, each paired with a binary lesion mask. Patches are tiled into 32 x 32 blocks, and tiles that are more than about 95% background are dropped.

**2. Feature extraction**
Three handcrafted descriptors per patch:

| Descriptor | Captures | Settings |
|---|---|---|
| GLCM | Second-order texture: contrast, homogeneity, energy, correlation | Multiple distances and orientations |
| LBP | Micro-texture around nuclei and cell boundaries | Radius 1, 8 neighbors |
| HOG | Macro-structure: edges and gradients | Default orientation bins |

**3. Classification**
XGBoost on the combined feature vectors, benchmarked against the deep CNNs.

```
n_estimators=250   learning_rate=0.05   max_depth=6   subsample=0.8
```

<details>
<summary><b>Model architectures</b></summary>

<br>

<p align="center">
  <img src="assets/xgboost_architecture.png" alt="XGBoost architecture" width="560">
</p>

<p align="center">
  <img src="assets/mobilenetv2_architecture.png" alt="MobileNetV2 architecture" width="680">
</p>

</details>

**4. Compression**
The strongest deep model is converted to TFLite, quantized and pruned, then benchmarked for inference time and weight size. The end target is an Android inference app.

<br>

## 🧬 Dataset

[SICAPv2](https://data.mendeley.com/datasets/9xxm58dvs3/1) contains 18,783 RGB prostate tissue patches with pixel-level cancer masks.

| Class | Patches |
|---|:---:|
| Lesion present | 8,063 |
| No lesion | 10,720 |

<br>

## 🧰 Tech stack

Python, scikit-learn, scikit-image, XGBoost, TensorFlow / Keras, OpenCV, TensorFlow Lite, TensorFlow Model Optimization

<br>

## 📁 Project structure

```
.
├── assets/
│   ├── methodology_pipeline.png
│   ├── xgboost_architecture.png
│   ├── mobilenetv2_architecture.png
│   ├── confusion_matrix.png
│   └── roc_curves.png
├── notebooks/
│   └── prostate_cancer_ml_eis_project.ipynb   features, models, quantization, benchmarking
├── paper/
│   └── prostate_cancer_xgboost.pdf            write-up of the approach and results
└── README.md
```

<br>

## 📄 Reference

Patil A., Revankar S., Patil V., Agasgekar A., Akkasaligar P. T., Gavade P. A.
*Automated System for Prostate Cancer Detection from Histopathology Images Using XGBoost.*
Department of CSE (AI), KLE Technological University, Dr. M. S. Sheshgiri Campus, Belagavi.
