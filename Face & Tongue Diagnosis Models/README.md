# 🩺 Face & Tongue Multi-Task Diagnosis Model

This module handles the multi-task feature extraction and classification pipeline for Chinese Traditional Medicine (CTM) face and tongue diagnosis. Designed to achieve high generalization under **limited sample size constraints**, this model maps localized CTM clinical indicators (118 finegrained labels) to categorical outputs.

---

## 🔬 Experimental Setup & Backbones

Due to dataset scale limitations, multiple modern computer vision architectures were evaluated to find the optimal balance between representation power and over-fitting resistance.

### Benchmark Comparison

| Backbone Family | Variant Tested | Performance / Generalization | Key Observations |
| :--- | :--- | :--- | :--- |
| **ResNet** | ResNet-50 / 101 | Baseline | Fast convergence, but prone to early overfitting on small classes. |
| **EfficientNet** | EfficientNet-B0 | Moderate | Low parameter count, but sensitive to hyperparameter tuning on noisy boundaries. |
| **DenseNet** | DenseNet-121 | Moderate | Feature reuse helped, but memory footprint during training was relatively high. |
| **ConvNeXt v1** | ConvNeXt-Tiny / Small | High | Significant bump in feature representation over traditional CNNs. |
| **ConvNeXt v2** | **ConvNeXt v2 (Selected)** | **Best Overall** | **Superior feature retention and cross-head accuracy under sparse data.** |

> **Selected Architecture:** **ConvNeXt v2** with Exponential Moving Average (EMA) weight smoothing achieved the highest mean F1-score across all task heads.

---

## 🎨 Data Augmentation Strategy

To mitigate overfitting caused by limited training samples, three complementary augmentation techniques were integrated into the input pipeline:

1. **RandAugment:** Applies automated, randomly chained geometric and color transformations (contrast, brightness, sharpness) to build invariant representations against varying clinical capture environments.
2. **MixUp:** Blends image pairs and their corresponding ground-truth label distributions linearly ($\lambda \cdot x_1 + (1-\lambda)x_2$), smoothing decision boundaries across CTM symptom classes.
3. **CutMix:** Replaces rectangular image regions with patches from other samples while adjusting label weights proportionally, encouraging the model to focus on subtle global/local CTM features rather than localized artifacts.

---

## 📐 Validation Scheme: 5-Fold Cross-Validation

To ensure robust performance estimates and prevent data leakage:

* **5-Fold Cross-Validation:** The entire dataset is partitioned into 5 folds to select the optimal training strategies.
* **Metric Aggregation:** Models are evaluated across all 5 folds, reporting out-of-fold mean precision, recall, and top-1/3 accuracy .
* **Checkpoint Selection:** Final production weights (`face_ema.pt` / `tongue_ema.pt`) are generated using Exponential Moving Average (EMA) decay during training on the whole dataset.

