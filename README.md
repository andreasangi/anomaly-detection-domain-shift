# Industrial Anomaly Detection under Domain Shift

An industrial image anomaly detection and localization system based on **ResNet-50** and **PatchCore**. This project evaluates the model's robustness against domain shifts—such as lighting variations, sensor noise, and perspective distortions—and proposes a memory bank augmentation strategy for mitigation.

The system is trained in an unsupervised manner using only defect-free images and evaluated on the **metal_nut** category of the **MVTec AD** dataset.
The full project report is available here: [Report.pdf](./Report.pdf)

## Table of Contents

- [Industrial Anomaly Detection under Domain Shift](#industrial-anomaly-detection-under-domain-shift)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Dataset](#dataset)
  - [Domain Shift](#domain-shift)
  - [Robustness Mitigation](#robustness-mitigation)
    - [Baseline Results](#baseline-results)
    - [Results After Full-Range Augmentation](#results-after-full-range-augmentation)
    - [Results After Mild (0.6) Augmentation and Generalization Test](#results-after-mild-06-augmentation-and-generalization-test)
  - [Repository Structure](#repository-structure)
    - [`patchcore.ipynb`](#patchcoreipynb)
    - [`domain-shift/`](#domain-shift-1)
  - [Requirements](#requirements)
  - [Usage](#usage)
  - [Results](#results)
  - [Perturbation Details](#perturbation-details)
    - [Exposure](#exposure)
    - [White Balance](#white-balance)
    - [Noise](#noise)
    - [Contrast](#contrast)
    - [Perspective](#perspective)
    - [Specular Highlights](#specular-highlights)
    - [Held-out transforms (evaluated, not used for memory bank augmentation)](#held-out-transforms-evaluated-not-used-for-memory-bank-augmentation)
    - [Other implemented transforms (not evaluated at all)](#other-implemented-transforms-not-evaluated-at-all)
  - [Authors](#authors)
  - [Citations \& Licenses](#citations--licenses)

## Overview

The pipeline consists of:

1. **Feature extraction** using a pre-trained ResNet-50.
2. Extraction of intermediate features from `layer2` and `layer3`.
3. Construction of a **memory bank** containing features from normal training images.
4. Comparison of test-image patches with the memory bank using Euclidean distance.
5. Computation of an image-level **anomaly score**.
6. Generation of **heatmaps** for approximate defect localization.

The classification threshold is estimated from a validation set containing only normal images.

## Dataset

The project uses the **MVTec Anomaly Detection (MVTec AD)** dataset, focusing on the `metal_nut` category.

The dataset contains:

* `good` - defect-free samples
* `bent`
* `color`
* `flip`
* `scratch`

The original training set contains only normal images and is divided into:

* **80% training** - used to build the memory bank
* **20% validation** - used to determine the anomaly threshold

## Domain Shift

The robustness of the anomaly detector is evaluated under different
acquisition conditions, simulating variations that can occur in a real
industrial camera setup.

Six transformations were used for the core robustness evaluation and for
memory bank mitigation:

* Exposure changes
* White balance variations
* Gaussian and salt-and-pepper noise
* Contrast modifications
* Perspective transformations
* Specular highlights

Two additional transformations, Gamma Correction and Affine misalignment,
were also implemented and evaluated, but were kept out of the memory bank
augmentation. They are used exclusively as a held-out test of
generalization to acquisition conditions the mitigated memory bank was
never exposed to (see [Robustness Mitigation](#robustness-mitigation)).

These transformations simulate realistic variations in illumination, sensor
quality, camera viewpoint, and surface reflectance, without introducing new
defects. All parameters are randomly sampled within physically motivated
ranges for each transformation. Baseline domain shift results use the
original memory bank and threshold, unchanged from nominal evaluation.

See [Perturbation Details](#perturbation-details) below for the model and
parameter ranges used for each transformation.

## Robustness Mitigation

To reduce the accuracy and precision drop observed under domain shift,
particularly for Contrast and Perspective, the memory bank was enriched
with augmented normal samples. For each of the 176 training images, one
randomly selected perturbation was applied, and the result was added to
the memory bank alongside the original, using
[`augment_train_set.py`](./domain-shift/augment_train_set.py). The anomaly
threshold was kept unchanged, estimated as before from the clean
validation set.

Using more than one augmented copy per image (2 or 3) was also tested, but
consistently reduced performance, likely because a smaller proportion of
clean normal patches survives the fixed-size random subsampling once the
source pool grows. One augmentation per image was kept as the final
configuration.

The validation percentile used to compute the threshold was also varied
across `{85, 87.5, 90, 92.5, 95}`. Lowering it improved Recall slightly on
some conditions but reduced Accuracy and Precision across the board,
including on the original test set. The 90th percentile remained the best
overall choice.

A second experiment used `augment_train_set.py`'s `AUGMENTATION_SCALE`
parameter to shrink every transform's range to 60% of its original width
around the same midpoint, so the memory bank is exposed only to a milder
version of each perturbation than the one actually used at test time. On
top of this mildly-augmented memory bank, two further checks were run:
Gamma Correction and Affine, neither used during augmentation at all, and
a combined Exposure + Perspective condition, to test generalization beyond
the transforms and magnitudes seen during mitigation.

### Baseline Results

Six core conditions plus Gamma and Affine (evaluated but not used for
augmentation). The average row is computed over the six core conditions
only (Exposure through Specular), so it stays comparable across all three
tables in this section.

| Condition | Acc. | Prec. | Rec. | F1 |
| --- | ---: | ---: | ---: | ---: |
| Original | 91.3% | 96.6% | 92.5% | 94.5% |
| Exposure | 89.0% | 91.6% | 95.0% | 93.3% |
| White Balance | 89.0% | 89.6% | 97.5% | 93.4% |
| Noise | 90.0% | 91.6% | 96.3% | 93.9% |
| Contrast | 85.0% | 84.2% | 100% | 91.4% |
| Perspective | 85.0% | 87.4% | 95.0% | 91.0% |
| Specular | 89.0% | 90.6% | 96.3% | 93.3% |
| **Average (shifted, 6 core conditions)** | **87.8%** | **89.2%** | **96.7%** | **92.7%** |
| Gamma *(held out)* | 87.0% | 87.6% | 97.5% | 92.3% |
| Affine *(held out)* | 83.0% | 82.5% | 100% | 90.4% |

Contrast and Perspective show the largest drops in Accuracy and Precision
compared with the nominal condition.

### Results After Full-Range Augmentation

Memory bank augmented using the same parameter ranges as the test
conditions (`AUGMENTATION_SCALE = 1.0`).

| Condition | Acc. | Prec. | Rec. | F1 |
| --- | ---: | ---: | ---: | ---: |
| Original | 93.0% | 98.9% | 92.5% | 95.6% |
| Exposure | 89.0% | 92.6% | 93.8% | 93.2% |
| White Balance | 93.0% | 97.4% | 93.8% | 95.5% |
| Noise | 92.0% | 97.4% | 92.5% | 94.9% |
| Contrast | 94.0% | 95.1% | 97.5% | 96.3% |
| Perspective | 91.0% | 97.3% | 91.3% | 94.2% |
| Specular | 91.0% | 96.1% | 92.5% | 94.3% |
| **Average (shifted)** | **91.7%** | **96.0%** | **93.6%** | **94.7%** |

Average Accuracy increases from 87.8% to 91.7%, Precision from 89.2% to
96.0%, and F1-score from 92.7% to 94.7%. Recall decreases slightly from
96.7% to 93.6%, indicating a trade-off between anomaly detection and
false-positive reduction. Contrast and Perspective, the weakest conditions
before mitigation, improve the most.

### Results After Mild (0.6) Augmentation and Generalization Test

Memory bank augmented using only 60% of each transform's original range
(`AUGMENTATION_SCALE = 0.6`), leaving the outer 40% of every range genuinely
unseen during training. Gamma, Affine, and the combined Exposure +
Perspective condition were never used for augmentation at all, and are
included here purely as a generalization check. The average row is
computed over the same six core conditions as the other two tables.

| Condition | Acc. | Prec. | Rec. | F1 |
| --- | ---: | ---: | ---: | ---: |
| Original | 92.2% | 97.7% | 92.5% | 95.0% |
| Exposure | 90.0% | 92.6% | 93.8% | 93.2% |
| White Balance | 93.0% | 96.1% | 91.3% | 93.6% |
| Noise | 88.0% | 97.2% | 87.5% | 92.1% |
| Contrast | 92.0% | 90.9% | 100% | 95.2% |
| Perspective | 90.0% | 96.1% | 91.3% | 93.6% |
| Specular | 94.0% | 97.4% | 95.0% | 96.2% |
| **Average (shifted, 6 core conditions)** | **91.2%** | **95.1%** | **93.2%** | **94.0%** |
| Gamma Correction *(held out)* | 86.0% | 92.3% | 90.0% | 91.1% |
| Affine *(held out)* | 87.0% | 86.8% | 98.8% | 92.4% |
| Exposure + Perspective *(held out, combined)* | 87.7% | 87.5% | 94.6% | 90.9% |

Using only 60% of the original range for augmentation recovers most of the
robustness improvement seen with full-range augmentation (average F1
94.0% versus 94.7%), while training on a narrower, genuinely different
distribution than the one evaluated at test time. The held-out results are
mixed: Affine improves over its unmitigated baseline (F1 90.4% → 92.4%),
while Gamma stays close to its baseline (F1 92.3% → 91.1%). This indicates
that memory bank augmentation can generalize to unseen acquisition
conditions, but not uniformly across all of them.

Increasing the number of augmented copies per image (2 or 3) was also
tested here and, as with full-range augmentation, consistently reduced
performance, so a single augmented copy per image was kept as the final
configuration.

## Repository Structure

```text
anomaly-detection-domain-shift/
│
├── patchcore.ipynb
│
├── domain-shift/
│   ├── domain_shift_functions.py
│   ├── domain_shift_test_set.py
│   └── augment_train_set.py
│
└── README.md
```

### `patchcore.ipynb`

Main notebook containing the anomaly detection pipeline:

* dataset preprocessing
* ResNet-50 feature extraction
* memory bank construction
* threshold estimation
* anomaly classification
* performance evaluation
* heatmap generation

### `domain-shift/`

Contains the functions used to generate modified versions of the test set
and evaluate the system under domain shift
(`domain_shift_functions.py`, `domain_shift_test_set.py`), along with the
script used to build an augmented pool of normal training images for the
robustness mitigation experiment (`augment_train_set.py`). The latter
supports an `AUGMENTATION_SCALE` parameter to shrink or expand the
perturbation ranges used for augmentation relative to the ones used at
test time.

## Requirements

The main Python libraries used are:

```text
torch
torchvision
numpy
Pillow
scikit-learn
opencv-python
matplotlib
```

The notebook can be executed using **Google Colab** with either CPU or GPU.

## Usage

Clone the repository:

```bash
git clone https://github.com/andreasangi/anomaly-detection-domain-shift.git
cd anomaly-detection-domain-shift
```

Download the **MVTec AD** dataset and place the `metal_nut` dataset in the desired directory.

Then update the dataset paths inside `patchcore.ipynb`, for example:

```python
metal_nut = "/path/to/metal_nut/train/good"
metal_nut_test = "/path/to/metal_nut/test"
```

Run the notebook cells sequentially to build the memory bank, estimate the threshold, and evaluate the model.

## Results

On the original `metal_nut` test set, the final configuration achieved:

| Metric    | Result |
| --------- | -----: |
| Accuracy  |  91.3% |
| Precision |  96.6% |
| Recall    |  92.5% |
| F1-score  |  94.5% |

The system also generates anomaly heatmaps that highlight the regions most strongly associated with detected defects.

Robustness under realistic acquisition variations, and the mitigation
applied to improve it, is covered separately in
[Robustness Mitigation](#robustness-mitigation).

## Perturbation Details

Each perturbation is implemented as a physically motivated transformation,
with parameters randomly sampled within a plausible range for industrial
acquisition conditions.

### Exposure

Simulates a linear sensor response shift: `new_pixel = alpha * old_pixel + beta`.
`alpha < 1` models under-exposure (e.g. fast shutter, low light), `alpha > 1`
models over-exposure. Reproduces incorrect shutter timing, lamp aging, and
voltage fluctuations in industrial lighting.
**Range:** `alpha ∈ [0.5, 1.7]`, `beta ∈ [-30, 30]`

### White Balance

Applies independent per-channel scaling to the R and B channels, with the G
channel kept close to stable (cameras are designed around the green
channel). Reproduces the switch between different lighting technologies
(fluorescent, LED, halogen), which shift the color temperature of the
scene.
**Range:** per-channel scale `∈ [0.70, 1.40]`

### Noise

Combines Gaussian read noise with salt-and-pepper dead/hot pixels, applied
together to model a single sensor acquisition. Gaussian sigma reproduces
thermal/read noise at high gain (low light); the salt-and-pepper fraction
reproduces permanently defective pixels on aging sensors.
**Range:** Gaussian `sigma ∈ [5, 40]`, defective pixel fraction `∈ [0, 0.005]`

### Contrast

Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) on the L
channel in LAB color space, producing a spatially non-uniform contrast
adjustment rather than a simple global one. Reproduces differing AGC/AES
histogram modes and tone-curve settings between camera units.
**Range:** clip limit `∈ [2.0, 6.0]`, tile size `∈ {8, 16, 32}`

### Perspective

Builds a homography from explicit physical camera pitch and roll angles
(pinhole camera model), so the warp corresponds to a real camera position
rather than an arbitrary distortion. Reproduces a camera remounted with a
slight tilt after maintenance, or a fixture that is not perfectly level.
**Range:** pitch, roll `∈ [-15°, 15°]`, sampled independently

### Specular Highlights

Adds a synthetic glare hotspot as a 2D Gaussian, blended additively so it
only ever adds light, never removes it. Constrained to the inner region of
the image, with independently sampled radii for a slightly elliptical
shape. Reproduces glare on glossy metal surfaces, relevant for `metal_nut`
and other reflective MVTec categories.
**Range:** hotspot radius `∈ [0.05, 0.25]` of image size, brightness `∈ [80, 220]`

### Held-out transforms (evaluated, not used for memory bank augmentation)

* **Gamma correction** - `new_pixel = 255 * (old_pixel / 255) ^ gamma`,
  reproducing a misconfigured or replaced camera with a different tone
  curve. **Range:** `gamma ∈ [0.45, 2.2]`
* **Affine misalignment** - translation, rotation, and shear around the
  image center, reproducing imperfect part placement on the inspection
  fixture or a camera that shifted between calibration and deployment.
  **Range:** translation `∈ [-40, 40]` px, rotation `∈ [-25°, 25°]`,
  shear `∈ [-0.08, 0.08]`

These two, plus a combined Exposure + Perspective condition, are used only
as a held-out generalization test after mitigation (see
[Robustness Mitigation](#robustness-mitigation)); they were never used to
build the augmented memory bank.

### Other implemented transforms (not evaluated at all)

* **JPEG compression** - reproduces bandwidth-limited camera links or
  on-device compression.
* **Blur (motion + defocus)** - reproduces vibration during exposure or
  object height variation on the conveyor.
* **Vignetting** - reproduces uneven lens illumination toward the image
  borders.

## Authors

* Andrea Sangineto
* Elia Francesco Vigè
* Matteo De Marco

## Citations & Licenses

This project builds on the following external resources:

* **Dataset**: MVTec AD [1], released under
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
  Used here for academic, non-commercial purposes only.
* **Feature extractor**: ResNet-50 [2], pre-trained on ImageNet [3],
  provided via `torchvision` (BSD-3-Clause).
* **Method**: PatchCore [4], re-implemented here based on the original paper.
  The official implementation is available at
  [amazon-science/patchcore-inspection](https://github.com/amazon-science/patchcore-inspection)
  (Apache-2.0).
* **Domain shift evaluation**: perturbation design informed by [5].

[1] P. Bergmann, M. Fauser, D. Sattlegger, and C. Steger. *MVTec AD - A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.* CVPR, 2019.
[2] K. He, X. Zhang, S. Ren, and J. Sun. *Deep Residual Learning for Image Recognition.* CVPR, 2016.
[3] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. *ImageNet: A Large-Scale Hierarchical Image Database.* CVPR, 2009.
[4] K. Roth, L. Pemula, J. Zepeda, B. Schölkopf, T. Brox, and P. Gehler. *Towards Total Recall in Industrial Anomaly Detection.* CVPR, 2022.
[5] Z. Zhang, Z. Zhao, X. Zhang, C. Sun, and X. Chen. *Industrial Anomaly Detection with Domain Shift: A Real-World Dataset and Masked Multi-Scale Reconstruction.* Computers in Industry, 151:103990, 2023.

This repository does not redistribute the MVTec AD dataset. Users must
download it directly from MVTec and agree to its license terms.