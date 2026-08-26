# Industrial Anomaly Detection under Domain Shift

An industrial image anomaly detection and localization project based on **ResNet-50** and a **PatchCore-based approach**.

The system is trained using only defect-free images and is evaluated on the **metal_nut** category of the **MVTec AD** dataset.

The full project report is available here: [Report.pdf](./Report.pdf)

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

* `good` — defect-free samples
* `bent`
* `color`
* `flip`
* `scratch`

The original training set contains only normal images and is divided into:

* **80% training** — used to build the memory bank
* **20% validation** — used to determine the anomaly threshold

## Domain Shift

The robustness of the anomaly detector is also evaluated under different
acquisition conditions, simulating variations that can occur in a real
industrial camera setup.

The main transformations considered are:

* Exposure changes
* White balance variations
* Gaussian and salt-and-pepper noise
* Contrast modifications
* Perspective transformations

These transformations simulate realistic variations in illumination, sensor
quality, and camera viewpoint without introducing new defects. All
parameters are randomly sampled within physically motivated ranges for each
transformation. During domain shift evaluation, the ResNet-50 feature
extractor, memory bank, and anomaly threshold remain unchanged: no
additional training or calibration is performed on the shifted images.

A broader set of transforms (gamma correction, JPEG compression, motion and
defocus blur, vignetting, and specular highlights) was also implemented but
not included in the final evaluation, since the selected five already cover
the main sources of domain shift relevant to this scenario.

See [Perturbation Details](#perturbation-details) below for the model and
parameter ranges used for each transformation.

## Repository Structure

```text
anomaly-detection-domain-shift/
│
├── patchcore.ipynb
│
├── domain-shift/
│   ├── domain_shift_functions.py
│   └── domain_shift_test_set.py
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

Contains the functions used to generate modified versions of the test set and evaluate the system under domain shift.

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

### Other implemented transforms (not evaluated)

* **Gamma correction** — reproduces a misconfigured or replaced camera with
  a different tone curve.
* **JPEG compression** — reproduces bandwidth-limited camera links or
  on-device compression.
* **Blur (motion + defocus)** — reproduces vibration during exposure or
  object height variation on the conveyor.
* **Vignetting** — reproduces uneven lens illumination toward the image
  borders.
* **Specular highlights** — reproduce glare on glossy metal surfaces,
  relevant for `metal_nut` and similar reflective categories.

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

[1] P. Bergmann, M. Fauser, D. Sattlegger, and C. Steger. *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.* CVPR, 2019.
[2] K. He, X. Zhang, S. Ren, and J. Sun. *Deep Residual Learning for Image Recognition.* CVPR, 2016.
[3] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. *ImageNet: A Large-Scale Hierarchical Image Database.* CVPR, 2009.
[4] K. Roth, L. Pemula, J. Zepeda, B. Schölkopf, T. Brox, and P. Gehler. *Towards Total Recall in Industrial Anomaly Detection.* CVPR, 2022.
[5] Z. Zhang, Z. Zhao, X. Zhang, C. Sun, and X. Chen. *Industrial Anomaly Detection with Domain Shift: A Real-World Dataset and Masked Multi-Scale Reconstruction.* Computers in Industry, 151:103990, 2023.

This repository does not redistribute the MVTec AD dataset. Users must
download it directly from MVTec and agree to its license terms.