# Industrial Anomaly Detection under Domain Shift

An industrial image anomaly detection and localization project based on **ResNet-50** and a **PatchCore-based approach**.

The system is trained using only defect-free images and is evaluated on the **metal_nut** category of the **MVTec AD** dataset.

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

The robustness of the anomaly detector is also evaluated under different acquisition conditions.

The main transformations considered are:

* Exposure changes
* White balance variations
* Gaussian and salt-and-pepper noise
* Contrast modifications
* Perspective transformations

These transformations simulate realistic variations in illumination, sensor quality, and camera viewpoint without introducing new defects.

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

## Authors

* Andrea Sangineto
* Elia Francesco Vigè
* Matteo De Marco
