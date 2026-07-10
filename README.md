# Optimized Hybrid Initialization for Deep CNNs

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Framework-EE4C2C?logo=pytorch&logoColor=white)
![Torchvision](https://img.shields.io/badge/Torchvision-Datasets%20%26%20Transforms-5C3EE8)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Optional%20Trials-FF6F00?logo=tensorflow&logoColor=white)
![CIFAR--10](https://img.shields.io/badge/CIFAR--10-Image%20Classification-1F77B4)
![CIFAR--100](https://img.shields.io/badge/CIFAR--100-Image%20Classification-2CA02C)
![QMNIST](https://img.shields.io/badge/QMNIST-Digit%20Classification-7F7F7F)
![STL10](https://img.shields.io/badge/STL10-Image%20Classification-D62728)
![SVHN](https://img.shields.io/badge/SVHN-Real--World%20Digits-17BECF)

## Summary
This repository documents our full research path toward a new initialization strategy. We first experimented with **Poisson-based** and **scaled-uniform** ideas, then systematically studied **Xavier** and **LSUV**, and finally developed a unique **Hybrid Orthogonal + He** initialization method.

Across these stages, we evaluate effects on training stability, convergence speed, final accuracy, and cross-architecture generalization.

The codebase includes experiments for CNN, VGG-like, ResNet variants, CaffeNet/AlexNet-style models, and GoogLeNet-style models on **CIFAR-10, CIFAR-100, QMNIST, STL10, and SVHN**.

---

## Requirements
- Python 3.8+
- Jupyter Notebook or JupyterLab (for `.ipynb` files)
- PyTorch
- torchvision
- matplotlib
- numpy
- scipy (needed in some SVHN workflows)
- tensorflow (only for TensorFlow-based exploratory files)

---

## Installation Instructions

### 1) Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

### 2) Install core dependencies

```bash
pip install torch torchvision matplotlib numpy scipy notebook
```

### 3) Install TensorFlow only if running TensorFlow exploratory files

```bash
pip install tensorflow
```

### 4) Run notebooks or Python scripts

Notebook:
```bash
jupyter notebook
```

Python script:
```bash
python path/to/file.py
```

## Running All Benchmark Experiments

All benchmark notebooks can be executed automatically using the provided runner:

```bash
python run_all_benchmarks.py
```

The script uses Jupyter `nbconvert` to execute every selected notebook sequentially without manually opening them.

Install notebook execution dependencies if required:

```bash
pip install notebook nbconvert
```

The runner executes experiments across:

- CIFAR-10
- CIFAR-100
- QMNIST
- STL-10
- SVHN

covering architectures including:

- CaffeNet / AlexNet
- VGG variants
- ResNet variants
- GoogLeNet variants
- Custom CNN models

To select specific experiments, open:

```text
run_all_benchmarks.py
```

and comment/uncomment notebook paths inside:

```python
notebooks = [
    ...
]
```

The runner automatically:

- Executes notebooks sequentially
- Saves executed notebook outputs in-place
- Records runtime for each benchmark
- Skips missing files
- Continues remaining experiments if one fails

Example output:

```text
======================================================================
STARTING: Final Benchmarks/CIFAR-10/CaffeNet.ipynb
======================================================================

FINISHED: Final Benchmarks/CIFAR-10/CaffeNet.ipynb
Runtime: 12.42 minutes

============================
ALL POSSIBLE RUNS COMPLETE
TOTAL TIME: 8.40 hours
============================
```

This allows full reproduction of reported accuracy, loss, and convergence benchmark results.

---

## Folder Overview (Dataset + Brief)

### Initial Research
- **Primary dataset:** CIFAR-10
- **Brief:** Baseline and exploratory experiments to test foundational methods (LSUV, Xavier) and custom ideas (Poisson-based, scaled-uniform) on CIFAR-10.

### Final Benchmarks/CIFAR-10
- **Dataset:** CIFAR-10 (32x32 RGB, 10 classes)
- **Brief:** Final comparative runs of multiple architectures under hybrid initialization settings for robust benchmarking.

### Final Benchmarks/CIFAR-100
- **Dataset:** CIFAR-100 (32x32 RGB, 100 classes)
- **Brief:** Harder classification benchmark to evaluate how initialization behaves with more classes and finer-grained categories.

### Final Benchmarks/QMNIST
- **Dataset:** QMNIST (extended MNIST digit dataset)
- **Brief:** Digit-recognition experiments to test architecture behavior on grayscale/converted-input pipelines.

### Final Benchmarks/STL10
- **Dataset:** STL10 (96x96 RGB, 10 classes)
- **Brief:** Medium-resolution benchmark to test generalization and optimization stability under stronger augmentations.

### Final Benchmarks/SVHN
- **Dataset:** SVHN (Street View House Numbers)
- **Brief:** Real-world digit classification benchmark to test robustness on naturally captured scene digits.

---

## File-wise Architecture Features

### 1. Initial Research

### `Initial Research/LSUV_CNN.ipynb`
- Architecture: Custom deep CNN (9-layer style).
- Features:
- LSUV initialization to normalize activation variance layer-by-layer.
- Orthogonal seeds before LSUV refinement.
- BatchNorm + Dropout regularization.
- Cosine annealing schedule for smoother optimization.

### `Initial Research/Xavier_CNN.py`
- Architecture: Custom CNN with stacked convolution blocks + FC classifier.
- Features:
- Xavier initialization for Conv/Linear layers.
- ReLU-based feature extractor.
- Adam optimization with straightforward training pipeline.

### `Initial Research/Xavier_ResNet50.py`
- Architecture: ResNet-family transfer/modified classifier head for CIFAR-10.
- Features:
- Xavier initialization applied to trainable Conv/Linear layers.
- Residual skip connections for gradient flow.
- CIFAR-10-adapted output head (10 classes).

### `Initial Research/Poisson_LSUV_SimpleNet.ipynb`
- Architecture: Fully connected / simple classifier baseline.
- Features:
- Custom Poisson-inspired initialization.
- Mish activation exploration.
- Metric-focused evaluation (accuracy, precision, recall, F1).

### `Initial Research/ScaledUniformInitializer_CNN.py`
- Architecture: CNN classifier (TensorFlow-based trial).
- Features:
- Custom scaled-uniform initializer.
- ReLU + Adam training setup.
- Lightweight, fast exploratory run configuration.

---

### 2. Final Benchmarks

#### 2.1 Final Benchmarks/CIFAR-10

### `Final Benchmarks/CIFAR-10/CaffeNet.ipynb`
- Architecture: CaffeNet/AlexNet-style deep CNN.
- Features:
- Sequential conv blocks with pooling and dropout.
- Strong baseline for classic deep CNN behavior.
- Hybrid init experiments (He in early layers, Orthogonal deeper).

### `Final Benchmarks/CIFAR-10/Custom_CNN.ipynb`
- Architecture: Custom 9-layer CNN.
- Features:
- Deep stacked conv design for CIFAR-scale images.
- BatchNorm and dropout regularization.
- Hybrid initialization sensitivity testing.

### `Final Benchmarks/CIFAR-10/GoogLeNet-Small.ipynb`
- Architecture: Lightweight Inception/GoogLeNet-style network.
- Features:
- Multi-branch feature extraction blocks.
- Parameter-efficient representation learning.
- Tests initialization effects in branch-heavy topology.

### `Final Benchmarks/CIFAR-10/Mini-ResNet.ipynb`
- Architecture: Compact ResNet variant.
- Features:
- Residual blocks to preserve gradient stability.
- Efficient depth/compute tradeoff.
- Useful for comparing init effects in skip-connected nets.

### `Final Benchmarks/CIFAR-10/Mini-VGG.ipynb`
- Architecture: Compact VGG-style CNN.
- Features:
- Repeated 3x3 conv patterns.
- Simple, controlled architecture for fair initialization comparisons.
- Strong baseline with predictable optimization behavior.

### `Final Benchmarks/CIFAR-10/Modified_VGG.ipynb`
- Architecture: Enhanced VGG-like model.
- Features:
- VGG block structure with modifications (regularization/width/depth tuning).
- Designed for improved convergence on CIFAR-10.
- Good candidate to assess deep-layer initialization policies.

---

#### 2.2 Final Benchmarks/CIFAR-100

### `Final Benchmarks/CIFAR-100/Mini-VGG.ipynb`
- Architecture: Compact VGG-like network.
- Features:
- Adapts VGG simplicity to 100-class classification.
- Useful for testing class-separation under initialization changes.

### `Final Benchmarks/CIFAR-100/ResNet-Mini.ipynb`
- Architecture: Small ResNet-style model.
- Features:
- Residual learning for deeper effective optimization.
- Stable training on harder fine-grained class space.

### `Final Benchmarks/CIFAR-100/VGG.ipynb`
- Architecture: Full/expanded VGG-style model.
- Features:
- Higher-capacity conv stacks.
- Strong capacity baseline for CIFAR-100 difficulty.

---

#### 2.3 Final Benchmarks/QMNIST

### `Final Benchmarks/QMNIST/CaffeNet.ipynb`
- Architecture: CaffeNet-style CNN adapted for digit data.
- Features:
- Handles grayscale pipeline (or converted channels).
- Classic conv-depth baseline for QMNIST.

### `Final Benchmarks/QMNIST/Custom_CNN.ipynb`
- Architecture: Custom deep CNN.
- Features:
- Flexible shape-handling and task-specific tuning.
- Good for observing initialization impact on digit features.

### `Final Benchmarks/QMNIST/GoogLeNet.ipynb`
- Architecture: GoogLeNet/Inception-style model.
- Features:
- Multi-scale branch processing for local and global cues.
- Tests initialization behavior in heterogeneous branch modules.

### `Final Benchmarks/QMNIST/ResNet-18.ipynb`
- Architecture: ResNet-18 (modified as needed for input channels/classes).
- Features:
- Standard residual backbone with robust optimization behavior.
- Strong reference architecture for fair cross-dataset comparison.

---

#### 2.4 Final Benchmarks/STL10

### `Final Benchmarks/STL10/CaffeNet.ipynb`
- Architecture: CaffeNet/AlexNet-style model.
- Features:
- Larger-input compatible conv pipeline.
- Dropout-enabled classifier head.
- Useful baseline for medium-resolution images.

### `Final Benchmarks/STL10/Modified_ResNet-18.ipynb`
- Architecture: ResNet-18 variant with modifications.
- Features:
- Residual backbone with tuned head/regularization.
- Designed for stronger generalization on STL10.

### `Final Benchmarks/STL10/VGG.ipynb`
- Architecture: VGG-style deep CNN.
- Features:
- Sequential deep conv stacks for hierarchical features.
- Clean architecture to study initialization effects at depth.

---

#### 2.5 Final Benchmarks/SVHN

### `Final Benchmarks/SVHN/Mini-VGG.ipynb`
- Architecture: Compact VGG-like network.
- Features:
- Efficient model for digit classification.
- Good tradeoff between speed and representational power.

### `Final Benchmarks/SVHN/ResNet-Mini.ipynb`
- Architecture: Small ResNet variant.
- Features:
- Residual links improve optimization on real-scene digits.
- Stable convergence benchmark for hybrid initialization.

### `Final Benchmarks/SVHN/VGG.ipynb`
- Architecture: VGG-like deep CNN.
- Features:
- Deeper sequential conv representation learning.
- Useful contrast against residual and inception-style models.

---

## Collaborators
- [Aadit Pani](https://github.com/AaditPani-RVU)
- [Ayush](https://github.com/Xalibur1)
- [Mehul Goyal](https://github.com/Arnidh)
