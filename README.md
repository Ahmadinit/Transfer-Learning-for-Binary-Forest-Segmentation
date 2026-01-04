# 🌲 Transfer Learning for Binary Forest Segmentation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/) [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/) [![Keras](https://img.shields.io/badge/Keras-2.x-red.svg)](https://keras.io) [![Notebook](https://img.shields.io/badge/Jupyter-Notebook-9C27B0.svg)](https://jupyter.org)

💡 A rigorous, reproducible comparison of training a U-Net segmentation model from scratch versus transfer learning with a pre-trained ResNet50 encoder for binary forest segmentation (forest vs. non-forest). The repo includes an end-to-end notebook, standardized data pipeline, metrics, and visualizations.

## 🧭 Quick Overview

- 🌳 Goal: Segment forest regions in RGB imagery
- 🧠 Models: U-Net (scratch) vs. U-Net + ResNet50 (transfer)
- 🎯 Metrics: IoU, Dice, Accuracy, Precision, Recall, F1
- ⚡ Strategy: Freeze encoder → fine-tune (100 total epochs)
- 📈 Outputs: Metrics table + side-by-side visual comparisons

## 🗺️ Table of Contents

- 🔎 Problem Statement
- 🎯 Objective
- 🧩 Scope & Contributions
- 🗃️ Dataset
- 🧱 Data Pipeline
- 🏗️ Model Architecture
- 🔬 Experimental Design
- 📐 Loss Function
- ⚙️ Optimization
- 📊 Evaluation Metrics
- 🏆 Results & Analysis
- 🔁 Reproducibility
- 📦 Installation
- 🚀 Usage
- 🗂️ Project Structure
- 🧪 Troubleshooting
- 📚 References

## 🔎 Problem Statement

Binary segmentation of forest regions in RGB imagery supports environmental monitoring, land-use planning, and remote sensing. Training deep models from scratch is data- and compute-intensive; transfer learning leverages pre-trained features to accelerate convergence and improve accuracy. This project quantifies these trade-offs on a consistent setup.

## 🎯 Objective

Compare two approaches under identical conditions:
- 🧪 U-Net with randomly initialized encoder (scratch)
- 🚀 U-Net with ResNet50 encoder pre-trained on ImageNet (transfer)

Evaluate both on:
- 🎯 Effectiveness: IoU, Dice, Accuracy, Precision, Recall, F1
- ⚡ Efficiency: training time and convergence behavior over 100 epochs

## 🧩 Scope & Contributions

- 🧰 Standardized data pipeline and training protocol for fair comparison
- 🧊→🔥 Phased transfer strategy (freeze, then fine-tune)
- 📑 Reproducible metrics and clear tabular reporting
- 🖼️ Side-by-side visuals for boundary quality and consistency

## 🗃️ Dataset

- 📦 Source: Augmented Forest Segmentation Dataset (Kaggle)
- 🎯 Task: Binary semantic segmentation (forest vs. non-forest)
- 🖼️ Input: RGB images
- 🎭 Output: Binary masks (1 = forest, 0 = background)

Assumptions:
- 🔗 Image/mask filenames are paired by stem
- 🎚️ Masks are single-channel binary
- 🚫 No leakage across train/val/test splits

## 🧱 Data Pipeline

Applied consistently across experiments:
- 📐 Resize images/masks to 128×128
- 🎛️ Normalize image pixels to [0, 1]
- 🧼 Ensure masks are binary (threshold if needed)
- 🔀 Split: train/val/test (e.g., 70/15/15)

Expected structure:

```
data/
  train/{images,masks}
  val/{images,masks}
  test/{images,masks}
```

## 🏗️ Model Architecture

U-Net decoder + ResNet50 encoder (`segmentation_models`, TF/Keras):

```
ResNet50 Encoder (ImageNet) → multi-scale features
U-Net Decoder → upsampling + skip connections → sigmoid output
```

Why this setup?
- 🧠 ResNet50 captures hierarchical features
- 🛠️ U-Net decoder restores spatial detail
- 🎯 Sigmoid suits binary masks

## 🔬 Experimental Design

### A) Scratch Training
- ⚙️ Encoder: random init
- 🏃 End-to-end training: 100 epochs
- 🎯 Baseline without prior knowledge

### B) Transfer Learning (Phased)
- 🧊 Phase 1 (Epochs 1–50): freeze encoder, train decoder
- 🔥 Phase 2 (Epochs 51–100): unfreeze encoder, full fine-tuning

Controls:
- 🧪 Same optimizer and batch size
- 🧪 Identical preprocessing, splits, metrics

## 📐 Loss Function

Composite loss:

L = L_BCE + L_Dice

Dice loss:

L_Dice = 1 − (2 × |P ∩ G|) / (|P| + |G|)

Where P = predicted mask (thresholded), G = ground truth.

## ⚙️ Optimization

- 🔧 Optimizer: Adam (SGD optional)
- 🎚️ LR: tuned empirically, recorded in the notebook
- 🧮 Batch size: adapted to memory constraints
- ⏱️ Callbacks: early stopping, checkpoints recommended

## 📊 Evaluation Metrics

- 🥇 IoU = |P ∩ G| / |P ∪ G|
- 📈 Dice = 1 − L_Dice
- ✅ Accuracy
- 🎯 Precision
- 🔁 Recall
- 🔷 F1-Score

## 🏆 Results & Analysis

### 📋 Test Metrics (placeholders)

| Model                | IoU   | Accuracy | Precision | Recall | F1-Score | Train Time |
|----------------------|-------|----------|-----------|--------|----------|------------|
| Scratch (U-Net)      | [TBD] | [TBD]    | [TBD]     | [TBD]  | [TBD]    | [TBD]      |
| Transfer (ResNet50)  | [TBD] | [TBD]    | [TBD]     | [TBD]  | [TBD]    | [TBD]      |

### 🔍 Discussion

- 🌟 Transfer typically improves IoU/Dice and boundary adherence
- ⚡ Faster early convergence with frozen encoder
- 🧪 Scratch may overfit or struggle on small structures

### 🖼️ Visuals

Saved in `results/visualizations/`:
1. 🖼️ Input image
2. 🎭 Ground truth
3. 🔵 Scratch prediction
4. 🟢 Transfer prediction

## 🔁 Reproducibility

- 🎲 Set seeds (NumPy, TensorFlow)
- 🧾 Log hyperparameters, LR schedules, splits
- 💾 Save checkpoints to `models/`
- 📤 Export metrics to `results/metrics.csv`

## 📦 Installation

Prereqs:

```bash
Python 3.8+

```

Install:

```bash
pip install -r requirements.txt
```

Note for `segmentation_models`:

```python
import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()
```

## 🚀 Usage

Run the notebook:

```bash
jupyter notebook "Transfer learning for segmentation.ipynb"
```

Ensure `data/` is structured as described. The notebook trains both setups, logs metrics, and produces visualizations.

## 🗂️ Project Structure

```
TL-Binary-Forest-Segmentation/
├── Transfer learning for segmentation.ipynb
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── train/{images,masks}
│   ├── val/{images,masks}
│   └── test/{images,masks}
├── models/
│   ├── scratch_model.h5
│   └── transfer_learning_model.h5
└── results/
    ├── metrics.csv
    └── visualizations/
```

## 🧪 Troubleshooting

- 🧠 OOM: reduce batch size or image size; try mixed precision
- 📉 Diverging loss: lower LR; weight decay; verify binary masks
- 🪚 Poor boundaries: extend fine-tuning; augment with flips/rotations/elastic

## 📚 References

- U-Net — https://arxiv.org/abs/1505.04597
- ResNet — https://arxiv.org/abs/1512.03385
- Segmentation Models — https://github.com/qubvel/segmentation_models

---

If this project helps, ⭐ star it and share feedback!

