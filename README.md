# MPS-RetNet: Multi-Scale Prototype-Guided Semi-Supervised Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.3+](https://img.shields.io/badge/PyTorch-2.3+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Elsevier-green.svg)](https://arxiv.org/abs/yourpaper)

> **A Multi-Scale Prototype-Guided Semi-Supervised Framework with Quality-Aware Learning for Robust Retinal Disease Classification**

Official PyTorch implementation of MPS-RetNet as described in our paper.

**Authors:** Maisam Abbas, Ran-Zan Wang  
**Affiliation:** Department of Computer Science and Engineering, Yuan Ze University, Taiwan

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset Setup](#dataset-setup)
- [Training](#training)
- [Evaluation](#evaluation)
- [Ablation Studies](#ablation-studies)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

---

## 🎯 Overview

MPS-RetNet is a semi-supervised deep learning framework designed for multi-class retinal disease classification under annotation-scarce conditions. The framework achieves **90.10% accuracy** and **87.04% Cohen's κ** on 4-class fundus image classification with only **26% labeled data**.

### Key Innovations

- **Multi-Scale Spatial Pooling**: Captures lesion patterns across multiple receptive field scales
- **Prototype-Guided Representation**: Enhances class separability through learned class anchors
- **Quality-Aware Learning**: Adaptively modulates supervision based on image reliability
- **Semi-Supervised Training**: Leverages unlabeled data via teacher-student consistency regularization

### Architecture Overview

```
Input Image (224×224)
    ↓
ConvNeXt-Base Backbone (Transfer Learning)
    ↓
Multi-Scale Spatial Pooling (ASPP)
    ↓
Spatial Attention Blocks (2 layers)
    ↓
Prototype-Guided Module (8 prototypes)
    ↓
Quality Estimation Branch → Quality Score
    ↓
Classification Head (4 classes)
```

---

## ✨ Key Features

### 🔬 Medical AI Innovations

- ✅ **Label Efficiency**: Achieves 90%+ accuracy with only 26% labeled data
- ✅ **Quality Robustness**: Validated quality scores (r=0.71 with BRISQUE)
- ✅ **Clinical Interpretability**: Saliency maps validated by 33 ophthalmologists (90.3% diagnostic agreement)
- ✅ **Multi-Disease Support**: Cataract, DR, Glaucoma, Normal (extensible)

### 🚀 Technical Highlights

- ✅ **State-of-the-Art Performance**: 90.10% accuracy, 90.17% F1, 87.04% κ
- ✅ **Well-Calibrated**: ECE=0.0261, Brier=0.1307
- ✅ **Cross-Validated**: 5-fold CV (91.84±1.43% accuracy)
- ✅ **Externally Validated**: 95% on Messidor-2 dataset
- ✅ **Efficient**: 87 images/second inference on NVIDIA H200

---

## 📦 Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU support)
- 16GB+ RAM
- NVIDIA GPU with 8GB+ VRAM (recommended)

### Step 1: Clone Repository

```bash
(https://github.com/Maisamilens/MPSRET-Net-for-retina-Classification.git)
cd mps-retnet
```

### Step 2: Create Environment

```bash
# Using conda (recommended)
conda create -n mpsretnet python=3.10
conda activate mpsretnet

# Or using venv
python -m venv mpsretnet
source mpsretnet/bin/activate  # Linux/Mac
# mpsretnet\Scripts\activate  # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary>Click to see requirements.txt</summary>

```
torch>=2.3.0
torchvision>=0.18.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
pillow>=10.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
scipy>=1.11.0
```

</details>

---

## 🚀 Quick Start

### Minimal Example (5 minutes)

```python
import torch
from mpsretnet import MPSRetNet, Config

# Load configuration
config = Config()

# Initialize model
model = MPSRetNet(config).cuda()

# Load pretrained weights
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Inference
from PIL import Image
from torchvision import transforms

img = Image.open('test_image.jpg')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

with torch.no_grad():
    output = model(transform(img).unsqueeze(0).cuda())
    pred = output['logits'].argmax(dim=1)
    
print(f"Predicted class: {config.classes[pred]}")
```

---

## 📁 Dataset Setup

### Dataset Structure

Organize your data as follows:

```
retina_disease_classification/
├── cataract/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
├── diabetic_retinopathy/
│   ├── image001.jpg
│   └── ...
├── glaucoma/
│   └── ...
└── normal/
    └── ...
```

### Download Datasets

**Training Dataset (Ocular Imaging Health - OIH)**
- **Size**: 4,217 images
- **Classes**: 4 (Cataract, DR, Glaucoma, Normal)
- **Download**: [Google Drive Link](https://drive.google.com/file/d/15cQdw9dOIrudXb4R_evF4dq34M6ZUV_n/view)

**External Validation (Messidor-2)**
- **Size**: 400 images (100 per class)
- **Download**: [Google Drive Link](https://drive.google.com/file/d/1VedrAfQimX4zNGL6nPAgmgtJ7vP8JAOd/view)

### Data Statistics

| Split | Cataract | DR | Glaucoma | Normal | Total | % |
|-------|----------|----|---------| -------|-------|---|
| Training | 224 | 224 | 208 | 220 | 876 | 20.8% |
| Validation | 46 | 62 | 54 | 58 | 220 | 5.2% |
| Testing | 104 | 110 | 101 | 107 | 422 | 10.0% |
| Unlabeled | - | - | - | - | 2,695 | 64.0% |
| **Total** | - | - | - | - | **4,217** | **100%** |

---

## 🎓 Training

### Full Training with Cross-Validation

```bash
# 5-fold cross-validation (reproduces paper results)
python mpsretnet_v3.1_final.py

# Expected output:
# Fold 1: 89.45% accuracy
# Fold 2: 91.59% accuracy
# Fold 3: 92.17% accuracy
# Fold 4: 93.36% accuracy
# Fold 5: 92.65% accuracy
# Mean: 91.84 ± 1.43%
```

### Single Training Run

```python
from mpsretnet import train_model, Config

config = Config()
config.num_epochs = 50  # Adjust as needed

model, history, metrics, data, calibration, train_time = train_model(config, seed=42)

print(f"Test Accuracy: {metrics['accuracy']*100:.2f}%")
print(f"Test F1: {metrics['f1']*100:.2f}%")
print(f"Cohen's Kappa: {metrics['kappa']*100:.2f}%")
```

### Custom Configuration

```python
from mpsretnet import Config

config = Config()

# Data settings
config.data_root = "path/to/your/data"
config.batch_size = 32
config.labeled_ratio = 0.3  # Use 30% labeled data

# Training settings
config.num_epochs = 100
config.learning_rate = 3e-5
config.early_stopping_patience = 25

# Model settings
config.num_heads = 8
config.num_prototypes = 8
config.embed_dim = 512
```

---

## 📊 Evaluation

### Test Set Evaluation

```bash
python evaluate.py --checkpoint checkpoints/best_model.pt --data_path retina_disease_classification
```

### External Validation

```bash
python evaluate.py --checkpoint checkpoints/best_model.pt --data_path messidor --external
```

### Generate Saliency Maps

```bash
python generate_saliency.py --checkpoint checkpoints/best_model.pt --image_path test_images/ --output_path saliency_maps/
```

---

## 🔬 Ablation Studies

We provide separate scripts for each ablation study from the paper:

### 1. Component Ablation

```bash
python ablation_components.py
```

**Tests**: Transfer Learning, Spatial Attention, Multi-Scale Pooling, Prototype Module, Quality Branch, EMA, Confidence Filtering

**Output**: `ablation_results/table_ablation_components.csv`

### 2. Hyperparameter Sensitivity

```bash
python ablation_hyperparameters.py
```

**Tests**: Attention Heads (4, 8, 16), Transformer Blocks (1, 2, 4), Prototypes (4, 8, 16), λ_proto (0.01, 0.1, 0.5)

**Output**: `ablation_results/table_hyperparameter_sensitivity.csv`

### 3. Backbone Comparison

```bash
python ablation_backbones_fixed.py
```

**Tests**: ResNet-50, ResNet-101, DenseNet-121, EfficientNet-B3, ViT-B/16, ConvNeXt-Small, ConvNeXt-Base

**Output**: `ablation_results/table_backbone_comparison.csv`

### 4. Label Efficiency

```bash
python ablation_label_efficiency.py
```

**Tests**: 5%, 10%, 25%, 50%, 70%, 100% labeled data

**Output**: `ablation_results/table_label_efficiency.csv`

### Run All Ablations

```bash
python run_all_ablations.py
```

**Outputs**: All CSV files + `SUMMARY_REPORT.csv` + `LATEX_TABLES.tex`

---

## 📈 Results

### Classification Performance

| Metric | Cataract | DR | Glaucoma | Normal | **Average** |
|--------|----------|----|---------| -------|-------------|
| Sensitivity (%) | 90.38 | 100.00 | 82.17 | 87.85 | **90.10** |
| Specificity (%) | 98.42 | 99.67 | 95.63 | 93.33 | **96.76** |
| Precision (%) | 94.94 | 99.09 | 85.56 | 81.73 | **90.33** |
| F1-Score (%) | 92.61 | 99.54 | 83.83 | 84.64 | **90.17** |

**Overall**: Accuracy=90.10%, Cohen's κ=87.04%, AUC=0.979

### Cross-Validation Results

| Fold | Seed | Accuracy | F1 | κ | AUC | ECE | Brier |
|------|------|----------|----|----|-----|-----|-------|
| 1 | 42 | 89.45% | 89.38% | 85.94% | 0.9838 | 0.0122 | 0.1475 |
| 2 | 123 | 91.59% | 91.44% | 88.77% | 0.9875 | 0.0199 | 0.1290 |
| 3 | 456 | 92.17% | 92.06% | 89.56% | 0.9882 | 0.0103 | 0.1209 |
| 4 | 789 | 93.36% | 93.21% | 91.14% | 0.9923 | 0.0283 | 0.0998 |
| 5 | 1024 | 92.65% | 92.51% | 90.19% | 0.9883 | 0.0184 | 0.1130 |
| **Mean±SD** | - | **91.84±1.43** | **91.72±1.41** | **89.52±2.06** | **0.9886±0.0027** | **0.0178±0.0065** | **0.1223±0.0175** |

### External Validation (Messidor-2)

| Class | Sensitivity | Specificity | F1-Score |
|-------|-------------|-------------|----------|
| Cataract | 95.00% | 97.33% | 93.60% |
| DR | 100.00% | 99.67% | 99.50% |
| Glaucoma | 97.00% | 96.33% | 93.27% |
| Normal | 88.00% | 100.00% | 93.62% |
| **Average** | **95.00%** | **98.33%** | **95.00%** |

**Overall**: Accuracy=95.00%, Cohen's κ=93.33%

### Comparison with State-of-the-Art

| Method | Type | Accuracy | F1 | κ |
|--------|------|----------|----|----|
| VAT | Semi-Sup | 72.58% | 71.94% | 63.44% |
| Mean Teacher | Semi-Sup | 88.42% | 88.15% | 84.56% |
| ResNet-50 | Supervised | 89.83% | 89.78% | 86.44% |
| Pan-Ret | Supervised | 90.45% | 90.11% | - |
| **MPS-RetNet (Ours)** | **Semi-Sup** | **90.10%** | **90.17%** | **87.04%** |

### Computational Efficiency

| Metric | Value |
|--------|-------|
| Parameters | 110.82M |
| FLOPs (Inference) | 15.8 G |
| Training Time (30 epochs) | ~45 min |
| Inference Time (per image) | 11.5 ms |
| Throughput | 87 images/s |
| GPU Memory (Training) | 18.4 GB |
| GPU Memory (Inference) | 4.2 GB |

**Hardware**: NVIDIA H200 NVL (144 GB VRAM), CUDA 13.0

---

## 📖 Citation

If you find this work useful for your research, please cite:

```bibtex
@article{abbas2025mpsretnet,
  title={MPS-RetNet: A Multi-Scale Prototype-Guided Semi-Supervised Framework with Quality-Aware Learning for Robust Retinal Disease Classification},
  author={Abbas, Maisam and Wang, Ran-Zan},
  journal={to be submitted},
  year={2025},
  publisher={wip}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📧 Contact

- **Maisam Abbas**: s1129105@mail.yzu.edu.tw

Department of Computer Science and Engineering  
Yuan Ze University, Taoyuan, Taiwan

---

## 🙏 Acknowledgments

- Dataset sources: ODIR, IDRiD, HRF, Messidor-2
- AIHub computing resources from Yuan Ze University
- PyTorch and torchvision teams
- Medical validation by 33 ophthalmologists

---

## 📚 Additional Resources

- [Paper (https://doi.org/10.1038/s41598-024-77464-w)
- [Supplementary Materials] (https://doi.org/10.1038/s41598-025-98021-z)
- Supporting Materials: https://arxiv.org/pdf/2105.08982

---

**Keywords**: Retinal Disease Classification, Semi-Supervised Learning, Deep Learning, Medical AI, Fundus Imaging, Quality-Aware Learning, Prototype Learning, Explainable AI
