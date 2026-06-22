# DAFNet: Dynamic Attention Fusion Network

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)

**DAFNet** is a lightweight and efficient CNN architecture designed for medical image classification. It achieves strong performance through a combination of:

- **Reparameterizable convolutions** (MobileOne + RepLKNet) for efficient training and fast inference
- **Channel shuffle** with **feature map rotation** for enhanced feature diversity
- **Dynamic kernel fusion** from grouped convolution branches
- **Channel attention** mechanism fusing AvgPool and MaxPool features

## Model Variants

| Model        | Params (3-class) | FLOPs  | Description          |
|-------------|------------------|--------|----------------------|
| DAFNet-Tiny | ~4.5M            | ~0.5G  | Lightweight baseline |
| DAFNet-Small| ~8.5M            | ~0.9G  | Balanced performance |
| DAFNet-Base | ~12.5M           | ~1.3G  | Higher capacity      |
| DAFNet-Large| ~20.0M           | ~2.1G  | Maximum accuracy     |

> Note: FLOPs measured at 224×224 input resolution. Actual parameter counts vary with `num_classes`.

## Installation

```bash
git clone https://github.com/yourusername/DAFNet.git
cd DAFNet
pip install -r requirements.txt
```

## Quick Start

```python
from dafnet import create_model, list_models

# List available models
print(list_models())  # ['dafnet_tiny', 'dafnet_small', 'dafnet_base', 'dafnet_large']

# Create a model
model = create_model('dafnet_tiny', num_classes=3)

# Forward pass
import torch
x = torch.randn(1, 3, 224, 224)
output = model(x)  # (1, 3)
```

### Reparameterization for Inference

```python
from dafnet import reparameterize_model

# Fuse all training branches into single convolutions
inference_model = reparameterize_model(model)
```

## Dataset Structure

Organize your dataset as follows:

```
dataset/
├── train_val/
│   ├── class_0/
│   │   ├── img001.jpg
│   │   └── ...
│   ├── class_1/
│   │   ├── img001.jpg
│   │   └── ...
│   └── ...
└── test/
    ├── class_0/
    │   └── ...
    ├── class_1/
    │   └── ...
    └── ...
```

## Training

```bash
python scripts/train.py \
    --dataset_path /path/to/dataset \
    --model_name dafnet_tiny \
    --num_classes 3 \
    --epochs 100 \
    --batch_size 32 \
    --cv_folds 5 \
    --learning_rate 0.001 \
    --model_save_path ./outputs
```

The training script:
- Runs **stratified K-fold cross-validation**
- Logs to **TensorBoard** and a text log file
- Saves **best and last checkpoints** per fold
- Outputs comprehensive metrics (accuracy, F1, AUC, MCC, specificity, etc.)
- Appends results to a **CSV file** for easy comparison

## Feature Visualization

```bash
python scripts/visualize.py \
    --checkpoint path/to/checkpoint.pth \
    --image path/to/image.jpg \
    --model_name dafnet_tiny \
    --num_classes 3 \
    --output_dir ./visualizations
```

## Metrics

The trainer computes and logs the following metrics per fold (mean ± std):

- **Accuracy** — overall classification accuracy
- **Recall** — macro-averaged recall
- **Precision** — macro-averaged precision
- **F1 Score** — macro-averaged F1
- **AUC** — macro-averaged area under ROC curve (one-vs-rest)
- **Specificity** — macro-averaged specificity
- **MCC** — Matthews correlation coefficient
- **Cohen's Kappa** — inter-rater agreement score

## Architecture

```
Input (3, 224, 224)
    │
    ▼
Stem (7×7 RepLKConv + MobileOne blocks)
    │  → (96, 56, 56)
    ▼
Stage 1: BaseBlock × N₁ + CPD downsampling
    │  → (192, 28, 28)
    ▼
Stage 2: BaseBlock × N₂ + CPD downsampling
    │  → (384, 14, 14)
    ▼
Stage 3: BaseBlock × N₃ + CPD downsampling
    │  → (768, 7, 7)
    ▼
Stage 4: BaseBlock × N₄ + FFEN (no downsampling)
    │  → (768, 7, 7)
    ▼
Classifier: 1×1 Conv → GAP → Linear
    │  → (num_classes,)
```

### BaseBlock Detail

Each `BaseBlock` performs:
1. **Grouped 1×1 conv** for inter-group channel mixing
2. **Channel shuffle** into 4 groups
3. **Feature map rotation** (4 orientations: identity, transpose, flip, flip+transpose)
4. **Depthwise 3×3 conv** on each group with BN fusion
5. **Dynamic kernel fusion** from the fused weights
6. **Channel attention** (AvgPool + MaxPool → sigmoid gating)
7. **Residual connection** with parallel depthwise conv path

## Citation

If you use DAFNet in your research, please cite:

```bibtex
@article{dafnet2024,
  title={DAFNet: Dynamic Attention Fusion Network for Medical Image Classification},
  author={...},
  journal={...},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [MobileOne](https://github.com/apple/ml-mobileone) — Apple's reparameterizable mobile architecture
- [RepLKNet](https://github.com/DingXiaoH/RepLKNet-pytorch) — Large kernel reparameterization
- [timm](https://github.com/huggingface/pytorch-image-models) — PyTorch image models library