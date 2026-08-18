# KLA PS01: AI-Based Restoration of Degraded Images for Semiconductor Inspection

**Hackathon:** I4C / SEMICON India Hackathon 2026  
**Problem Statement 1 (KLA):** AI-Based Restoration of Degraded Images for Semiconductor Inspection  
**Task Type:** Joint 2x Super-Resolution and Image Denoising  

---

## Overview

High-resolution, noise-free imaging is critical for sub-nanometer defect inspection in modern semiconductor fabrication. This repository implements a deep-learning restoration pipeline that reconstructs high-resolution ($256 \times 256$) ground-truth wafer inspection patterns from noisy, low-resolution ($128 \times 128$) scanning measurements.

---

## Dataset Structure

The project consumes binary NumPy arrays (`.npy`) extracted directly from the official dataset:

```
KLA_Semicon_PS01/
├── data_sets/                          # Unmodified Official Dataset
│   ├── train/train/
│   │   ├── GT/                         # 3,200 Ground-Truth arrays (256x256, float32, [0, 1])
│   │   └── NoisyLR/                    # 3,200 Input Noisy arrays (128x128, float32)
│   └── Test_NoisyLR/NoisyLR/           # 400 Test Noisy arrays (128x128, float32)
├── models/
│   └── baseline_unet.py                # 2x SR-UNet Architecture with PixelShuffle
├── utils/
│   ├── losses.py                       # Charbonnier + SSIM Loss
│   └── metrics.py                      # PSNR & SSIM Evaluation Metrics
├── weights/                            # Trained model checkpoints (.pth)
├── restored_outputs/                   # Evaluated output arrays (.npy)
├── dataset.py                          # PyTorch Dataset Loader with 90/10 reproducible split
├── train.py                            # Training pipeline
├── inference.py                        # Single-image inference CLI
├── evaluation.py                       # Standalone KLA benchmark evaluation CLI
├── requirements.txt
└── README.md
```

### Verified Empirical Dataset Properties
- **GT Target Shape**: `(256, 256)`, 1-channel, `float32`, bounded in `[0.0, 1.0]`.
- **NoisyLR Input Shape**: `(128, 128)`, 1-channel, `float32`, unclipped noise values in `[-0.10, 1.95]`.
- **Test Set**: 400 images `(128, 128)`, 1-channel, `float32`.
- **Pairing**: 100% paired by matching filenames (`000000.npy` to `003199.npy`).
- **Baseline Bicubic PSNR**: ~23.15 dB.

---

## Setup & Installation

### Environment Setup
Python 3.10+ and PyTorch 2.0+ are recommended. Install dependencies via `pip`:

```bash
pip install -r requirements.txt
```

---

## Usage Commands

### 1. Training Command
Train the baseline 2x SR-UNet model on the paired training dataset:

```bash
python train.py --epochs 10 --batch_size 8 --lr 0.0001 --data_dir ../data_sets --weights_dir weights
```

### 2. Single-Image Inference Command
Restores a single `.npy` test/noisy file to a $256 \times 256$ float32 array:

```bash
python inference.py \
  --input ../data_sets/Test_NoisyLR/NoisyLR/000000.npy \
  --output restored_outputs/000000.npy \
  --weights weights/best_model.pth
```

### 3. Official KLA Standalone Evaluation Command
Runs batch inference over all 400 test images in `Test_NoisyLR/NoisyLR` and saves restored $256 \times 256$ `.npy` outputs with matching filenames:

```bash
python evaluation.py \
  --test_dir ../data_sets/Test_NoisyLR/NoisyLR \
  --output_dir ./restored_outputs \
  --weights ./weights/best_model.pth
```

---

## Model Architecture & Tensor Dimensions

The `BaselineSRUNet` model performs joint feature denoising and learnable 2x spatial super-resolution:

```
Input: Tensor (B, 1, 128, 128) [Noisy Low-Resolution Input]
  │
  ├── 3x3 Conv Feature Head -> (B, 64, 128, 128)
  ├── Encoder Stage 1 (ResBlock + DownConv) -> (B, 128, 64, 64)
  ├── Encoder Stage 2 & Bottleneck (ResBlocks) -> (B, 128, 64, 64)
  ├── Decoder Stage 1 (TransposedConv + Skip Connection) -> (B, 64, 128, 128)
  ├── 2x Learnable Super-Resolution (Conv3x3 + PixelShuffle(2)) -> (B, 64, 256, 256)
  └── 3x3 Tail Conv + Clamp[0.0, 1.0] -> (B, 1, 256, 256)
  │
Output: Tensor (B, 1, 256, 256) [Restored High-Resolution Image]
```

---

## Loss Functions & Metrics

### Loss Function
Combination of pixel-level Charbonnier loss ($\epsilon = 10^{-3}$) and structural SSIM loss:

$$\mathcal{L}_{total} = \mathcal{L}_{charbonnier} + 0.2 \cdot \mathcal{L}_{ssim}$$

### Validation Metrics
- **PSNR (Peak Signal-to-Noise Ratio)**: Measured in dB on validation set ($256 \times 256$ float32 in $[0, 1]$).
- **SSIM (Structural Similarity Index)**: Evaluates structural edge fidelity of semiconductor geometries.

---

## Hardware Requirements

- **System RAM**: 8 GB+ (16 GB recommended).
- **GPU VRAM**: 6 GB+ recommended (e.g. NVIDIA RTX 3060/3080/4070/4090 or T4/V100/A100 on Google Colab).
- **Disk Storage**: ~2 GB for dataset, checkpoints, and output arrays.
