# Precise Retinal Blood Vessel Segmentation from Color Fundus Images Using U-Net Built and Trained from Scratch in PyTorch

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-U--Net-EE4C2C?logo=pytorch&logoColor=white)
![Task](https://img.shields.io/badge/Task-Image%20Segmentation-success)
![Platform](https://img.shields.io/badge/Platform-Kaggle%20Notebook-20BEFF?logo=kaggle&logoColor=white)

This repository contains a single, self-contained Kaggle notebook — [`u-net-main-kaggle.ipynb`](./u-net-main-kaggle.ipynb) — that loads paired retinal fundus images and vessel masks, trains a U-Net to segment the vessels pixel-by-pixel, and visualizes the results. Vessel segmentation like this is a standard building block in ophthalmic image analysis, used in research on diabetic retinopathy, hypertension, and glaucoma.

|  |  |
|---|---|
| **Task** | Binary semantic segmentation (vessel vs. background) |
| **Model** | U-Net, from scratch, ~31M parameters |
| **Framework** | PyTorch / torchvision |
| **Input** | RGB fundus image → 512×512 |
| **Data** | 40 labeled fundus images + vessel masks |
| **Environment** | Kaggle Notebook, GPU-enabled |

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Setup](#training-setup)
- [Results](#results)
- [References](#references)
- [License](#license)

## Overview

Given a color fundus photo, the model predicts a mask marking every pixel that belongs to a blood vessel. End to end, the notebook:

1. Loads paired fundus images and hand-labeled vessel masks from disk.
2. Resizes everything to 512×512 and converts it to tensors (image → RGB, mask → single-channel, binarized).
3. Can split the images into a train/test set, or train on the whole folder at once (controlled by a `mode` flag — see [Results](#results) for how this was actually used).
4. Trains a from-scratch U-Net using a combined Dice + Jaccard loss.
5. Tracks accuracy, sensitivity (recall), and F1 after every epoch.
6. Runs inference on a held-out labeled split *and* a separate folder of unlabeled test images, saving prediction plots to disk.
7. Includes a small visual-diagnostics cell that breaks a sample image into its R/G/B channels — useful here since the green channel gives the strongest vessel-to-background contrast in fundus photos.

It was built and run as a GPU-enabled Kaggle notebook, so the easiest way to reproduce it is to reopen it there

## Dataset

The notebook expects a folder of fundus images and matching vessel masks laid out like this:

```
retinal-vessels/
├── all_label/
│   ├── Images/     # st01.jpg ... st40.jpg (some also present as .tif)
│   └── Labels/     # st_label01.jpg ... (some as .gif)
└── test/           # separate, unlabeled images used only for inference
```

**At a glance:**

| | |
|---|---|
| Total labeled images | 40 (default split used in the notebook: 28 train / 12 test, random, seed = 2) |
| Image sizes | 700 × 605 px or 565 × 584 px |
| Image channels | RGB photo, single-channel (grayscale) mask |
| Resized to | 512 × 512 for training |

## Model Architecture

The core model (the `UNet` class) is a standard encoder–decoder with skip connections, built entirely from scratch — no pretrained backbone:

- **Encoder** — 4 stages of `[Conv3×3 → BatchNorm → ReLU] × 2 → Dropout(0.5)`, doubling channels each stage (3 → 64 → 128 → 256 → 512), with 2×2 max-pooling between stages.
- **Bottleneck** — the same double-conv block at 512 → 1024 channels.
- **Decoder** — 4 stages of `2×2 transposed convolution` (upsample) → concatenate with the matching encoder feature map (skip connection) → the same double-conv block, mirroring the encoder back down to 64 channels.
- **Output head** — a 1×1 convolution to a single channel, passed through `sigmoid` to produce a per-pixel vessel probability.

Each `conv_block` in the diagram is the full `Conv → BN → ReLU → Conv → BN → ReLU → Dropout(0.5)` unit described above — the diagram shows the block-level flow, not every internal layer. In total the model has **~31.0M trainable parameters**, matching the channel widths of the original Ronneberger et al. U-Net.

A second variant, `ModifiedUNet`, is also defined in the notebook as an experiment: it upsamples the level-2 encoder features and merges them with the level-1 features into one combined tensor, then feeds that combined tensor into the final decoder block in place of the plain level-1 skip connection used in the base `UNet`. It's implemented but **not** the model used in the main training run; swap `model = UNet(...)` for `model = ModifiedUNet(...)` in the `__main__` block to try it.

## Training Setup

| | |
|---|---|
| Input size | 512 × 512 |
| Batch size | 4 |
| Optimizer | Adam |
| Learning rate | 1e-5 |
| Epochs | 50 |
| Loss | `DiceLoss()` + `JaccardLoss()` (summed; both apply `sigmoid` internally) |
| Metrics tracked | Accuracy, Sensitivity (Recall), F1 — computed from TP/FP/FN/TN at a fixed 0.5 threshold |
| Reproducibility | `torch.manual_seed(2)`, `random.seed(2)`, `np.random.seed(2)` |
| Hardware | CUDA GPU (Kaggle GPU accelerator) |

## References

- Ronneberger, O., Fischer, P., & Brox, T. (2015). [*U-Net: Convolutional Networks for Biomedical Image Segmentation.*](https://arxiv.org/abs/1505.04597) MICCAI.
- [STARE](https://cecas.clemson.edu/~ahoover/stare/) — Structured Analysis of the Retina dataset.
- [DRIVE](https://drive.grand-challenge.org/) — Digital Retinal Images for Vessel Extraction dataset.

## License

No license file is currently included in this repository. Until one is added, please check with the repository owner before reusing this code.
