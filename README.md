# FracTrans

**Fractional Fourier Transformer for Medical Hyperspectral Image Classification**

FracTrans is a transformer framework for medical hyperspectral image (MHSI) classification. It models non-stationary spatial–frequency correlations with a learnable Fractional Fourier Transform (FrFT), so that fine pathological details can be separated from global anatomical structures.

> Paper status: under revision for *IEEE Transactions on Image Processing* (TIP).

---

## Highlights

| Module | What it does |
|---|---|
| **Learnable FrFT Attention** | Adaptive spatial–frequency rotation with a trainable order \(\alpha\) |
| **Stochastic Frequency Sampling** | Randomly keeps \(K\) modes (\(K=64\)) to avoid \(\mathcal{O}(N^2)\) attention |
| **LSA + GSA** | Sliding local spectral attention + dual-channel global spectral attention |
| **Multi-scale Fusion** | Patch embedding, cross-scale fusion, and pathology-focused pooling |

On choledoch (MDC) and gastric precancerous lesion (PLGC) datasets, FracTrans outperforms strong CNN / Transformer / frequency-domain baselines. Integrating the FrTrans module yields a **+7.35%** accuracy gain in ablation.

---

## Architecture

<p align="center">
  <img src="assets/architecture.png" width="92%" alt="FracTrans architecture"/>
</p>

<p align="center"><em>Overall FracTrans pipeline: PatchEmbed → FrTrans → L-GSA → multi-scale fusion → classification.</em></p>

### Fractional Fourier vs. standard Fourier

<p align="center">
  <img src="assets/frft_vs_ft.png" width="72%" alt="FrFT vs FT"/>
</p>

<p align="center"><em>Adaptive FrFT better preserves high-frequency pathological textures under the same mode dropping ratio.</em></p>

### Local–Global Spectral Attention

<p align="center">
  <img src="assets/local_global.png" width="70%" alt="LSA GSA"/>
  &nbsp;
  <img src="assets/gsa_dc.png" width="26%" alt="Dual-channel GSA"/>
</p>

### Feature fusion & ablation

<p align="center">
  <img src="assets/fusion.png" width="48%" alt="Fusion"/>
  <img src="assets/ablation.png" width="48%" alt="Ablation"/>
</p>

---

## Repository Structure

```text
FracTrans/
├── model/
│   ├── FracTrans.py          # main model (updated)
│   └── __init__.py
├── dataset/
│   ├── DGA_dataset.py        # MDC / choledoch loader
│   └── PLGC_dataset.py       # PLGC loader
├── train_DGA.py              # training script (MDC)
├── train_PLGC.py             # training script (PLGC)
assets/                       # figures used in this README
└── README.md
```

---

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.12 (CUDA recommended)
- NumPy

```bash
pip install torch torchvision numpy
```

---

## Quick Start

```python
import torch
from FracTrans.model.FracTrans import get_SFT_Swin, get_SFT_PLGC_Swin

# MDC / choledoch: 60 bands, binary classification
model = get_SFT_Swin(in_channels=60, num_classes=2, image_size=256)
x = torch.randn(1, 60, 256, 256)
logits, aux_logits = model(x)

# PLGC: 40 bands, 3-class classification
model_plgc = get_SFT_PLGC_Swin(in_channels=40, num_classes=3, image_size=256)
```

Self-check:

```bash
cd FracTrans/model
python FracTrans.py
```

Training (after preparing your dataset paths):

```bash
python train_DGA.py
python train_PLGC.py
```

---

## Feature Visualization

t-SNE embeddings of the learned representations. FracTrans forms clearer class clusters than competing MHSI baselines.

<p align="center">
  <img src="assets/tsne.png" width="92%" alt="t-SNE visualization"/>
</p>

---

## Contact

- Qikai Chen — `qikaichen@bit.edu.cn`
- Corresponding author: Huan Liu — `huanliu233@gmail.com`

Code is released for academic research.
