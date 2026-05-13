# SCAFBTSRegressor

**Riemannian Filter Bank Tangent Space for EEG-Based Vigilance Estimation**

A Python implementation of Riemannian geometry-based EEG features for continuous vigilance, drowsiness, and emotion estimation. Submitted to *Journal of Neural Engineering*.

---

## Key Finding

**FBTS+EOG fusion significantly outperforms DE+EOF** (COR 0.618 vs 0.586, p=0.044) because FBTS encodes inter-channel covariance structure that complements ocular information.

---

## Overview

This repository implements the **Filter Bank Tangent Space (FBTS)** pipeline for EEG-based regression tasks:

```
Raw EEG → Filter Bank → SPD Covariance Matrices → Riemannian Tangent Space → Feature Selection → SVR/Ridge Regression
```

### Core Innovation

- **Pre-computed architecture**: Filter + covariance computed once, reused across cross-validation folds → **6.5× speedup**
- **Channel-agnostic**: Adapts to 4–62 channels automatically (tangent space dimension scales with C²)
- **Task-agnostic**: Same features work for both regression (SVR) and classification (SVC)
- **Multimodal fusion**: Seamlessly combines EEG (FBTS/DE/PSD) with EOG features

---

## Datasets

| Dataset | Task | Subjects | Channels | Epochs | Labels |
|---------|------|----------|----------|--------|--------|
| **SEED-VIG** | Vigilance regression | 23 | 17 | 885 × 8s | PERCLOS [0,1] |
| **DROZY** | Drowsiness detection | 14 | 5 | ~75/test × 8s | KSS 1–9 |
| **SEED** | Emotion classification | 15 × 3 sessions | 62 | variable × 4s | ±1/0 |

**Note**: Dataset access requires separate requests to BCML (SEED-VIG/SEED) and ULg (DROZY).

---

## Quick Start

```bash
# Install dependencies
pip install numpy scipy scikit-learn pyriemann mne joblib matplotlib

# Full experiment (SEED-VIG + DROZY, all subjects)
python run_all.py --n-jobs 4

# Quick validation (2 subjects per dataset)
python run_all.py --quick

# SEED-VIG only with Riemannian features
python run_final.py --n-jobs 4

# Multimodal fusion experiments
python run_fusion.py --n-jobs 4

# LOSO cross-subject evaluation
python run_loso.py

# Binary classification (alert vs drowsy)
python run_binary.py
```

---

## Project Structure

```
SCAFBTSRegressor/
├── sca_fbts_fast.py            # ★ Core regressor (pre-computed, 6.5× faster)
├── sca_fbts_regressor.py       # Reference implementation (per-epoch)
├── sca_fbts_torch.py           # GPU-accelerated covariance (experimental)
├── data_loader.py              # SEED-VIG: .mat loading, epoch building
├── drozy_loader.py             # DROZY: .edf loading, KSS labels
├── utils.py                    # COR/RMSE/MAE metrics, 5-fold temporal splits
├── run_all.py                  # ★ One-click multi-dataset experiments
├── run_final.py                # SEED-VIG full evaluation
├── run_fusion.py               # Multimodal fusion experiments
├── run_loso.py                 # Leave-One-Subject-Out cross-subject
├── run_binary.py               # Binary alert/drowsy classification
├── generate_figures.py         # Paper figures + statistical tests
├── paper/                      # JNE submission (12 pages, 5 tables)
└── results/                    # Experiment results (JSON)
```

---

## Core API

```python
from sca_fbts_fast import SCAFBTSRegressorFast

# Initialize
clf = SCAFBTSRegressorFast(
    freq_bands='5band',     # 5 bands: δ, θ, α, β, γ
    estimator='oas',        # OAS shrinkage covariance
    metric='riemann',       # Affine-invariant Riemannian metric
    regressor='svr',        # SVR with RBF kernel
    n_features=100,         # Select top-100 features
    fs=200,                 # Sampling rate
    temporal_smoothing=True, # Moving average smoothing
    smoothing_window=3,
)

# Pre-compute (one-time, fold-independent)
clf.precompute(raw_eeg, fs=200)

# Train/predict per fold
clf.fit(train_indices, y[train_indices])
y_pred = clf.predict(test_indices)
```

---

## Method Details

### Filter Bank Configuration

| Preset | Bands | Dimensions (17ch) |
|--------|-------|-------------------|
| `5band` | δ(1-4), θ(4-8), α(8-14), β(14-31), γ(31-50) | 5 × 153 = 765d |
| `8band` | δ, low-θ, high-θ, low-α, high-α, σ, low-β, high-β | 8 × 153 = 1224d |

### Compared Methods

| Method | Description | Dim (17ch) |
|--------|-------------|------------|
| **DE** | Differential Entropy (current SOTA) | 85d |
| **PSD** | Power Spectral Density | 85d |
| **FBTS** | Riemannian Filter Bank Tangent Space | 765d→100d |
| **EOG** | ICA-separated ocular features | 36d |

---

## Experimental Results

### Main Results (SEED-VIG, 23 subjects)

| Configuration | COR | RMSE | Notes |
|--------------|-----|------|-------|
| DE (17ch) | 0.521 | — | Current SOTA |
| FBTS (17ch, 5band) | 0.520 | — | Equivalent to DE |
| **FBTS+EOG (17ch)** | **0.618** | — | **Best, p=0.044 vs DE+EOG** |
| DE+EOG (17ch) | 0.586 | — | |
| FBTS (4ch forehead) | 0.613 | — | Wearable-ready |

### Statistical Significance

- FBTS ≈ DE in single-modality (p=0.984)
- **FBTS+EOG significantly outperforms DE+EOG** (p=0.044, Cohen's d=0.45)
- 4-channel forehead FBTS achieves 99% of 17-channel performance

---

## Paper

**Riemannian Filter Bank Tangent Space Features for EEG-Based Vigilance Estimation with Multimodal Fusion**

X. Li, H. Tan, L. Zhang

*Submitted to Journal of Neural Engineering, 2026*

---

## Citation

```bibtex
@article{li2026riemannian,
  title={Riemannian Filter Bank Tangent Space Features for EEG-Based Vigilance Estimation with Multimodal Fusion},
  author={Li, X. and Tan, H. and Zhang, L.},
  journal={Journal of Neural Engineering},
  year={2026}
}
```

---

## License

Research use only. Dataset access requires separate agreements with BCML and ULg.
