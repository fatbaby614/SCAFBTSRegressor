# SCAFBTSRegressor

**Riemannian Filter Bank Tangent Space for EEG-Based Vigilance Estimation**

A Python implementation of Riemannian geometry-based EEG features for continuous vigilance, drowsiness, and emotion estimation. Under major revision at the *Journal of Neural Engineering* (manuscript JNE-110905).

---

## Key Finding

**FBTS is statistically equivalent to DE as a single-modality EEG encoder, and the ocular channel is the dominant source of achievable gain.** In the leakage-free re-computation (scaler, tangent-space reference mean and feature selection all fitted inside each training fold):

- Within-subject: FBTS 0.510 vs DE 0.517 COR — no significant difference ($t(22)=-0.41$, $p=0.68$).
- EOG dominates the fusion gain: DE+EOG improves over DE by **+0.080** COR ($p<10^{-4}$, $d_z=1.11$), FBTS+EOG over FBTS by **+0.101** ($p=0.0008$, $d_z=0.81$); once EOG is present the second EEG encoder adds essentially nothing (FBTS+EOG 0.611 vs DE+EOG 0.596, $p=0.32$).
- Under matched LOSO, an **EOG-only baseline reaches COR 0.765 — 94% of the 0.813 of the best full fusion (DE+EOG)**, and exceeds DE-only by +0.185 ($p=0.0097$).

> An earlier claim — "FBTS+EOG significantly outperforms DE+EOG" — was an artefact of cross-fold leakage in the feature pipeline and has been withdrawn.

---

## Overview

This repository implements the **Filter Bank Tangent Space (FBTS)** pipeline for EEG-based regression tasks:

```
Raw EEG → Filter Bank → SPD Covariance Matrices → Riemannian Tangent Space → Feature Selection → SVR/Ridge Regression
```

### Core Innovation

- **Pre-computed architecture**: Filter + covariance computed once, reused across cross-validation folds → **6.7× speedup** (460.8 s → 69.1 s over a full 5-fold CV)
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

**Note**: Dataset access requires separate requests to BCML (SEED-VIG/SEED) and ULg (DROZY). Result JSON files (`results/`) are gitignored and not committed.

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

# LOSO cross-subject evaluation (matched protocol)
python run_loso.py

# Binary classification (alert vs drowsy)
python run_binary.py
```

### Major-revision experiments (JNE-110905)

```bash
# Subject-level statistics (paired t / Wilcoxon / Cohen's dz / BH-FDR)
python run_stats.py

# PCA-based dimensionality sensitivity analysis (Table 11)
python run_pca_ablation.py

# SEED 62-channel fair comparison (DE-62 vs FBTS-62 ± PCA)
python run_seed_62ch.py

# DROZY label/epoch mapping audit + test-level aggregation
python run_drozy_testlevel.py

# PERCLOS intermediate-range and subject-selection sensitivity
python run_perclos_sensitivity.py

# Decision-level fusion baseline (exploratory, inner-CV weight selection)
python run_decision_fusion.py --n-subjects 5

# Stage-wise latency benchmark with hardware disclosure
python run_latency_benchmark.py --repeats 100
```

---

## Project Structure

```
SCAFBTSRegressor/
├── sca_fbts_fast.py            # ★ Core regressor (pre-computed, 6.7× faster)
├── sca_fbts_regressor.py       # Reference implementation (per-epoch)
├── sca_fbts_torch.py           # GPU-accelerated covariance (experimental)
├── data_loader.py              # SEED-VIG: .mat loading, epoch building
├── drozy_loader.py             # DROZY: .edf loading, KSS labels (with test/epoch audit)
├── utils.py                    # COR/RMSE/MAE metrics, 5-fold temporal splits
├── run_all.py                  # ★ One-click multi-dataset experiments
├── run_final.py                # SEED-VIG full evaluation
├── run_fusion.py               # Multimodal fusion experiments
├── run_loso.py                 # Leave-One-Subject-Out (matched protocol)
├── run_binary.py               # Binary alert/drowsy classification
├── run_stats.py                # Subject-level statistics + paper-ready tables
├── run_pca_ablation.py         # PCA dimensionality sensitivity analysis
├── run_seed_62ch.py            # SEED 62-channel fair comparison
├── run_drozy_testlevel.py      # DROZY label audit + test-level aggregation
├── run_perclos_sensitivity.py  # PERCLOS range / subject-selection sensitivity
├── run_decision_fusion.py      # Decision-level fusion baseline (exploratory)
├── run_latency_benchmark.py    # Stage-wise latency measurement
├── generate_figures.py         # Paper figures + statistical tests
├── precompute_figures.py       # Figure data pre-computation (npz cache)
├── paper/                      # JNE submission (LaTeX source + figures)
└── results/                    # Experiment results (JSON, gitignored)
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

### 1. Within-subject (SEED-VIG, 23 subjects, 5-fold chronological CV, leakage-free)

Mean subject-wise COR:

| Configuration | COR | Global COR | Notes |
|--------------|-----|-----------|-------|
| DE (17ch) | 0.517 | 0.660 | Standard baseline |
| FBTS (17ch, 5band) | 0.510 | 0.654 | Equivalent to DE (p=0.68) |
| DE+EOG (17ch) | 0.596 | 0.689 | +0.080 over DE (p<10⁻⁴, d_z=1.11) |
| FBTS+EOG (17ch, 5band) | 0.611 | 0.726 | +0.101 over FBTS (p=0.0008, d_z=0.81) |
| FBTS+EOG (17ch, 8band) | 0.614 | 0.727 | Best within-subject configuration |
| FBTS+EOG vs DE+EOG | +0.015 | — | Not significant (p=0.32, FDR q=0.52) |
| FBTS+EOG (6ch temporal) | 0.612 | 0.713 | +0.029 over DE+EOG 6ch (p=0.026, FDR q=0.086) |
| DE+FBTS+EOG (4ch anterior, 8band) | 0.605 | 0.706 | Wearable-ready, 136 input dims |

Tri-modal fusion is redundant (p=0.97); a PCA-based dimensionality sensitivity analysis (10 configurations) shows the FBTS-vs-DE comparison is not a dimensionality artefact.

### 2. Cross-subject LOSO (matched protocol, identical folds/labels/regressor for all rows)

| Method | COR | Notes |
|--------|-----|-------|
| DE+EOG (17ch) | **0.813** | Best cross-subject result |
| DE+FBTS+EOG (6ch temp.) | 0.795 | |
| FBTS+EOG (6ch temp.) | 0.788 | vs DE+EOG not significant (p=0.96) |
| FBTS+EOG (17ch) | 0.748 | vs DE+EOG p=0.073, FDR q=0.20 |
| **EOG only** | 0.765 | 94% of the best; > DE-only by +0.185 (p=0.0097, d_z=0.59) |
| DE only (17ch) | 0.580 | |
| FBTS only (17ch) | 0.571 | |

FBTS+EOG does not differ significantly from DE+EOG in any configuration ($p=0.073$–$0.96$, FDR $q\ge0.20$); the dominant effect is the ocular channel, not the EEG encoder.

### 3. Cross-task transfer (SEED, 62 channels, trial-level 5-fold CV)

- 10-channel comparison is a channel-count artefact: FBTS-10 64.7% vs DE-10 66.5% (p=0.19); expanding FBTS to 62 channels gains **+10.3 pp** ($p<10^{-4}$, $d_z=2.15$).
- At 62 channels DE remains stronger: 80.2% vs 75.0% ($p=0.026$); PCA compression (50/100/200) does not close the gap.
- SEED is treated as a cross-task generalisation test, not an independent vigilance replication.

### 4. Added in the major revision

- **DROZY test-level audit**: label/epoch mapping audit + test-level aggregation (exploratory; 14 subjects, 36 independent test points).
- **PERCLOS sensitivity**: intermediate-range (0.4–0.6) exclusion effects quantified; five-subject sensitivity analysis reported cautiously ($t(4)=-1.01$, $p=0.37$).
- **Decision-level fusion**: exploratory baseline on a 5-subject computational-cost subset (subjects 10–14); weight selected by inner CV on training folds only.
- **Stage-wise latency**: filtering → covariance → tangent mapping → inference benchmarked with hardware/software disclosure.
- **Subject-level statistics** everywhere: paired t-test, Wilcoxon, Cohen's $d_z$, 95% CI, Benjamini–Hochberg FDR.

---

## Paper

**Riemannian Filter Bank Tangent Space Features for EEG-Based Vigilance Estimation with Multimodal Fusion**

Xiangzhu Li, Huang Tan, Xiaoyu Yang, Guangmou Cai, Li Li and Li Zhang

*Under major revision, Journal of Neural Engineering, 2026*

---

## Citation

```bibtex
@article{li2026riemannian,
  title={Riemannian Filter Bank Tangent Space Features for EEG-Based Vigilance Estimation with Multimodal Fusion},
  author={Li, Xiangzhu and Tan, Huang and Yang, Xiaoyu and Cai, Guangmou and Li, Li and Zhang, Li},
  journal={Journal of Neural Engineering},
  year={2026},
  note={Manuscript JNE-110905, under major revision}
}
```

---

## License

Research use only. Dataset access requires separate agreements with BCML and ULg.
