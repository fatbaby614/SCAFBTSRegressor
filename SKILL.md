---
name: riemann-vigilance
description: Riemannian FBTS pipeline for EEG vigilance, drowsiness & emotion. SEED-VIG/DROZY/SEED. Complete experiment matrix: within-subject, fusion, LOSO, binary, SEED cross-task, 7 ablations. JNE paper (12pp, 5 tables, 32 refs). Submitted 2026.
allowed-tools: Read Write Edit
metadata:
  author: Huang Tan, Xiangzhu Li, Li Zhang
  version: "3.1.0"
  domain: bci-eeg riemannian-geometry
  last_updated: "2026-05-12"
  paper: "submitted to J. Neural Eng."
  paper_status: draft-complete
  datasets: SEED-VIG DROZY SEED
---

# Riemannian FBTS — EEG Vigilance, Drowsiness & Emotion Estimation

## Quick Commands

```bash
python run_all.py --n-jobs 8              # Full experiment matrix
python run_all.py --quick                 # Quick (2 subjects/dataset)
python run_all.py --skip-extra            # Core only (Tables 1-5)
python run_ablation_bands.py --n-subjects 5
python run_feature_provenance.py --n-subjects 5
python _compute_figures.py && python generate_figures.py
cd paper && pdflatex paper_jne.tex && pdflatex paper_jne.tex
```

## Core Pipeline

```
Raw EEG → Butterworth Filter Bank → SPD Covariance (OAS) → Riemannian Tangent Space → SelectKBest → SVR/Ridge
```

## Experiment Matrix (15 experiments)

### Paper (Tables 1-5, always run)
| # | Experiment | Script |
|---|---|---|
| 1 | Within-subject regression (DE/PSD/FBTS) | run_all.py / run_final.py |
| 2 | Multimodal fusion (~40 configs) | run_fusion.py |
| 3 | Riemannian metric ablation | run_fusion.py |
| 4 | LOSO cross-subject | run_loso.py |
| 5 | PERCLOS binary classification | run_binary.py |
| 6 | DROZY LOTO | run_all.py inline |
| 7 | SEED cross-task | run_seed_fast.py / run_seed_fbts.py |

### Ablation (--skip-extra to skip)
| ID | Experiment | Script | N | ~Time |
|---|---|---|---|---|
| A | Covariance estimator | run_ablation_estimator.py | 5 | 5 min |
| B | Frequency band (LOBO) | run_ablation_bands.py | 5 | 5 min |
| C | Regressor choice | run_ablation_regressor.py | 5 | 3 min |
| D | Temporal smoothing | run_ablation_smoothing.py | 5 | 5 min |
| E | Feature provenance | run_feature_provenance.py | 5 | 1 min |
| F | Data efficiency | run_data_efficiency.py | 5 | 5 min |
| G | Computational efficiency | run_efficiency_benchmark.py | 1 | 2 min |

## Key Results (from Linux workstation run)

| Finding | Value |
|---|---|
| FBTS ≈ DE single-modality | COR 0.520 vs 0.521 |
| FBTS+EOG > DE+EOG | COR 0.618 vs 0.586 |
| LOSO FBTS+EOG 6ch (strongest) | COR 0.835 |
| 4ch forehead ≈ 17ch | COR 0.613 vs 0.618 |
| Binary FBTS+EOG 17ch | ACC 0.993, F1 0.981 |
| DROZY all negative | COR -0.89 to -0.94 |
| SEED DE 62ch | ACC 86.1%, F1 0.846 |
| Fast vs Base speedup | 6.4× |
| OAS best estimator | COR 0.599 |
| RidgeCV best regressor | COR 0.609 |

## Architecture

```
config.py          ← Path config (Linux/Windows auto-detect)
sca_fbts_fast.py   ← Main engine (pre-compute, 6.5× faster)
run_all.py         ← One-click orchestrator
data_loader.py     ← SEED-VIG loader
drozy_loader.py    ← DROZY loader
utils.py           ← COR/RMSE/MAE + chronological 5-fold
generate_figures.py ← Paper figures + statistics
paper/paper_jne.tex ← JNE manuscript
```

## Datasets

| | SEED-VIG | DROZY | SEED |
|---|---|---|---|
| Subjects | 23 | 14 | 15×3 |
| Channels | 17 | 5 | 62 |
| Epochs | 885×8s | ~75/test×8s | var×4s |
| Labels | PERCLOS [0,1] | KSS 1-9 | ±1/0 |
| CV | 5-fold chronological | LOTO | Trial-level 5-fold |

Linux paths: `/mnt/data1/home/tanhuang/datasets/{SEED-VIG,DROZY,SEED}/`

## Changelog

| Version | Date | Changes |
|---|---|---|
| 3.1.0 | 2026-05-12 | PSD baseline, smoothing/data efficiency, ablation section in paper, dataUsed archive |
| 3.0.0 | 2026-05-12 | config.py, estimator/band/regressor ablation, feature provenance, efficiency benchmark |
| 2.1.0 | 2026-05-10 | SEED cross-task, DROZY analysis |
| 2.0.0 | 2026-05-09 | Fast pre-compute, fusion, LOSO |
| 1.0.0 | 2026-05-01 | Initial: base regressor, SEED-VIG 5-fold |
