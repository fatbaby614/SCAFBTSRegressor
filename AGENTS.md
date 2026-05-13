# AGENTS.md — SCAFBTSRegressor

> AI agent onboarding document. Read this first before making any changes.

---

## 1. What This Project Is

A research codebase for **Riemannian Filter Bank Tangent Space (FBTS)** features applied to EEG vigilance estimation. Submitted to *Journal of Neural Engineering* (JNE), IOP Publishing.

**One-sentence core finding:** FBTS+EOG fusion significantly outperforms DE+EOG (COR 0.618 vs 0.586, p=0.044) because FBTS encodes inter-channel covariance structure that complements ocular information, while DE's per-channel spectral energy is redundant with EOG.

**Three datasets:** SEED-VIG (primary, 23 subjects × 17ch × 885 epochs), DROZY (validation, 14 subjects × 5ch), SEED (cross-task, 15×3 sessions × 62ch).

---

## 2. Architecture (three-layer)

```
Core Engine          →  data layer      →  experiment scripts
sca_fbts_fast.py ★     data_loader.py      run_all.py ★
sca_fbts_regressor.py  drozy_loader.py     run_final.py
sca_fbts_torch.py      utils.py            run_fusion.py
                       config.py           run_loso.py ...
```

### Core Engine

Three implementations of the same FBTS pipeline:
- **`sca_fbts_fast.py`** — Production version. Pre-computes filter + covariance once, reuses across folds. **6.5× faster.** Use this.
- **`sca_fbts_regressor.py`** — Reference implementation. Per-epoch computation. Only for debugging.
- **`sca_fbts_torch.py`** — GPU-accelerated covariance via `torch.bmm`. Experimental.

Key API pattern (Fast version):
```python
clf = SCAFBTSRegressorFast(freq_bands='5band', estimator='oas', ...)
clf.precompute(raw_eeg, fs=200)          # One-time: filter + covariance
clf.fit(train_indices, y[train_indices])  # Per-fold: tangent space + SVR
y_pred = clf.predict(test_indices)
```

### Data Layer

- **`config.py`** — Centralized path constants. Auto-detects Linux/Windows via `platform.system()`. **Edit only this file to change paths.**
- **`data_loader.py`** — SEED-VIG loading: `load_raw_eeg()`, `load_perclos()`, `build_epochs_from_raw()`, `build_de_features_for_baseline()`, `load_eog_features()`.
- **`drozy_loader.py`** — DROZY loading: `load_drozy_subject()`, `get_kss_label()`, `extract_de_features()`. Handles EDF→epochs conversion.
- **`utils.py`** — `cor()`, `rmse()`, `mae()`, `get_5fold_splits(n_samples, n_folds)` — chronological, NOT shuffled.

---

## 3. Experiment Matrix

### Paper experiments (Tables 1-5) — always run

| # | What | Script | Table |
|---|---|---|---|
| 1 | Within-subject regression (DE/PSD/FBTS × ch × bands) | `run_all.py` inline | Table 1 |
| 2 | Multimodal fusion (~40 configs) | `run_fusion.py` | Table 2 |
| 3 | Riemannian metric ablation | `run_fusion.py` | Table 3 |
| 4 | LOSO cross-subject | `run_loso.py` | Table 4 |
| 5 | PERCLOS binary classification | `run_binary.py` | — |
| 6 | DROZY LOTO | `run_all.py` inline | Table 5 |
| 7 | SEED cross-task | `run_seed_fast.py` + `run_seed_fbts.py` | — |

### Ablation experiments — `--skip-extra` to skip

| Letter | What | Script | Subjects | ~Time |
|---|---|---|---|---|
| A | Covariance estimator (OAS/LWF/SCM/cov/corr) | `run_ablation_estimator.py` | 5 | 5 min |
| B | Leave-One-Band-Out | `run_ablation_bands.py` | 5 | 5 min |
| C | Regressor (SVR/Ridge/RidgeCV/RFR) | `run_ablation_regressor.py` | 5 | 3 min |
| D | Smoothing window (1/3/5/7/9) | `run_ablation_smoothing.py` | 5 | 5 min |
| E | Feature provenance (SelectKBest→bands) | `run_feature_provenance.py` | 5 | 1 min |
| F | Data efficiency (10%→100%) | `run_data_efficiency.py` | 5 | 5 min |
| G | Computational efficiency (Base/Fast/Torch) | `run_efficiency_benchmark.py` | 1 | 2 min |

---

## 4. Critical Design Decisions (do NOT change)

| Decision | Why |
|---|---|
| Pre-compute architecture | Filter+covariance is 80% of runtime, 100% fold-independent |
| SelectKBest per fold | Feature selection must NOT leak across CV folds |
| Chronological 5-fold (not shuffled) | Consecutive EEG epochs are highly correlated |
| SEED trial-level CV | Same-trial segments must NOT cross folds |
| DROZY negative COR kept | Honest reporting of label sparsity limitation |
| Paths in config.py only | One source of truth; auto-detects OS |

---

## 5. Code Conventions

### Naming
- `run_*.py` — executable experiment scripts (importable, `if __name__ == '__main__'`)
- `_*.py` — analysis/utility scripts (not in main pipeline)
- `sca_fbts_*.py` — core engine modules
- Functions prefixed with `_` are internal to the module

### Return types
- All experiment results are `dict` with keys: `cor_mean`, `cor_std`, `rmse_mean`, `rmse_std`, `time_sec`
- Saved as JSON in `results/results_{type}_{timestamp}.json`

### Args convention
- `--n-subjects N` — number of subjects for ablation (default 5)
- `--n-jobs N` — joblib parallel workers (default -1 = all cores)
- `--quick` — fast mode (2 subjects per dataset)
- `--skip-*` — skip a category of experiments

---

## 6. How to Run

```bash
# One command, everything
python run_all.py --n-jobs 8

# Quick check (2 subjects, ~10 min)
python run_all.py --quick

# Only paper experiments (skip ablations)
python run_all.py --skip-extra

# Single ablation
python run_ablation_bands.py --n-subjects 5
python run_feature_provenance.py --n-subjects 5
```

Linux workstation paths auto-detected. On Windows, `config.py` falls back to `D:\EEG\datasets\`.

---

## 7. Adding a New Experiment

1. Create `run_new_experiment.py`:
   ```python
   import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
   from config import SEED_VIG_ROOT, RESULTS_DIR, DEFAULT_QUICK_N
   # ... your experiment logic ...
   if __name__ == '__main__': main()
   ```
2. Add to `run_all.py` in the `_scripts` list (with `--n-subjects {_nsubj}` if applicable)
3. Add row to `SKILL.md` §3.2 and `README.md` §4 ablation table
4. Update `AGENTS.md` §3 if it's a paper-relevant ablation

---

## 8. Changing Data Paths

Edit ONLY `config.py`. It auto-detects OS:
```python
if platform.system() == 'Linux':
    SEED_VIG_ROOT = '/mnt/data1/home/tanhuang/datasets/SEED-VIG'
else:
    SEED_VIG_ROOT = r'D:\EEG\datasets\SEED-VIG'
```
All other files import from `config`. **Do NOT hardcode paths in scripts.**

---

## 9. Building the Paper

```bash
python _compute_figures.py               # Generate cache (once, slow)
python generate_figures.py               # Figures from cache (fast)
cd paper && pdflatex paper_jne.tex && pdflatex paper_jne.tex
```

Paper structure: `paper/paper_jne.tex` — IOP template, 12pp, 5 tables, 32 refs.

---

## 10. Key Metrics

| Metric | Code | Direction |
|---|---|---|
| COR (Pearson r) | `utils.cor()` | Higher = better |
| RMSE | `utils.rmse()` | Lower = better |
| MAE | `utils.mae()` | Lower = better |

---

## 11. Dataset Quick Reference

| | SEED-VIG | DROZY | SEED |
|---|---|---|---|
| Task | Vigilance regression | Drowsiness detection | Emotion classification |
| Subjects | 23 | 14 | 15×3 sessions |
| Channels | 17 | 5 (Fz,Cz,C3,C4,Pz) | 62 |
| Epochs | 885×8s | ~75/test×8s | var×4s |
| Labels | PERCLOS [0,1] | KSS 1-9 (per-test) | ±1/0 (per-trial) |
| CV | 5-fold chronological | Leave-One-Test-Out | Trial-level 5-fold |

SEED-VIG channel subsets: `all`=0–16, `temporal`=0–5, `forehead`=0–3.

---

## 12. Troubleshooting

| Symptom | Check |
|---|---|
| DROZY negative COR | Expected — confirmed label sparsity limitation |
| `FileNotFoundError` SEED-VIG | Verify `config.py` paths; check `SEED_VIG_ROOT/Raw_Data/` exists |
| `MemoryError` SEED 62ch | FBTS cov dimension = 9765d; use `run_seed_fbts.py` (10ch) instead |
| `joblib` pickle error | Lambda in config not serializable; use `--n-jobs 1` |
| Import errors | Run from project root; all scripts use `sys.path.insert(0, ...)` |

---

## 13. File Map

```
config.py              ← PATH ONLY FILE (edit for new environments)
sca_fbts_fast.py       ← MAIN ENGINE (use this)
run_all.py             ← ORCHESTRATOR (one-click everything)
run_fusion.py          ← FUSION + METRIC ABLATION (Tables 2-3)
run_loso.py            ← CROSS-SUBJECT (Table 4)
run_binary.py          ← ALERT vs DROWSY
data_loader.py         ← SEED-VIG data
drozy_loader.py        ← DROZY data
utils.py               ← COR/RMSE/MAE + splits
generate_figures.py    ← PAPER FIGURES
SKILL.md               ← PROJECT SKILL (human reference)
README.md              ← PROJECT README (Chinese)
doc/DATASETS.md        ← DATASET SPECS
config.py              ← PATH CONFIG (edit once)
```
