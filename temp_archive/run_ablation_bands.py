#!/usr/bin/env python3
"""Leave-One-Band-Out ablation: quantify each frequency band's contribution."""
import sys, os, json, time, numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SEED_VIG_ROOT, RESULTS_DIR, BANDS_5, BAND_NAMES_5, DEFAULT_QUICK_N
from data_loader import list_subjects, load_raw_eeg, load_perclos
from utils import cor, get_5fold_splits
from sca_fbts_fast import SCAFBTSRegressorFast

def evaluate_bands(n_subjects, bands):
    subjects = list_subjects(SEED_VIG_ROOT)[:n_subjects]
    all_cors = []
    for subj in subjects:
        raw_s, sr_s = load_raw_eeg(SEED_VIG_ROOT, subj)
        y_s = load_perclos(SEED_VIG_ROOT, subj)
        clf = SCAFBTSRegressorFast(freq_bands=bands, estimator='oas',
            metric='riemann', regressor='svr', n_features=100, fs=sr_s,
            temporal_smoothing=True, smoothing_window=3)
        clf.precompute(raw_s, fs=sr_s)
        splits = get_5fold_splits(len(y_s), 5)
        cors = []
        for tr_idx, te_idx in splits:
            clf.fit(tr_idx, y_s[tr_idx])
            y_pred = clf.predict(te_idx)
            if clf.temporal_smoothing and len(y_pred) > 1:
                from scipy.ndimage import uniform_filter1d
                y_pred = uniform_filter1d(y_pred, size=clf.smoothing_window)
            cors.append(cor(y_s[te_idx], y_pred))
        all_cors.append(np.mean(cors))
    return float(np.mean(all_cors)), float(np.std(all_cors))

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--n-subjects', type=int, default=DEFAULT_QUICK_N)
    args = p.parse_args()
    n_subj = args.n_subjects
    print(f"Leave-One-Band-Out Ablation: {n_subj} subjects")
    # Baseline
    print(f"  [baseline] All 5 bands ...", end=' ', flush=True)
    t0 = time.time()
    base_cor, base_std = evaluate_bands(n_subj, BANDS_5)
    print(f"COR={base_cor:.4f}+/-{base_std:.4f} ({time.time()-t0:.0f}s)")
    results = [{'condition': 'all_5band', 'label': 'All 5 bands',
                 'cor_mean': base_cor, 'cor_std': base_std, 'cor_drop': 0}]
    # Leave one out
    for i, (band, name) in enumerate(zip(BANDS_5, BAND_NAMES_5)):
        bands_without = [b for j, b in enumerate(BANDS_5) if j != i]
        print(f"  [without_{i}] Without {name} ...", end=' ', flush=True)
        t0 = time.time()
        c, s = evaluate_bands(n_subj, bands_without)
        drop = base_cor - c
        results.append({'condition': f'without_band{i}', 'label': f'Without {name}',
            'removed_band': name, 'cor_mean': c, 'cor_std': s,
            'cor_drop': drop, 'time_sec': time.time()-t0})
        print(f"COR={c:.4f}+/-{s:.4f} (drop={drop:+.4f})")
    print(f"\n{'Condition':<28s} {'COR':>8s} {'CORstd':>8s} {'Drop':>8s}")
    for r in results:
        print(f"{r['label']:<28s} {r['cor_mean']:>8.4f} {r['cor_std']:>8.4f} {r['cor_drop']:>+8.4f}")
    lo = [r for r in results if r['cor_drop'] > 0]
    if lo:
        worst = max(lo, key=lambda x: x['cor_drop'])
        print(f"\nMost impactful: {worst['removed_band']} (drop={worst['cor_drop']:.4f})")
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(os.path.join(RESULTS_DIR, f'results_ablation_bands_{ts}.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: results_ablation_bands_{ts}.json")

if __name__ == '__main__':
    main()
