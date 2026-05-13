#!/usr/bin/env python3
"""Temporal smoothing ablation: test different smoothing window sizes."""
import sys, os, json, time, numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SEED_VIG_ROOT, RESULTS_DIR, DEFAULT_QUICK_N
from data_loader import list_subjects, load_raw_eeg, load_perclos
from utils import cor, get_5fold_splits
from sca_fbts_fast import SCAFBTSRegressorFast

WINDOWS = [1, 3, 5, 7, 9]  # 1 = no smoothing

def evaluate_window(n_subjects, win):
    subjects = list_subjects(SEED_VIG_ROOT)[:n_subjects]
    all_cors = []
    for subj in subjects:
        raw_s, sr_s = load_raw_eeg(SEED_VIG_ROOT, subj)
        y_s = load_perclos(SEED_VIG_ROOT, subj)
        clf = SCAFBTSRegressorFast(freq_bands='5band', estimator='oas',
            metric='riemann', regressor='svr', n_features=100, fs=sr_s,
            temporal_smoothing=win > 1, smoothing_window=win, scaler=True)
        clf.precompute(raw_s, fs=sr_s)
        splits = get_5fold_splits(len(y_s), 5)
        cors = []
        for tr_idx, te_idx in splits:
            clf.fit(tr_idx, y_s[tr_idx])
            y_pred = clf.predict(te_idx)
            cors.append(cor(y_s[te_idx], y_pred))
        all_cors.append(np.mean(cors))
    return float(np.mean(all_cors)), float(np.std(all_cors))

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--n-subjects', type=int, default=DEFAULT_QUICK_N)
    args = p.parse_args()
    n_subj = args.n_subjects
    print(f"Temporal Smoothing Ablation: {n_subj} subjects, windows {WINDOWS}")
    results = []
    for win in WINDOWS:
        print(f"  window={win} ...", end=' ', flush=True)
        t0 = time.time()
        c, s = evaluate_window(n_subj, win)
        r = {'window': win, 'cor_mean': c, 'cor_std': s, 'time_sec': time.time()-t0}
        results.append(r)
        label = 'off' if win == 1 else f'{win}-point'
        print(f"COR={c:.4f}+/-{s:.4f} ({label}, {r['time_sec']:.0f}s)")
    print(f"\n{'Window':<12s} {'COR':>8s} {'CORstd':>8s}")
    for r in results:
        wl = 'off' if r['window']==1 else f"{r['window']}-pt"
        print(f"{wl:<12s} {r['cor_mean']:>8.4f} {r['cor_std']:>8.4f}")
    best = max(results, key=lambda x: x['cor_mean'])
    print(f"\nBest: window={best['window']} (COR={best['cor_mean']:.4f})")
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(os.path.join(RESULTS_DIR, f'results_ablation_smoothing_{ts}.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: results_ablation_smoothing_{ts}.json")

if __name__ == '__main__':
    main()
