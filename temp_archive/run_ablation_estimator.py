#!/usr/bin/env python3
"""Covariance estimator ablation: compare 5 SPD estimators for FBTS regression."""
import sys, os, json, time, numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SEED_VIG_ROOT, RESULTS_DIR, ESTIMATORS, DEFAULT_QUICK_N, DEFAULT_N_JOBS
from data_loader import list_subjects, load_raw_eeg, load_perclos
from utils import cor, get_5fold_splits
from sca_fbts_fast import SCAFBTSRegressorFast

def evaluate_one_estimator(n_subjects, estimator):
    subjects = list_subjects(SEED_VIG_ROOT)[:n_subjects]
    all_cors = []
    for subj in subjects:
        raw_s, sr_s = load_raw_eeg(SEED_VIG_ROOT, subj)
        y_s = load_perclos(SEED_VIG_ROOT, subj)
        clf = SCAFBTSRegressorFast(freq_bands='5band', estimator=estimator,
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
    return {'cor_mean': float(np.mean(all_cors)), 'cor_std': float(np.std(all_cors))}

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--n-subjects', type=int, default=DEFAULT_QUICK_N)
    p.add_argument('--n-jobs', type=int, default=DEFAULT_N_JOBS)
    args = p.parse_args()
    n_subj = args.n_subjects
    print(f"Covariance Estimator Ablation: {n_subj} subjects x {len(ESTIMATORS)} estimators")
    results = []
    for est in ESTIMATORS:
        print(f"  [{est}] ...", end=' ', flush=True)
        t0 = time.time()
        try:
            r = evaluate_one_estimator(n_subj, est)
            r['estimator'] = est
            r['time_sec'] = time.time() - t0
            results.append(r)
            print(f"COR={r['cor_mean']:.4f}+/-{r['cor_std']:.4f} ({r['time_sec']:.0f}s)")
        except Exception as e:
            print(f"ERROR: {e}")
    results.sort(key=lambda x: -(x['cor_mean']))
    print(f"\n{'Estimator':<12s} {'COR':>8s} {'CORstd':>8s}")
    for r in results:
        print(f"{r['estimator']:<12s} {r['cor_mean']:>8.4f} {r['cor_std']:>8.4f}")
    print(f"\nBest: {results[0]['estimator']} ({results[0]['cor_mean']:.4f})")
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(os.path.join(RESULTS_DIR, f'results_ablation_estimator_{ts}.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: results_ablation_estimator_{ts}.json")

if __name__ == '__main__':
    main()
