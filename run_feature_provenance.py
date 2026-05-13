#!/usr/bin/env python3
"""Feature provenance: map SelectKBest scores back to frequency bands."""
import sys, os, json, time, numpy as np
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SEED_VIG_ROOT, RESULTS_DIR, BANDS_5, BAND_NAMES_5, DEFAULT_QUICK_N
from data_loader import list_subjects, load_raw_eeg, load_perclos
from utils import get_5fold_splits
from sca_fbts_fast import SCAFBTSRegressorFast
from pyriemann.tangentspace import TangentSpace
from sklearn.feature_selection import SelectKBest, f_regression

def compute_subject_scores(subject_id):
    raw, sr = load_raw_eeg(SEED_VIG_ROOT, subject_id)
    y = load_perclos(SEED_VIG_ROOT, subject_id)
    n_ch = raw.shape[1]
    band_dim = n_ch * (n_ch + 1) // 2
    clf = SCAFBTSRegressorFast(freq_bands='5band', estimator='oas', metric='riemann',
        regressor='svr', n_features=None, fs=sr, temporal_smoothing=False, scaler=False)
    clf.precompute(raw, fs=sr)
    splits = get_5fold_splits(len(y), 5)
    all_scores = []
    for tr_idx, te_idx in splits:
        feats_list = []
        for (low, high) in clf.freq_bands:
            covs_tr = clf._epochs_cov[(low, high)][tr_idx]
            ts = TangentSpace(metric='riemann')
            ts_feats = ts.fit_transform(covs_tr, y[tr_idx])
            feats_list.append(ts_feats)
        X = np.hstack(feats_list)
        sel = SelectKBest(score_func=f_regression, k='all')
        sel.fit(X, y[tr_idx])
        all_scores.append(sel.scores_)
    avg_scores = np.mean(all_scores, axis=0)
    # Per-band stats
    band_stats = {}
    for i, (band, name) in enumerate(zip(BANDS_5, BAND_NAMES_5)):
        start = i * band_dim
        end = (i + 1) * band_dim
        band_stats[name] = {
            'mean_score': float(np.mean(avg_scores[start:end])),
            'sum_score': float(np.sum(avg_scores[start:end])),
        }
    # Top-100 count per band
    top100_idx = np.argsort(avg_scores)[-100:]
    for i, name in enumerate(BAND_NAMES_5):
        start, end = i * band_dim, (i + 1) * band_dim
        band_stats[name]['top100_count'] = int(np.sum((top100_idx >= start) & (top100_idx < end)))
    return band_stats

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--n-subjects', type=int, default=DEFAULT_QUICK_N)
    args = p.parse_args()
    n_subj = args.n_subjects
    subjects = list_subjects(SEED_VIG_ROOT)[:n_subj]
    print(f"Feature Provenance: {n_subj} subjects")
    all_band = defaultdict(list)
    for i, subj in enumerate(subjects):
        print(f"  [{i+1}/{n_subj}] {subj} ...", end=' ', flush=True)
        t0 = time.time()
        try:
            bs = compute_subject_scores(subj)
            for name, s in bs.items():
                all_band[name].append(s['top100_count'])
            dist = [bs[n]['top100_count'] for n in BAND_NAMES_5]
            print(" | ".join(f"{BAND_NAMES_5[j].split('(')[0]}={dist[j]}" for j in range(5)),
                  f"({time.time()-t0:.0f}s)")
        except Exception as e:
            print(f"ERROR: {e}")
    print(f"\nTop-100 Feature Distribution by Frequency Band:")
    print(f"{'Band':<20s} {'Count':>8s} {'Pct':>8s} {'Neurophysiology'}")
    print("-"*70)
    neuro = {
        'Delta(1-4Hz)': 'deep sleep, minimal vigilance relevance',
        'Theta(4-8Hz)': 'drowsiness marker (Strijkstra 2003)',
        'Alpha(8-14Hz)': 'alpha anteriorization, inter-hemispheric coherence',
        'Beta(14-31Hz)': 'alertness maintenance',
        'Gamma(31-50Hz)': 'high-freq, low SNR in scalp EEG',
    }
    summary = {}
    for name in BAND_NAMES_5:
        vals = all_band[name]
        if vals:
            mean_cnt = np.mean(vals)
            pct = mean_cnt / 100 * 100
            summary[name] = {'count': float(mean_cnt), 'pct': float(pct)}
            print(f"{name:<20s} {mean_cnt:>7.1f} {pct:>7.1f}%  {neuro.get(name,'')}")
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(os.path.join(RESULTS_DIR, f'results_feature_provenance_{ts}.json'), 'w') as f:
        json.dump({'n_subjects': n_subj, 'summary': summary, 'per_subject': all_band}, f, indent=2)
    print(f"Saved: results_feature_provenance_{ts}.json")

if __name__ == '__main__':
    main()
