#!/usr/bin/env python3
"""Data efficiency curve: how COR changes with more training data."""
import sys, os, json, time, numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SEED_VIG_ROOT, RESULTS_DIR, DEFAULT_QUICK_N, BANDS_5
from data_loader import list_subjects, load_raw_eeg, load_perclos
from utils import cor, get_5fold_splits
from sca_fbts_fast import SCAFBTSRegressorFast
from pyriemann.tangentspace import TangentSpace
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

FRACTIONS = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]

def evaluate_fraction(subject_id, fraction):
    raw, sr = load_raw_eeg(SEED_VIG_ROOT, subject_id)
    y = load_perclos(SEED_VIG_ROOT, subject_id)
    clf = SCAFBTSRegressorFast(freq_bands='5band', estimator='oas',
        metric='riemann', regressor='svr', n_features=None, fs=sr,
        temporal_smoothing=False, scaler=False)
    clf.precompute(raw, fs=sr)
    n_total = len(y)
    n_train = int(n_total * fraction)
    if n_train < 50:
        return None
    splits = get_5fold_splits(n_total, 5)
    cors = []
    for tr_idx, te_idx in splits:
        tr_sub = tr_idx[:int(len(tr_idx)*fraction)]
        if len(tr_sub) < 10:
            continue
        feats_list = []
        for (low, high) in clf.freq_bands:
            covs_tr = clf._epochs_cov[(low, high)][tr_sub]
            ts = TangentSpace(metric='riemann')
            ts_feats = ts.fit_transform(covs_tr, y[tr_sub])
            feats_list.append(ts_feats)
        X_tr = np.hstack(feats_list)
        sel = SelectKBest(f_regression, k=min(100, X_tr.shape[1]))
        X_tr_s = sel.fit_transform(X_tr, y[tr_sub])
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr_s)
        svr = SVR(kernel='rbf', C=1.0, gamma='scale')
        svr.fit(X_tr_s, y[tr_sub])
        feats_te = []
        for (low, high) in clf.freq_bands:
            covs_te = clf._epochs_cov[(low, high)][te_idx]
            ts2 = TangentSpace(metric='riemann')
            f_te = ts2.fit_transform(covs_te, y[te_idx])
            feats_te.append(f_te)
        X_te = np.hstack(feats_te)
        X_te_s = sel.transform(X_te)
        X_te_s = sc.transform(X_te_s)
        y_pred = svr.predict(X_te_s)
        from scipy.ndimage import uniform_filter1d
        y_pred = uniform_filter1d(y_pred, size=3)
        cors.append(cor(y[te_idx], y_pred))
    return float(np.mean(cors)) if cors else None

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--n-subjects', type=int, default=DEFAULT_QUICK_N)
    args = p.parse_args()
    n_subj = args.n_subjects
    subjects = list_subjects(SEED_VIG_ROOT)[:n_subj]
    print(f"Data Efficiency Curve: {n_subj} subjects, fractions {FRACTIONS}")
    all_cors = {f: [] for f in FRACTIONS}
    for i, subj in enumerate(subjects):
        print(f"  [{i+1}/{n_subj}] {subj}")
        for frac in FRACTIONS:
            c = evaluate_fraction(subj, frac)
            if c is not None:
                all_cors[frac].append(c)
        dist = " | ".join(f"{int(f*100)}%={np.mean(all_cors[f]):.3f}" if all_cors[f] else f"{int(f*100)}%=N/A" for f in FRACTIONS)
        print(f"    {dist}")
    print(f"\n{'Fraction':<12s} {'COR_mean':>10s} {'COR_std':>10s}")
    results = []
    for frac in FRACTIONS:
        vals = all_cors[frac]
        if vals:
            m, s = float(np.mean(vals)), float(np.std(vals))
            results.append({'fraction': frac, 'cor_mean': m, 'cor_std': s, 'n': len(vals)})
            print(f"{int(frac*100):>3d}%       {m:>10.4f} {s:>10.4f}")
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(os.path.join(RESULTS_DIR, f'results_data_efficiency_{ts}.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: results_data_efficiency_{ts}.json")

if __name__ == '__main__':
    main()
