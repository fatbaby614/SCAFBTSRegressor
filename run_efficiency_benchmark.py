#!/usr/bin/env python3
"""Computational efficiency benchmark: Base vs Fast vs Torch."""
import sys, os, json, time, numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SEED_VIG_ROOT, RESULTS_DIR, BANDS_5
from data_loader import list_subjects, load_raw_eeg, load_perclos
from utils import cor, get_5fold_splits
from sca_fbts_regressor import SCAFBTSRegressor, apply_bandpass_filter
from sca_fbts_fast import SCAFBTSRegressorFast
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

def benchmark_base(subject_id):
    raw, sr = load_raw_eeg(SEED_VIG_ROOT, subject_id)
    y = load_perclos(SEED_VIG_ROOT, subject_id)
    epoch_len = int(sr * 8)
    n_epochs = raw.shape[0] // epoch_len
    raw_T = raw.T.astype(np.float64)
    epochs = np.array([raw_T[:, i*epoch_len:(i+1)*epoch_len] for i in range(n_epochs)])
    splits = get_5fold_splits(n_epochs, 5)
    tr_idx, te_idx = splits[0]
    X_tr, y_tr = epochs[tr_idx], y[tr_idx]
    timings = {}
    t0 = time.time()
    for low, high in BANDS_5:
        for trial in X_tr:
            apply_bandpass_filter(trial, low, high, sr)
    timings['filter'] = time.time() - t0
    t0 = time.time()
    cov_est = Covariances(estimator='oas')
    for low, high in BANDS_5:
        Xb = np.array([apply_bandpass_filter(t, low, high, sr) for t in X_tr])
        cov_est.fit_transform(Xb)
    timings['covariance'] = time.time() - t0
    t0 = time.time()
    feats = []
    for low, high in BANDS_5:
        Xb = np.array([apply_bandpass_filter(t, low, high, sr) for t in X_tr])
        covs = cov_est.transform(Xb)
        ts = TangentSpace(metric='riemann')
        feats.append(ts.fit_transform(covs, y_tr))
    Xc = np.hstack(feats)
    sel = SelectKBest(f_regression, k=100).fit_transform(Xc, y_tr)
    sc = StandardScaler().fit_transform(sel)
    SVR(kernel='rbf', C=1.0, gamma='scale').fit(sc, y_tr)
    timings['tangent_regr'] = time.time() - t0
    timings['total_per_fold'] = sum(timings.values())
    return timings, n_epochs

def benchmark_fast(subject_id):
    raw, sr = load_raw_eeg(SEED_VIG_ROOT, subject_id)
    y = load_perclos(SEED_VIG_ROOT, subject_id)
    n_ch = raw.shape[1]
    timings = {}
    t0 = time.time()
    clf = SCAFBTSRegressorFast(freq_bands='5band', estimator='oas', metric='riemann',
        regressor='svr', n_features=100, fs=sr, temporal_smoothing=False, scaler=True)
    clf.precompute(raw, fs=sr)
    timings['precompute'] = time.time() - t0
    splits = get_5fold_splits(clf._n_epochs, 5)
    tr_idx, te_idx = splits[0]
    t0 = time.time()
    clf.fit(tr_idx, y[tr_idx])
    _ = clf.predict(te_idx)
    timings['tangent_regr'] = time.time() - t0
    # Estimate per-stage breakdown
    raw_T = raw.T.astype(np.float64)
    t0 = time.time()
    for low, high in BANDS_5:
        apply_bandpass_filter(raw_T, low, high, sr)
    timings['filter_est'] = time.time() - t0
    epoch_len = int(sr * 8)
    filtered = apply_bandpass_filter(raw_T, BANDS_5[0][0], BANDS_5[0][1], sr)
    epochs_arr = np.array([filtered[:, i*epoch_len:(i+1)*epoch_len]
                           for i in range(clf._n_epochs)], dtype=np.float64)
    t0 = time.time()
    Covariances(estimator='oas').transform(epochs_arr)
    timings['cov_est'] = (time.time() - t0) * len(BANDS_5)
    return timings

def benchmark_torch(subject_id):
    raw, sr = load_raw_eeg(SEED_VIG_ROOT, subject_id)
    has_torch = False
    try:
        import torch
        has_torch = True
    except ImportError:
        pass
    timings = {'torch_available': has_torch}
    if has_torch:
        raw_T = raw.T.astype(np.float32)
        epoch_len = int(sr * 8)
        n_epochs = raw_T.shape[1] // epoch_len
        filtered = apply_bandpass_filter(raw_T, BANDS_5[0][0], BANDS_5[0][1], sr)
        epochs_t = np.array([filtered[:, i*epoch_len:(i+1)*epoch_len]
                             for i in range(n_epochs)], dtype=np.float32)
        t0 = time.time()
        from sca_fbts_torch import batch_covariance
        _ = batch_covariance(epochs_t, estimator='scm')
        timings['torch_cov'] = time.time() - t0
        t0 = time.time()
        Covariances(estimator='oas').transform(epochs_t.astype(np.float64))
        timings['pyriemann_cov'] = time.time() - t0
    return timings

def main():
    subjects = list_subjects(SEED_VIG_ROOT)
    subj = subjects[0]
    print(f"Efficiency Benchmark: {subj}")
    all_r = {}
    print(f"\n--- Base ---")
    try:
        bt, ne = benchmark_base(subj)
        all_r['base'] = bt
        for k, v in bt.items():
            print(f"  {k:<20s}: {v:>8.2f}s")
        print(f"  5-fold estimate: {bt['total_per_fold']*5:.0f}s")
    except Exception as e:
        print(f"  ERROR: {e}")
    print(f"\n--- Fast ---")
    try:
        ft = benchmark_fast(subj)
        all_r['fast'] = ft
        for k, v in ft.items():
            if isinstance(v, (int, float)):
                print(f"  {k:<20s}: {v:>8.2f}s")
    except Exception as e:
        print(f"  ERROR: {e}")
    print(f"\n--- Torch ---")
    try:
        tt = benchmark_torch(subj)
        all_r['torch'] = tt
        for k, v in tt.items():
            print(f"  {k:<20s}: {v!r:>8}")
    except Exception as e:
        print(f"  ERROR: {e}")
    if 'base' in all_r and 'fast' in all_r:
        base_5fold = all_r['base']['total_per_fold'] * 5
        fast_5fold = all_r['fast']['precompute'] + all_r['fast']['tangent_regr'] * 5
        if fast_5fold > 0:
            print(f"\n5-fold total: Base={base_5fold:.1f}s, Fast={fast_5fold:.1f}s ({base_5fold/fast_5fold:.1f}x)")
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(os.path.join(RESULTS_DIR, f'results_efficiency_{ts}.json'), 'w') as f:
        json.dump(all_r, f, indent=2)
    print(f"Saved: results_efficiency_{ts}.json")

if __name__ == '__main__':
    main()
