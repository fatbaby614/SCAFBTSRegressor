"""
论文图表数据预计算（执行一次，缓存到 paper/figures/cache/）
=============================================================
生成:
  - cors_data.npz       → 预测曲线 + 散点图
  - heatmap_data.npz    → DE 热力图 (Alert vs Drowsy)
  - ablation_data.npz   → 特征数量消融曲线
  - fbts_connectivity.npz → FBTS 通道对可解释性图
"""

import sys, os, json, time
import numpy as np
from collections import defaultdict
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    SEED_VIG_ROOT, CACHE_DIR, BANDS_5, BAND_NAMES_5,
)
from data_loader import list_subjects, load_raw_eeg, load_perclos, load_eog_features, load_eeg_features
from utils import cor, get_5fold_splits
from sca_fbts_fast import SCAFBTSRegressorFast
from pyriemann.tangentspace import TangentSpace
from sklearn.feature_selection import SelectKBest, f_regression

os.makedirs(CACHE_DIR, exist_ok=True)
N_FOLDS = 5; N_FEATURES = 100


def compute_prediction_cache():
    """FBTS+EOG 5band 17ch: 23 被试 true/predicted + COR"""
    print("Computing prediction cache...")
    subjects = list_subjects(SEED_VIG_ROOT)
    cors_list, y_true_list, y_pred_list = [], [], []
    for i, subj in enumerate(subjects):
        raw, sr = load_raw_eeg(SEED_VIG_ROOT, subj)
        y = load_perclos(SEED_VIG_ROOT, subj)
        eog = load_eog_features(SEED_VIG_ROOT, subj)
        clf = SCAFBTSRegressorFast(
            freq_bands='5band', estimator='oas', metric='riemann',
            regressor='svr', n_features=100, fs=sr,
            temporal_smoothing=False, scaler=True,
        )
        clf.precompute(raw, fs=sr)
        splits = get_5fold_splits(clf._n_epochs, 5)
        ya, yp = np.zeros(clf._n_epochs), np.zeros(clf._n_epochs)
        for tr_idx, te_idx in splits:
            clf.fit(tr_idx, y[tr_idx], X_eog_extra=eog[tr_idx])
            yp[te_idx] = clf.predict(te_idx, X_eog_extra=eog[te_idx])
            ya[te_idx] = y[te_idx]
        ys = uniform_filter1d(yp, size=3)
        cors_list.append(cor(ya, ys))
        y_true_list.append(ya); y_pred_list.append(ys)
        if (i+1) % 5 == 0: print(f"  [{i+1}/{len(subjects)}]")
    np.savez(os.path.join(CACHE_DIR, 'cors_data.npz'),
             subjects=np.array(subjects, dtype=object),
             cors=np.array(cors_list),
             y_true=np.array(y_true_list, dtype=object),
             y_pred=np.array(y_pred_list, dtype=object))
    print(f"  Saved cors_data.npz")


def compute_heatmap_cache():
    """DE Alert vs Drowsy 热力图"""
    print("Computing heatmap cache...")
    subjects = list_subjects(SEED_VIG_ROOT)
    al, dr = [], []
    for subj in subjects:
        y = load_perclos(SEED_VIG_ROOT, subj)
        feats = load_eeg_features(SEED_VIG_ROOT, subj,
                                  feature_dir='EEG_Feature_5Bands',
                                  feature_type='de_movingAve')  # (17, 885, 5)
        al.append(feats[:, y < 0.4, :].mean(axis=1))
        dr.append(feats[:, y > 0.6, :].mean(axis=1))
    np.savez(os.path.join(CACHE_DIR, 'heatmap_data.npz'),
             de_alert=np.mean(al, axis=0), de_drowsy=np.mean(dr, axis=0))
    print(f"  Saved heatmap_data.npz")


def compute_ablation_cache():
    """特征数量消融曲线 (23 被试)"""
    print("Computing ablation cache (23 subjects)...")
    subjects = list_subjects(SEED_VIG_ROOT)
    nfs = [10, 25, 50, 100, 200, 400, 765]
    means, stds = [], []
    for nf in nfs:
        cors = []
        for subj in subjects:
            raw, sr = load_raw_eeg(SEED_VIG_ROOT, subj)
            y = load_perclos(SEED_VIG_ROOT, subj)
            clf = SCAFBTSRegressorFast(
                freq_bands='5band', estimator='oas', metric='riemann',
                regressor='svr', n_features=nf, fs=sr,
                temporal_smoothing=True, smoothing_window=3,
            )
            clf.precompute(raw, fs=sr)
            fc = []
            for tr, te in get_5fold_splits(len(y), 5):
                clf.fit(tr, y[tr]); yp = clf.predict(te)
                if len(yp) > 1: yp = uniform_filter1d(yp, size=3)
                fc.append(cor(y[te], yp))
            cors.append(np.mean(fc))
        means.append(np.mean(cors)); stds.append(np.std(cors))
        print(f"  n={nf:3d} COR={np.mean(cors):.4f}")
    np.savez(os.path.join(CACHE_DIR, 'ablation_data.npz'),
             labels=np.array([str(n) for n in nfs], dtype=object),
             means=np.array(means), stds=np.array(stds))
    print(f"  Saved ablation_data.npz")


def compute_fbts_connectivity():
    """FBTS 通道对可解释性: top-100 特征回溯到 (band, channel_pair)"""
    print("Computing FBTS connectivity...")
    subjects = list_subjects(SEED_VIG_ROOT)
    n_ch = 17
    band_dim = n_ch * (n_ch + 1) // 2
    n_bands = len(BANDS_5)
    accum = np.zeros((n_bands, n_ch, n_ch))
    band_top100 = np.zeros(n_bands)
    n_folds_total = 0

    for si, subj in enumerate(subjects):
        raw, sr = load_raw_eeg(SEED_VIG_ROOT, subj)
        y = load_perclos(SEED_VIG_ROOT, subj)
        clf = SCAFBTSRegressorFast(
            freq_bands='5band', estimator='oas', metric='riemann',
            regressor='svr', n_features=None, fs=sr,
            temporal_smoothing=False, scaler=False,
        )
        clf.precompute(raw, fs=sr)
        for tr_idx, te_idx in get_5fold_splits(len(y), 5):
            feats = []
            for (lo, hi) in clf.freq_bands:
                ts = TangentSpace(metric='riemann')
                feats.append(ts.fit_transform(clf._epochs_cov[(lo, hi)][tr_idx], y[tr_idx]))
            X = np.hstack(feats)
            sel = SelectKBest(score_func=f_regression, k=N_FEATURES)
            sel.fit(X, y[tr_idx])
            for g in np.argsort(sel.scores_)[-N_FEATURES:]:
                bi = int(g // band_dim)
                pi = int(g % band_dim)
                # pi → (i,j) upper-triangular index
                i = int(np.floor((2*n_ch + 1 - np.sqrt((2*n_ch+1)**2 - 8*pi)) / 2))
                j = int(pi - i * n_ch + i * (i - 1) // 2 + i)
                if bi < n_bands and 0 <= i < n_ch and 0 <= j < n_ch:
                    accum[bi, i, j] += 1.0
                    band_top100[bi] += 1
            n_folds_total += 1
        if (si+1) % 5 == 0: print(f"  [{si+1}/{len(subjects)}]")

    if n_folds_total > 0:
        accum /= n_folds_total
    total = band_top100.sum()
    bp = band_top100 / total * 100 if total > 0 else np.zeros(n_bands)
    for bi, n in enumerate(BAND_NAMES_5):
        print(f"  {n:<15s} {bp[bi]:.1f}%")

    np.savez(os.path.join(CACHE_DIR, 'fbts_connectivity.npz'),
             importance=accum, band_pct=bp,
             band_names=np.array(BAND_NAMES_5, dtype=object),
             ch_names=np.array(['FT7','FT8','T7','T8','TP7','TP8',
                                'CP1','CP2','P1','PZ','P2',
                                'PO3','POZ','PO4','O1','OZ','O2'], dtype=object))
    print(f"  Saved fbts_connectivity.npz")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    for s in ['prediction','heatmap','ablation','connectivity']:
        p.add_argument(f'--skip-{s}', action='store_true')
    args = p.parse_args()
    t0 = time.time()
    print("="*60+"\nPrecomputing figure data\n"+"="*60)
    if not args.skip_prediction: compute_prediction_cache()
    if not args.skip_heatmap: compute_heatmap_cache()
    if not args.skip_ablation: compute_ablation_cache()
    if not args.skip_connectivity: compute_fbts_connectivity()
    print(f"\nDone in {time.time()-t0:.0f}s -> {CACHE_DIR}")
