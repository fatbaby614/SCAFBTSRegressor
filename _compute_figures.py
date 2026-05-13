"""
图表数据预计算（跑一次即可）
==========================
将所有耗时计算保存到 paper/figures/cache/，供 generate_figures.py 快速加载。
运行: python _compute_figures.py
"""

import sys, os, json
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import (
    list_subjects, load_raw_eeg, load_perclos,
    build_de_features_for_baseline, load_eog_features
)
from utils import cor, get_5fold_splits
from sca_fbts_fast import SCAFBTSRegressorFast
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from pyriemann.tangentspace import TangentSpace

from config import SEED_VIG_ROOT as SEED_VIG, CACHE_DIR as CACHE, FIGURES_DIR as OUTPUT

ROOT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(CACHE, exist_ok=True)

BANDS_5 = [(1, 4), (4, 8), (8, 14), (14, 31), (31, 50)]


# ═══════════════════════════════════════════════════
# 1. 预测曲线数据：23 被试 × FBTS+EOG 5-fold
# ═══════════════════════════════════════════════════

def compute_prediction_curves():
    """对每个被试跑 FBTS+EOG 5-fold，收集 (subj, cor, y_true, y_pred)。"""
    print("Computing prediction curves (23 subjects, FBTS+EOG 5-fold)...")
    subjects = list_subjects(SEED_VIG)
    n_subj = len(subjects)
    n_epochs = 885

    subjects_arr = []
    cors_arr = np.zeros(n_subj, dtype=np.float64)
    yt_arr = np.zeros((n_subj, n_epochs), dtype=np.float64)
    yp_arr = np.zeros((n_subj, n_epochs), dtype=np.float64)

    for i, subj in enumerate(subjects):
        print(f"  [{i+1}/{n_subj}] {subj}")
        raw, sr = load_raw_eeg(SEED_VIG, subj)
        y = load_perclos(SEED_VIG, subj)
        eog = load_eog_features(SEED_VIG, subj)
        eog = StandardScaler().fit_transform(eog)

        clf = SCAFBTSRegressorFast(
            freq_bands='5band', estimator='oas', metric='riemann',
            regressor='svr', n_features=100, fs=sr,
            temporal_smoothing=False, scaler=True,
        )
        clf.precompute(raw, fs=sr)

        splits = get_5fold_splits(n_epochs, 5)
        y_pred_all = np.zeros(n_epochs)

        for tr_idx, te_idx in splits:
            feats_list = []
            for b in BANDS_5:
                covs_tr = clf._epochs_cov[b][tr_idx]
                covs_te = clf._epochs_cov[b][te_idx]
                ts = TangentSpace(metric='riemann')
                f_tr = ts.fit_transform(covs_tr, y[tr_idx])
                f_te = ts.transform(covs_te)
                feats_list.append((f_tr, f_te))
            X_tr = np.hstack([f[0] for f in feats_list])
            X_te = np.hstack([f[1] for f in feats_list])
            X_tr = np.hstack([X_tr, eog[tr_idx]])
            X_te = np.hstack([X_te, eog[te_idx]])

            sel = SelectKBest(f_regression, k=min(150, X_tr.shape[1]))
            X_tr = sel.fit_transform(X_tr, y[tr_idx])
            X_te = sel.transform(X_te)
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
            svr = SVR(kernel='rbf', C=1.0, gamma='scale')
            svr.fit(X_tr, y[tr_idx])
            y_pred_all[te_idx] = svr.predict(X_te)

        subjects_arr.append(subj)
        cors_arr[i] = cor(y, y_pred_all)
        yt_arr[i] = y
        yp_arr[i] = y_pred_all

    # 保存
    np.savez(os.path.join(CACHE, 'cors_data.npz'),
             subjects=np.array(subjects_arr),
             cors=cors_arr,
             y_true=yt_arr, y_pred=yp_arr)
    print(f"  Saved: {CACHE}/cors_data.npz")


# ═══════════════════════════════════════════════════
# 2. 热力图数据：Alert vs Drowsy 平均 DE 特征
# ═══════════════════════════════════════════════════

def compute_heatmap():
    """计算各通道×频段的平均 DE 特征值 (alert < 0.4 vs drowsy > 0.6)。"""
    print("Computing heatmap data (23 subjects DE features)...")
    subjects = list_subjects(SEED_VIG)
    all_de_alert = []
    all_de_drowsy = []

    for subj in subjects:
        feats = build_de_features_for_baseline(SEED_VIG, subj,
                                                channels='all',
                                                feature_type='de_LDS')[0]
        y = load_perclos(SEED_VIG, subj)
        mask_alert = y < 0.4
        mask_drowsy = y > 0.6
        if mask_alert.sum() > 10:
            all_de_alert.append(feats[mask_alert].mean(axis=0))
        if mask_drowsy.sum() > 10:
            all_de_drowsy.append(feats[mask_drowsy].mean(axis=0))

    de_alert = np.mean(all_de_alert, axis=0).reshape(17, 5)
    de_drowsy = np.mean(all_de_drowsy, axis=0).reshape(17, 5)

    np.savez(os.path.join(CACHE, 'heatmap_data.npz'),
             de_alert=de_alert, de_drowsy=de_drowsy)
    print(f"  Saved: {CACHE}/heatmap_data.npz")


# ═══════════════════════════════════════════════════
# 3. 特征维度消融数据
# ═══════════════════════════════════════════════════

def compute_ablation():
    """不同 n_features 对 COR 的影响 (5 被试)。"""
    print("Computing feature ablation (5 subjects, 10 n_features values)...")
    subjects = list_subjects(SEED_VIG)[:5]
    n_features_list = [10, 25, 50, 75, 100, 150, 200, 300, 500, None]

    results_riemann = {str(n) if n else 'All': [] for n in n_features_list}

    for subj in subjects:
        print(f"  {subj}")
        raw, sr = load_raw_eeg(SEED_VIG, subj)
        y = load_perclos(SEED_VIG, subj)

        clf = SCAFBTSRegressorFast(
            freq_bands='5band', estimator='oas', metric='riemann',
            regressor='svr', n_features=None, fs=sr, temporal_smoothing=False,
        )
        clf.precompute(raw, fs=sr)

        splits = get_5fold_splits(885, 5)
        for n_feat in n_features_list:
            key = str(n_feat) if n_feat else 'All'
            cor_list = []
            for tr_idx, te_idx in splits:
                feats_list = []
                for b in BANDS_5:
                    covs_tr = clf._epochs_cov[b][tr_idx]
                    covs_te = clf._epochs_cov[b][te_idx]
                    ts = TangentSpace(metric='riemann')
                    f_tr = ts.fit_transform(covs_tr, y[tr_idx])
                    f_te = ts.transform(covs_te)
                    feats_list.append((f_tr, f_te))
                X_tr = np.hstack([f[0] for f in feats_list])
                X_te = np.hstack([f[1] for f in feats_list])

                if n_feat and X_tr.shape[1] > n_feat:
                    sel = SelectKBest(f_regression, k=n_feat)
                    X_tr = sel.fit_transform(X_tr, y[tr_idx])
                    X_te = sel.transform(X_te)
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr)
                X_te = scaler.transform(X_te)
                svr = SVR(kernel='rbf', C=1.0, gamma='scale')
                svr.fit(X_tr, y[tr_idx])
                y_pred = svr.predict(X_te)
                cor_list.append(cor(y[te_idx], y_pred))
            results_riemann[key].append(np.mean(cor_list))

    # 转换为 means + stds
    keys_sorted = sorted(results_riemann.keys(), key=lambda k: (
        float(k) if k != 'All' else 1e9))
    means = np.array([np.mean(results_riemann[k]) for k in keys_sorted])
    stds = np.array([np.std(results_riemann[k]) for k in keys_sorted])
    labels = np.array(keys_sorted)

    np.savez(os.path.join(CACHE, 'ablation_data.npz'),
             labels=labels, means=means, stds=stds)
    print(f"  Saved: {CACHE}/ablation_data.npz")


# ═══════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("Precomputing figure data")
    print("=" * 60)

    compute_prediction_curves()
    compute_heatmap()
    compute_ablation()

    print(f"\nDone. Cache saved to: {CACHE}")