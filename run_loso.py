"""
跨被试 Leave-One-Subject-Out (LOSO) 实验
========================================
对比 DE vs Riemannian FBTS 在跨被试泛化上的性能。
Riemannian 的优势: 切空间投影可以对齐不同被试的 SPD 流形分布。
"""

import sys
import os
import json
import time
import numpy as np
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import (
    list_subjects, load_raw_eeg, load_perclos, build_de_features_for_baseline,
    load_eog_features
)
from utils import cor, rmse, mae
from sca_fbts_fast import SCAFBTSRegressorFast, apply_bandpass_filter
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.feature_selection import SelectKBest, f_regression

from config import SEED_VIG_ROOT, RESULTS_DIR as OUTPUT_DIR


# ── Ridge 基线 (对齐 Zheng & Lu 2017 的 DE+Ridge LOSO) ──
from sklearn.linear_model import Ridge


def run_loso_de(subjects, feature_type='de_LDS', channels='all',
                regressor='svr', verbose=True):
    """DE 特征 LOSO 评测。

    对每个被试: 用其余 22 人训练 → 测试该被试。
    """
    cor_list, rmse_list = [], []

    for i, test_subj in enumerate(subjects):
        # 收集训练数据
        X_train_list, y_train_list = [], []
        for train_subj in subjects:
            if train_subj == test_subj:
                continue
            X_s, y_s, _ = build_de_features_for_baseline(
                SEED_VIG_ROOT, train_subj,
                channels=channels, feature_type=feature_type
            )
            X_train_list.append(X_s)
            y_train_list.append(y_s)

        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list)

        # 测试数据
        X_test, y_test, _ = build_de_features_for_baseline(
            SEED_VIG_ROOT, test_subj,
            channels=channels, feature_type=feature_type
        )

        # 标准化 (训练集拟合)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # 训练
        if regressor == 'svr':
            clf = SVR(kernel='rbf', C=1.0, gamma='scale')
        else:
            clf = Ridge(alpha=1.0)
        clf.fit(X_train_s, y_train)

        # 预测
        y_pred = clf.predict(X_test_s)
        from scipy.ndimage import uniform_filter1d
        y_pred = uniform_filter1d(y_pred, size=3)

        c = cor(y_test, y_pred)
        r = rmse(y_test, y_pred)
        cor_list.append(c)
        rmse_list.append(r)

        if verbose:
            print(f"  [{i+1}/{len(subjects)}] {test_subj}: COR={c:.4f}, RMSE={r:.4f}")

    return {
        'cor_mean': float(np.mean(cor_list)),
        'cor_std': float(np.std(cor_list)),
        'cor_all': [float(x) for x in cor_list],
        'rmse_mean': float(np.mean(rmse_list)),
        'rmse_std': float(np.std(rmse_list)),
    }


def run_loso_riemann(subjects, bands='5band', channels='all', verbose=True):
    """Riemannian FBTS LOSO 评测。

    策略:
    1. 每个被试预计算协方差缓存 (滤波 + cov)
    2. LOSO: 池化训练被试的协方差 → 切空间 → 特征选择 → SVR
    """
    print(f"  Precomputing all {len(subjects)} subjects...")

    # 通道选择
    if channels == 'temporal':
        ch_idx = [0, 1, 2, 3, 4, 5]
    elif channels == 'forehead':
        ch_idx = [0, 1, 2, 3]
    else:
        ch_idx = None

    # 第一步: 预计算每个被试的滤波数据 + 协方差
    freq_bands = [(1,4),(4,8),(8,14),(14,31),(31,50)] if bands == '5band' else \
                 [(1,4),(4,6),(6,8),(8,10),(10,12),(12,14),(14,20),(20,30)]

    all_covs = {}   # {subject: {band: covs}}
    all_labels = {}  # {subject: y}
    fs = 200

    for subj in subjects:
        raw, sr = load_raw_eeg(SEED_VIG_ROOT, subj)
        y = load_perclos(SEED_VIG_ROOT, subj)
        all_labels[subj] = y

        if ch_idx is not None:
            raw = raw[:, ch_idx]

        raw_T = raw.T.astype(np.float64)
        epoch_len = int(sr * 8)
        n_epochs = raw.shape[0] // epoch_len
        n_ch = raw.shape[1]

        cov_dict = {}
        cov_est = Covariances(estimator='oas')

        for low, high in freq_bands:
            filtered = apply_bandpass_filter(raw_T, low, high, sr)
            epochs = np.array([
                filtered[:, i * epoch_len:(i + 1) * epoch_len]
                for i in range(n_epochs)
            ], dtype=np.float64)
            cov_dict[(low, high)] = cov_est.transform(epochs)

        all_covs[subj] = cov_dict

    print(f"  Precomputed. Running LOSO...")

    # 第二步: LOSO
    cor_list, rmse_list = [], []

    for i, test_subj in enumerate(subjects):
        # 池化训练被试的协方差
        train_covs_per_band = {b: [] for b in freq_bands}
        train_labels = []

        for train_subj in subjects:
            if train_subj == test_subj:
                continue
            train_labels.append(all_labels[train_subj])
            for b in freq_bands:
                train_covs_per_band[b].append(all_covs[train_subj][b])

        y_train = np.concatenate(train_labels)

        # 对每个频段: 拼接协方差 → 切空间
        features_list = []
        for b in freq_bands:
            covs_pooled = np.concatenate(train_covs_per_band[b], axis=0)
            ts = TangentSpace(metric='riemann')
            ts_feats = ts.fit_transform(covs_pooled, y_train)
            # 测试被试的协方差 → 切空间投影
            test_covs = all_covs[test_subj][b]
            test_feats = ts.transform(test_covs)
            features_list.append((ts_feats, test_feats))

        # 拼接特征
        X_train = np.hstack([f[0] for f in features_list])
        X_test = np.hstack([f[1] for f in features_list])
        y_test = all_labels[test_subj]

        # 特征选择
        if X_train.shape[1] > 100:
            sel = SelectKBest(f_regression, k=100)
            X_train = sel.fit_transform(X_train, y_train)
            X_test = sel.transform(X_test)

        # 标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 训练
        clf = SVR(kernel='rbf', C=1.0, gamma='scale')
        clf.fit(X_train, y_train)

        # 预测
        y_pred = clf.predict(X_test)
        from scipy.ndimage import uniform_filter1d
        y_pred = uniform_filter1d(y_pred, size=3)

        c = cor(y_test, y_pred)
        r = rmse(y_test, y_pred)
        cor_list.append(c)
        rmse_list.append(r)

        if verbose:
            print(f"  [{i+1}/{len(subjects)}] {test_subj}: COR={c:.4f}, RMSE={r:.4f}")

    return {
        'cor_mean': float(np.mean(cor_list)),
        'cor_std': float(np.std(cor_list)),
        'cor_all': [float(x) for x in cor_list],
        'rmse_mean': float(np.mean(rmse_list)),
        'rmse_std': float(np.std(rmse_list)),
    }


def run_loso_riemann_eog(subjects, bands='5band', channels='all', verbose=True):
    """Riemannian FBTS + EOG LOSO 评测。"""
    print(f"  Precomputing {len(subjects)} subjects (FBTS + EOG)...")

    # EOG 特征加载
    all_eog = {}
    for subj in subjects:
        eog = load_eog_features(SEED_VIG_ROOT, subj, method='features_table_ica')
        all_eog[subj] = StandardScaler().fit_transform(eog.astype(np.float64))

    # Riemannian 预计算
    if channels == 'temporal':
        ch_idx = [0, 1, 2, 3, 4, 5]
    elif channels == 'forehead':
        ch_idx = [0, 1, 2, 3]
    else:
        ch_idx = None

    freq_bands = [(1, 4), (4, 8), (8, 14), (14, 31), (31, 50)] if bands == '5band' else \
                 [(1, 4), (4, 6), (6, 8), (8, 10), (10, 12),
                  (12, 14), (14, 20), (20, 30)]

    all_covs = {}
    all_labels = {}

    for subj in subjects:
        raw, sr = load_raw_eeg(SEED_VIG_ROOT, subj)
        y = load_perclos(SEED_VIG_ROOT, subj)
        all_labels[subj] = y

        if ch_idx is not None:
            raw = raw[:, ch_idx]

        raw_T = raw.T.astype(np.float64)
        epoch_len = int(sr * 8)
        n_epochs = raw.shape[0] // epoch_len
        cov_dict = {}
        cov_est = Covariances(estimator='oas')

        for low, high in freq_bands:
            filtered = apply_bandpass_filter(raw_T, low, high, sr)
            epochs = np.array([
                filtered[:, i * epoch_len:(i + 1) * epoch_len]
                for i in range(n_epochs)
            ], dtype=np.float64)
            cov_dict[(low, high)] = cov_est.transform(epochs)
        all_covs[subj] = cov_dict

    print(f"  Precomputed. Running LOSO...")

    cor_list, rmse_list = [], []

    for i, test_subj in enumerate(subjects):
        train_covs_per_band = {b: [] for b in freq_bands}
        train_eog_list = []
        train_labels = []

        for train_subj in subjects:
            if train_subj == test_subj:
                continue
            train_labels.append(all_labels[train_subj])
            train_eog_list.append(all_eog[train_subj])
            for b in freq_bands:
                train_covs_per_band[b].append(all_covs[train_subj][b])

        y_train = np.concatenate(train_labels)

        # 切空间特征
        feats_list = []
        for b in freq_bands:
            covs_pooled = np.concatenate(train_covs_per_band[b], axis=0)
            ts = TangentSpace(metric='riemann')
            ts_feats = ts.fit_transform(covs_pooled, y_train)
            test_feats = ts.transform(all_covs[test_subj][b])
            feats_list.append((ts_feats, test_feats))

        X_train_riem = np.hstack([f[0] for f in feats_list])
        X_test_riem = np.hstack([f[1] for f in feats_list])

        # 拼接 EOG
        X_train_eog = np.concatenate(train_eog_list, axis=0)
        X_test_eog = all_eog[test_subj]

        X_train = np.hstack([X_train_riem, X_train_eog])
        X_test = np.hstack([X_test_riem, X_test_eog])
        y_test = all_labels[test_subj]

        if X_train.shape[1] > 150:
            sel = SelectKBest(f_regression, k=150)
            X_train = sel.fit_transform(X_train, y_train)
            X_test = sel.transform(X_test)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = Ridge(alpha=1.0)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        from scipy.ndimage import uniform_filter1d
        y_pred = uniform_filter1d(y_pred, size=3)

        c = cor(y_test, y_pred)
        r = rmse(y_test, y_pred)
        cor_list.append(c)
        rmse_list.append(r)

        if verbose:
            print(f"  [{i+1}/{len(subjects)}] {test_subj}: COR={c:.4f}, RMSE={r:.4f}")

    return {
        'cor_mean': float(np.mean(cor_list)),
        'cor_std': float(np.std(cor_list)),
        'cor_all': [float(x) for x in cor_list],
        'rmse_mean': float(np.mean(rmse_list)),
        'rmse_std': float(np.std(rmse_list)),
    }


def main():
    print(f"{'='*70}")
    print(f"SEED-VIG LOSO (Leave-One-Subject-Out) Experiment")
    print(f"{'='*70}")

    subjects = list_subjects(SEED_VIG_ROOT)
    print(f"Subjects: {len(subjects)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = []

    # ── DE 基线 (SVR + Ridge) ─────────────────────────────
    for ch_name, ch_key in [('17ch', 'all'), ('6ch_temporal', 'temporal')]:
        for reg in ['svr', 'ridge']:
            name = f'LOSO_DE_{reg}_{ch_name}'
            print(f"\n{name}")
            t0 = time.time()
            r = run_loso_de(subjects, feature_type='de_LDS',
                           channels=ch_key, regressor=reg)
            elapsed = time.time() - t0
            print(f"  -> COR={r['cor_mean']:.4f}+/-{r['cor_std']:.4f}, "
                  f"RMSE={r['rmse_mean']:.4f}, {elapsed:.0f}s")
            all_results.append({
                'exp_name': name,
                'cor_mean': r['cor_mean'], 'cor_std': r['cor_std'],
                'rmse_mean': r['rmse_mean'], 'rmse_std': r['rmse_std'],
                'cor_all': r['cor_all'], 'time_sec': elapsed,
            })

    # ── Riemannian LOSO ───────────────────────────────────
    for ch_name, ch_key in [('17ch', 'all'), ('6ch_temporal', 'temporal')]:
        for bands in ['5band', '8band']:
            name = f'LOSO_Riemann_{bands}_{ch_name}'
            print(f"\n{name}")
            t0 = time.time()
            r = run_loso_riemann(subjects, bands=bands, channels=ch_key)
            elapsed = time.time() - t0
            print(f"  -> COR={r['cor_mean']:.4f}+/-{r['cor_std']:.4f}, "
                  f"RMSE={r['rmse_mean']:.4f}, {elapsed:.0f}s")
            all_results.append({
                'exp_name': name,
                'cor_mean': r['cor_mean'], 'cor_std': r['cor_std'],
                'rmse_mean': r['rmse_mean'], 'rmse_std': r['rmse_std'],
                'cor_all': r['cor_all'], 'time_sec': elapsed,
            })

    # ── Riemannian + EOG LOSO ────────────────────────────
    for ch_name, ch_key in [('17ch', 'all'), ('6ch_temporal', 'temporal')]:
        for bands in ['5band', '8band']:
            name = f'LOSO_Riemann+EOG_{bands}_{ch_name}'
            print(f"\n{name}")
            t0 = time.time()
            r = run_loso_riemann_eog(subjects, bands=bands, channels=ch_key)
            elapsed = time.time() - t0
            print(f"  -> COR={r['cor_mean']:.4f}+/-{r['cor_std']:.4f}, "
                  f"RMSE={r['rmse_mean']:.4f}, {elapsed:.0f}s")
            all_results.append({
                'exp_name': name,
                'cor_mean': r['cor_mean'], 'cor_std': r['cor_std'],
                'rmse_mean': r['rmse_mean'], 'rmse_std': r['rmse_std'],
                'cor_all': r['cor_all'], 'time_sec': elapsed,
            })

    # ── 汇总 ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"LOSO Results Summary")
    print(f"{'='*70}")
    print(f"{'Experiment':<30s} {'COR':>8s} {'COR_std':>8s} {'RMSE':>8s} {'Time':>8s}")
    print("-" * 65)
    for r in sorted(all_results, key=lambda x: -(x.get('cor_mean', 0))):
        print(f"{r['exp_name']:<30s} {r['cor_mean']:>8.4f} {r['cor_std']:>8.4f} "
              f"{r['rmse_mean']:>8.4f} {r.get('time_sec', 0):>7.0f}s")

    # 保存
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(OUTPUT_DIR, f'results_loso_{ts}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()