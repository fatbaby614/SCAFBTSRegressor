"""
SEED 数据集加载器 + 跨数据集情绪分类实验
=========================================
SEED: 15 subjects × 3 sessions, 62ch EEG, 5频段 DE 特征
任务: 3类情绪分类 (positive/neutral/negative)
"""

import os, sys, json, time, glob
import numpy as np
import scipy.io as sio
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import cor, rmse, get_5fold_splits
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sca_fbts_fast import SCAFBTSRegressorFast, apply_bandpass_filter
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from config import SEED_ROOT, RESULTS_DIR as OUTPUT

# 每 session 15 个 trial 的情绪标签
TRIAL_LABELS = np.array([1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1])


def list_seed_sessions():
    """列出所有 SEED session 文件 (不含 label.mat/readme)."""
    feat_dir = os.path.join(SEED_ROOT, 'ExtractedFeatures')
    files = sorted([f for f in os.listdir(feat_dir)
                    if f.endswith('.mat') and f not in ('label.mat', 'readme.txt')])
    return [os.path.join(feat_dir, f) for f in files]


def load_seed_de_features(filepath, feature_type='de_LDS'):
    """加载一个 session 的 DE 特征 (所有 15 个 trial).

    Returns:
        X: (n_total_samples, 62*5) DE 特征
        y: (n_total_samples,) 标签 (1,0,-1)
    """
    data = sio.loadmat(filepath)
    X_list, y_list = [], []

    for trial_idx in range(1, 16):
        key = f'{feature_type}{trial_idx}'
        if key not in data:
            continue
        feats = data[key]  # (62, n_segments, 5)
        # 转置: (n_segments, 62, 5) → (n_segments, 310)
        feats_2d = feats.transpose(1, 0, 2).reshape(feats.shape[1], -1)
        X_list.append(feats_2d)
        y_list.extend([TRIAL_LABELS[trial_idx - 1]] * feats.shape[1])

    X = np.concatenate(X_list, axis=0)
    y = np.array(y_list)
    return X.astype(np.float64), y


def load_seed_raw_eeg(filepath, fs=200):
    """加载一个 session 的原始 EEG (所有 15 trial).

    Returns:
        X: (n_total_samples, 62, epoch_len) epoch 格式
        y: (n_total_samples,) 标签
    """
    eeg_dir = os.path.join(SEED_ROOT, 'Preprocessed_EEG')
    fname = os.path.basename(filepath)
    eeg_path = os.path.join(eeg_dir, fname)
    if not os.path.exists(eeg_path):
        return None, None

    data = sio.loadmat(eeg_path)
    epoch_len = fs * 4  # 4s epochs
    X_list, y_list = [], []

    for trial_idx in range(1, 16):
        key = f'ww_eeg{trial_idx}'
        if key not in data:
            continue
        raw = data[key]  # (62, n_times)
        n_times = raw.shape[1]
        n_epochs = n_times // epoch_len
        if n_epochs == 0:
            continue

        # 切分: (n_epochs, 62, epoch_len)
        epochs = np.array([
            raw[:, i * epoch_len:(i + 1) * epoch_len]
            for i in range(n_epochs)
        ], dtype=np.float64)
        X_list.append(epochs)
        y_list.extend([TRIAL_LABELS[trial_idx - 1]] * n_epochs)

    X = np.concatenate(X_list, axis=0)
    y = np.array(y_list)
    return X, y


def evaluate_de_classification(sessions, feature_type='de_LDS', verbose=True):
    """DE 特征情绪分类 (session-level 交叉验证).

    策略: 每 session 独立评测 (session 内按 trial 分组 CV).
    """
    acc_list, f1_list = [], []

    for i, fpath in enumerate(sessions):
        X, y = load_seed_de_features(fpath, feature_type)
        if X is None or len(np.unique(y)) < 3:
            continue

        # 标准化
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # 特征选择
        sel = SelectKBest(f_classif, k=min(200, X.shape[1]))
        X = sel.fit_transform(X, y)

        # Trial-level 5-fold CV (按 trial 分, 避免数据泄露)
        n_trials = 15
        trial_splits = get_5fold_splits(n_trials, 5)

        acc_fold, f1_fold = [], []
        for tr_trials, te_trials in trial_splits:
            tr_mask = np.zeros(len(y), dtype=bool)
            te_mask = np.zeros(len(y), dtype=bool)
            samples_per_trial = len(y) // 15
            for t in tr_trials:
                tr_mask[t * samples_per_trial:(t+1) * samples_per_trial] = True
            for t in te_trials:
                te_mask[t * samples_per_trial:(t+1) * samples_per_trial] = True

            X_tr, y_tr = X[tr_mask], y[tr_mask]
            X_te, y_te = X[te_mask], y[te_mask]

            clf = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced')
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)
            acc_fold.append(accuracy_score(y_te, y_pred))
            f1_fold.append(f1_score(y_te, y_pred, average='macro', zero_division=0))

        acc_list.append(np.mean(acc_fold))
        f1_list.append(np.mean(f1_fold))

        if verbose and (i % 5 == 0 or i == len(sessions) - 1):
            print(f"  [{i+1}/{len(sessions)}] {os.path.basename(fpath)}: "
                  f"ACC={acc_list[-1]:.3f}, F1={f1_list[-1]:.3f}")

    return {
        'acc_mean': float(np.mean(acc_list)),
        'acc_std': float(np.std(acc_list)),
        'f1_mean': float(np.mean(f1_list)),
        'f1_std': float(np.std(f1_list)),
        'acc_all': [float(x) for x in acc_list],
    }


def evaluate_riemann_classification(sessions, bands='5band', verbose=True):
    """Riemannian FBTS 情绪分类."""
    freq_bands = [(1,4),(4,8),(8,14),(14,31),(31,50)] if bands == '5band' else \
                 [(1,4),(4,6),(6,8),(8,10),(10,12),(12,14),(14,20),(20,30)]

    acc_list, f1_list = [], []

    for i, fpath in enumerate(sessions):
        X_raw, y = load_seed_raw_eeg(fpath)
        if X_raw is None or len(np.unique(y)) < 3:
            continue

        n_ch = X_raw.shape[1]
        cov_est = Covariances(estimator='oas')

        # 预计算协方差
        all_covs = {}
        for low, high in freq_bands:
            filtered = np.array([
                apply_bandpass_filter(e, low, high, 200) for e in X_raw
            ], dtype=np.float64)
            all_covs[(low, high)] = cov_est.transform(filtered)

        # Trial-level 5-fold CV
        n_trials = 15
        samples_per_trial = len(y) // 15
        trial_splits = get_5fold_splits(n_trials, 5)

        acc_fold, f1_fold = [], []
        for tr_trials, te_trials in trial_splits:
            tr_mask = np.zeros(len(y), dtype=bool)
            te_mask = np.zeros(len(y), dtype=bool)
            for t in tr_trials:
                tr_mask[t * samples_per_trial:(t+1) * samples_per_trial] = True
            for t in te_trials:
                te_mask[t * samples_per_trial:(t+1) * samples_per_trial] = True

            # 切空间
            feats_tr, feats_te = [], []
            for b in freq_bands:
                ts = TangentSpace(metric='riemann')
                f_tr = ts.fit_transform(all_covs[b][tr_mask], y[tr_mask])
                f_te = ts.transform(all_covs[b][te_mask])
                feats_tr.append(f_tr); feats_te.append(f_te)

            X_tr = np.hstack(feats_tr)
            X_te = np.hstack(feats_te)

            # 特征选择
            if X_tr.shape[1] > 200:
                sel = SelectKBest(f_classif, k=200)
                X_tr = sel.fit_transform(X_tr, y[tr_mask])
                X_te = sel.transform(X_te)

            # 标准化
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)

            clf = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced')
            clf.fit(X_tr, y[tr_mask])
            y_pred = clf.predict(X_te)
            acc_fold.append(accuracy_score(y[te_mask], y_pred))
            f1_fold.append(f1_score(y[te_mask], y_pred, average='macro', zero_division=0))

        acc_list.append(np.mean(acc_fold))
        f1_list.append(np.mean(f1_fold))

        if verbose and (i % 3 == 0 or i == len(sessions) - 1):
            print(f"  [{i+1}/{len(sessions)}] {os.path.basename(fpath)}: "
                  f"ACC={acc_list[-1]:.3f}, F1={f1_list[-1]:.3f}")

    return {
        'acc_mean': float(np.mean(acc_list)),
        'acc_std': float(np.std(acc_list)),
        'f1_mean': float(np.mean(f1_list)),
        'f1_std': float(np.std(f1_list)),
        'acc_all': [float(x) for x in acc_list],
    }


def main():
    print(f"{'='*60}")
    print(f"SEED Emotion Classification (Cross-Dataset Validation)")
    print(f"{'='*60}")

    sessions = list_seed_sessions()
    print(f"Sessions: {len(sessions)} (15 subjects × 3)")

    os.makedirs(OUTPUT, exist_ok=True)
    all_results = []

    # ── DE 基线 ──
    for ft in ['de_LDS', 'psd_LDS']:
        tag = 'DE' if 'de' in ft else 'PSD'
        print(f"\n{tag} baseline...")
        t0 = time.time()
        r = evaluate_de_classification(sessions, feature_type=ft)
        elapsed = time.time() - t0
        print(f"  ACC={r['acc_mean']:.4f}+/-{r['acc_std']:.4f}, "
              f"F1={r['f1_mean']:.4f}, {elapsed:.0f}s")
        all_results.append(dict(exp_name=f'SEED_{tag}', acc_mean=r['acc_mean'],
                                acc_std=r['acc_std'], f1_mean=r['f1_mean'],
                                f1_std=r['f1_std'], time_sec=elapsed))

    # ── Riemannian ──
    for bands in ['5band', '8band']:
        print(f"\nRiemannian FBTS {bands}...")
        t0 = time.time()
        r = evaluate_riemann_classification(sessions, bands=bands)
        elapsed = time.time() - t0
        print(f"  ACC={r['acc_mean']:.4f}+/-{r['acc_std']:.4f}, "
              f"F1={r['f1_mean']:.4f}, {elapsed:.0f}s")
        all_results.append(dict(exp_name=f'SEED_Riemann_{bands}',
                                acc_mean=r['acc_mean'], acc_std=r['acc_std'],
                                f1_mean=r['f1_mean'], f1_std=r['f1_std'],
                                time_sec=elapsed))

    # ── 汇总 ──
    print(f"\n{'='*60}")
    print(f"SEED Results Summary")
    print(f"{'='*60}")
    print(f"{'Experiment':<25s} {'ACC':>8s} {'ACCstd':>8s} {'F1':>8s} {'Time':>8s}")
    print("-" * 60)
    for r in sorted(all_results, key=lambda x: -(x.get('acc_mean', 0))):
        print(f"{r['exp_name']:<25s} {r['acc_mean']:>8.4f} {r['acc_std']:>8.4f} "
              f"{r['f1_mean']:>8.4f} {r.get('time_sec', 0):>7.0f}s")

    # 保存
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(OUTPUT, f'results_seed_{ts}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()