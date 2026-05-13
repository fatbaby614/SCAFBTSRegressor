"""SEED Riemannian FBTS 10ch 情绪分类 —— 跨数据集验证"""
import os, sys, json, time, numpy as np
import scipy.io as sio
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import get_5fold_splits
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score
from sca_fbts_fast import SCAFBTSRegressorFast, apply_bandpass_filter
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from config import SEED_ROOT as SEED, RESULTS_DIR as OUT

TRIAL_LABELS = np.array([1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1])

# 10 个前额/额叶认知通道 (SEED 62ch 标准排序中的索引 0-9)
CH_10_INDICES = list(range(10))  # FP1,FPZ,FP2,AF3,AF4,F7,F5,F3,F1,FZ
BANDS_5 = [(1, 4), (4, 8), (8, 14), (14, 31), (31, 50)]


def list_sessions():
    d = os.path.join(SEED, 'ExtractedFeatures')
    return sorted([os.path.join(d, f) for f in os.listdir(d)
                   if f.endswith('.mat') and f not in ('label.mat', 'readme.txt')])


def load_seed_raw_10ch(feature_filepath):
    """从 Preprocessed_EEG 加载对应 session 的 10ch 原始 EEG.

    Returns:
        X: (n_total_samples, 10, epoch_len) 或 None
        y: (n_total_samples,)
    """
    eeg_dir = os.path.join(SEED, 'Preprocessed_EEG')
    fname = os.path.basename(feature_filepath)
    eeg_path = os.path.join(eeg_dir, fname)
    if not os.path.exists(eeg_path):
        return None, None

    data = sio.loadmat(eeg_path)
    fs = 200
    epoch_len = fs * 4  # 4s epochs
    X_list, y_list = [], []

    for trial_idx in range(1, 16):
        key = f'ww_eeg{trial_idx}'
        if key not in data:
            continue
        raw = data[key]  # (62, n_times)
        # 选 10 个前额/额叶通道
        raw_10ch = raw[CH_10_INDICES]  # (10, n_times)
        n_times = raw_10ch.shape[1]
        n_epochs = n_times // epoch_len
        if n_epochs == 0:
            continue

        epochs = np.array([
            raw_10ch[:, i * epoch_len:(i + 1) * epoch_len]
            for i in range(n_epochs)
        ], dtype=np.float64)
        X_list.append(epochs)
        y_list.extend([TRIAL_LABELS[trial_idx - 1]] * n_epochs)

    if not X_list:
        return None, None
    X = np.concatenate(X_list, axis=0)
    y = np.array(y_list)
    return X, y


def run_riemann(sessions, bands='5band', verbose=True):
    """Riemannian FBTS 10ch 情绪分类."""
    freq_bands = BANDS_5 if bands == '5band' else \
        [(1, 4), (4, 6), (6, 8), (8, 10), (10, 12),
         (12, 14), (14, 20), (20, 30)]

    acc_list, f1_list = [], []

    for i, fpath in enumerate(sessions):
        X_raw, y = load_seed_raw_10ch(fpath)
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
        sp = len(y) // n_trials
        if sp == 0:
            continue

        trial_splits = get_5fold_splits(n_trials, 5)
        acc_fold, f1_fold = [], []

        for tr_trials, te_trials in trial_splits:
            tr_mask = np.zeros(len(y), dtype=bool)
            te_mask = np.zeros(len(y), dtype=bool)
            for t in tr_trials:
                tr_mask[t * sp:(t + 1) * sp] = True
            for t in te_trials:
                te_mask[t * sp:(t + 1) * sp] = True

            feats_tr, feats_te = [], []
            for b in freq_bands:
                ts = TangentSpace(metric='riemann')
                f_tr = ts.fit_transform(all_covs[b][tr_mask], y[tr_mask])
                f_te = ts.transform(all_covs[b][te_mask])
                feats_tr.append(f_tr)
                feats_te.append(f_te)

            X_tr = np.hstack(feats_tr)
            X_te = np.hstack(feats_te)

            if X_tr.shape[1] > 100:
                sel = SelectKBest(f_classif, k=100)
                X_tr = sel.fit_transform(X_tr, y[tr_mask])
                X_te = sel.transform(X_te)

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
    }


def main():
    print(f"{'='*60}")
    print(f"SEED 10ch Riemannian FBTS Emotion Classification")
    print(f"{'='*60}")

    sessions = list_sessions()
    print(f"Sessions: {len(sessions)} (15 subjects x 3)")

    os.makedirs(OUT, exist_ok=True)
    all_results = []

    for bands in ['5band', '8band']:
        print(f"\nRiemannian FBTS {bands} (10ch frontal)...")
        t0 = time.time()
        r = run_riemann(sessions, bands=bands)
        elapsed = time.time() - t0
        print(f"  ACC={r['acc_mean']:.4f}+/-{r['acc_std']:.4f}, "
              f"F1={r['f1_mean']:.4f}+/-{r['f1_std']:.4f}, {elapsed:.0f}s")
        all_results.append(dict(
            exp_name=f'SEED_Riemann_{bands}_10ch',
            acc_mean=r['acc_mean'], acc_std=r['acc_std'],
            f1_mean=r['f1_mean'], f1_std=r['f1_std'],
            time_sec=elapsed
        ))

    # 汇总
    print(f"\n{'='*60}")
    print(f"SEED 10ch Results Summary")
    print(f"{'='*60}")
    for r in sorted(all_results, key=lambda x: -(x.get('acc_mean', 0))):
        print(f"  {r['exp_name']}: ACC={r['acc_mean']:.4f}+/-{r['acc_std']:.4f}, "
              f"F1={r['f1_mean']:.4f}, {r.get('time_sec', 0):.0f}s")

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(OUT, f'results_seed_fbts_{ts}.json')
    with open(path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()