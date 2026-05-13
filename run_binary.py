"""从融合实验 output 计算 PERCLOS 二值分类指标（30s 快速跑）"""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import list_subjects, load_raw_eeg, load_perclos, load_eog_features
from utils import get_5fold_splits, cor, rmse
from sca_fbts_fast import SCAFBTSRegressorFast
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from pyriemann.tangentspace import TangentSpace
from datetime import datetime

from config import SEED_VIG_ROOT as SEED_VIG, RESULTS_DIR as OUT

BANDS_5 = [(1, 4), (4, 8), (8, 14), (14, 31), (31, 50)]


def evaluate_binary(y_true, y_pred, threshold_alert=0.4, threshold_drowsy=0.6):
    """PERCLOS 二值化评测。
    
    排除中间区域 (0.4-0.6) 以避免类边界模糊。
    """
    from scipy.ndimage import uniform_filter1d
    y_pred = uniform_filter1d(y_pred, size=3)

    # 阈值化
    y_true_bin = np.full_like(y_true, -1, dtype=int)
    y_pred_bin = np.full_like(y_pred, -1, dtype=int)

    y_true_bin[y_true <= threshold_alert] = 0
    y_true_bin[y_true >= threshold_drowsy] = 1
    y_pred_bin[y_pred <= threshold_alert] = 0
    y_pred_bin[y_pred >= threshold_drowsy] = 1

    mask = (y_true_bin >= 0) & (y_pred_bin >= 0)
    if mask.sum() < 10:
        return None

    yt = y_true_bin[mask]
    yp = y_pred_bin[mask]

    if len(np.unique(yt)) < 2 or len(np.unique(yp)) < 2:
        return None

    return {
        'acc': float(accuracy_score(yt, yp)),
        'f1': float(f1_score(yt, yp, average='binary', zero_division=0)),
        'bac': float(balanced_accuracy_score(yt, yp)),
        'n_samples': int(mask.sum()),
        'n_alert': int((yt == 0).sum()),
        'n_drowsy': int((yt == 1).sum()),
    }


def process_subject(subject_id, config_name, channels='all'):
    """单被试 FBTS+EOG 二值分类."""
    raw, sr = load_raw_eeg(SEED_VIG, subject_id)
    y = load_perclos(SEED_VIG, subject_id)
    eog = load_eog_features(SEED_VIG, subject_id, method='features_table_ica')
    eog = StandardScaler().fit_transform(eog.astype(np.float64))

    if channels == 'temporal':
        raw = raw[:, [0, 1, 2, 3, 4, 5]]

    clf = SCAFBTSRegressorFast(
        freq_bands='5band', estimator='oas', metric='riemann',
        regressor='svr', n_features=100, fs=sr,
        temporal_smoothing=False, scaler=True,
    )
    clf.precompute(raw, fs=sr)

    splits = get_5fold_splits(885, 5)
    y_pred_all = np.zeros(885)
    y_true_all = y.copy()

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

    bin_result = evaluate_binary(y_true_all, y_pred_all)
    reg_cor = cor(y_true_all, y_pred_all)

    if bin_result is None:
        return None

    return {
        'subject': subject_id, 'config': config_name,
        'cor': float(reg_cor),
        'acc': bin_result['acc'], 'f1': bin_result['f1'], 'bac': bin_result['bac'],
        'n_samples': bin_result['n_samples'],
        'n_alert': bin_result['n_alert'], 'n_drowsy': bin_result['n_drowsy'],
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--subjects', nargs='+', default=None)
    p.add_argument('--quick', action='store_true')
    args = p.parse_args()

    subjects = args.subjects or list_subjects(SEED_VIG)
    if args.quick:
        subjects = subjects[:5]

    all_results = []

    for ch_name, ch_key in [('17ch', 'all'), ('6ch_temporal', 'temporal')]:
        config_name = f'FBTS+EOG_{ch_name}'
        print(f"\n{config_name} ({len(subjects)} subjects)")

        for subj in subjects:
            t0 = time.time()
            try:
                r = process_subject(subj, config_name, channels=ch_key)
                if r:
                    r['time_sec'] = time.time() - t0
                    all_results.append(r)
                    print(f"  {subj}: ACC={r['acc']:.3f} F1={r['f1']:.3f} "
                          f"COR={r['cor']:.3f} ({r['time_sec']:.0f}s)")
            except Exception as e:
                print(f"  {subj}: ERROR {e}")

    # 汇总
    from collections import defaultdict
    agg = defaultdict(list)
    for r in all_results:
        name = r['config']
        agg[name].append(r)

    print(f"\n{'='*60}")
    print(f"PERCLOS Binary Classification (Alert ≤0.4 vs Drowsy ≥0.6)")
    print(f"{'='*60}")
    for name in sorted(agg.keys()):
        items = agg[name]
        acc = np.mean([r['acc'] for r in items])
        f1 = np.mean([r['f1'] for r in items])
        bac = np.mean([r['bac'] for r in items])
        cor_ = np.mean([r['cor'] for r in items])
        n = len(items)
        print(f"  {name}: N={n} ACC={acc:.4f} F1={f1:.4f} BAC={bac:.4f} COR={cor_:.4f}")

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(OUT, f'results_binary_{ts}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()