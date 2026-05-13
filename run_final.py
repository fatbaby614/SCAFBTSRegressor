"""
最终加速版 — pyriemann Fast + joblib 并行
=========================================
结合两者优势:
- sca_fbts_fast.py: 预滤波 + 预计算协方差 (6.5× 加速)
- joblib: 多被试并行 (N_cores× 加速)

用法:
  python run_final.py                          # 全量 23 被试并行
  python run_final.py --subjects s1 s2 s3      # 指定被试
  python run_final.py --n-jobs 4               # 4 进程并行
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
    list_subjects, load_raw_eeg, load_perclos, build_de_features_for_baseline
)
from utils import cor, rmse, mae, get_5fold_splits, evaluate_de_baseline
from sca_fbts_fast import SCAFBTSRegressorFast
from sklearn.svm import SVR

from config import SEED_VIG_ROOT as DATA_ROOT, RESULTS_DIR as OUTPUT_DIR


def evaluate_fast(clf, y):
    """预计算数据 5-fold 评测 (静默)。"""
    splits = get_5fold_splits(len(y), 5)
    cor_list, rmse_list, mae_list = [], [], []
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in splits:
        clf.fit(train_idx, y[train_idx])
        y_pred = clf.predict(test_idx)

        if clf.temporal_smoothing and len(y_pred) > 1:
            from scipy.ndimage import uniform_filter1d
            y_pred = uniform_filter1d(y_pred, size=clf.smoothing_window)

        y_test = y[test_idx]
        cor_list.append(cor(y_test, y_pred))
        rmse_list.append(rmse(y_test, y_pred))
        mae_list.append(mae(y_test, y_pred))
        y_true_all.append(y_test)
        y_pred_all.append(y_pred)

    yt = np.concatenate(y_true_all)
    yp = np.concatenate(y_pred_all)
    return {
        'cor_mean': float(np.mean(cor_list)),
        'cor_std': float(np.std(cor_list)),
        'cor_all': [float(x) for x in cor_list],
        'rmse_mean': float(np.mean(rmse_list)),
        'rmse_std': float(np.std(rmse_list)),
        'global_cor': float(cor(yt, yp)),
        'global_rmse': float(rmse(yt, yp)),
        'mae_mean': float(np.mean(mae_list)),
    }


def process_subject(args):
    """单被试处理 (joblib 可序列化)。"""
    subject_id, configs = args
    results = []

    for cfg in configs:
        t0 = time.time()
        try:
            if cfg['type'] == 'de_baseline':
                X, y, _ = build_de_features_for_baseline(
                    DATA_ROOT, subject_id,
                    channels=cfg['channels'],
                    feature_type=cfg.get('feature_type', 'de_LDS')
                )
                r = evaluate_de_baseline(
                    lambda: SVR(kernel='rbf', C=1.0, gamma='scale'),
                    X, y, verbose=False
                )
            elif cfg['type'] == 'riemann_fast':
                raw, sr = load_raw_eeg(DATA_ROOT, subject_id)
                y = load_perclos(DATA_ROOT, subject_id)
                ch = cfg.get('channels', 'all')
                if ch == 'temporal':
                    raw = raw[:, [0,1,2,3,4,5]]
                elif ch == 'forehead':
                    raw = raw[:, [0,1,2,3]]
                clf = SCAFBTSRegressorFast(
                    freq_bands=cfg['freq_bands'],
                    estimator=cfg.get('estimator', 'oas'),
                    metric=cfg.get('metric', 'riemann'),
                    regressor=cfg.get('regressor', 'svr'),
                    n_features=cfg.get('n_features', 100),
                    fs=sr,
                    temporal_smoothing=cfg.get('temporal_smooth', True),
                    smoothing_window=cfg.get('smoothing_window', 3),
                )
                clf.precompute(raw, fs=sr)
                r = evaluate_fast(clf, y)
            else:
                continue

            results.append({
                'subject': subject_id,
                'exp_name': cfg['name'],
                'cor_mean': r['cor_mean'],
                'cor_std': r['cor_std'],
                'global_cor': r['global_cor'],
                'rmse_mean': r['rmse_mean'],
                'rmse_std': r['rmse_std'],
                'mae_mean': r['mae_mean'],
                'time_sec': time.time() - t0,
            })
        except Exception as e:
            import traceback; traceback.print_exc()
            results.append({
                'subject': subject_id, 'exp_name': cfg['name'],
                'cor_mean': None, 'error': str(e),
            })
    return results


def get_configs():
    cfgs = []
    # DE基线
    for ch_name, ch_key in [('17ch', 'all'), ('6ch_temporal', 'temporal')]:
        cfgs.append(dict(name=f'DE_SVR_{ch_name}', type='de_baseline',
                        channels=ch_key, feature_type='de_LDS',
                        temporal_smooth=True, smoothing_window=3))
    # Riemannian Fast
    for ch_name, ch_key in [('17ch', 'all'), ('6ch_temporal', 'temporal')]:
        for bands in ['5band', '8band']:
            cfgs.append(dict(name=f'RFast_{bands}_{ch_name}', type='riemann_fast',
                            channels=ch_key, freq_bands=bands,
                            estimator='oas', metric='riemann', regressor='svr',
                            n_features=100, temporal_smooth=True, smoothing_window=3))
    # 前额4ch
    for bands in ['5band', '8band']:
        cfgs.append(dict(name=f'RFast_{bands}_4ch_forehead', type='riemann_fast',
                        channels='forehead', freq_bands=bands,
                        estimator='oas', metric='riemann', regressor='svr',
                        n_features=100, temporal_smooth=True, smoothing_window=3))
    return cfgs


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--subjects', nargs='+', default=None)
    p.add_argument('--n-jobs', type=int, default=-1)
    args = p.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    subjects = args.subjects or list_subjects(DATA_ROOT)
    configs = get_configs()

    print(f"SEED-VIG Final: {len(subjects)} subjects x {len(configs)} configs, "
          f"{args.n_jobs} workers")

    t0 = time.time()

    if args.n_jobs in (1, -1) and len(subjects) == 1:
        # 单被试串行
        all_results = process_subject((subjects[0], configs))
    else:
        from joblib import Parallel, delayed
        tasks = [(s, configs) for s in subjects]
        nested = Parallel(n_jobs=args.n_jobs, verbose=10)(
            delayed(process_subject)(t) for t in tasks
        )
        all_results = [r for subj_res in nested for r in subj_res]

    elapsed = time.time() - t0

    # 汇总
    agg = defaultdict(lambda: {'cor': [], 'rmse': [], 'gcor': [], 'time': []})
    for r in all_results:
        if r.get('cor_mean') is not None:
            n = r['exp_name']
            agg[n]['cor'].append(r['cor_mean'])
            agg[n]['rmse'].append(r['rmse_mean'])
            agg[n]['gcor'].append(r['global_cor'])
            agg[n]['time'].append(r.get('time_sec', 0))

    print(f"\n{'='*70}")
    print(f"Results: {len(subjects)} subjects, {elapsed:.0f}s total")
    print(f"{'='*70}")
    print(f"{'Experiment':<40s} {'N':>4s} {'COR':>8s} {'COR_std':>8s} "
          f"{'RMSE':>8s} {'G.COR':>8s} {'Time':>8s}")
    print("-" * 85)

    for name in sorted(agg.keys()):
        s = agg[name]
        n = len(s['cor'])
        print(f"{name:<40s} {n:>4d} {np.mean(s['cor']):>8.4f} "
              f"{np.std(s['cor']):>8.4f} {np.mean(s['rmse']):>8.4f} "
              f"{np.mean(s['gcor']):>8.4f} {np.mean(s['time']):>7.1f}s")

    # 保存结果
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(OUTPUT_DIR, f'results_final_{ts}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()