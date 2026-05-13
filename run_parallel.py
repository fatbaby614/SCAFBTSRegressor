"""
并行实验脚本 — Torch 协方差 + Joblib 多被试并行
==============================================
每条被试独立运行，joblib 自动分配 CPU 核心。
支持 GPU 加速 (torch.cuda) + CPU 多进程并行。
"""

import sys
import os
import json
import time
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import (
    list_subjects, load_raw_eeg, load_perclos, build_de_features_for_baseline
)
from utils import cor, rmse, mae, get_5fold_splits, evaluate_de_baseline
from sca_fbts_torch import SCAFBTSRegressorTorch
from sklearn.svm import SVR

from config import SEED_VIG_ROOT as DATA_ROOT, RESULTS_DIR as OUTPUT_DIR


# ── 评测函数 (torch 版) ─────────────────────────────────

def evaluate_riemann_torch(clf, y, verbose=False):
    """预计算数据的 5-fold 评测 (无打印版，用于并行)。"""
    n_epochs = len(y)
    splits = get_5fold_splits(n_epochs, 5)
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

    y_true_cat = np.concatenate(y_true_all)
    y_pred_cat = np.concatenate(y_pred_all)

    return {
        'cor_mean': float(np.mean(cor_list)),
        'cor_std': float(np.std(cor_list)),
        'cor_all': [float(x) for x in cor_list],
        'rmse_mean': float(np.mean(rmse_list)),
        'rmse_std': float(np.std(rmse_list)),
        'global_cor': float(cor(y_true_cat, y_pred_cat)),
        'global_rmse': float(rmse(y_true_cat, y_pred_cat)),
        'mae_mean': float(np.mean(mae_list)),
    }


# ── 单被试任务 ───────────────────────────────────────────

def process_subject(args):
    """单个被试的处理函数 (joblib 可序列化)。

    Args:
        args: (subject_id, configs_list, verbose)

    Returns:
        list of result dicts
    """
    subject_id, configs, verbose = args
    results = []

    for cfg in configs:
        name = cfg['name']
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
                elapsed = time.time() - t0
                results.append({
                    'subject': subject_id, 'exp_name': name,
                    'cor_mean': r['cor_mean'], 'cor_std': r['cor_std'],
                    'global_cor': r['global_cor'], 'rmse_mean': r['rmse_mean'],
                    'rmse_std': r['rmse_std'], 'mae_mean': r['mae_mean'],
                    'time_sec': elapsed,
                })

            elif cfg['type'] == 'riemann_torch':
                raw, sr = load_raw_eeg(DATA_ROOT, subject_id)
                y = load_perclos(DATA_ROOT, subject_id)

                # 通道选择
                ch = cfg.get('channels', 'all')
                if ch == 'temporal':
                    raw = raw[:, [0,1,2,3,4,5]]
                elif ch == 'forehead':
                    raw = raw[:, [0,1,2,3]]

                clf = SCAFBTSRegressorTorch(
                    freq_bands=cfg['freq_bands'],
                    estimator=cfg.get('estimator', 'scm'),
                    metric=cfg.get('metric', 'riemann'),
                    regressor=cfg.get('regressor', 'svr'),
                    n_features=cfg.get('n_features', 100),
                    fs=sr,
                    temporal_smoothing=cfg.get('temporal_smooth', True),
                    smoothing_window=cfg.get('smoothing_window', 3),
                )
                clf.precompute(raw, fs=sr)
                r = evaluate_riemann_torch(clf, y)
                elapsed = time.time() - t0
                results.append({
                    'subject': subject_id, 'exp_name': name,
                    'cor_mean': r['cor_mean'], 'cor_std': r['cor_std'],
                    'global_cor': r['global_cor'], 'rmse_mean': r['rmse_mean'],
                    'rmse_std': r['rmse_std'], 'mae_mean': r['mae_mean'],
                    'time_sec': elapsed,
                })

        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({
                'subject': subject_id, 'exp_name': name,
                'cor_mean': None, 'error': str(e),
            })

    return results


# ── 配置 ─────────────────────────────────────────────────

def get_configs():
    cfgs = []

    # DE 基线
    for ch_name, ch_key in [('17ch', 'all'), ('6ch_temporal', 'temporal')]:
        cfgs.append({
            'name': f'DE_SVR_{ch_name}', 'type': 'de_baseline',
            'channels': ch_key, 'feature_type': 'de_LDS',
            'temporal_smooth': True, 'smoothing_window': 3,
        })

    # Riemannian Torch
    for ch_name, ch_key in [('17ch', 'all'), ('6ch_temporal', 'temporal')]:
        for bands in ['5band', '8band']:
            cfgs.append({
                'name': f'RTorch_{bands}_{ch_name}',
                'type': 'riemann_torch', 'channels': ch_key,
                'freq_bands': bands, 'estimator': 'scm',
                'metric': 'riemann', 'regressor': 'svr',
                'n_features': 100, 'temporal_smooth': True,
                'smoothing_window': 3,
            })

    # 前额 4ch
    for bands in ['5band', '8band']:
        cfgs.append({
            'name': f'RTorch_{bands}_4ch_forehead',
            'type': 'riemann_torch', 'channels': 'forehead',
            'freq_bands': bands, 'estimator': 'scm',
            'metric': 'riemann', 'regressor': 'svr',
            'n_features': 100, 'temporal_smooth': True,
            'smoothing_window': 3,
        })

    return cfgs


# ── 主函数 ─────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--subjects', nargs='+', default=None)
    parser.add_argument('--n-jobs', type=int, default=-1,
                       help='并行进程数 (-1=全部核心)')
    parser.add_argument('--sequential', action='store_true',
                       help='串行模式 (调试用)')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    subjects = args.subjects or list_subjects(DATA_ROOT)
    configs = get_configs()
    n_jobs = 1 if args.sequential else args.n_jobs

    print(f"{'='*70}")
    print(f"SEED-VIG 并行实验 (Torch + Joblib)")
    print(f"被试: {len(subjects)}, 配置: {len(configs)}, 并行: {n_jobs} 核")
    if not args.sequential:
        try:
            import torch
            print(f"Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
        except:
            print("Torch: 未安装 (fallback CPU)")
    print(f"{'='*70}")

    t_start = time.time()

    if args.sequential:
        # 串行模式 (调试)
        all_results = []
        for subj in subjects:
            print(f"\n[串行] {subj}")
            res = process_subject((subj, configs, True))
            all_results.extend(res)
            # 打印当前被试结果
            for r in res:
                if r.get('cor_mean'):
                    print(f"  {r['exp_name']}: COR={r['cor_mean']:.4f}, "
                          f"RMSE={r['rmse_mean']:.4f}, {r['time_sec']:.1f}s")
    else:
        # 并行模式
        from joblib import Parallel, delayed
        tasks = [(subj, configs, False) for subj in subjects]
        all_nested = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(process_subject)(t) for t in tasks
        )
        all_results = []
        for subj_results in all_nested:
            all_results.extend(subj_results)

    elapsed = time.time() - t_start

    # ── 汇总 ────────────────────────────────────────────
    from collections import defaultdict
    agg = defaultdict(lambda: {'cor': [], 'rmse': [], 'gcor': [], 'time': []})

    for r in all_results:
        if r.get('cor_mean') is not None:
            n = r['exp_name']
            agg[n]['cor'].append(r['cor_mean'])
            agg[n]['rmse'].append(r['rmse_mean'])
            agg[n]['gcor'].append(r['global_cor'])
            agg[n]['time'].append(r.get('time_sec', 0))

    print(f"\n{'='*70}")
    print(f"汇总 ({len(subjects)} 被试, 总耗时 {elapsed:.0f}s)")
    print(f"{'='*70}")
    print(f"{'实验':<40s} {'N':>4s} {'COR':>8s} {'RMSE':>8s} "
          f"{'G.COR':>8s} {'Time':>8s}")
    print("-" * 80)

    for name in sorted(agg.keys()):
        s = agg[name]
        n = len(s['cor'])
        print(f"{name:<40s} {n:>4d} {np.mean(s['cor']):>8.4f} "
              f"{np.mean(s['rmse']):>8.4f} {np.mean(s['gcor']):>8.4f} "
              f"{np.mean(s['time']):>7.1f}s")

    # 保存
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(OUTPUT_DIR, f'results_parallel_{ts}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存: {path}")


if __name__ == '__main__':
    main()