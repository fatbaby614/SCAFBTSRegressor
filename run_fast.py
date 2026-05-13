"""
优化版评测 + 快速实验脚本
========================
支持 SCAFBTSRegressorFast 的预计算模式。
"""

import sys
import os
import json
import time
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import (
    list_subjects, load_raw_eeg, load_perclos,
    build_de_features_for_baseline
)
from utils import cor, rmse, mae, get_5fold_splits, evaluate_de_baseline
from sca_fbts_fast import SCAFBTSRegressorFast
from sklearn.svm import SVR
from sklearn.linear_model import Ridge

from config import SEED_VIG_ROOT as DATA_ROOT, RESULTS_DIR as OUTPUT_DIR
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def evaluate_riemann_fast(clf, y, verbose=True):
    """使用预计算数据的 5-fold 评测。

    Args:
        clf: SCAFBTSRegressorFast (已调用 precompute)
        y: (n_epochs,) PERCLOS 标签
        verbose: 是否打印
    Returns:
        results dict
    """
    n_epochs = len(y)
    splits = get_5fold_splits(n_epochs, 5)
    cor_list, rmse_list, mae_list = [], [], []
    y_true_all, y_pred_all = [], []

    for fold, (train_idx, test_idx) in enumerate(splits):
        # fit: 传入 epoch 索引 (不是数据)
        clf.fit(train_idx, y[train_idx])

        # predict: 传入 epoch 索引
        y_pred = clf.predict(test_idx)

        # 时间平滑
        if clf.temporal_smoothing and len(y_pred) > 1:
            from scipy.ndimage import uniform_filter1d
            y_pred = uniform_filter1d(y_pred, size=clf.smoothing_window)

        y_test = y[test_idx]
        c = cor(y_test, y_pred)
        r = rmse(y_test, y_pred)
        m = mae(y_test, y_pred)
        cor_list.append(c)
        rmse_list.append(r)
        mae_list.append(m)
        y_true_all.append(y_test)
        y_pred_all.append(y_pred)

        if verbose:
            print(f"  Fold {fold+1}/5: COR={c:.4f}, RMSE={r:.4f}, MAE={m:.4f}")

    y_true_cat = np.concatenate(y_true_all)
    y_pred_cat = np.concatenate(y_pred_all)
    global_cor = cor(y_true_cat, y_pred_cat)
    global_rmse = rmse(y_true_cat, y_pred_cat)

    results = {
        'cor_mean': np.mean(cor_list), 'cor_std': np.std(cor_list),
        'cor_all': cor_list,
        'rmse_mean': np.mean(rmse_list), 'rmse_std': np.std(rmse_list),
        'rmse_all': rmse_list,
        'mae_mean': np.mean(mae_list), 'mae_std': np.std(mae_list),
        'global_cor': global_cor, 'global_rmse': global_rmse,
        'y_true_all': y_true_cat, 'y_pred_all': y_pred_cat,
    }

    if verbose:
        print(f"  Avg: COR={results['cor_mean']:.4f}+/-{results['cor_std']:.4f}, "
              f"RMSE={results['rmse_mean']:.4f}+/-{results['rmse_std']:.4f}")
        print(f"  Global: COR={global_cor:.4f}, RMSE={global_rmse:.4f}")

    return results


def get_fast_configs():
    """精简版实验配置 (仅最优组合，用于快速验证)。"""
    configs = []

    # DE 基线
    for ch_name, ch_key in [('17ch', 'all'), ('6ch_temporal', 'temporal')]:
        configs.append({
            'name': f'DE_SVR_{ch_name}',
            'type': 'de_baseline',
            'channels': ch_key,
            'feature_type': 'de_LDS',
            'regressor_factory': lambda: SVR(kernel='rbf', C=1.0, gamma='scale'),
            'temporal_smooth': True, 'smoothing_window': 3,
        })

    # Riemannian Fast (仅最优组合)
    for ch_name, ch_key in [('17ch', 'all'), ('6ch_temporal', 'temporal')]:
        for bands in ['5band', '8band']:
            configs.append({
                'name': f'RiemannFast_{bands}_{ch_name}',
                'type': 'riemann_fast',
                'channels': ch_key,
                'freq_bands': bands,
                'estimator': 'oas',
                'metric': 'riemann',
                'regressor': 'svr',
                'n_features': 100,
                'temporal_smooth': True, 'smoothing_window': 3,
                'scaler': True,
            })

    # 前额 4ch
    for bands in ['5band', '8band']:
        configs.append({
            'name': f'RiemannFast_{bands}_4ch_forehead',
            'type': 'riemann_fast',
            'channels': 'forehead',
            'freq_bands': bands,
            'estimator': 'oas', 'metric': 'riemann',
            'regressor': 'svr', 'n_features': 100,
            'temporal_smooth': True, 'smoothing_window': 3,
            'scaler': True,
        })

    return configs


def run_single(config, subject_id, verbose=True):
    """运行单个实验配置。"""
    cfg_type = config['type']

    if cfg_type == 'de_baseline':
        X, y, _ = build_de_features_for_baseline(
            DATA_ROOT, subject_id,
            channels=config['channels'],
            feature_type=config.get('feature_type', 'de_LDS')
        )
        reg_factory = config['regressor_factory']
        return evaluate_de_baseline(
            reg_factory, X, y,
            temporal_smooth=config.get('temporal_smooth', True),
            smoothing_window=config.get('smoothing_window', 3),
            verbose=verbose
        )

    elif cfg_type == 'riemann_fast':
        # 加载原始数据
        raw, sr = load_raw_eeg(DATA_ROOT, subject_id)
        y = load_perclos(DATA_ROOT, subject_id)

        # 通道选择
        channels = config.get('channels', 'all')
        if channels == 'temporal':
            ch_idx = [0, 1, 2, 3, 4, 5]
        elif channels == 'forehead':
            ch_idx = [0, 1, 2, 3]
        else:
            ch_idx = list(range(raw.shape[1]))
        raw = raw[:, ch_idx]

        # 创建 + 预计算
        clf = SCAFBTSRegressorFast(
            freq_bands=config['freq_bands'],
            estimator=config.get('estimator', 'oas'),
            metric=config.get('metric', 'riemann'),
            regressor=config.get('regressor', 'svr'),
            n_features=config.get('n_features', 100),
            fs=sr,
            temporal_smoothing=config.get('temporal_smooth', True),
            smoothing_window=config.get('smoothing_window', 3),
            scaler=config.get('scaler', True),
        )
        clf.precompute(raw, fs=sr)

        # 评测
        return evaluate_riemann_fast(clf, y, verbose=verbose)

    else:
        raise ValueError(f"Unknown type: {cfg_type}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--subjects', nargs='+', default=None)
    parser.add_argument('--fast-only', action='store_true',
                       help='只跑 Riemannian Fast (跳过 DE 基线)')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    subjects = args.subjects or list_subjects(DATA_ROOT)
    configs = get_fast_configs()

    if args.fast_only:
        configs = [c for c in configs if c['type'] == 'riemann_fast']

    print(f"{'='*70}")
    print(f"SEED-VIG 快速实验 (预计算加速)")
    print(f"被试: {len(subjects)}, 配置: {len(configs)}")
    print(f"{'='*70}")

    summary = []

    for si, subj in enumerate(subjects):
        print(f"\n{'='*70}")
        print(f"[{si+1}/{len(subjects)}] {subj}")
        print(f"{'='*70}")

        for ci, cfg in enumerate(configs):
            name = cfg['name']
            print(f"\n  [{ci+1}/{len(configs)}] {name}")
            t0 = time.time()

            try:
                results = run_single(cfg, subj, verbose=True)
                elapsed = time.time() - t0
                row = {
                    'subject': subj, 'exp_name': name,
                    'cor_mean': results['cor_mean'],
                    'cor_std': results['cor_std'],
                    'global_cor': results['global_cor'],
                    'rmse_mean': results['rmse_mean'],
                    'rmse_std': results['rmse_std'],
                    'mae_mean': results['mae_mean'],
                    'time_sec': elapsed,
                }
                summary.append(row)
                print(f"  -> {elapsed:.1f}s, COR={row['cor_mean']:.4f}, "
                      f"RMSE={row['rmse_mean']:.4f}")
            except Exception as e:
                print(f"  -> ERROR: {e}")
                import traceback; traceback.print_exc()
                summary.append({'subject': subj, 'exp_name': name,
                               'cor_mean': None, 'error': str(e)})

        # 保存中间结果
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(OUTPUT_DIR, f'results_fast_{ts}.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    # 汇总
    print(f"\n{'='*70}")
    print("汇总 (预计算加速版)")
    print(f"{'='*70}")
    from collections import defaultdict
    agg = defaultdict(lambda: {'cor': [], 'rmse': [], 'gcor': []})
    for row in summary:
        if row.get('cor_mean') is not None:
            n = row['exp_name']
            agg[n]['cor'].append(row['cor_mean'])
            agg[n]['rmse'].append(row['rmse_mean'])
            agg[n]['gcor'].append(row['global_cor'])

    print(f"{'实验':<45s} {'N':>4s} {'COR':>8s} {'RMSE':>8s} {'G.COR':>8s} {'Time':>8s}")
    print("-" * 85)
    for name in sorted(agg.keys()):
        s = agg[name]
        n = len(s['cor'])
        times = [r['time_sec'] for r in summary if r.get('exp_name') == name and r.get('time_sec')]
        avg_t = np.mean(times) if times else 0
        print(f"{name:<45s} {n:>4d} {np.mean(s['cor']):>8.4f} "
              f"{np.mean(s['rmse']):>8.4f} {np.mean(s['gcor']):>8.4f} {avg_t:>7.1f}s")


if __name__ == '__main__':
    main()
