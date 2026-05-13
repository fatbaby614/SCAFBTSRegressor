"""
SEED-VIG 警觉度估计实验主脚本
=============================
对照实验:
  A) DE 特征 + SVR         (标准基线, Zheng & Lu 2017)
  B) DE 特征 + Ridge        (线性基线)
  C) SCAFBTS (Riemannian)  + SVR   (本文方法)
  D) SCAFBTS (Riemannian)  + Ridge

频道配置:
  1. 全脑 17ch
  2. 颞区 6ch
  3. 前额 4ch (可穿戴)

频段配置:
  - 5band, 8band, 25band

输出:
  - 逐被试、逐条件的 COR/RMSE 汇总
  - 自动保存结果到 results/ 目录
"""

import sys
import os
import json
import time
import numpy as np
from datetime import datetime

# 确保模块可导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import (
    list_subjects, load_raw_eeg, load_perclos, build_epochs_from_raw,
    build_de_features_for_baseline
)
from utils import (
    evaluate_regressor, evaluate_de_baseline, cor, rmse
)
from sca_fbts_regressor import SCAFBTSRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# ── 配置 ─────────────────────────────────────────────────────
from config import SEED_VIG_ROOT as DATA_ROOT, RESULTS_DIR as OUTPUT_DIR
# OUTPUT_DIR imported from config
SUBJECTS = None  # None = 自动检测所有被试; 或指定列表如 ['10_20151125_noon']


def run_experiment(subject_id, config, verbose=True):
    """对单个被试运行一组实验配置。

    Args:
        subject_id: 如 '10_20151125_noon'
        config: dict with keys:
            - 'name': 实验名称
            - 'type': 'de_baseline' | 'riemann'
            - 以及其他该类型需要的参数

    Returns:
        results: dict with COR/RMSE 等
    """
    exp_type = config['type']

    if exp_type == 'de_baseline':
        # ── DE 特征基线 ──
        X, y, ch_names = build_de_features_for_baseline(
            DATA_ROOT, subject_id,
            channels=config.get('channels', 'all'),
            feature_type=config.get('feature_type', 'de_LDS')
        )
        regressor_factory = config['regressor_factory']
        results = evaluate_de_baseline(
            regressor_factory, X, y,
            temporal_smooth=config.get('temporal_smooth', True),
            smoothing_window=config.get('smoothing_window', 3),
            verbose=verbose
        )

    elif exp_type == 'riemann':
        # ── Riemannian 方法 ──
        # 加载原始 EEG
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

        raw_subset = raw[:, ch_idx]
        X, y = build_epochs_from_raw(raw_subset, y, fs=sr)

        # 构建回归器
        clf = SCAFBTSRegressor(
            freq_bands=config.get('freq_bands', '5band'),
            estimator=config.get('estimator', 'oas'),
            metric=config.get('metric', 'riemann'),
            regressor=config.get('regressor', 'svr'),
            n_features=config.get('n_features', 100),
            fs=sr,
            temporal_smoothing=config.get('temporal_smooth', True),
            smoothing_window=config.get('smoothing_window', 3),
            scaler=config.get('scaler', True),
        )
        results = evaluate_regressor(
            clf, X, y,
            temporal_smooth=config.get('temporal_smooth', True),
            smoothing_window=config.get('smoothing_window', 3),
            verbose=verbose
        )

    else:
        raise ValueError(f"Unknown experiment type: {exp_type}")

    # 附加元信息
    results['subject'] = subject_id
    results['config'] = config
    return results


# ── 实验定义 ─────────────────────────────────────────────────

def get_experiment_configs():
    """返回所有待运行的实验配置列表。"""
    configs = []

    # ── A: DE 特征基线 ──────────────────────────────────────
    for ch_name, ch_key in [('17ch', 'all'), ('6ch_temporal', 'temporal')]:
        for reg_name, reg_factory in [
            ('SVR', lambda: SVR(kernel='rbf', C=1.0, gamma='scale')),
            ('Ridge', lambda: Ridge(alpha=1.0)),
        ]:
            configs.append({
                'name': f'DE_{reg_name}_{ch_name}',
                'type': 'de_baseline',
                'channels': ch_key,
                'feature_type': 'de_LDS',
                'regressor_factory': reg_factory,
                'temporal_smooth': True,
                'smoothing_window': 3,
            })

    # ── B: Riemannian (SCAFBTS) ─────────────────────────────
    for ch_name, ch_key in [('17ch', 'all'), ('6ch_temporal', 'temporal')]:
        for reg in ['svr', 'ridge']:
            for bands in ['5band', '8band']:
                for metric in ['riemann', 'euclid']:
                    configs.append({
                        'name': f'Riemann_{reg}_{metric}_{bands}_{ch_name}',
                        'type': 'riemann',
                        'channels': ch_key,
                        'freq_bands': bands,
                        'estimator': 'oas',
                        'metric': metric,
                        'regressor': reg,
                        'n_features': 100,
                        'temporal_smooth': True,
                        'smoothing_window': 3,
                        'scaler': True,
                    })

    # ── C: Riemannian 前额4ch (可穿戴场景) ──────────────────
    for bands in ['5band', '8band']:
        for metric in ['riemann']:
            configs.append({
                'name': f'Riemann_svr_{metric}_{bands}_4ch_forehead',
                'type': 'riemann',
                'channels': 'forehead',
                'freq_bands': bands,
                'estimator': 'oas',
                'metric': metric,
                'regressor': 'svr',
                'n_features': 100,
                'temporal_smooth': True,
                'smoothing_window': 3,
                'scaler': True,
            })

    return configs


# ── 主函数 ───────────────────────────────────────────────────

def main():
    # 准备
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    subjects = SUBJECTS or list_subjects(DATA_ROOT)
    configs = get_experiment_configs()

    print(f"=" * 70)
    print(f"SEED-VIG 警觉度估计实验")
    print(f"被试数: {len(subjects)}")
    print(f"实验配置数: {len(configs)}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"=" * 70)

    # 结果汇总
    summary = []  # list of dicts: {subject, exp_name, cor_mean, rmse_mean, ...}

    for subj_idx, subject_id in enumerate(subjects):
        print(f"\n{'='*70}")
        print(f"[{subj_idx+1}/{len(subjects)}] 被试: {subject_id}")
        print(f"{'='*70}")

        for cfg_idx, config in enumerate(configs):
            exp_name = config['name']
            print(f"\n  [{cfg_idx+1}/{len(configs)}] {exp_name}")

            t0 = time.time()
            try:
                results = run_experiment(subject_id, config, verbose=True)
                elapsed = time.time() - t0

                row = {
                    'subject': subject_id,
                    'exp_name': exp_name,
                    'cor_mean': results['cor_mean'],
                    'cor_std': results['cor_std'],
                    'global_cor': results['global_cor'],
                    'rmse_mean': results['rmse_mean'],
                    'rmse_std': results['rmse_std'],
                    'global_rmse': results['global_rmse'],
                    'mae_mean': results['mae_mean'],
                    'time_sec': elapsed,
                }
                summary.append(row)
                print(f"  -> 耗时 {elapsed:.1f}s, "
                      f"COR={row['cor_mean']:.4f}, RMSE={row['rmse_mean']:.4f}")

            except Exception as e:
                print(f"  -> ERROR: {e}")
                import traceback
                traceback.print_exc()
                summary.append({
                    'subject': subject_id,
                    'exp_name': exp_name,
                    'cor_mean': None,
                    'error': str(e),
                })

        # 每完成一个被试，保存中间结果
        _save_summary(summary)

    # ── 汇总统计 ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"全部实验完成！汇总结果:")
    print(f"{'='*70}")

    _print_aggregate_results(summary)
    _save_summary(summary, final=True)
    print(f"\n结果已保存至: {OUTPUT_DIR}")


def _save_summary(summary, final=False):
    """保存结果到 JSON。"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    suffix = 'final' if final else 'checkpoint'
    path = os.path.join(OUTPUT_DIR, f'results_{suffix}_{timestamp}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    return path


def _print_aggregate_results(summary):
    """打印按实验名称汇总的平均 COR/RMSE。"""
    from collections import defaultdict
    agg = defaultdict(lambda: {'cor': [], 'rmse': [], 'global_cor': []})

    for row in summary:
        if row['cor_mean'] is not None:
            name = row['exp_name']
            agg[name]['cor'].append(row['cor_mean'])
            agg[name]['rmse'].append(row['rmse_mean'])
            agg[name]['global_cor'].append(row['global_cor'])

    print(f"\n{'实验名称':<50s} {'被试数':>5s} {'COR_mean':>8s} {'COR_std':>8s} "
          f"{'RMSE':>8s} {'GlobalCOR':>8s}")
    print("-" * 90)

    for name in sorted(agg.keys()):
        stats = agg[name]
        n = len(stats['cor'])
        if n == 0:
            continue
        c_mean = np.mean(stats['cor'])
        c_std = np.std(stats['cor'])
        r_mean = np.mean(stats['rmse'])
        g_mean = np.mean(stats['global_cor'])
        print(f"{name:<50s} {n:>5d} {c_mean:>8.4f} {c_std:>8.4f} "
              f"{r_mean:>8.4f} {g_mean:>8.4f}")


# ── 快速测试 (单被试) ──────────────────────────────────────

def quick_test():
    """单被试快速验证——确保 pipeline 能跑通。"""
    subject_id = '10_20151125_noon'
    print(f"快速测试被试: {subject_id}\n")

    # 1. DE baseline
    print("=== DE + SVR (17ch) ===")
    X, y, _ = build_de_features_for_baseline(DATA_ROOT, subject_id)
    results_de = evaluate_de_baseline(
        lambda: SVR(kernel='rbf', C=1.0, gamma='scale'),
        X, y, verbose=True
    )

    # 2. Riemannian 5band
    print("\n=== Riemannian SVR (5band, 17ch) ===")
    raw, sr = load_raw_eeg(DATA_ROOT, subject_id)
    y = load_perclos(DATA_ROOT, subject_id)
    X, y = build_epochs_from_raw(raw, y, fs=sr)
    clf = SCAFBTSRegressor(
        freq_bands='5band', estimator='oas', metric='riemann',
        regressor='svr', n_features=50, fs=sr
    )
    results_r = evaluate_regressor(clf, X, y, verbose=True)

    print(f"\n快速测试完成:")
    print(f"  DE+SVR:        COR={results_de['cor_mean']:.4f}, RMSE={results_de['rmse_mean']:.4f}")
    print(f"  Riemannian+SVR: COR={results_r['cor_mean']:.4f}, RMSE={results_r['rmse_mean']:.4f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='快速单被试测试')
    parser.add_argument('--subjects', nargs='+', default=None, help='指定被试列表')
    parser.add_argument('--exp', type=str, default=None, help='只运行指定实验 (名称前缀)')
    args = parser.parse_args()

    if args.quick:
        quick_test()
    else:
        if args.subjects:
            SUBJECTS = args.subjects
        main()
