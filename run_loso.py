"""
跨被试 Leave-One-Subject-Out (LOSO) 实验 — JNE 修回 Experiment A
================================================================
匹配协议多组对照 (Table 4 缺口: DE+EOG):

  A  eog          EOG-only 基线
  B  de           DE-only 基线
  C  fbts         FBTS-only
  D  de_eog       DE+EOG        (审稿人要求补齐的关键对照)
  E  fbts_eog     FBTS+EOG      (主方法)
  F  de_fbts_eog  DE+FBTS+EOG   (三模态)

统一协议 (六组完全一致):
- 相同 LOSO folds、相同标签与指标;
- 协方差块的切空间参考均值仅用训练被试拟合;
- DE/EOG/FBTS 全部特征的 z-score 仅用训练被试池拟合;
- F-test top-K (K=100, 维度不足时全保留) 仅用训练 fold 拟合;
- 统一 Ridge(alpha=1.0) 回归器 (无组间调参预算差异);
- 因果 3 点滑动平均 (仅当前及过去 epoch), 同时记录未平滑 (raw) 指标。
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
from utils import cor, rmse, causal_ma
from sca_fbts_fast import apply_bandpass_filter
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.feature_selection import SelectKBest, f_regression

from config import SEED_VIG_ROOT, RESULTS_DIR as OUTPUT_DIR


FREQ_BANDS = {
    '5band': [(1,4),(4,8),(8,14),(14,31),(31,50)],
    '8band': [(1,4),(4,6),(6,8),(8,10),(10,12),(12,14),(14,20),(20,30)],
}

GROUPS = ['eog', 'de', 'fbts', 'de_eog', 'fbts_eog', 'de_fbts_eog']
GROUP_PARTS = {
    'eog':          ['eog'],
    'de':           ['de'],
    'fbts':         ['covs'],
    'de_eog':       ['de', 'eog'],
    'fbts_eog':     ['covs', 'eog'],
    'de_fbts_eog':  ['de', 'covs', 'eog'],
}


def _channel_idx(channels):
    if channels == 'temporal':
        return [0, 1, 2, 3, 4, 5]
    if channels == 'forehead':
        return [0, 1, 2, 3]
    return None


def precompute_subjects(subjects, channels, bands, verbose=True):
    """预计算所有被试的 DE / EOG / 各频段协方差。

    DE 与 EOG 截断到公共长度 n = min(len(y), len(de), len(eog))。
    """
    freq_bands = FREQ_BANDS[bands]
    ch_idx = _channel_idx(channels)
    data = {}

    for subj in subjects:
        raw, sr = load_raw_eeg(SEED_VIG_ROOT, subj)
        y = load_perclos(SEED_VIG_ROOT, subj)
        de, y_de, _ = build_de_features_for_baseline(
            SEED_VIG_ROOT, subj, channels=channels)
        eog = load_eog_features(SEED_VIG_ROOT, subj,
                                method='features_table_ica')

        n = int(min(len(y), len(de), len(eog)))
        y = np.asarray(y[:n], dtype=np.float64)
        de = np.asarray(de[:n], dtype=np.float64)
        eog = np.asarray(eog[:n], dtype=np.float64)

        # 协方差 (与 DE/EOG 同步截断)
        if ch_idx is not None:
            raw = raw[:, ch_idx]
        raw_T = raw.T.astype(np.float64)
        epoch_len = int(sr * 8)
        n_epochs = raw.shape[0] // epoch_len
        n = min(n, n_epochs)
        y, de, eog = y[:n], de[:n], eog[:n]

        cov_dict = {}
        cov_est = Covariances(estimator='oas')
        for low, high in freq_bands:
            filtered = apply_bandpass_filter(raw_T, low, high, sr)
            epochs = np.array([
                filtered[:, i * epoch_len:(i + 1) * epoch_len]
                for i in range(n)
            ], dtype=np.float64)
            cov_dict[(low, high)] = cov_est.transform(epochs)

        data[subj] = dict(de=de, eog=eog, covs=cov_dict, y=y)

    if verbose:
        print(f"  Precomputed {len(subjects)} subjects "
              f"({channels}, {bands}, n={n})")
    return data


def build_subject_blocks(data, test_subj, freq_bands):
    """一次构建某测试被试下各组的基础特征块。

    协方差块的切空间参考均值仅用训练被试池拟合 (每个测试被试只拟合一次,
    供 fbts / fbts_eog / de_fbts_eog 三组复用)。

    Returns:
        blocks: {'de': (X_tr, X_te), 'eog': (...), 'covs': (...)}
        y_tr, y_te
    """
    train_subs = [s for s in data.keys() if s != test_subj]
    blocks = {}

    for part in ('de', 'eog'):
        blocks[part] = (
            np.concatenate([data[s][part] for s in train_subs], axis=0),
            data[test_subj][part],
        )

    tr_list, te_list = [], []
    for b in freq_bands:
        covs_tr = np.concatenate(
            [data[s]['covs'][b] for s in train_subs], axis=0)
        ts = TangentSpace(metric='riemann')
        tr_list.append(ts.fit_transform(covs_tr))
        te_list.append(ts.transform(data[test_subj]['covs'][b]))
    blocks['covs'] = (np.hstack(tr_list), np.hstack(te_list))

    y_tr = np.concatenate([data[s]['y'] for s in train_subs])
    y_te = data[test_subj]['y']
    return blocks, y_tr, y_te


def fit_eval_group(X_tr, X_te, y_tr, y_te, k_features=100, alpha=1.0,
                   smooth_win=3):
    """匹配协议管线: F-test → z-score → Ridge → 因果平滑评测.

    Returns:
        (cor, rmse, cor_raw, rmse_raw, n_selected)
    """
    X_tr = np.asarray(X_tr, dtype=np.float64)
    X_te = np.asarray(X_te, dtype=np.float64)

    # F-test 特征选择 (仅训练池拟合)
    n_selected = X_tr.shape[1]
    if X_tr.shape[1] > k_features:
        sel = SelectKBest(f_regression, k=k_features)
        X_tr = sel.fit_transform(X_tr, y_tr)
        X_te = sel.transform(X_te)
        n_selected = X_tr.shape[1]

    # z-score (仅训练池拟合)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    # 统一回归器
    reg = Ridge(alpha=alpha)
    reg.fit(X_tr, y_tr)
    y_pred_raw = reg.predict(X_te)

    y_pred = causal_ma(y_pred_raw, window=smooth_win) \
        if len(y_pred_raw) > 1 else y_pred_raw

    return (cor(y_te, y_pred), rmse(y_te, y_pred),
            cor(y_te, y_pred_raw), rmse(y_te, y_pred_raw), n_selected)


def run_loso_matched(subjects, channels='temporal', bands='8band',
                     k_features=100, alpha=1.0, verbose=True):
    """匹配协议 LOSO: 六组 (A-F) 一次预计算、统一评测."""
    freq_bands = FREQ_BANDS[bands]
    data = precompute_subjects(subjects, channels, bands, verbose=verbose)

    results = {}
    acc = {g: defaultdict(list) for g in GROUPS}

    for i, test_subj in enumerate(subjects):
        blocks, y_tr, y_te = build_subject_blocks(
            data, test_subj, freq_bands)

        for group in GROUPS:
            parts = GROUP_PARTS[group]
            X_tr = np.hstack([blocks[p][0] for p in parts])
            X_te = np.hstack([blocks[p][1] for p in parts])
            raw_dim = X_tr.shape[1]
            c, r, c_raw, r_raw, n_sel = fit_eval_group(
                X_tr, X_te, y_tr, y_te,
                k_features=k_features, alpha=alpha)
            a = acc[group]
            a['cor'].append(c); a['rmse'].append(r)
            a['raw_cor'].append(c_raw); a['raw_rmse'].append(r_raw)
            a['n_sel'].append(n_sel)
            a['raw_dim'].append(int(raw_dim))

        if verbose:
            line = '  '.join(
                f"{g}:{acc[g]['cor'][-1]:.3f}" for g in GROUPS)
            print(f"  [{i+1}/{len(subjects)}] {test_subj}  {line}")

    for group in GROUPS:
        a = acc[group]
        results[group] = {
            'exp_name': f'{group}_{channels}_{bands}',
            'group': group, 'channels': channels, 'bands': bands,
            'cor_mean': float(np.mean(a['cor'])),
            'cor_std': float(np.std(a['cor'])),
            'cor_all': [float(x) for x in a['cor']],
            'rmse_mean': float(np.mean(a['rmse'])),
            'rmse_std': float(np.std(a['rmse'])),
            'raw_cor_mean': float(np.mean(a['raw_cor'])),
            'raw_cor_std': float(np.std(a['raw_cor'])),
            'raw_rmse_mean': float(np.mean(a['raw_rmse'])),
            'n_selected': int(np.mean(a['n_sel'])),
            'n_features': int(np.mean(a['raw_dim'])),
            'subjects': list(subjects),
            'cor_per_subject': {s: float(c)
                                for s, c in zip(subjects, a['cor'])},
            'raw_cor_per_subject': {s: float(c)
                                    for s, c in zip(subjects,
                                                    a['raw_cor'])},
            'rmse_per_subject': {s: float(r)
                                 for s, r in zip(subjects, a['rmse'])},
        }

    return results


def main():
    import argparse
    p = argparse.ArgumentParser(description='LOSO matched experiment (A-F)')
    p.add_argument('--channels', nargs='+', default=['temporal', 'all'],
                   choices=['temporal', 'all', 'forehead'])
    p.add_argument('--bands', nargs='+', default=['8band', '5band'],
                   choices=['5band', '8band'])
    p.add_argument('--quick', action='store_true',
                   help='仅前 3 名被试 (冒烟测试)')
    args = p.parse_args()

    subjects = list_subjects(SEED_VIG_ROOT)
    if args.quick:
        subjects = subjects[:3]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print(f"LOSO matched experiment (A-F): {len(subjects)} subjects")
    print(f"channels={args.channels}, bands={args.bands}, "
          f"K=100, Ridge(alpha=1), causal MA(3)")
    print("=" * 70)

    all_results = {}
    t_start = time.time()
    for ch in args.channels:
        for b in args.bands:
            res = run_loso_matched(subjects, channels=ch, bands=b)
            for k, v in res.items():
                all_results[v['exp_name']] = v

    # 汇总表
    print(f"\n{'='*78}")
    print(f"{'Experiment':<28s} {'COR':>8s} {'std':>7s} "
          f"{'rawCOR':>8s} {'RMSE':>7s} {'dim':>5s}")
    print("-" * 78)
    for name in sorted(all_results.keys()):
        r = all_results[name]
        print(f"{name:<28s} {r['cor_mean']:>8.4f} {r['cor_std']:>7.4f} "
              f"{r['raw_cor_mean']:>8.4f} {r['rmse_mean']:>7.4f} "
              f"{r['n_selected']:>5d}")

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(OUTPUT_DIR, f'results_loso_matched_{ts}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nTotal: {time.time()-t_start:.0f}s")
    print(f"Saved: {path}")


if __name__ == '__main__':
    main()
