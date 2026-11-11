"""
PERCLOS 中间区间与被试筛选敏感性分析 (JNE 修回 Experiment F)
=============================================================
问题: 现稿主分析为连续 PERCLOS 回归, 但二分类分析排除了 0.4-0.6 中间区间
并因此剔除 8-9 名被试。审稿人要求说明纳入/排除中间区间对主结论的影响,
并报告每名被试的有效 epoch 数、PERCLOS 分布与排除原因。

本脚本给出四个分析条件 (同一 5-fold 时序协议, 仅标签筛选不同):
  all      : 全部 epoch (主分析)
  excl_mid : 排除 PERCLOS 0.4-0.6 中间区间
  mid_only : 仅保留中间区间 (诊断用)
  narrow   : 排除标签范围过窄 (< 0.3 标准差) 的被试

每个条件报告:
  - 有效被试数 / 有效 epoch 数
  - COR (被试级均值 ± 标准差) 与 pooled COR
  - 逐被试明细: 有效 epoch 数、PERCLOS 分布 (min/max/mean/std)、COR

用法:
  python run_perclos_sensitivity.py                # 全部 23 被试
  python run_perclos_sensitivity.py --n-subjects 5
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
    build_de_features_for_baseline, load_eog_features,
)
from utils import cor, rmse, get_5fold_splits, causal_ma
from sca_fbts_fast import SCAFBTSRegressorFast
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from pyriemann.tangentspace import TangentSpace

from config import SEED_VIG_ROOT, RESULTS_DIR


BANDS_5 = [(1, 4), (4, 8), (8, 14), (14, 31), (31, 50)]
MID_LOW, MID_HIGH = 0.4, 0.6
NARROW_STD = 0.15
K_FEATURES = 100


def prepare_subject(subject_id):
    """载入被试并准备 DE / EOG / 各频段协方差。"""
    raw, sr = load_raw_eeg(SEED_VIG_ROOT, subject_id)
    y = load_perclos(SEED_VIG_ROOT, subject_id)
    de, _, _ = build_de_features_for_baseline(SEED_VIG_ROOT, subject_id)
    eog = load_eog_features(SEED_VIG_ROOT, subject_id,
                            method='features_table_ica')

    n = int(min(len(y), len(de), len(eog)))
    y = np.asarray(y[:n], dtype=np.float64)
    de = np.asarray(de[:n], dtype=np.float64)
    eog = np.asarray(eog[:n], dtype=np.float64)

    clf = SCAFBTSRegressorFast(
        freq_bands='5band', estimator='oas', metric='riemann',
        regressor='svr', n_features=None, fs=sr,
        temporal_smoothing=False, scaler=False,
    )
    clf.precompute(raw, fs=sr)
    covs = {b: clf._epochs_cov[b].astype(np.float64) for b in clf.freq_bands}
    return de, eog, covs, y


def eval_subject(de, eog, covs, y, mask, smooth_win=3):
    """在给定样本掩码下做 5-fold 时序评测。

    返回 (cor_mean, cor_std, rmse_mean, pooled_cor, pooled_rmse, cor_per_fold)。
    """
    idx_all = np.where(mask)[0]
    if len(idx_all) < 20:
        return None

    # 在保留的 epoch 上重新做时序 5-fold (保持时序性)
    n = len(idx_all)
    fold_size = n // 5
    cors, rmses, yt_all, yp_all = [], [], [], []

    for i in range(5):
        te_pos = np.arange(i * fold_size, (i + 1) * fold_size)
        tr_pos = np.concatenate([np.arange(0, i * fold_size),
                                 np.arange((i + 1) * fold_size, n)])
        tr_idx = idx_all[tr_pos]
        te_idx = idx_all[te_pos]
        if len(tr_idx) == 0 or len(te_idx) == 0:
            continue

        X_tr_parts, X_te_parts = [], []
        for band in BANDS_5:
            ts = TangentSpace(metric='riemann')
            X_tr_parts.append(ts.fit_transform(covs[band][tr_idx]))
            X_te_parts.append(ts.transform(covs[band][te_idx]))
        X_tr = np.hstack([np.hstack(X_tr_parts), de[tr_idx], eog[tr_idx]])
        X_te = np.hstack([np.hstack(X_te_parts), de[te_idx], eog[te_idx]])

        sel = SelectKBest(f_regression, k=min(K_FEATURES, X_tr.shape[1]))
        X_tr = sel.fit_transform(X_tr, y[tr_idx])
        X_te = sel.transform(X_te)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        # SVR epsilon: 默认 0.1 与正文协议一致, 仅当训练折标签标准差小于
        # epsilon 时 (mid_only 条件, std≈0.05) 按比例缩小, 否则预测退化为
        # 常数, COR 恒为 0。all / excl_mid 条件下 std≥0.115, epsilon 保持 0.1,
        # 与论文其他实验可比。
        y_scale = float(y[tr_idx].std())
        eps = 0.1 if y_scale >= 0.1 else max(1e-3, 0.1 * y_scale)
        clf = SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=eps)
        clf.fit(X_tr, y[tr_idx])
        y_pred_raw = clf.predict(X_te)
        y_pred = causal_ma(y_pred_raw, window=smooth_win) \
            if len(y_pred_raw) > 1 else y_pred_raw

        cors.append(cor(y[te_idx], y_pred))
        rmses.append(rmse(y[te_idx], y_pred))
        yt_all.append(y[te_idx])
        yp_all.append(y_pred)

    if not cors:
        return None

    yt = np.concatenate(yt_all)
    yp = np.concatenate(yp_all)
    return {
        'cor_mean': float(np.mean(cors)),
        'cor_std': float(np.std(cors)),
        'rmse_mean': float(np.mean(rmses)),
        'pooled_cor': float(cor(yt, yp)),
        'pooled_rmse': float(rmse(yt, yp)),
        'cor_per_fold': [float(c) for c in cors],
        'y_pred_std': float(yp.std()),
        'y_true_std': float(yt.std()),
        # SVR 默认 epsilon=0.1: 标签范围远小于 epsilon 时预测退化为常数
        'degenerate_prediction': bool(yp.std() < 1e-6),
    }


def process_subject(args):
    subject_id, conditions = args
    try:
        de, eog, covs, y = prepare_subject(subject_id)
    except Exception as e:
        import traceback; traceback.print_exc()
        return [{'subject': subject_id, 'error': str(e)}]

    out = []
    for cond in conditions:
        if cond == 'all':
            mask = np.ones(len(y), dtype=bool)
        elif cond == 'excl_mid':
            mask = ~((y >= MID_LOW) & (y <= MID_HIGH))
        elif cond == 'mid_only':
            mask = (y >= MID_LOW) & (y <= MID_HIGH)
        elif cond == 'narrow':
            mask = np.ones(len(y), dtype=bool)
        else:
            continue

        r = eval_subject(de, eog, covs, y, mask)
        if r is None:
            out.append({
                'subject': subject_id, 'condition': cond,
                'n_valid_epochs': int(mask.sum()), 'status': 'insufficient',
            })
            continue

        out.append({
            'subject': subject_id,
            'condition': cond,
            'n_valid_epochs': int(mask.sum()),
            'n_excluded_epochs': int((~mask).sum()),
            'perclos_min': float(y[mask].min()),
            'perclos_max': float(y[mask].max()),
            'perclos_mean': float(y[mask].mean()),
            'perclos_std': float(y[mask].std()),
            'cor_mean': r['cor_mean'],
            'cor_std': r['cor_std'],
            'rmse_mean': r['rmse_mean'],
            'pooled_cor': r['pooled_cor'],
            'pooled_rmse': r['pooled_rmse'],
            'cor_per_fold': r['cor_per_fold'],
            'y_pred_std': r['y_pred_std'],
            'y_true_std': r['y_true_std'],
            'degenerate_prediction': r['degenerate_prediction'],
        })
    return out


def main():
    import argparse
    p = argparse.ArgumentParser(
        description='PERCLOS mid-range / subject-filter sensitivity (Exp F)')
    p.add_argument('--n-subjects', type=int, default=0)
    p.add_argument('--quick', action='store_true')
    p.add_argument('--n-jobs', type=int, default=-1)
    args = p.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    subjects = list_subjects(SEED_VIG_ROOT)
    if args.quick:
        subjects = subjects[:3]
    elif args.n_subjects:
        subjects = subjects[:args.n_subjects]

    conditions = ['all', 'excl_mid', 'mid_only']
    print('=' * 84)
    print(f'PERCLOS sensitivity analysis: {len(subjects)} subjects, '
          f'conditions={conditions}')
    print('=' * 84)

    t0 = time.time()
    tasks = [(s, conditions) for s in subjects]
    if args.n_jobs == 1 or len(subjects) == 1:
        nested = [process_subject(t) for t in tasks]
    else:
        from joblib import Parallel, delayed
        nested = Parallel(n_jobs=args.n_jobs, verbose=10)(
            delayed(process_subject)(t) for t in tasks)
    all_results = [r for sub in nested for r in sub]

    # ── 标签范围过窄的被试 (narrow 条件的排除依据) ──
    per_subject_labels = {}
    for r in all_results:
        if r.get('condition') == 'all' and r.get('perclos_std') is not None:
            per_subject_labels[r['subject']] = r['perclos_std']
    narrow_subjects = sorted([s for s, sd in per_subject_labels.items()
                              if sd < NARROW_STD])

    # ── 汇总 ──
    from collections import defaultdict
    by_cond = defaultdict(list)
    for r in all_results:
        if r.get('cor_mean') is not None:
            by_cond[r['condition']].append(r)

    print(f"\n{'=' * 92}")
    print(f"{'Condition':<12s} {'N subj':>7s} {'N epoch':>9s} {'COR':>8s} "
          f"{'std':>7s} {'RMSE':>8s} {'pooledCOR':>10s}")
    print('-' * 92)
    summary = []
    for cond in conditions + ['narrow']:
        rows = by_cond.get(cond, [])
        if cond == 'narrow':
            rows = [r for r in by_cond.get('all', [])
                    if r['subject'] not in narrow_subjects]
        if not rows:
            continue
        cors = np.array([r['cor_mean'] for r in rows])
        rmses = np.array([r['rmse_mean'] for r in rows])
        pooled = [r['pooled_cor'] for r in rows if r.get('pooled_cor') is not None]
        n_ep = sum(r['n_valid_epochs'] for r in rows)
        print(f"{cond:<12s} {len(rows):>7d} {n_ep:>9d} {cors.mean():>8.4f} "
              f"{cors.std():>7.4f} {rmses.mean():>8.4f} "
              f"{(np.mean(pooled) if pooled else float('nan')):>10.4f}")
        summary.append({
            'condition': cond, 'n_subjects': len(rows), 'n_epochs': int(n_ep),
            'cor_mean': float(cors.mean()), 'cor_std': float(cors.std()),
            'rmse_mean': float(rmses.mean()),
            'pooled_cor_mean': float(np.mean(pooled)) if pooled else None,
        })

    print(f"\n标签范围过窄 (PERCLOS std < {NARROW_STD}) 的被试 "
          f"({len(narrow_subjects)}/{len(subjects)}): {narrow_subjects}")

    # ── 排除中间区间后的配对比较 ──
    stat_rows = []
    corr_by = defaultdict(dict)
    for r in all_results:
        if r.get('cor_mean') is not None:
            corr_by[r['condition']][r['subject']] = r['cor_mean']
    if 'all' in corr_by and 'excl_mid' in corr_by:
        from scipy import stats as _st
        common = sorted(set(corr_by['all']) & set(corr_by['excl_mid']))
        if len(common) >= 3:
            a = np.array([corr_by['all'][s] for s in common])
            b = np.array([corr_by['excl_mid'][s] for s in common])
            t_stat, p_val = _st.ttest_rel(a, b)
            diff = a - b
            sd = diff.std(ddof=1)
            stat_rows.append({
                'comparison': 'all vs excl_mid',
                'n': len(common), 'mean_diff': float(diff.mean()),
                't_stat': float(t_stat), 'p_value': float(p_val),
                'cohens_dz': float(diff.mean() / sd) if sd > 0 else 0.0,
                'subjects': common,
            })
            print(f"\nall vs excl_mid: diff={diff.mean():+.4f} "
                  f"t({len(common)-1})={t_stat:.3f} p={p_val:.4f} "
                  f"d_z={diff.mean() / sd if sd > 0 else 0:.3f}")

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(RESULTS_DIR, f'results_perclos_sensitivity_{ts}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'statistics': stat_rows,
                   'narrow_subjects': narrow_subjects,
                   'per_subject': all_results},
                  f, indent=2, ensure_ascii=False, default=str)
    print(f'\nSaved: {path}')
    print(f'Total: {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
