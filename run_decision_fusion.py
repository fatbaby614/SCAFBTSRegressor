"""
Decision-level fusion 基线 (JNE 修回 Reviewer 2)
===============================================
现稿用的是 feature-level (early) fusion: EEG 与 EOG 特征拼接后送入同一模型。
审稿人指出这只是透明的融合基线, 并建议补一个简单的 decision-level baseline。

本脚本在 SEED-VIG 被试内 5-fold 协议下比较:
  feature-level : [FBTS 切空间 | EOG] → F-test(K) → 标准化 → SVR
  decision-level : EEG 模型与 EOG 模型分别训练, 预测值按权重 w 线性组合
                   w 在训练 fold 内通过内层 CV 选出 (在训练集上做 5-fold
                   选 w, 不用测试 fold 的标签)

融合权重 w 的内层选择:
  - 网格 w ∈ {0.0, 0.1, ..., 1.0}
  - 对每个 w, 在训练 fold 内部再做 5-fold 时序 CV, 计算 pooled MSE
  - 取 MSE 最小的 w, 用它组合测试 fold 的两个模型预测
  - EEG 与 EOG 的分量分别标准化后再加权 (避免量纲主导)

对照组:
  EOG-only / FBTS-only / feature-level FBTS+EOG / decision-level FBTS+EOG

用法:
  python run_decision_fusion.py                  # 全部 23 被试
  python run_decision_fusion.py --n-subjects 5
"""

import sys
import os
import json
import time
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import (
    list_subjects, load_raw_eeg, load_perclos, load_eog_features,
)
from utils import cor, rmse, get_5fold_splits, causal_ma
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from pyriemann.tangentspace import TangentSpace

from config import SEED_VIG_ROOT, RESULTS_DIR


BANDS_5 = [(1, 4), (4, 8), (8, 14), (14, 31), (31, 50)]
K_FEATURES = 100
W_GRID = np.round(np.arange(0, 1.01, 0.1), 2)


def prepare_subject(subject_id):
    """载入被试, 计算 DE/EOG 与各频段协方差。"""
    from sca_fbts_fast import apply_bandpass_filter
    from pyriemann.estimation import Covariances

    raw, sr = load_raw_eeg(SEED_VIG_ROOT, subject_id)
    y = load_perclos(SEED_VIG_ROOT, subject_id)
    eog = load_eog_features(SEED_VIG_ROOT, subject_id,
                            method='features_table_ica')

    n = int(min(len(y), len(eog)))
    y = np.asarray(y[:n], dtype=np.float64)
    eog = np.asarray(eog[:n], dtype=np.float64)

    epoch_len = int(sr * 8)
    n_epochs = raw.shape[0] // epoch_len
    n = min(n, n_epochs)
    y, eog = y[:n], eog[:n]

    raw_T = raw.T.astype(np.float64)
    cov_est = Covariances(estimator='oas')
    covs = {}
    for low, high in BANDS_5:
        filtered = apply_bandpass_filter(raw_T, low, high, sr)
        epochs = np.array([filtered[:, i * epoch_len:(i + 1) * epoch_len]
                           for i in range(n)], dtype=np.float64)
        covs[(low, high)] = cov_est.transform(epochs)

    return covs, eog, y


def _tangent(covs, tr_idx, te_idx):
    """切空间: 参考均值仅用 tr_idx 拟合。"""
    ftr, fte = [], []
    for band in BANDS_5:
        ts = TangentSpace(metric='riemann')
        ftr.append(ts.fit_transform(covs[band][tr_idx]))
        fte.append(ts.transform(covs[band][te_idx]))
    return np.hstack(ftr), np.hstack(fte)


def _fit_model(X, y, k_features=K_FEATURES):
    """F-test + 标准化 + SVR (全部在给定训练集内拟合)。"""
    X = np.asarray(X, dtype=np.float64)
    sel = None
    if X.shape[1] > k_features:
        sel = SelectKBest(f_regression, k=k_features)
        X = sel.fit_transform(X, y)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # epsilon 与正文协议一致 (0.1); 标签标准差更小时按比例缩小
    y_scale = float(np.std(y))
    eps = 0.1 if y_scale >= 0.1 else max(1e-3, 0.1 * y_scale)
    model = SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=eps)
    model.fit(X, y)
    return model, scaler, sel


def _predict(model, scaler, sel, X):
    X = np.asarray(X, dtype=np.float64)
    if sel is not None:
        X = sel.transform(X)
    return model.predict(scaler.transform(X))


def decision_level_fold(covs, eog, y, tr_idx, te_idx):
    """一个 fold 内的 decision-level fusion。

    Returns:
        dict of method -> prediction array
    """
    # EEG 分量 (FBTS)
    fb_tr, fb_te = _tangent(covs, tr_idx, te_idx)
    eeg_model, eeg_scaler, eeg_sel = _fit_model(fb_tr, y[tr_idx])
    p_eeg = _predict(eeg_model, eeg_scaler, eeg_sel, fb_te)

    # EOG 分量
    eog_model, eog_scaler, eog_sel = _fit_model(eog[tr_idx], y[tr_idx])
    p_eog = _predict(eog_model, eog_scaler, eog_sel, eog[te_idx])

    # 分量标准化 (用训练集预测的均值/标准差), 避免量纲主导
    inner = get_5fold_splits(len(tr_idx), 5)
    eeg_tr_pred = np.zeros(len(tr_idx))
    eog_tr_pred = np.zeros(len(tr_idx))
    for itr, ite in inner:
        eeg_model_i, sc_i, sel_i = _fit_model(fb_tr[itr], y[tr_idx][itr])
        eeg_tr_pred[ite] = _predict(eeg_model_i, sc_i, sel_i, fb_tr[ite])
        eog_model_i, sc_i, sel_i = _fit_model(eog[tr_idx][itr], y[tr_idx][itr])
        eog_tr_pred[ite] = _predict(eog_model_i, sc_i, sel_i, eog[tr_idx][ite])

    mu_e, sd_e = eeg_tr_pred.mean(), eeg_tr_pred.std(ddof=1)
    mu_o, sd_o = eog_tr_pred.mean(), eog_tr_pred.std(ddof=1)
    sd_e = sd_e if sd_e > 1e-12 else 1.0
    sd_o = sd_o if sd_o > 1e-12 else 1.0

    z_eeg_tr = (eeg_tr_pred - mu_e) / sd_e
    z_eog_tr = (eog_tr_pred - mu_o) / sd_o
    z_eeg_te = (p_eeg - mu_e) / sd_e
    z_eog_te = (p_eog - mu_o) / sd_o

    # 网格搜索融合权重 (用训练集内层 CV 的 MSE)
    best_w, best_mse = 0.5, np.inf
    for w in W_GRID:
        combo = w * z_eeg_tr + (1 - w) * z_eog_tr
        mse = float(np.mean((combo - y[tr_idx]) ** 2))
        if mse < best_mse:
            best_mse, best_w = mse, float(w)

    # 融合预测反变换回 PERCLOS 尺度: 用训练折线性回归标定
    combo_tr = best_w * z_eeg_tr + (1 - best_w) * z_eog_tr
    A = np.vstack([combo_tr, np.ones(len(combo_tr))]).T
    coef, *_ = np.linalg.lstsq(A, y[tr_idx], rcond=None)
    combo_te = best_w * z_eeg_te + (1 - best_w) * z_eog_te
    p_fused = coef[0] * combo_te + coef[1]

    # feature-level fusion 对照
    fb_eog_tr = np.hstack([fb_tr, eog[tr_idx]])
    fb_eog_te = np.hstack([fb_te, eog[te_idx]])
    feat_model, feat_scaler, feat_sel = _fit_model(fb_eog_tr, y[tr_idx])
    p_feat = _predict(feat_model, feat_scaler, feat_sel, fb_eog_te)

    return {
        'FBTS-only': p_eeg,
        'EOG-only': p_eog,
        'feature-level FBTS+EOG': p_feat,
        'decision-level FBTS+EOG': p_fused,
        'fusion_weight': np.array([best_w]),
    }


def process_subject(args):
    subject_id = args
    try:
        covs, eog, y = prepare_subject(subject_id)
    except Exception as e:
        import traceback; traceback.print_exc()
        return [{'subject': subject_id, 'exp_name': 'ALL', 'error': str(e)}]

    out = []
    for tr_idx, te_idx in get_5fold_splits(len(y), 5):
        preds = decision_level_fold(covs, eog, y, tr_idx, te_idx)
        w = float(preds.pop('fusion_weight')[0])
        for name, p in preds.items():
            p_smooth = causal_ma(p, window=3) if len(p) > 1 else p
            out.append({
                'subject': subject_id,
                'exp_name': name,
                'fold_cor': float(cor(y[te_idx], p_smooth)),
                'fold_rmse': float(rmse(y[te_idx], p_smooth)),
                'fusion_weight': w,
            })
    return out


def main():
    import argparse
    p = argparse.ArgumentParser(description='Decision-level fusion baseline')
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

    print('=' * 78)
    print(f'Decision-level fusion baseline: {len(subjects)} subjects')
    print('=' * 78)

    t0 = time.time()
    if args.n_jobs == 1 or len(subjects) == 1:
        nested = [process_subject(s) for s in subjects]
    else:
        from joblib import Parallel, delayed
        nested = Parallel(n_jobs=args.n_jobs, verbose=10)(
            delayed(process_subject)(s) for s in subjects)
    all_results = [r for sub in nested for r in sub]

    from collections import defaultdict
    from scipy import stats as _st
    agg = defaultdict(list)
    weights = []
    for r in all_results:
        if r.get('fold_cor') is not None:
            agg[r['exp_name']].append(r['fold_cor'])
            if r['exp_name'] == 'decision-level FBTS+EOG':
                weights.append(r['fusion_weight'])

    print(f"\n{'Method':<28s} {'N':>4s} {'COR':>8s} {'std':>8s}")
    print('-' * 52)
    summary = []
    for name in sorted(agg):
        vals = np.array(agg[name])
        print(f"{name:<28s} {len(vals):>4d} {vals.mean():>8.4f} {vals.std():>8.4f}")
        summary.append({'exp_name': name, 'n': len(vals),
                        'cor_mean': float(vals.mean()),
                        'cor_std': float(vals.std())})

    # decision-level vs feature-level 配对检验（subject-level：先对每被试的5折取平均再检验）
    by_name = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        if r.get('fold_cor') is not None:
            by_name[r['exp_name']][r['subject']].append(r['fold_cor'])
    stat_rows = []
    a = 'decision-level FBTS+EOG'
    b = 'feature-level FBTS+EOG'
    if a in by_name and b in by_name:
        common = sorted(set(by_name[a]) & set(by_name[b]))
        if len(common) >= 3:
            va = np.array([np.mean(by_name[a][s]) for s in common])
            vb = np.array([np.mean(by_name[b][s]) for s in common])
            t_stat, p_val = _st.ttest_rel(va, vb)
            diff = va - vb
            sd = diff.std(ddof=1)
            dz = diff.mean() / sd if sd > 0 else 0.0
            print(f"\ndecision-level vs feature-level (subject-level, n={len(common)}): "
                  f"diff={diff.mean():+.4f} t({len(common)-1})={t_stat:.3f} "
                  f"p={p_val:.4f} dz={dz:+.3f}")
            stat_rows.append({
                'comparison': 'decision-level vs feature-level (subject-level)',
                'n': len(common), 'mean_diff': float(diff.mean()),
                't_stat': float(t_stat), 'p_value': float(p_val),
                'cohens_dz': float(dz),
            })

    if weights:
        w = np.array(weights)
        print(f"\n融合权重 w (FBTS 分量占比): mean={w.mean():.3f} "
              f"median={np.median(w):.3f} std={w.std():.3f}")

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(RESULTS_DIR, f'results_decision_fusion_{ts}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'statistics': stat_rows,
                   'fusion_weights': weights, 'per_fold': all_results},
                  f, indent=2, ensure_ascii=False, default=str)
    print(f'\nSaved: {path}')
    print(f'Total: {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
