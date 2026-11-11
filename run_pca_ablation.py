"""
特征维度与融合公平性消融 (JNE 修回 Experiment D)
================================================
问题: 现稿 Table 2 中 FBTS+EOG (801 维) 与 DE+EOG (121 维) 维度差异近 7 倍,
三模态 (886 维) 反而略低, 审稿人质疑维度不平衡。

本脚本在同一被试内 5-fold 协议下比较各维度控制方案, 并同时报告
"理论维度" 与 "实际入模维度" (F-test top-K 与 PCA 之后的实际列数):

  A  DE                 (85 维)
  B  FBTS               (765 维 → F-test K=100)
  C  FBTS-PCA50         (切空间后 PCA 至 50 维, 再 F-test K=50)
  D  FBTS-PCA85         (与 DE 维度对齐: 85 维)
  E  FBTS-PCA100        (与主实验 F-test K=100 对齐: 100 维)
  F  DE+EOG             (121 维 → F-test K=100)
  G  FBTS-PCA85+EOG     (85+36=121 维)
  H  FBTS-PCA100+EOG    (100+36=136 维)
  I  DE+FBTS+EOG        (886 维 → F-test K=100)
  J  DE+FBTS-PCA85+EOG  (85+85+36=206 维)

关键约束:
  - PCA 只在训练 fold 内拟合, 测试 fold 只 transform;
  - F-test (SelectKBest) 只在训练 fold 内拟合;
  - 标准化只在训练 fold 内拟合;
  - 切空间参考均值只在训练 fold 内拟合;
  - PCA 与 F-test 是不同操作, 分别报告, 不混称为"降维".

用法:
  python run_pca_ablation.py                  # 全部 23 被试
  python run_pca_ablation.py --n-subjects 5   # 5 被试消融
  python run_pca_ablation.py --quick
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
from utils import cor, rmse, mae, get_5fold_splits, causal_ma
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from pyriemann.tangentspace import TangentSpace

from config import SEED_VIG_ROOT, RESULTS_DIR


BANDS_5 = [(1, 4), (4, 8), (8, 14), (14, 31), (31, 50)]
K_DEFAULT = 100


# ── 配置 ─────────────────────────────────────────────────────

def build_configs():
    """返回 [(name, blocks, pca_dims, ftest_k, dim_note)]。

    blocks: list of ('de'|'eog'|'fbts',) — 'fbts' 段整体做 PCA
    pca_dims: 对 FBTS 段做 PCA 的目标维度 (None = 不做)
    ftest_k: F-test 保留维度 (None = 不做 F-test)
    """
    cfgs = [
        ('DE',                 ['de'],               None, None, '85'),
        ('FBTS',               ['fbts'],             None, K_DEFAULT, '765'),
        ('FBTS-PCA50',         ['fbts'],             50,  50,       '765→50'),
        ('FBTS-PCA85',         ['fbts'],             85,  85,       '765→85'),
        ('FBTS-PCA100',        ['fbts'],             100, 100,      '765→100'),
        ('DE+EOG',             ['de', 'eog'],        None, K_DEFAULT, '121'),
        ('FBTS-PCA85+EOG',     ['fbts', 'eog'],      85,  None,      '85+36=121'),
        ('FBTS-PCA100+EOG',    ['fbts', 'eog'],      100, None,      '100+36=136'),
        ('DE+FBTS+EOG',        ['de', 'fbts', 'eog'], None, K_DEFAULT, '886'),
        ('DE+FBTS-PCA85+EOG',  ['de', 'fbts', 'eog'], 85,  None,      '85+85+36=206'),
    ]
    return cfgs


# ── 数据准备 ─────────────────────────────────────────────────

def prepare_subject(subject_id, channels='all'):
    """返回 (de, eog, covs_by_band, y)。"""
    raw, sr = load_raw_eeg(SEED_VIG_ROOT, subject_id)
    y = load_perclos(SEED_VIG_ROOT, subject_id)
    de, _, _ = build_de_features_for_baseline(SEED_VIG_ROOT, subject_id,
                                              channels=channels)
    eog = load_eog_features(SEED_VIG_ROOT, subject_id,
                            method='features_table_ica')
    eog = np.asarray(eog, dtype=np.float64)

    n = int(min(len(y), len(de), len(eog)))
    y = np.asarray(y[:n], dtype=np.float64)
    de = np.asarray(de[:n], dtype=np.float64)
    eog = eog[:n]

    from sca_fbts_fast import apply_bandpass_filter
    epoch_len = int(sr * 8)
    n_epochs = raw.shape[0] // epoch_len
    n = min(n, n_epochs)
    y, de, eog = y[:n], de[:n], eog[:n]

    raw_T = raw.T.astype(np.float64)
    covs = {}
    for low, high in BANDS_5:
        filtered = apply_bandpass_filter(raw_T, low, high, sr)
        epochs = np.array([filtered[:, i * epoch_len:(i + 1) * epoch_len]
                           for i in range(n)], dtype=np.float64)
        from pyriemann.estimation import Covariances
        covs[(low, high)] = Covariances(estimator='oas').transform(epochs)

    return de, eog, covs, y


# ── 评测 ─────────────────────────────────────────────────────

def evaluate_config(de, eog, covs, y, blocks, pca_dims, ftest_k,
                    smooth_win=3):
    """单被试 5-fold 评测。返回 (cor, rmse, mae, actual_dim, raw_cor)。"""
    splits = get_5fold_splits(len(y), 5)
    cors, rmses, maes, raw_cors, dims = [], [], [], [], []

    for tr_idx, te_idx in splits:
        X_tr_parts, X_te_parts = [], []

        for b in blocks:
            if b == 'de':
                X_tr_parts.append(de[tr_idx])
                X_te_parts.append(de[te_idx])
            elif b == 'eog':
                X_tr_parts.append(eog[tr_idx])
                X_te_parts.append(eog[te_idx])
            elif b == 'fbts':
                ftr, fte = [], []
                for band in BANDS_5:
                    ts = TangentSpace(metric='riemann')
                    ftr.append(ts.fit_transform(covs[band][tr_idx]))
                    fte.append(ts.transform(covs[band][te_idx]))
                fb_tr = np.hstack(ftr)
                fb_te = np.hstack(fte)
                # PCA: 仅训练 fold 拟合
                if pca_dims is not None:
                    k = min(pca_dims, fb_tr.shape[1], fb_tr.shape[0])
                    pca = PCA(n_components=k, random_state=42)
                    fb_tr = pca.fit_transform(fb_tr)
                    fb_te = pca.transform(fb_te)
                X_tr_parts.append(fb_tr)
                X_te_parts.append(fb_te)

        X_tr = np.hstack(X_tr_parts).astype(np.float64)
        X_te = np.hstack(X_te_parts).astype(np.float64)

        # F-test: 仅训练 fold 拟合
        if ftest_k is not None and X_tr.shape[1] > ftest_k:
            sel = SelectKBest(f_regression, k=ftest_k)
            X_tr = sel.fit_transform(X_tr, y[tr_idx])
            X_te = sel.transform(X_te)

        # 标准化: 仅训练 fold 拟合
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        # SVR epsilon: 默认 0.1 与正文协议一致; 仅当训练折标签标准差更小
        # 时按比例缩小, 避免预测退化为常数 (标签范围窄的被试)。
        y_scale = float(y[tr_idx].std())
        eps = 0.1 if y_scale >= 0.1 else max(1e-3, 0.1 * y_scale)
        clf = SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=eps)
        clf.fit(X_tr, y[tr_idx])
        y_pred_raw = clf.predict(X_te)
        y_pred = causal_ma(y_pred_raw, window=smooth_win) \
            if len(y_pred_raw) > 1 else y_pred_raw

        dims.append(X_tr.shape[1])
        cors.append(cor(y[te_idx], y_pred))
        rmses.append(rmse(y[te_idx], y_pred))
        maes.append(mae(y[te_idx], y_pred))
        raw_cors.append(cor(y[te_idx], y_pred_raw))

    return (float(np.mean(cors)), float(np.mean(rmses)), float(np.mean(maes)),
            int(np.mean(dims)), float(np.mean(raw_cors)))


def process_subject(args):
    """joblib 可序列化的单被试处理。"""
    subject_id, configs = args
    results = []
    try:
        de, eog, covs, y = prepare_subject(subject_id)
    except Exception as e:
        import traceback; traceback.print_exc()
        return [{'subject': subject_id, 'exp_name': 'ALL', 'error': str(e)}]

    for name, blocks, pca_dims, ftest_k, dim_note in configs:
        t0 = time.time()
        try:
            c, r, m, dim, raw_c = evaluate_config(de, eog, covs, y,
                                                  blocks, pca_dims, ftest_k)
            results.append({
                'subject': subject_id,
                'exp_name': name,
                'cor_mean': c, 'rmse_mean': r, 'mae_mean': m,
                'raw_cor_mean': raw_c,
                'n_features_actual': dim,
                'dim_note': dim_note,
                'time_sec': time.time() - t0,
            })
        except Exception as e:
            import traceback; traceback.print_exc()
            results.append({
                'subject': subject_id, 'exp_name': name,
                'cor_mean': None, 'error': str(e),
            })
    return results


# ── 主流程 ───────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser(description='Feature dimension / PCA ablation (Exp D)')
    p.add_argument('--n-subjects', type=int, default=0,
                   help='仅跑前 N 名被试 (0 = 全部 23)')
    p.add_argument('--quick', action='store_true')
    p.add_argument('--n-jobs', type=int, default=-1)
    args = p.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    subjects = list_subjects(SEED_VIG_ROOT)
    if args.quick:
        subjects = subjects[:3]
    elif args.n_subjects:
        subjects = subjects[:args.n_subjects]

    configs = build_configs()
    print('=' * 76)
    print(f'Feature dimension / PCA ablation: {len(subjects)} subjects x '
          f'{len(configs)} configs')
    print('=' * 76)

    t0 = time.time()
    tasks = [(s, configs) for s in subjects]
    if args.n_jobs == 1 or len(subjects) == 1:
        nested = [process_subject(t) for t in tasks]
    else:
        from joblib import Parallel, delayed
        nested = Parallel(n_jobs=args.n_jobs, verbose=10)(
            delayed(process_subject)(t) for t in tasks)
    all_results = [r for sub in nested for r in sub]

    # ── 汇总 ──
    from collections import defaultdict
    agg = defaultdict(list)
    dim_agg = defaultdict(list)
    for r in all_results:
        if r.get('cor_mean') is not None:
            agg[r['exp_name']].append(r)
            dim_agg[r['exp_name']].append(r.get('n_features_actual'))

    print(f"\n{'=' * 92}")
    print(f"{'Config':<22s} {'N':>3s} {'COR':>9s} {'std':>8s} {'RMSE':>8s} "
          f"{'rawCOR':>8s} {'theoDim':>9s} {'actDim':>7s}")
    print('-' * 92)
    summary = []
    for name, _, _, _, dim_note in configs:
        rows = agg.get(name, [])
        if not rows:
            continue
        cors = np.array([r['cor_mean'] for r in rows])
        rmses = np.array([r['rmse_mean'] for r in rows])
        raws = np.array([r['raw_cor_mean'] for r in rows])
        dims = np.array(dim_agg[name])
        print(f"{name:<22s} {len(rows):>3d} {cors.mean():>9.4f} {cors.std():>8.4f} "
              f"{rmses.mean():>8.4f} {raws.mean():>8.4f} {dim_note:>9s} "
              f"{int(dims.mean()):>7d}")
        summary.append({
            'exp_name': name, 'n_subjects': len(rows),
            'cor_mean': float(cors.mean()), 'cor_std': float(cors.std()),
            'rmse_mean': float(rmses.mean()), 'raw_cor_mean': float(raws.mean()),
            'theoretical_dim': dim_note,
            'actual_dim_mean': int(dims.mean()),
        })

    # ── 配对检验 (被试级) ──
    from scipy import stats as _st
    pairs = [
        ('FBTS-PCA85 vs DE', 'FBTS-PCA85', 'DE'),
        ('FBTS-PCA85+EOG vs DE+EOG', 'FBTS-PCA85+EOG', 'DE+EOG'),
        ('FBTS-PCA100+EOG vs DE+EOG', 'FBTS-PCA100+EOG', 'DE+EOG'),
        ('FBTS vs DE', 'FBTS', 'DE'),
        ('DE+FBTS+EOG vs FBTS', 'DE+FBTS+EOG', 'FBTS'),
    ]
    by_name = {name: {r['subject']: r['cor_mean'] for r in agg.get(name, [])}
               for name, _, _, _, _ in configs}
    print(f"\n{'=' * 92}")
    print(f"{'Comparison':<34s} {'diff':>8s} {'t':>8s} {'p':>9s} {'d_z':>7s}")
    print('-' * 92)
    stat_rows = []
    for desc, a, b in pairs:
        common = sorted(set(by_name.get(a, {})) & set(by_name.get(b, {})))
        if len(common) < 3:
            continue
        ca = np.array([by_name[a][s] for s in common])
        cb = np.array([by_name[b][s] for s in common])
        t_stat, p_val = _st.ttest_rel(ca, cb)
        diff = ca - cb
        sd = diff.std(ddof=1)
        d_z = diff.mean() / sd if sd > 0 else 0.0
        print(f"{desc:<34s} {diff.mean():>8.4f} {t_stat:>8.3f} {p_val:>9.4f} "
              f"{d_z:>7.3f}")
        stat_rows.append({'comparison': desc, 'method_a': a, 'method_b': b,
                          'n': len(common), 'mean_diff': float(diff.mean()),
                          't_stat': float(t_stat), 'p_value': float(p_val),
                          'cohens_dz': float(d_z)})

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(RESULTS_DIR, f'results_pca_ablation_{ts}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'statistics': stat_rows,
                   'per_subject': all_results},
                  f, indent=2, ensure_ascii=False, default=str)
    print(f'\nSaved: {path}')
    print(f'Total: {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
