"""
融合实验 — DE/PSD + Riemannian + EOG 全面对比
===============================================
单模态 + 双模态融合 + 三模态融合, 5-fold 被试内评测.
"""

import sys, os, json, time, numpy as np
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import (
    list_subjects, load_raw_eeg, load_perclos,
    build_de_features_for_baseline, load_eog_features
)
from utils import cor, rmse, mae, get_5fold_splits
from sca_fbts_fast import SCAFBTSRegressorFast
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

from config import SEED_VIG_ROOT as SEED_VIG, RESULTS_DIR as OUTPUT


# ── 特征提取 ──────────────────────────────────────────────

def extract_de_features(subject_id, channels='all', feature_type='de_LDS'):
    """提取 DE 或 PSD 特征 (标准化后)."""
    X, y, ch_names = build_de_features_for_baseline(
        SEED_VIG, subject_id, channels=channels, feature_type=feature_type
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X.astype(np.float64), y, scaler


def extract_eog_features(subject_id):
    """提取 EOG 特征 (标准化后)."""
    X = load_eog_features(SEED_VIG, subject_id, method='features_table_ica')
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X.astype(np.float64), scaler


def extract_riemann_features(subject_id, channels='all', bands='5band'):
    """预计算 Riemannian 切空间特征."""
    freq_bands = [(1,4),(4,8),(8,14),(14,31),(31,50)] if bands == '5band' else \
                 [(1,4),(4,6),(6,8),(8,10),(10,12),(12,14),(14,20),(20,30)]

    raw, sr = load_raw_eeg(SEED_VIG, subject_id)
    y = load_perclos(SEED_VIG, subject_id)

    if channels == 'temporal':
        raw = raw[:, [0,1,2,3,4,5]]
    elif channels == 'forehead':
        raw = raw[:, [0,1,2,3]]

    clf = SCAFBTSRegressorFast(
        freq_bands=bands, estimator='oas', metric='riemann',
        regressor='svr', n_features=None, fs=sr,
        temporal_smoothing=False, scaler=False,
    )
    clf.precompute(raw, fs=sr)

    # 提取全部切空间特征 (不选特征, 不做标准化 — 留给融合管线统一处理)
    features_list = []
    for (low, high) in clf.freq_bands:
        covs = clf._epochs_cov[(low, high)]
        # 对全量数据拟合切空间 (用全量 y 做参考点)
        from pyriemann.tangentspace import TangentSpace
        ts = TangentSpace(metric='riemann')
        feats = ts.fit_transform(covs, y)
        features_list.append(feats)

    X = np.hstack(features_list).astype(np.float64)
    return X, y


# ── 5-fold 评测 (支持特征拼接融合) ─────────────────────────

def evaluate_fusion_5fold(X, y, n_features=150, temporal_smooth=True,
                          smooth_win=3, verbose=False):
    """对 (n_samples, n_features) 特征矩阵做 5-fold 评测."""
    splits = get_5fold_splits(len(y), 5)
    cor_list, rmse_list, mae_list = [], [], []
    yt_all, yp_all = [], []

    for tr_idx, te_idx in splits:
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]

        # 特征选择
        if n_features and X_tr.shape[1] > n_features:
            sel = SelectKBest(f_regression, k=min(n_features, X_tr.shape[1]))
            X_tr = sel.fit_transform(X_tr, y_tr)
            X_te = sel.transform(X_te)

        # 标准化
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        # SVR
        clf = SVR(kernel='rbf', C=1.0, gamma='scale')
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)

        if temporal_smooth and len(y_pred) > 1:
            from scipy.ndimage import uniform_filter1d
            y_pred = uniform_filter1d(y_pred, size=smooth_win)

        cor_list.append(cor(y_te, y_pred))
        rmse_list.append(rmse(y_te, y_pred))
        mae_list.append(mae(y_te, y_pred))
        yt_all.append(y_te); yp_all.append(y_pred)

    yt = np.concatenate(yt_all); yp = np.concatenate(yp_all)
    return {
        'cor_mean': float(np.mean(cor_list)),
        'cor_std': float(np.std(cor_list)),
        'rmse_mean': float(np.mean(rmse_list)),
        'rmse_std': float(np.std(rmse_list)),
        'mae_mean': float(np.mean(mae_list)),
        'global_cor': float(cor(yt, yp)),
        'global_rmse': float(rmse(yt, yp)),
    }


# ── 单被试实验 ────────────────────────────────────────────

def process_subject(subject_id, configs, cache=None):
    """处理一个被试的全部配置。cache 避免重复提取相同特征."""
    results = []
    feat_cache = {} if cache is None else cache

    for cfg in configs:
        t0 = time.time()
        try:
            parts = cfg['features']  # list of feature specs
            feature_blocks = []

            for p in parts:
                cache_key = f"{p['type']}_{p.get('channels','')}_{p.get('bands','')}_{p.get('ftype','')}"
                if cache_key not in feat_cache:
                    if p['type'] == 'de':
                        X_f, y, _ = extract_de_features(
                            subject_id, channels=p.get('channels', 'all'),
                            feature_type=p.get('ftype', 'de_LDS'))
                    elif p['type'] == 'psd':
                        X_f, y, _ = extract_de_features(
                            subject_id, channels=p.get('channels', 'all'),
                            feature_type='psd_LDS')
                    elif p['type'] == 'eog':
                        X_f, _ = extract_eog_features(subject_id)
                        _, y, _ = extract_de_features(subject_id)
                    elif p['type'] == 'riemann':
                        X_f, y = extract_riemann_features(
                            subject_id, channels=p.get('channels', 'all'),
                            bands=p.get('bands', '5band'))
                    else:
                        continue
                    feat_cache[cache_key] = (X_f, y)
                else:
                    X_f, y = feat_cache[cache_key]

                feature_blocks.append(X_f)

            # 拼接
            X = np.hstack(feature_blocks).astype(np.float64)
            r = evaluate_fusion_5fold(X, y, n_features=cfg.get('n_features', 150))

            results.append({
                'subject': subject_id, 'exp_name': cfg['name'],
                'cor_mean': r['cor_mean'], 'cor_std': r['cor_std'],
                'rmse_mean': r['rmse_mean'], 'rmse_std': r['rmse_std'],
                'global_cor': r['global_cor'], 'global_rmse': r['global_rmse'],
                'mae_mean': r['mae_mean'], 'n_features': X.shape[1],
                'time_sec': time.time() - t0,
            })
        except Exception as e:
            import traceback; traceback.print_exc()
            results.append({
                'subject': subject_id, 'exp_name': cfg['name'],
                'cor_mean': None, 'error': str(e),
            })
    return results


# ── 配置定义 ───────────────────────────────────────────────

def get_fusion_configs():
    """定义全部实验配置: 单模态 + 双模态 + 三模态."""
    cfgs = []

    # ── 单模态基线 ──────────────────────────────────────
    for ch in ['all', 'temporal']:
        for ft in ['de_LDS', 'psd_LDS']:
            tag = 'DE' if 'de' in ft else 'PSD'
            cfgs.append(dict(
                name=f'{tag}_{ch[:6]}',
                features=[dict(type='de', channels=ch, ftype=ft)],
                n_features=150))

    # Riemannian 单模态
    for ch in ['all', 'temporal', 'forehead']:
        for b in ['5band', '8band']:
            cfgs.append(dict(
                name=f'Riem_{b}_{ch[:6]}',
                features=[dict(type='riemann', channels=ch, bands=b)],
                n_features=150))

    # ── 双模态融合 ──────────────────────────────────────
    for ch in ['all', 'temporal']:
        for b in ['5band', '8band']:
            # DE + Riemannian
            cfgs.append(dict(
                name=f'DE+Riem_{b}_{ch[:6]}',
                features=[
                    dict(type='de', channels=ch, ftype='de_LDS'),
                    dict(type='riemann', channels=ch, bands=b),
                ], n_features=200))
            # Riemannian + EOG
            cfgs.append(dict(
                name=f'Riem+EOG_{b}_{ch[:6]}',
                features=[
                    dict(type='riemann', channels=ch, bands=b),
                    dict(type='eog'),
                ], n_features=200))

    # DE + EOG (不使用频带, 移出循环避免重复)
    for ch in ['all', 'temporal']:
        cfgs.append(dict(
            name=f'DE+EOG_{ch[:6]}',
            features=[
                dict(type='de', channels=ch, ftype='de_LDS'),
                dict(type='eog'),
            ], n_features=200))

    # ── 三模态融合 ──────────────────────────────────────
    for ch in ['all', 'temporal']:
        for b in ['5band', '8band']:
            cfgs.append(dict(
                name=f'DE+Riem+EOG_{b}_{ch[:6]}',
                features=[
                    dict(type='de', channels=ch, ftype='de_LDS'),
                    dict(type='riemann', channels=ch, bands=b),
                    dict(type='eog'),
                ], n_features=200))

    # ── 前额4ch 融合 (可穿戴场景) ────────────────────────
    for b in ['5band', '8band']:
        cfgs.append(dict(
            name=f'DE+Riem_{b}_4ch_fh',
            features=[
                dict(type='de', channels='forehead', ftype='de_LDS'),
                dict(type='riemann', channels='forehead', bands=b),
            ], n_features=100))
        cfgs.append(dict(
            name=f'DE+Riem+EOG_{b}_4ch_fh',
            features=[
                dict(type='de', channels='forehead', ftype='de_LDS'),
                dict(type='riemann', channels='forehead', bands=b),
                dict(type='eog'),
            ], n_features=100))

    return cfgs


# ── 主函数 ─────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--subjects', nargs='+', default=None)
    p.add_argument('--quick', action='store_true')
    p.add_argument('--n-jobs', type=int, default=-1)
    args = p.parse_args()

    os.makedirs(OUTPUT, exist_ok=True)
    subjects = args.subjects or list_subjects(SEED_VIG)
    if args.quick:
        subjects = subjects[:3]
    configs = get_fusion_configs()

    print(f"{'='*70}")
    print(f"Fusion Experiment: {len(subjects)} subjects x {len(configs)} configs")
    print(f"{'='*70}")

    t0 = time.time()

    if len(subjects) == 1:
        all_results = process_subject(subjects[0], configs)
    else:
        from joblib import Parallel, delayed
        tasks = [(s, configs) for s in subjects]
        nested = Parallel(n_jobs=args.n_jobs, verbose=10)(
            delayed(process_subject)(t[0], t[1]) for t in tasks)
        all_results = [r for subj_res in nested for r in subj_res]

    elapsed = time.time() - t0

    # 汇总
    agg = defaultdict(lambda: {'cor': [], 'rmse': [], 'gcor': [], 'time': []})
    for r in all_results:
        if r.get('cor_mean') is not None:
            agg[r['exp_name']]['cor'].append(r['cor_mean'])
            agg[r['exp_name']]['rmse'].append(r.get('rmse_mean', 0))
            agg[r['exp_name']]['gcor'].append(r.get('global_cor', 0))
            agg[r['exp_name']]['time'].append(r.get('time_sec', 0))

    print(f"\n{'='*70}")
    print(f"Fusion Results: {len(subjects)} subjects, {elapsed:.0f}s")
    print(f"{'='*70}")
    print(f"{'Experiment':<30s} {'N':>4s} {'COR':>8s} {'CORstd':>8s} "
          f"{'RMSE':>8s} {'G.COR':>8s} {'dim':>6s} {'Time':>8s}")
    print("-" * 90)

    # 排序: COR 降序
    sorted_names = sorted(agg.keys(), key=lambda n: -np.mean(agg[n]['cor']))
    for name in sorted_names:
        s = agg[name]
        n = len(s['cor'])
        if n == 0:
            continue
        dims = [r.get('n_features', 0) for r in all_results
                if r.get('exp_name') == name and r.get('n_features')]
        avg_dim = np.mean(dims) if dims else 0
        print(f"{name:<30s} {n:>4d} {np.mean(s['cor']):>8.4f} "
              f"{np.std(s['cor']):>8.4f} {np.mean(s['rmse']):>8.4f} "
              f"{np.mean(s['gcor']):>8.4f} {avg_dim:>5.0f} "
              f"{np.mean(s['time']):>7.1f}s")

    # 保存
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(OUTPUT, f'results_fusion_{ts}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()