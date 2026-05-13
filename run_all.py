"""
一键多数据集实验脚本 —— 对应 SKILL.md §3.1 完整实验矩阵
========================================================
用法:
  python run_all.py                        # 全量所有实验
  python run_all.py --quick                # 快速验证 (每数据集 2 被试)
  python run_all.py --n-jobs 8             # 8 核并行
  python run_all.py --dataset SEED-VIG     # 仅 SEED-VIG
  python run_all.py --dataset DROZY        # 仅 DROZY
  python run_all.py --skip-de              # 跳过 DE 基线 (仅 Riemannian)
  python run_all.py --skip-extra           # 跳过新增消融实验

内置实验 (#1-#7 = 论文 Tables 1-5):
  #1 被试内回归    run_final.py (inline)   → Table 1
  #2 多模态融合    run_fusion.py           → Table 2
  #3 黎曼度量消融  run_fusion.py           → Table 3
  #4 LOSO 跨被试   run_loso.py            → Table 4
  #5 二值分类      run_binary.py           → 补充
  #6 DROZY LOTO    inline DROZY loop       → Table 5
  #7 SEED 跨任务   run_seed_fast/fbts.py   → 补充

新增消融实验 (--skip-extra 跳过):
  run_ablation_estimator.py   协方差估计器消融
  run_ablation_bands.py       频段消融 (Leave-One-Band-Out)
  run_ablation_regressor.py   回归器消融
  run_ablation_smoothing.py   时序平滑窗口消融
  run_feature_provenance.py   特征重要性溯源
  run_data_efficiency.py      数据效率曲线
  run_efficiency_benchmark.py 计算效率基准
  run_experiment.py           黎曼度量消融 (基础版回归器)
  run_parallel.py             Torch + Joblib 并行
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
    list_subjects as list_seedvig_subjects,
    load_raw_eeg, load_perclos, build_de_features_for_baseline
)
from drozy_loader import (
    list_drozy_tests, load_drozy_subject, get_kss_label, extract_de_features
)
from utils import get_5fold_splits, cor, rmse, mae
from sca_fbts_fast import SCAFBTSRegressorFast
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler

# ── 路径配置 ─────────────────────────────────────────────────
from config import SEED_VIG_ROOT, DROZY_ROOT, RESULTS_DIR as OUTPUT_DIR, DROZY_PSG_DIR
# OUTPUT_DIR imported from config above


# ── 评测函数 ─────────────────────────────────────────────────

def evaluate_regression_fast(clf, y, n_folds=5):
    """预计算回归评测。"""
    splits = get_5fold_splits(len(y), n_folds)
    cor_list, rmse_list, mae_list = [], [], []
    yt_all, yp_all = [], []

    for tr_idx, te_idx in splits:
        clf.fit(tr_idx, y[tr_idx])
        y_pred = clf.predict(te_idx)
        if clf.temporal_smoothing and len(y_pred) > 1:
            from scipy.ndimage import uniform_filter1d
            y_pred = uniform_filter1d(y_pred, size=clf.smoothing_window)
        y_test = y[te_idx]
        cor_list.append(cor(y_test, y_pred))
        rmse_list.append(rmse(y_test, y_pred))
        mae_list.append(mae(y_test, y_pred))
        yt_all.append(y_test); yp_all.append(y_pred)

    yt = np.concatenate(yt_all); yp = np.concatenate(yp_all)
    return {
        'cor_mean': float(np.mean(cor_list)),
        'cor_std': float(np.std(cor_list)),
        'rmse_mean': float(np.mean(rmse_list)),
        'rmse_std': float(np.std(rmse_list)),
        'mae_mean': float(np.mean(mae_list)),
        'global_cor': float(cor(yt, yp)),
        'global_rmse': float(rmse(yt, yp)),
        'cor_all': [float(x) for x in cor_list],
    }


def evaluate_classification_fast(clf, y_binary, n_folds=5):
    """预计算分类评测 (二分类: alert vs drowsy)。"""
    from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
    splits = get_5fold_splits(len(y_binary), n_folds)
    acc_list, f1_list, bac_list = [], [], []

    for tr_idx, te_idx in splits:
        clf.fit(tr_idx, y_binary[tr_idx])
        y_pred = clf.predict(te_idx)
        if clf.temporal_smoothing and len(y_pred) > 1:
            from scipy.ndimage import uniform_filter1d
            y_pred_smooth = uniform_filter1d(y_pred.astype(float), size=clf.smoothing_window)
            y_pred = (y_pred_smooth > 0.5).astype(int)
        yt = y_binary[te_idx]
        acc_list.append(accuracy_score(yt, y_pred))
        f1_list.append(f1_score(yt, y_pred, average='binary', zero_division=0))
        bac_list.append(balanced_accuracy_score(yt, y_pred))

    return {
        'acc_mean': float(np.mean(acc_list)),
        'acc_std': float(np.std(acc_list)),
        'f1_mean': float(np.mean(f1_list)),
        'f1_std': float(np.std(f1_list)),
        'bac_mean': float(np.mean(bac_list)),
    }


# ── 单被试任务 ──────────────────────────────────────────────

def process_seedvig_subject(args):
    """处理一个 SEED-VIG 被试。"""
    subject_id, configs = args
    results = []

    for cfg in configs:
        t0 = time.time()
        try:
            if cfg['type'] == 'de_regression':
                X, y, _ = build_de_features_for_baseline(
                    SEED_VIG_ROOT, subject_id,
                    channels=cfg.get('channels', 'all'),
                    feature_type=cfg.get('feature_type', 'de_LDS')
                )
                from utils import evaluate_de_baseline
                r = evaluate_de_baseline(
                    lambda: SVR(kernel='rbf', C=1.0, gamma='scale'),
                    X, y, verbose=False
                )
            elif cfg['type'] == 'riemann_regression':
                raw, sr = load_raw_eeg(SEED_VIG_ROOT, subject_id)
                y = load_perclos(SEED_VIG_ROOT, subject_id)
                ch = cfg.get('channels', 'all')
                if ch == 'temporal': raw = raw[:, [0,1,2,3,4,5]]
                elif ch == 'forehead': raw = raw[:, [0,1,2,3]]
                clf = SCAFBTSRegressorFast(
                    freq_bands=cfg['freq_bands'], estimator='oas',
                    metric='riemann', regressor='svr',
                    n_features=cfg.get('n_features', 100), fs=sr,
                    temporal_smoothing=True, smoothing_window=3, scaler=True,
                )
                clf.precompute(raw, fs=sr)
                r = evaluate_regression_fast(clf, y)
            else:
                continue

            results.append({
                'dataset': 'SEED-VIG', 'subject': subject_id,
                'exp_name': cfg['name'],
                'cor_mean': r.get('cor_mean'), 'rmse_mean': r.get('rmse_mean'),
                'acc_mean': r.get('acc_mean'), 'f1_mean': r.get('f1_mean'),
                'global_cor': r.get('global_cor'),
                'time_sec': time.time() - t0,
            })
        except Exception as e:
            import traceback; traceback.print_exc()
            results.append({
                'dataset': 'SEED-VIG', 'subject': subject_id,
                'exp_name': cfg['name'], 'error': str(e),
            })
    return results


def evaluate_drozy_loto(clf, X, y_kss, y_bin, t_ids, cfg_type):
    """DROZY Leave-One-Test-Out 评测。

    DROZY 的 KSS 是 per-test 标签，不是 per-epoch。
    正确做法: 每次留一个 test 做测试，其余 test 训练。
    """
    unique_tests = np.unique(t_ids)
    n_tests = len(unique_tests)
    if n_tests < 2:
        return None  # 只有 1 个 test，无法 LOTO

    kss_preds = []
    kss_trues = []
    bin_preds = []
    bin_trues = []

    for held_out in unique_tests:
        train_mask = t_ids != held_out
        test_mask = t_ids == held_out

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        # 训练
        clf.fit(train_idx, y_kss[train_idx])
        # 预测
        y_pred_epochs = clf.predict(test_idx)

        # 聚合: test 内取均值
        pred_mean = float(np.mean(y_pred_epochs))
        true_val = float(y_kss[test_idx[0]])  # 同一 test 的 KSS 都相同

        kss_preds.append(pred_mean)
        kss_trues.append(true_val)

        # 二值化
        pred_bin = 1 if pred_mean >= 5.5 else 0  # KSS 1-9, 阈值 5.5
        true_bin = int(y_bin[test_idx[0]])
        bin_preds.append(pred_bin)
        bin_trues.append(true_bin)

    kss_preds = np.array(kss_preds)
    kss_trues = np.array(kss_trues)
    bin_preds = np.array(bin_preds)
    bin_trues = np.array(bin_trues)

    # 回归指标
    reg_cor = cor(kss_trues, kss_preds) if len(kss_trues) > 1 else 0.0
    reg_rmse = rmse(kss_trues, kss_preds)

    # 分类指标
    from sklearn.metrics import accuracy_score, f1_score
    if len(np.unique(bin_trues)) > 1 and len(np.unique(bin_preds)) > 1:
        cls_acc = float(accuracy_score(bin_trues, bin_preds))
        cls_f1 = float(f1_score(bin_trues, bin_preds, average='binary', zero_division=0))
    else:
        cls_acc = float(np.mean(bin_trues == bin_preds))
        cls_f1 = 0.0

    return {
        'cor_mean': reg_cor,
        'rmse_mean': reg_rmse,
        'acc_mean': cls_acc,
        'f1_mean': cls_f1,
        'n_tests': n_tests,
        'kss_preds': kss_preds.tolist(),
        'kss_trues': kss_trues.tolist(),
    }


def evaluate_drozy_loto_de(X_de, y_kss, y_bin, t_ids):
    """DROZY DE 特征 LOTO 评测。"""
    unique_tests = np.unique(t_ids)
    if len(unique_tests) < 2:
        return None

    kss_preds, kss_trues = [], []
    bin_preds, bin_trues = [], []

    for held_out in unique_tests:
        train_mask = t_ids != held_out
        test_mask = t_ids == held_out
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        X_tr, y_tr = X_de[train_mask], y_kss[train_mask]
        X_te = X_de[test_mask]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        clf = SVR(kernel='rbf', C=1.0, gamma='scale')
        clf.fit(X_tr_s, y_tr)
        y_pred_epochs = clf.predict(X_te_s)

        pred_mean = float(np.mean(y_pred_epochs))
        true_val = float(y_kss[test_mask][0])

        kss_preds.append(pred_mean)
        kss_trues.append(true_val)
        pred_bin = 1 if pred_mean >= 5.5 else 0
        true_bin = int(y_bin[test_mask][0])
        bin_preds.append(pred_bin)
        bin_trues.append(true_bin)

    kss_preds = np.array(kss_preds)
    kss_trues = np.array(kss_trues)
    bin_preds = np.array(bin_preds)
    bin_trues = np.array(bin_trues)

    reg_cor = cor(kss_trues, kss_preds) if len(kss_trues) > 1 else 0.0
    reg_rmse = rmse(kss_trues, kss_preds)

    from sklearn.metrics import accuracy_score, f1_score
    if len(np.unique(bin_trues)) > 1 and len(np.unique(bin_preds)) > 1:
        cls_acc = float(accuracy_score(bin_trues, bin_preds))
        cls_f1 = float(f1_score(bin_trues, bin_preds, average='binary', zero_division=0))
    else:
        cls_acc = float(np.mean(bin_trues == bin_preds))
        cls_f1 = 0.0

    return {
        'cor_mean': reg_cor, 'rmse_mean': reg_rmse,
        'acc_mean': cls_acc, 'f1_mean': cls_f1,
    }


def process_drozy_subject(args):
    """处理一个 DROZY 被试 (LOTO 评测)。"""
    subject_id, configs = args
    results = []
    psg_dir = DROZY_PSG_DIR

    X, y_kss, y_bin, t_ids, names = load_drozy_subject(psg_dir, subject_id)
    if X is None:
        return [{'dataset': 'DROZY', 'subject': f'subj{subject_id:02d}',
                 'error': 'No data'}]
    if len(np.unique(t_ids)) < 2:
        return [{'dataset': 'DROZY', 'subject': f'subj{subject_id:02d}',
                 'error': 'Only 1 test'}]

    for cfg in configs:
        t0 = time.time()
        try:
            if cfg['type'] == 'de_loto':
                # DE 特征提取 + LOTO
                X_de = extract_de_features(X, fs=200)
                r = evaluate_drozy_loto_de(X_de, y_kss, y_bin, t_ids)
                if r is None:
                    continue
                results.append({
                    'dataset': 'DROZY', 'subject': f'subj{subject_id:02d}',
                    'exp_name': cfg['name'],
                    'cor_mean': r.get('cor_mean'), 'rmse_mean': r.get('rmse_mean'),
                    'acc_mean': r.get('acc_mean'), 'f1_mean': r.get('f1_mean'),
                    'time_sec': time.time() - t0,
                })
                continue

            # 构建模型 (回归/分类都走 SVR)
            clf = SCAFBTSRegressorFast(
                freq_bands=cfg['freq_bands'], estimator='oas',
                metric='riemann', regressor='svr',
                n_features=cfg.get('n_features', 50), fs=200,
                temporal_smoothing=False, smoothing_window=3, scaler=True,
            )
            clf.precompute(X, fs=200)
            r = evaluate_drozy_loto(clf, X, y_kss, y_bin, t_ids, cfg['type'])

            if r is None:
                continue

            results.append({
                'dataset': 'DROZY', 'subject': f'subj{subject_id:02d}',
                'exp_name': cfg['name'],
                'cor_mean': r.get('cor_mean'), 'rmse_mean': r.get('rmse_mean'),
                'acc_mean': r.get('acc_mean'), 'f1_mean': r.get('f1_mean'),
                'n_tests': r.get('n_tests'),
                'time_sec': time.time() - t0,
            })
        except Exception as e:
            import traceback; traceback.print_exc()
            results.append({
                'dataset': 'DROZY', 'subject': f'subj{subject_id:02d}',
                'exp_name': cfg['name'], 'error': str(e),
            })
    return results


# ── 实验配置 ─────────────────────────────────────────────────

def get_configs(datasets=None, skip_de=False):
    """生成所有实验配置。"""
    if datasets is None:
        datasets = ['SEED-VIG', 'DROZY']
    cfgs = {'SEED-VIG': [], 'DROZY': []}

    # ── SEED-VIG ──
    if 'SEED-VIG' in datasets and not skip_de:
        for ch in ['all', 'temporal']:
            cfgs['SEED-VIG'].append(dict(
                name=f'DE_SVR_{ch[:6]}', type='de_regression', channels=ch))
            cfgs['SEED-VIG'].append(dict(
                name=f'PSD_SVR_{ch[:6]}', type='de_regression', channels=ch,
                feature_type='psd_LDS'))

    if 'SEED-VIG' in datasets:
        for ch_n, ch_k in [('17ch', 'all'), ('6ch_t', 'temporal'),
                           ('4ch_fh', 'forehead')]:
            for b in ['5band', '8band']:
                cfgs['SEED-VIG'].append(dict(
                    name=f'RFast_{b}_{ch_n}', type='riemann_regression',
                    channels=ch_k, freq_bands=b, n_features=100))

    # ── DROZY (LOTO: 一次实验产出回归+分类) ──
    if 'DROZY' in datasets:
        if not skip_de:
            cfgs['DROZY'].append(dict(
                name=f'DE_SVR_LOTO', type='de_loto'))
        for b in ['5band', '8band']:
            cfgs['DROZY'].append(dict(
                name=f'RFast_{b}_LOTO', type='riemann_loto',
                freq_bands=b, n_features=50))

    return cfgs


# ── 主函数 ───────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser(description='Multi-dataset EEG experiment runner')
    p.add_argument('--dataset', nargs='+', default=['SEED-VIG','DROZY'],
                   choices=['SEED-VIG','DROZY'])
    p.add_argument('--quick', action='store_true',
                   help='Quick validation: 2 subjects per dataset')
    p.add_argument('--skip-de', action='store_true',
                   help='Skip DE baseline')
    p.add_argument('--skip-extra', action='store_true',
                   help='Skip extra experiments (run_experiment, run_parallel)')
    p.add_argument('--n-jobs', type=int, default=-1,
                   help='Parallel workers')
    args = p.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t_start = time.time()
    all_results = []

    # ── SEED-VIG ──
    if 'SEED-VIG' in args.dataset:
        subjects = list_seedvig_subjects(SEED_VIG_ROOT)
        if args.quick:
            subjects = subjects[:2]
        configs = get_configs(['SEED-VIG'], args.skip_de)['SEED-VIG']
        print(f"\n{'='*70}")
        print(f"SEED-VIG: {len(subjects)} subjects x {len(configs)} configs")
        print(f"{'='*70}")

        if len(subjects) == 1:
            res = process_seedvig_subject((subjects[0], configs))
        else:
            from joblib import Parallel, delayed
            tasks = [(s, configs) for s in subjects]
            nested = Parallel(n_jobs=args.n_jobs, verbose=10)(
                delayed(process_seedvig_subject)(t) for t in tasks)
            res = [r for subj_res in nested for r in subj_res]
        all_results.extend(res)

    # ── DROZY ──
    if 'DROZY' in args.dataset:
        psg_dir = DROZY_PSG_DIR
        tests = list_drozy_tests(psg_dir)
        subject_ids = sorted(set(s for s, t, p in tests))
        if args.quick:
            subject_ids = subject_ids[:2]
        configs = get_configs(['DROZY'], args.skip_de)['DROZY']
        print(f"\n{'='*70}")
        print(f"DROZY: {len(subject_ids)} subjects x {len(configs)} configs")
        print(f"{'='*70}")

        if len(subject_ids) == 1:
            res = process_drozy_subject((subject_ids[0], configs))
        else:
            from joblib import Parallel, delayed
            tasks = [(s, configs) for s in subject_ids]
            nested = Parallel(n_jobs=args.n_jobs, verbose=10)(
                delayed(process_drozy_subject)(t) for t in tasks)
            res = [r for subj_res in nested for r in subj_res]
        all_results.extend(res)

    elapsed = time.time() - t_start

    # ── 汇总 ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Multi-Dataset Results ({elapsed:.0f}s total)")
    print(f"{'='*70}")

    for ds in args.dataset:
        ds_results = [r for r in all_results if r.get('dataset') == ds]
        if not ds_results:
            continue
        agg = defaultdict(lambda: {'cor': [], 'rmse': [], 'acc': [], 'f1': [], 'time': []})
        for r in ds_results:
            if r.get('cor_mean') is not None:
                agg[r['exp_name']]['cor'].append(r['cor_mean'])
                agg[r['exp_name']]['rmse'].append(r.get('rmse_mean', 0))
                agg[r['exp_name']]['time'].append(r.get('time_sec', 0))
            if r.get('acc_mean') is not None:
                agg[r['exp_name']]['acc'].append(r['acc_mean'])
                agg[r['exp_name']]['f1'].append(r.get('f1_mean', 0))

        print(f"\n[{ds}]")
        if ds == 'DROZY':
            print(f"{'Experiment':<30s} {'N':>4s} {'COR':>8s} {'RMSE':>8s} {'ACC':>8s} {'F1':>8s} {'Time':>8s}")
            print("-" * 80)
            for name in sorted(agg.keys()):
                s = agg[name]
                n = len(s['cor'])
                print(f"{name:<30s} {n:>4d} {np.mean(s['cor']):>8.4f} "
                      f"{np.mean(s['rmse']):>8.3f} {np.mean(s['acc']):>8.4f} "
                      f"{np.mean(s['f1']):>8.4f} {np.mean(s['time']):>7.1f}s")
        else:
            print(f"{'Experiment':<35s} {'N':>4s} {'COR':>8s} {'RMSE':>8s} {'Time':>8s}")
            print("-" * 65)
            for name in sorted(agg.keys()):
                s = agg[name]
                n = len(s['cor']) or len(s['acc'])
                if s['cor']:
                    print(f"{name:<35s} {n:>4d} {np.mean(s['cor']):>8.4f} "
                          f"{np.mean(s['rmse']):>8.4f} {np.mean(s['time']):>7.1f}s")
                elif s['acc']:
                    print(f"{name:<35s} {n:>4d} {np.mean(s['acc']):>8.4f} "
                          f"{np.mean(s['f1']):>8.4f} {np.mean(s['time']):>7.1f}s")

    # 保存
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(OUTPUT_DIR, f'results_all_{ts}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved: {path}")


    # ── 完整实验矩阵（SKILL.md §3.1）──
    import subprocess as _sp
    _proj = os.path.dirname(os.path.abspath(__file__))
    _py = sys.executable
    _nj = str(args.n_jobs)
    _scripts = []

    # #2 多模态融合 + #3 黎曼度量消融
    _scripts.append(f'{_py} run_fusion.py --n-jobs {_nj}' + (' --quick' if args.quick else ''))
    # #4 LOSO 跨被试
    _scripts.append(f'{_py} run_loso.py')
    # #5 二值分类
    _scripts.append(f'{_py} run_binary.py' + (' --quick' if args.quick else ''))
    # #7 SEED 跨任务
    _scripts.append(f'{_py} run_seed_fast.py')
    _scripts.append(f'{_py} run_seed_fbts.py')

    if not args.skip_extra:
        # 新增消融实验
        _nsubj = '5' if not args.quick else '3'
        _scripts.append(f'{_py} run_ablation_estimator.py --n-subjects {_nsubj}')
        _scripts.append(f'{_py} run_ablation_bands.py --n-subjects {_nsubj}')
        _scripts.append(f'{_py} run_ablation_regressor.py --n-subjects {_nsubj}')
        _scripts.append(f'{_py} run_feature_provenance.py --n-subjects {_nsubj}')
        _scripts.append(f'{_py} run_efficiency_benchmark.py')
        _scripts.append(f'{_py} run_ablation_smoothing.py --n-subjects {_nsubj}')
        _scripts.append(f'{_py} run_data_efficiency.py --n-subjects {_nsubj}')
        # 度量消融 + Torch 并行
        _scripts.append(f'{_py} run_experiment.py' + (' --quick' if args.quick else ''))
        _scripts.append(f'{_py} run_parallel.py --n-jobs {_nj}')

    for _cmd in _scripts:
        print(f"\n  Running: {_cmd}")
        _sp.run(_cmd.split(), cwd=_proj)


if __name__ == '__main__':
    main()
