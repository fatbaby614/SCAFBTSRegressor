"""
DROZY test-level 聚合复现 + 标签/epoch 映射审计 (JNE 修回 Experiment B)
=======================================================================
背景: 现稿 Table 5 报告 DROZY LOTO 的强负相关 (COR -0.89 ~ -0.94), 并归因于
test-level KSS 标签稀疏造成的统计伪象。该解释尚未验证。本脚本按修回计划 §4
分两阶段执行, 结果出来之前不预设结论。

阶段 1 — 标签与 test/epoch 映射审计
  逐被试逐 test 输出审计表:
    subject / test_id / KSS / 二值标签 / epoch 数 / 有效性与排除原因
  一致性检查:
    - epoch 是否按 test 分块且 test 顺序单调 (无交错)
    - 同一 test 内 KSS 是否恒定
    - test -> KSS 序列 (核对 KSS 方向: 高 = 更困倦)

阶段 2 — test-level 聚合主分析 (特征单位与标签单位对齐)
  P1  LOSO_test_level : 先聚合 test 内 epoch 特征; 训练集 = 其他被试的全部 test
  P2  LOTO_test_level : 先聚合 test 内 epoch 特征; 训练集 = 同被试其余 test
  P3  LOSO_epoch_avg  : epoch 级模型, test 内取预测均值; 训练集 = 其他被试
  P4  LOTO_epoch_avg  : epoch 级模型, test 内取预测均值; 训练集 = 同被试其余 test

  特征集 : FBTS (切空间 75d) / DE (25d) / FBTS+DE (100d)
  聚合   : mean / median / mean+std (test_level 协议)
  回归器 : Ridge(alpha=1.0) 与 SVR(RBF), 与 LOSO 匹配协议一致
  切空间参考均值: 仅用训练数据拟合 (无泄漏)

统计单位: 被试 (或独立 test)。不做"已证实统计伪象"的表述, 只报数值与不确定度。

用法:
  python run_drozy_testlevel.py            # 全量 14 被试
  python run_drozy_testlevel.py --quick    # 前 4 被试冒烟
"""

import sys
import os
import json
import time
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drozy_loader import (
    list_drozy_tests, load_drozy_subject, get_kss_label, extract_de_features,
)
from utils import cor, rmse
from sca_fbts_fast import SCAFBTSRegressorFast
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace

from config import DROZY_PSG_DIR, RESULTS_DIR


N_CH = 5
FBTS_DIM = 5 * (N_CH * (N_CH + 1) // 2)   # 5 bands × 15 = 75
DE_DIM = N_CH * 5                          # 25


# ── 阶段 1: 标签 / epoch 映射审计 ────────────────────────────

def audit_labels(psg_dir):
    """建立 subject/test/KSS/epoch 审计表。"""
    from drozy_loader import load_edf_eeg

    tests = list_drozy_tests(psg_dir)
    by_subject = {}
    for sid, tid, path in tests:
        by_subject.setdefault(sid, []).append((tid, path))

    audit_rows = []
    for sid in sorted(by_subject):
        for tid, path in sorted(by_subject[sid]):
            kss, binary = get_kss_label(sid, tid)
            row = {
                'subject': f'subj{sid:02d}', 'subject_id': sid, 'test_id': tid,
                'kss': int(kss), 'kss_binary': int(binary),
                'n_epochs': None, 'valid': False, 'exclusion_reason': None,
            }
            if kss == 0:
                row['exclusion_reason'] = 'KSS missing (0) in authoritative annotation'
                audit_rows.append(row)
                continue
            try:
                epochs, _, names = load_edf_eeg(path, target_fs=200, epoch_sec=8)
            except Exception as e:
                row['exclusion_reason'] = f'EDF load failed: {e}'
                audit_rows.append(row)
                continue
            row['n_epochs'] = int(len(epochs))
            row['channels'] = list(names)
            row['valid'] = True
            audit_rows.append(row)
    return audit_rows


def audit_consistency(psg_dir):
    """对载入的被试做 epoch 顺序 / KSS 一致性检查。"""
    checks = []
    for sid in range(1, 15):
        X, y_kss, y_bin, t_ids, names = load_drozy_subject(psg_dir, sid)
        if X is None:
            continue
        # epoch 按 test 分块: 连续段序列展开后应与 t_ids 逐元素相等
        blocks = []
        block_seq = []
        cur = t_ids[0]
        for t in t_ids:
            if t != cur:
                blocks.append(cur)
                cur = t
            block_seq.append(cur)
        blocks.append(cur)
        grouped = bool(np.array_equal(np.array(block_seq), t_ids))
        monotone = list(blocks) == sorted(blocks)
        kss_ok = all(len(np.unique(y_kss[t_ids == t])) == 1
                     for t in np.unique(t_ids))
        seq = [(int(t), int(y_kss[t_ids == t][0])) for t in np.unique(t_ids)]
        checks.append({
            'subject': f'subj{sid:02d}',
            'n_epochs': int(len(y_kss)),
            'n_tests': int(len(np.unique(t_ids))),
            'epochs_grouped_by_test': bool(grouped),
            'test_order_monotone': bool(monotone),
            'kss_constant_within_test': bool(kss_ok),
            'test_kss_sequence': seq,
        })
    return checks


# ── 阶段 2: 特征准备与切空间 ─────────────────────────────────

def _regularize_covs(covs, eps=1e-6):
    """迹归一化 + 绝对正则下界, 保证黎曼几何数值安全。

    C <- C / tr(C) 消除跨 epoch/被试的尺度差异 (黎曼 BCI 标准做法,
    TangentSpace 对整体缩放不变); 再加 eps*I 使 min_eig >= eps,
    条件数有界。仅做相对正则 (eps*tr(C)*I) 时, trace~1e-11 的退化
    被试 (如 subj04) 会在黎曼均值定点迭代中特征值下溢为 0,
    isqrt 除零 -> Cm12 含 inf/NaN -> logm 抛
    'Matrices must be positive definite'。退化矩阵 (非有限/trace<=0)
    用中位 trace 单位阵替代 (归一化后即单位阵)。
    """
    covs = np.asarray(covs, dtype=np.float64)
    ch = covs.shape[-1]
    tr = np.trace(covs, axis1=-2, axis2=-1) / ch            # (N,)
    scale = float(np.median(tr[np.isfinite(tr) & (tr > 0)])) if np.any(tr > 0) else 1.0
    eye = np.eye(ch)
    bad = ~np.isfinite(covs).all(axis=(1, 2)) | ~np.isfinite(tr) | (tr <= 0)
    out = covs.copy()
    out[bad] = scale * eye
    tr = np.where(bad, scale, tr)
    out = out / tr[:, None, None]                           # 迹归一化: 总迹 = ch
    return (1.0 - eps) * out + eps * eye


def prepare_subject(psg_dir, sid):
    """载入单被试并提取 DE 与各频段协方差。"""
    X, y_kss, y_bin, t_ids, names = load_drozy_subject(psg_dir, sid)
    if X is None:
        return None
    de = extract_de_features(X, fs=200)

    clf = SCAFBTSRegressorFast(
        freq_bands='5band', estimator='oas', metric='riemann',
        regressor='svr', n_features=None, fs=200,
        temporal_smoothing=False, scaler=False,
    )
    clf.precompute(X, fs=200)
    covs = {b: _regularize_covs(clf._epochs_cov[b]) for b in clf.freq_bands}

    return {'de': de, 'covs': covs, 'y_kss': y_kss.astype(np.float64),
            'y_bin': y_bin, 't_ids': t_ids, 'names': names}


def fit_ts_transform(fit_pool, targets):
    """按频段拟合切空间参考均值并变换目标。

    Args:
        fit_pool: {band: (N_fit, ch, ch)} — 仅训练数据的协方差
        targets: list of (key, {band: (n, ch, ch)})

    Returns:
        dict: key -> (n, FBTS_DIM)
    """
    out = {}
    for key, covs in targets:
        feats = []
        for band in fit_pool:
            ts = TangentSpace(metric='riemann')
            ts.fit(fit_pool[band])
            feats.append(ts.transform(covs[band]))
        out[key] = np.hstack(feats)
    return out


def aggregate_by_test(X_epoch, t_ids, mode):
    """把 epoch 级特征聚合为 test 级特征。行序与 np.unique(t_ids) 一致。"""
    rows = []
    for t in np.unique(t_ids):
        seg = X_epoch[t_ids == t]
        if mode == 'mean':
            rows.append(seg.mean(axis=0))
        elif mode == 'median':
            rows.append(np.median(seg, axis=0))
        elif mode == 'mean_std':
            rows.append(np.hstack([seg.mean(axis=0), seg.std(axis=0)]))
        else:
            raise ValueError(mode)
    return np.vstack(rows)


def test_level_labels(y_kss, t_ids):
    """每个 test 的 KSS (同一 test 内恒定)。"""
    return np.array([y_kss[t_ids == t][0] for t in np.unique(t_ids)])


def _fit_predict(X_tr, y_tr, X_te, regressor='ridge'):
    """标准化在训练集内拟合。"""
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_tr)
    Xte = scaler.transform(X_te)
    if regressor == 'ridge':
        model = Ridge(alpha=1.0)
    elif regressor == 'svr':
        model = SVR(kernel='rbf', C=1.0, gamma='scale')
    else:
        raise ValueError(regressor)
    model.fit(Xtr, y_tr)
    return model.predict(Xte)


FEATURE_SETS = ('FBTS', 'DE', 'FBTS+DE')


def _assemble(feat_name, fbts, de):
    if feat_name == 'FBTS':
        return fbts
    if feat_name == 'DE':
        return de
    return np.hstack([fbts, de])


# ── P1/P2: test-level 聚合协议 ───────────────────────────────

def protocol_test_level(subjects, mode='loso', agg='mean', regressor='ridge',
                        feat_name='FBTS'):
    """mode='loso': 训练集 = 其他被试全部 test;  mode='loto': 训练集 = 同被试其余 test。"""
    sids = sorted(subjects.keys())
    preds, trues, owner = [], [], []

    for held in sids:
        held_data = subjects[held]

        if mode == 'loso':
            pool = {band: np.concatenate(
                [subjects[s]['covs'][band] for s in sids if s != held
                 and subjects[s] is not None])
                for band in held_data['covs']}
            targets = [(s, subjects[s]['covs']) for s in sids if s != held]
            targets.append((held, held_data['covs']))
            ts_out = fit_ts_transform(pool, targets)

            X_tr_blocks, y_tr_blocks = [], []
            for s in sids:
                if s == held:
                    continue
                X_tr_blocks.append(aggregate_by_test(ts_out[s], subjects[s]['t_ids'], agg))
                y_tr_blocks.append(test_level_labels(subjects[s]['y_kss'], subjects[s]['t_ids']))
            Xtr_fb = np.vstack(X_tr_blocks)
            ytr = np.concatenate(y_tr_blocks)
            Xte_fb = aggregate_by_test(ts_out[held], held_data['t_ids'], agg)
            yte = test_level_labels(held_data['y_kss'], held_data['t_ids'])
            Xtr_de = np.vstack([aggregate_by_test(subjects[s]['de'], subjects[s]['t_ids'], agg)
                                for s in sids if s != held])
            Xte_de = aggregate_by_test(held_data['de'], held_data['t_ids'], agg)

            y_pred = _fit_predict(
                _assemble(feat_name, Xtr_fb, Xtr_de), ytr,
                _assemble(feat_name, Xte_fb, Xte_de), regressor)
            for p, t in zip(y_pred, yte):
                preds.append(float(p)); trues.append(float(t))
                owner.append(f'subj{held:02d}')

        else:  # loto
            n_tests = len(np.unique(held_data['t_ids']))
            if n_tests < 2:
                # LOTO 需要 >=2 个 test: 仅 1 个 test 时训练池为空,
                # TangentSpace.fit 空输入会产生无效参考均值
                print(f'  [LOTO] skip subj{held:02d}: only {n_tests} valid test(s)')
                continue
            for held_test in np.unique(held_data['t_ids']):
                pool = {band: held_data['covs'][band][held_data['t_ids'] != held_test]
                        for band in held_data['covs']}
                ts_out = fit_ts_transform(pool, [(held, held_data['covs'])])
                fb_all = ts_out[held]

                tr_mask = held_data['t_ids'] != held_test
                Xtr_fb = aggregate_by_test(fb_all[tr_mask], held_data['t_ids'][tr_mask], agg)
                ytr = test_level_labels(held_data['y_kss'][tr_mask], held_data['t_ids'][tr_mask])
                Xte_fb = aggregate_by_test(fb_all[~tr_mask], held_data['t_ids'][~tr_mask], agg)
                yte = test_level_labels(held_data['y_kss'][~tr_mask], held_data['t_ids'][~tr_mask])
                Xtr_de = aggregate_by_test(held_data['de'][tr_mask],
                                           held_data['t_ids'][tr_mask], agg)
                Xte_de = aggregate_by_test(held_data['de'][~tr_mask],
                                           held_data['t_ids'][~tr_mask], agg)

                y_pred = _fit_predict(
                    _assemble(feat_name, Xtr_fb, Xtr_de), ytr,
                    _assemble(feat_name, Xte_fb, Xte_de), regressor)
                for p, t in zip(y_pred, yte):
                    preds.append(float(p)); trues.append(float(t))
                    owner.append(f'subj{held:02d}')

    return _summarize(preds, trues, owner,
                      f'{mode}_test_level', agg, regressor, feat_name)


# ── P3/P4: epoch 级模型 + test 内预测均值 ────────────────────

def protocol_epoch_avg(subjects, mode='loso', regressor='ridge'):
    """epoch 级模型, test 内取预测均值。"""
    sids = sorted(subjects.keys())
    store = {f: [] for f in FEATURE_SETS}

    for held in sids:
        held_data = subjects[held]
        n_ep = len(held_data['y_kss'])

        if mode == 'loso':
            pool = {band: np.concatenate(
                [subjects[s]['covs'][band] for s in sids if s != held
                 and subjects[s] is not None])
                for band in held_data['covs']}
            targets = [(s, subjects[s]['covs']) for s in sids if s != held]
            targets.append((held, held_data['covs']))
            ts_out = fit_ts_transform(pool, targets)

            tr_fb = np.vstack([ts_out[s] for s in sids if s != held])
            tr_de = np.vstack([subjects[s]['de'] for s in sids if s != held])
            tr_y = np.concatenate([subjects[s]['y_kss'] for s in sids if s != held])
            te_fb = ts_out[held]
            te_de = held_data['de']
            te_y = held_data['y_kss']
            te_t = held_data['t_ids']
        else:
            ut = np.unique(held_data['t_ids'])
            if len(ut) < 2:
                print(f'  [LOTO epoch_avg] skip subj{held:02d}: only {len(ut)} valid test(s)')
                continue
            for held_test in ut:
                tr_mask = held_data['t_ids'] != held_test
                pool = {band: held_data['covs'][band][tr_mask]
                        for band in held_data['covs']}
                ts_out = fit_ts_transform(pool, [(held, held_data['covs'])])
                tr_fb = ts_out[held][tr_mask]
                tr_de = held_data['de'][tr_mask]
                tr_y = held_data['y_kss'][tr_mask]
                te_fb = ts_out[held][~tr_mask]
                te_de = held_data['de'][~tr_mask]
                te_y = held_data['y_kss'][~tr_mask]
                te_t = held_data['t_ids'][~tr_mask]

                for feat_name in FEATURE_SETS:
                    y_pred_ep = _fit_predict(
                        _assemble(feat_name, tr_fb, tr_de), tr_y,
                        _assemble(feat_name, te_fb, te_de), regressor)
                    uts = np.unique(te_t)
                    p_test = np.array([y_pred_ep[te_t == t].mean() for t in uts])
                    t_test = np.array([te_y[te_t == t][0] for t in uts])
                    store[feat_name].append((f'subj{held:02d}', p_test, t_test))
            continue

        for feat_name in FEATURE_SETS:
            y_pred_ep = _fit_predict(
                _assemble(feat_name, tr_fb, tr_de), tr_y,
                _assemble(feat_name, te_fb, te_de), regressor)
            uts = np.unique(te_t)
            p_test = np.array([y_pred_ep[te_t == t].mean() for t in uts])
            t_test = np.array([te_y[te_t == t][0] for t in uts])
            store[feat_name].append((f'subj{held:02d}', p_test, t_test))

    out = {}
    for feat_name, rows in store.items():
        preds, trues, owner = [], [], []
        for subj, p, t in rows:
            preds.extend(p.tolist()); trues.extend(t.tolist())
            owner.extend([subj] * len(p))
        out[feat_name] = _summarize(preds, trues, owner,
                                    f'{mode}_epoch_avg', 'mean', regressor, feat_name)
    return out


def _summarize(preds, trues, owner, protocol, agg, regressor, feature_set):
    preds = np.asarray(preds, dtype=float)
    trues = np.asarray(trues, dtype=float)
    owner = np.asarray(owner)

    per_subject = {}
    for s in np.unique(owner):
        m = owner == s
        per_subject[s] = {
            'cor': float(cor(trues[m], preds[m])) if m.sum() > 1 else None,
            'rmse': float(rmse(trues[m], preds[m])),
            'n_tests': int(m.sum()),
        }
    subj_cors = [v['cor'] for v in per_subject.values() if v['cor'] is not None]
    return {
        'protocol': protocol,
        'feature_set': feature_set,
        'agg_mode': agg,
        'regressor': regressor,
        'pooled_cor': float(cor(trues, preds)) if len(trues) > 2 else None,
        'pooled_rmse': float(rmse(trues, preds)),
        'mean_subject_cor': float(np.mean(subj_cors)) if subj_cors else None,
        'n_test_points': int(len(trues)),
        'per_subject': per_subject,
    }


# ── 主流程 ───────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser(description='DROZY test-level aggregation (Exp B)')
    p.add_argument('--quick', action='store_true', help='前 4 被试')
    args = p.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()

    print('=' * 74)
    print('DROZY test-level 聚合复现 + 标签审计 (Experiment B)')
    print('=' * 74)

    print('\n[阶段 1] 标签与 test/epoch 映射审计')
    audit_rows = audit_labels(DROZY_PSG_DIR)
    for r in audit_rows:
        status = 'OK' if r['valid'] else f"EXCLUDED ({r['exclusion_reason']})"
        print(f"  {r['subject']} test{r['test_id']}: KSS={r['kss']} "
              f"binary={r['kss_binary']} epochs={r['n_epochs']} -> {status}")

    checks = audit_consistency(DROZY_PSG_DIR)
    n_grp = sum(c['epochs_grouped_by_test'] for c in checks)
    n_kss = sum(c['kss_constant_within_test'] for c in checks)
    print(f"\n  一致性检查: {n_grp}/{len(checks)} 被试 epoch 按 test 分块, "
          f"{n_kss}/{len(checks)} 被试 test 内 KSS 恒定, "
          f"{sum(c['test_order_monotone'] for c in checks)}/{len(checks)} test 顺序单调")

    print('\n[阶段 2] 特征准备 (DE + FBTS 各频段协方差)')
    sids = list(range(1, 15))
    if args.quick:
        sids = sids[:4]
    subjects = {}
    for sid in sids:
        data = prepare_subject(DROZY_PSG_DIR, sid)
        if data is None:
            print(f'  subj{sid:02d}: 无有效 test, 跳过')
            continue
        subjects[sid] = data
        print(f"  subj{sid:02d}: epochs={len(data['y_kss'])} "
              f"tests={len(np.unique(data['t_ids']))} "
              f"KSS={sorted(set(data['y_kss'].tolist()))}")
    print(f'  载入 {len(subjects)} 个被试 ({time.time() - t0:.0f}s)')

    results = {'audit_table': audit_rows, 'audit_checks': checks, 'protocols': []}

    print('\n[阶段 2] 协议执行')
    for regressor in ['ridge', 'svr']:
        for agg in ['mean', 'median', 'mean_std']:
            for mode in ['loso', 'loto']:
                t1 = time.time()
                for feat in FEATURE_SETS:
                    r = protocol_test_level(subjects, mode=mode, agg=agg,
                                            regressor=regressor,
                                            feat_name=feat)
                    results['protocols'].append(r)
                print(f'  {mode.upper():5s} test_level {agg:9s} {regressor:5s} '
                      f'({time.time() - t1:.0f}s)')

        for mode in ['loso', 'loto']:
            t1 = time.time()
            out = protocol_epoch_avg(subjects, mode=mode, regressor=regressor)
            for feat, r in out.items():
                r['feature_set'] = feat
                results['protocols'].append(r)
            print(f'  {mode.upper():5s} epoch_avg     {regressor:5s} '
                  f'({time.time() - t1:.0f}s)')

    print(f"\n{'=' * 100}")
    print(f"{'Protocol':<22s} {'Feat':<9s} {'Agg':<9s} {'Reg':<6s} "
          f"{'pooledCOR':>11s} {'subjCOR':>9s} {'RMSE':>8s} {'N':>4s}")
    print('-' * 100)
    for r in results['protocols']:
        pc = r['pooled_cor']
        sc = r['mean_subject_cor']
        print(f"{r['protocol']:<22s} {r['feature_set']:<9s} {r['agg_mode']:<9s} "
              f"{r['regressor']:<6s} "
              f"{'n/a' if pc is None else f'{pc:11.4f}':>11s} "
              f"{'n/a' if sc is None else f'{sc:9.4f}':>9s} "
              f"{r['pooled_rmse']:8.3f} {r['n_test_points']:4d}")

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(RESULTS_DIR, f'results_drozy_testlevel_{ts}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f'\nSaved: {path}')
    print(f'Total: {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
