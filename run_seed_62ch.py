"""
SEED 62-channel FBTS 公平对照 (JNE 修回 Experiment C)
======================================================
问题: 现稿仅报告 10 通道 FBTS (ACC 55.8%) 与 62 通道 DE (ACC 86.1%) 对比,
通道数与特征维度不同, 不能据此判定 FBTS 方法本身较差。

本脚本在同一被试/session 划分、同一分类器、同一评价指标下比较:
  DE-62        : 62ch DE (ExtractedFeatures, 5 bands)      → 310 维
  DE-10        : 10ch DE (同 10 通道子集)                   → 50 维
  FBTS-10      : 10ch FBTS 切空间 (5 bands)                  → 5×55 = 275 维
  FBTS-62      : 62ch FBTS 切空间 (5 bands)                  → 5×1953 = 9765 维
  FBTS-62-PCA  : FBTS-62 后接 PCA (50/100/200 维)

关键约束:
  - PCA 只能在训练 fold 内拟合 (绝不 fit_transform 全数据);
  - 切空间参考均值只在训练 fold 内拟合;
  - 特征选择 / 标准化只在训练 fold 内拟合;
  - Trial-level 5-fold CV (同一 trial 的 segment 不跨 fold);
  - 62ch 9765 维远超样本量, PCA 分支是必要的对照条件。

用法:
  python run_seed_62ch.py                # 全部 45 个 session
  python run_seed_62ch.py --n-sessions 9 # 前 9 个 session 冒烟
"""

import os
import re
import sys
import json
import time
import numpy as np
import scipy.io as sio
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from sca_fbts_fast import apply_bandpass_filter

from config import SEED_ROOT, RESULTS_DIR


TRIAL_LABELS = np.array([1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1])

CH_10 = list(range(10))          # FP1,FPZ,FP2,AF3,AF4,F7,F5,F3,F1,FZ
BANDS_5 = [(1, 4), (4, 8), (8, 14), (14, 31), (31, 50)]
PCA_DIMS = [50, 100, 200]
FS = 200
EPOCH_SEC = 4


def list_sessions():
    d = os.path.join(SEED_ROOT, 'ExtractedFeatures')
    return sorted([os.path.join(d, f) for f in os.listdir(d)
                   if f.endswith('.mat') and f not in ('label.mat', 'readme.txt')])


def load_de_session(feature_filepath):
    """加载一个 session 的 DE 特征 (62ch)。

    ExtractedFeatures 的 de_LDS{trial} 形状为 (62, n_segments, 5):
    通道 × 段 × 频段。按 trial 展开为 (n_segments, 310)。
    """
    data = sio.loadmat(feature_filepath)
    X, y = [], []
    for t in range(1, 16):
        key = f'de_LDS{t}'
        if key not in data:
            continue
        arr = data[key]                    # (62, n_seg, 5)
        feats = arr.transpose(1, 0, 2).reshape(arr.shape[1], -1)   # (n_seg, 310)
        X.append(feats)
        y.extend([TRIAL_LABELS[t - 1]] * feats.shape[0])
    return np.vstack(X), np.array(y)


def load_raw_session(feature_filepath, ch_idx=None):
    """加载一个 session 的原始 EEG 并切 4s epochs。

    Preprocessed_EEG 的 ww_eeg{trial} 形状为 (62, n_times)。
    返回 (X, y, trial_of_sample)。
    """
    fname = os.path.basename(feature_filepath)
    eeg_path = os.path.join(SEED_ROOT, 'Preprocessed_EEG', fname)
    if not os.path.exists(eeg_path):
        return None, None, None

    data = sio.loadmat(eeg_path)
    epoch_len = FS * EPOCH_SEC
    X, y, trial_ids = [], [], []
    # SEED 变量前缀为被试姓名缩写 (如 ww_eeg1 / wsf_eeg1 / zjy_eeg1),
    # 按正则匹配 '<letters>_eeg<trial>' 并按 trial 号索引
    seg_keys = {}
    for k in data:
        m = re.fullmatch(r'([a-z]+_eeg)(\d+)', k)
        if m:
            seg_keys[int(m.group(2))] = k
    for t in range(1, 16):
        key = seg_keys.get(t)
        if key is None:
            continue
        raw = data[key]
        if ch_idx is not None:
            raw = raw[ch_idx]
        n_epochs = raw.shape[1] // epoch_len
        if n_epochs == 0:
            continue
        epochs = np.stack([raw[:, i * epoch_len:(i + 1) * epoch_len]
                           for i in range(n_epochs)], axis=0)
        X.append(epochs)
        y.extend([TRIAL_LABELS[t - 1]] * n_epochs)
        trial_ids.extend([t - 1] * n_epochs)
    return (np.concatenate(X, axis=0), np.array(y), np.array(trial_ids))


def trial_5fold_splits(n_trials=15, n_folds=5):
    """Trial-level 时序 5-fold (同一 trial 的 segment 不跨 fold)。"""
    fold_size = n_trials // n_folds
    idx = np.arange(n_trials)
    splits = []
    for i in range(n_folds):
        te = idx[i * fold_size:(i + 1) * fold_size]
        tr = np.concatenate([idx[:i * fold_size], idx[(i + 1) * fold_size:]])
        splits.append((tr, te))
    return splits


def trial_masks(trial_of_sample, tr_trials, te_trials):
    """按 trial 索引生成样本级掩码 (同一 trial 的样本不跨 fold)。"""
    tr_mask = np.isin(trial_of_sample, tr_trials)
    te_mask = np.isin(trial_of_sample, te_trials)
    return tr_mask, te_mask


def run_de(X, y, trial_of_sample, n_features=200):
    """DE 基线: StandardScaler + SelectKBest(f_classif) + SVC。"""
    accs, f1s = [], []
    for tr_trials, te_trials in trial_5fold_splits():
        tr_mask, te_mask = trial_masks(trial_of_sample, tr_trials, te_trials)
        if tr_mask.sum() == 0 or te_mask.sum() == 0:
            continue
        Xtr, Xte = X[tr_mask], X[te_mask]
        ytr, yte = y[tr_mask], y[te_mask]

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)

        k = min(n_features, Xtr.shape[1])
        sel = SelectKBest(f_classif, k=k)
        Xtr = sel.fit_transform(Xtr, ytr)
        Xte = sel.transform(Xte)

        clf = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced')
        clf.fit(Xtr, ytr)
        yp = clf.predict(Xte)
        accs.append(accuracy_score(yte, yp))
        f1s.append(f1_score(yte, yp, average='macro', zero_division=0))
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(f1s))


def compute_covs(X_raw):
    """各频段协方差 (滤波 + OAS), 一次计算供多个 PCA 变体复用。"""
    cov_est = Covariances(estimator='oas')
    covs = {}
    for low, high in BANDS_5:
        filtered = np.stack([apply_bandpass_filter(e, low, high, FS)
                             for e in X_raw], axis=0)
        covs[(low, high)] = cov_est.transform(filtered)
    return covs


def build_fold_features(covs, y, trial_of_sample, metric='riemann'):
    """预计算 5-fold 的切空间特征。

    每个 fold 的切空间参考均值仅用训练 fold 拟合 (无泄漏)。结果缓存后
    供 FBTS-62 / FBTS-62-PCA{50,100,200} 各变体复用, 避免重复计算。

    metric='riemann' 与正文协议一致 (默认); 'logeuclid' 参考均值有闭式解,
    62ch 场景下快得多, 可用 --metric logeuclid 切换做速度/稳健性对比。

    Returns:
        list of (tr_mask, te_mask, Xtr, Xte, ytr, yte)
    """
    folds = []
    for tr_trials, te_trials in trial_5fold_splits():
        tr_mask, te_mask = trial_masks(trial_of_sample, tr_trials, te_trials)
        if tr_mask.sum() == 0 or te_mask.sum() == 0:
            continue
        ytr, yte = y[tr_mask], y[te_mask]
        feats_tr, feats_te = [], []
        for b in BANDS_5:
            ts = TangentSpace(metric=metric)
            feats_tr.append(ts.fit_transform(covs[b][tr_mask], ytr))
            feats_te.append(ts.transform(covs[b][te_mask]))
        folds.append((tr_mask, te_mask, np.hstack(feats_tr), np.hstack(feats_te),
                      ytr, yte))
    return folds


def evaluate_folds(folds, pca_dims=None, n_features=100):
    """在预计算的切空间特征上做 PCA + 特征选择 + 分类。

    PCA 与特征选择均只在训练 fold 内拟合。
    """
    accs, f1s, explained = [], [], []
    for _, _, Xtr0, Xte0, ytr, yte in folds:
        Xtr, Xte = Xtr0, Xte0

        if pca_dims is not None:
            k = min(pca_dims, Xtr.shape[1], Xtr.shape[0])
            pca = PCA(n_components=k, random_state=42)
            Xtr = pca.fit_transform(Xtr)
            Xte = pca.transform(Xte)
            explained.append(float(pca.explained_variance_ratio_.sum()))

        if Xtr.shape[1] > n_features:
            sel = SelectKBest(f_classif, k=n_features)
            Xtr = sel.fit_transform(Xtr, ytr)
            Xte = sel.transform(Xte)

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)

        clf = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced')
        clf.fit(Xtr, ytr)
        yp = clf.predict(Xte)
        accs.append(accuracy_score(yte, yp))
        f1s.append(f1_score(yte, yp, average='macro', zero_division=0))

    return (float(np.mean(accs)), float(np.std(accs)), float(np.mean(f1s)),
            float(np.mean(explained)) if explained else None)


def main():
    import argparse
    p = argparse.ArgumentParser(description='SEED 62ch FBTS fair comparison (Exp C)')
    p.add_argument('--n-sessions', type=int, default=0,
                   help='仅跑前 N 个 session (0 = 全部 45)')
    p.add_argument('--metric', default='riemann',
                   choices=['riemann', 'logeuclid'],
                   help='切空间度量 (默认 riemann 与正文一致)')
    args = p.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    sessions = list_sessions()
    if args.n_sessions and args.n_sessions < len(sessions):
        sessions = sessions[:args.n_sessions]
    print('=' * 70)
    print(f'SEED 62-channel FBTS fair comparison: {len(sessions)} sessions')
    print(f"  metric = {args.metric}")
    print('=' * 70)

    results = []
    t_start = time.time()

    for i, fpath in enumerate(sessions):
        name = os.path.basename(fpath)
        row = {'session': name, 'metric': args.metric}

        # ── DE 62ch ──
        Xde, yde = load_de_session(fpath)
        if Xde is None:
            print(f'  [{i+1}/{len(sessions)}] {name}: DE 特征缺失, 跳过')
            continue
        # DE 的 trial 归属: 每 trial 的段数可能不同, 用累计长度还原
        trial_of_de = []
        data = sio.loadmat(fpath)
        for t in range(1, 16):
            key = f'de_LDS{t}'
            if key not in data:
                continue
            n_seg = data[key].shape[1]
            trial_of_de.extend([t - 1] * n_seg)
        trial_of_de = np.array(trial_of_de)

        t0 = time.time()
        acc, accs, f1 = run_de(Xde, yde, trial_of_de)
        row.update(de62_acc_mean=acc, de62_acc_std=accs, de62_f1_mean=f1,
                   de62_dim=int(Xde.shape[1]), de62_time=time.time() - t0)
        print(f'  [{i+1}/{len(sessions)}] {name} DE-62 ACC={acc:.4f} F1={f1:.4f}')

        # ── DE 10ch (通道子集) ──
        Xde10 = Xde.reshape(len(Xde), 62, 5)[:, CH_10, :].reshape(len(Xde), -1)
        t0 = time.time()
        acc, accs, f1 = run_de(Xde10, yde, trial_of_de)
        row.update(de10_acc_mean=acc, de10_acc_std=accs, de10_f1_mean=f1,
                   de10_dim=int(Xde10.shape[1]), de10_time=time.time() - t0)
        print(f'      DE-10 ACC={acc:.4f} F1={f1:.4f}')

        # ── FBTS 10ch ──
        Xr10, yr10, tr10 = load_raw_session(fpath, ch_idx=CH_10)
        if Xr10 is not None:
            t0 = time.time()
            covs10 = compute_covs(Xr10)
            folds10 = build_fold_features(covs10, yr10, tr10, metric=args.metric)
            row['fbts10_feature_time'] = time.time() - t0
            acc, accs, f1, _ = evaluate_folds(folds10)
            row.update(fbts10_acc_mean=acc, fbts10_acc_std=accs,
                       fbts10_f1_mean=f1, fbts10_dim=5 * (10 * 11 // 2),
                       fbts10_time=time.time() - t0)
            print(f'      FBTS-10 ACC={acc:.4f} F1={f1:.4f} '
                  f'({time.time() - t0:.0f}s)')
            del covs10, folds10

        # ── FBTS 62ch (含 PCA 分支, 切空间特征只算一次) ──
        Xr62, yr62, tr62 = load_raw_session(fpath, ch_idx=None)
        if Xr62 is not None:
            t0 = time.time()
            covs62 = compute_covs(Xr62)
            row['fbts62_cov_time'] = time.time() - t0

            t0 = time.time()
            folds62 = build_fold_features(covs62, yr62, tr62, metric=args.metric)
            row['fbts62_feature_time'] = time.time() - t0
            print(f'      FBTS-62 features built ({time.time() - t0:.0f}s)')

            t0 = time.time()
            acc, accs, f1, _ = evaluate_folds(folds62)
            row.update(fbts62_acc_mean=acc, fbts62_acc_std=accs,
                       fbts62_f1_mean=f1, fbts62_dim=5 * (62 * 63 // 2),
                       fbts62_time=time.time() - t0)
            print(f'      FBTS-62 ACC={acc:.4f} F1={f1:.4f} '
                  f'({time.time() - t0:.0f}s)')

            for pd in PCA_DIMS:
                t0 = time.time()
                acc, accs, f1, ev = evaluate_folds(folds62, pca_dims=pd)
                row.update(**{f'fbts62_pca{pd}_acc_mean': acc,
                              f'fbts62_pca{pd}_acc_std': accs,
                              f'fbts62_pca{pd}_f1_mean': f1,
                              f'fbts62_pca{pd}_dim': pd,
                              f'fbts62_pca{pd}_explained_var': ev,
                              f'fbts62_pca{pd}_time': time.time() - t0})
                print(f'      FBTS-62-PCA{pd} ACC={acc:.4f} F1={f1:.4f} '
                      f'(expl.var={ev:.3f}, {time.time() - t0:.0f}s)')
            del covs62, folds62

        results.append(row)

    # ── 汇总 ──
    def agg(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)

    summary = []
    for label, key in [('DE-62', 'de62_acc_mean'),
                       ('DE-10', 'de10_acc_mean'),
                       ('FBTS-10', 'fbts10_acc_mean'),
                       ('FBTS-62', 'fbts62_acc_mean'),
                       ('FBTS-62-PCA50', 'fbts62_pca50_acc_mean'),
                       ('FBTS-62-PCA100', 'fbts62_pca100_acc_mean'),
                       ('FBTS-62-PCA200', 'fbts62_pca200_acc_mean')]:
        acc_m, acc_s = agg(key)
        f1_m, _ = agg(key.replace('_acc_mean', '_f1_mean'))
        dim = None
        for r in results:
            if r.get(key) is not None:
                dim = r.get(key.replace('_acc_mean', '_dim'))
                break
        summary.append({'method': label, 'acc_mean': acc_m, 'acc_std': acc_s,
                        'f1_mean': f1_m, 'dim': dim})

    print(f"\n{'=' * 78}")
    print(f"{'Method':<16s} {'Dim':>7s} {'ACC':>9s} {'ACCstd':>9s} {'F1':>9s}")
    print('-' * 78)
    for s in summary:
        if s['acc_mean'] is None:
            continue
        print(f"{s['method']:<16s} {str(s['dim']):>7s} {s['acc_mean']:>9.4f} "
              f"{s['acc_std']:>9.4f} {s['f1_mean']:>9.4f}")

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(RESULTS_DIR, f'results_seed_62ch_{ts}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'per_session': results},
                  f, indent=2, ensure_ascii=False, default=str)
    print(f'\nSaved: {path}')
    print(f'Total: {time.time() - t_start:.0f}s')


def _trial_of_raw(feature_filepath):
    """原始 EEG 的 trial 归属 (按每 trial 的 4s epoch 数展开)。"""
    _, _, trial_ids = load_raw_session(feature_filepath, ch_idx=None)
    return trial_ids


if __name__ == '__main__':
    main()
