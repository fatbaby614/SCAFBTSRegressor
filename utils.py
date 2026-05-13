"""
评估工具函数
============
SEED-VIG 标准评测: 5-fold 时序交叉验证, COR + RMSE.
"""

import numpy as np
from scipy.stats import pearsonr


def cor(y_true, y_pred):
    """Pearson 相关系数 (COR)。

    Args:
        y_true: (n,) 真实值
        y_pred: (n,) 预测值

    Returns:
        float
    """
    # 处理常值预测
    if np.std(y_pred) < 1e-10 or np.std(y_true) < 1e-10:
        return 0.0
    return pearsonr(y_true, y_pred)[0]


def rmse(y_true, y_pred):
    """均方根误差。

    Args:
        y_true: (n,) 真实值
        y_pred: (n,) 预测值

    Returns:
        float
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    """平均绝对误差。

    Args:
        y_true: (n,) 真实值
        y_pred: (n,) 预测值

    Returns:
        float
    """
    return np.mean(np.abs(y_true - y_pred))


def get_5fold_splits(n_samples=885, n_folds=5):
    """生成时序保持的 5-fold 划分索引。

    SEED-VIG 标准: 885 个样本按时序等分为 5 段 (每段 177 个)，
    1 段作测试，4 段作训练。时序不能打乱。

    Args:
        n_samples: 总样本数 (默认 885)
        n_folds: 折数 (默认 5)

    Returns:
        list of (train_idx, test_idx) tuples
    """
    fold_size = n_samples // n_folds  # 177
    splits = []
    indices = np.arange(n_samples)

    for i in range(n_folds):
        test_start = i * fold_size
        test_end = test_start + fold_size
        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([
            indices[:test_start],
            indices[test_end:]
        ])
        splits.append((train_idx, test_idx))

    return splits


def evaluate_regressor(clf, X, y, n_folds=5, temporal_smooth=True,
                       smoothing_window=3, verbose=True):
    """标准 5-fold 交叉验证评测。

    Args:
        clf: 回归器对象 (需实现 fit/predict)
        X: (n_samples, n_channels, n_times) 或 (n_samples, n_features)
        y: (n_samples,)
        n_folds: 折数
        temporal_smooth: 是否对预测做时间平滑
        smoothing_window: 平滑窗口大小
        verbose: 是否打印进度

    Returns:
        results: dict with keys:
            - 'cor_mean', 'cor_std', 'cor_all': COR 统计
            - 'rmse_mean', 'rmse_std', 'rmse_all': RMSE 统计
            - 'mae_mean', 'mae_std': MAE 统计 (新增)
            - 'y_true_all', 'y_pred_all': 拼接全部预测用于全局 COR
    """
    splits = get_5fold_splits(len(y), n_folds)
    cor_list, rmse_list, mae_list = [], [], []
    y_true_all, y_pred_all = [], []

    for fold, (train_idx, test_idx) in enumerate(splits):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # 训练
        clf.fit(X_train, y_train)

        # 预测 (新模型实例不携带 fit 后的 classifier，直接用 predict)
        y_pred = clf.predict(X_test)

        # 时间平滑
        if temporal_smooth and len(y_pred) > 1:
            from scipy.ndimage import uniform_filter1d
            y_pred = uniform_filter1d(y_pred, size=smoothing_window)

        # 评测
        c = cor(y_test, y_pred)
        r = rmse(y_test, y_pred)
        m = mae(y_test, y_pred)
        cor_list.append(c)
        rmse_list.append(r)
        mae_list.append(m)
        y_true_all.append(y_test)
        y_pred_all.append(y_pred)

        if verbose:
            print(f"  Fold {fold+1}/{n_folds}: COR={c:.4f}, RMSE={r:.4f}, MAE={m:.4f}")

    # 全局 COR (拼接所有 fold)
    y_true_cat = np.concatenate(y_true_all)
    y_pred_cat = np.concatenate(y_pred_all)
    global_cor = cor(y_true_cat, y_pred_cat)
    global_rmse = rmse(y_true_cat, y_pred_cat)

    results = {
        'cor_mean': np.mean(cor_list),
        'cor_std': np.std(cor_list),
        'cor_all': cor_list,
        'rmse_mean': np.mean(rmse_list),
        'rmse_std': np.std(rmse_list),
        'rmse_all': rmse_list,
        'mae_mean': np.mean(mae_list),
        'mae_std': np.std(mae_list),
        'global_cor': global_cor,
        'global_rmse': global_rmse,
        'y_true_all': y_true_cat,
        'y_pred_all': y_pred_cat,
    }

    if verbose:
        print(f"  Avg: COR={results['cor_mean']:.4f}±{results['cor_std']:.4f}, "
              f"RMSE={results['rmse_mean']:.4f}±{results['rmse_std']:.4f}")
        print(f"  Global: COR={global_cor:.4f}, RMSE={global_rmse:.4f}")

    return results


def evaluate_de_baseline(regressor, X, y, n_folds=5, temporal_smooth=True,
                         smoothing_window=3, verbose=True):
    """DE 特征基线的 5-fold 评测 (2D 输入)。

    与 evaluate_regressor 相同逻辑，但输入是 (n, features) 而非 (n, ch, time)。
    每个 fold 需要新建回归器 (sklearn 风格 fit/predict)。

    Args:
        regressor: sklearn 回归器工厂函数，如 lambda: SVR(kernel='rbf')
        X: (n_samples, n_features)
        y: (n_samples,)
        n_folds, temporal_smooth, smoothing_window, verbose: 同上

    Returns:
        results: 同 evaluate_regressor
    """
    splits = get_5fold_splits(len(y), n_folds)
    cor_list, rmse_list, mae_list = [], [], []
    y_true_all, y_pred_all = [], []

    for fold, (train_idx, test_idx) in enumerate(splits):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # 新建并训练回归器
        clf = regressor()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        if temporal_smooth and len(y_pred) > 1:
            from scipy.ndimage import uniform_filter1d
            y_pred = uniform_filter1d(y_pred, size=smoothing_window)

        c = cor(y_test, y_pred)
        r = rmse(y_test, y_pred)
        m = mae(y_test, y_pred)
        cor_list.append(c)
        rmse_list.append(r)
        mae_list.append(m)
        y_true_all.append(y_test)
        y_pred_all.append(y_pred)

        if verbose:
            print(f"  Fold {fold+1}/{n_folds}: COR={c:.4f}, RMSE={r:.4f}, MAE={m:.4f}")

    y_true_cat = np.concatenate(y_true_all)
    y_pred_cat = np.concatenate(y_pred_all)
    global_cor = cor(y_true_cat, y_pred_cat)
    global_rmse = rmse(y_true_cat, y_pred_cat)

    results = {
        'cor_mean': np.mean(cor_list),
        'cor_std': np.std(cor_list),
        'cor_all': cor_list,
        'rmse_mean': np.mean(rmse_list),
        'rmse_std': np.std(rmse_list),
        'rmse_all': rmse_list,
        'mae_mean': np.mean(mae_list),
        'mae_std': np.std(mae_list),
        'global_cor': global_cor,
        'global_rmse': global_rmse,
        'y_true_all': y_true_cat,
        'y_pred_all': y_pred_cat,
    }

    if verbose:
        print(f"  Avg: COR={results['cor_mean']:.4f}±{results['cor_std']:.4f}, "
              f"RMSE={results['rmse_mean']:.4f}±{results['rmse_std']:.4f}")
        print(f"  Global: COR={global_cor:.4f}, RMSE={global_rmse:.4f}")

    return results


# ── 测试 ─────────────────────────────────────────────────────
if __name__ == '__main__':
    # 测试 5-fold
    splits = get_5fold_splits(885)
    for i, (train, test) in enumerate(splits):
        print(f"Fold {i+1}: train {len(train)} ({train[0]}..{train[-1]}), "
              f"test {len(test)} ({test[0]}..{test[-1]})")

    # 测试 COR/RMSE
    y_true = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    y_pred = np.array([0.15, 0.25, 0.55, 0.65, 0.95])
    print(f"\nCOR: {cor(y_true, y_pred):.4f}")
    print(f"RMSE: {rmse(y_true, y_pred):.4f}")
    print(f"MAE: {mae(y_true, y_pred):.4f}")
