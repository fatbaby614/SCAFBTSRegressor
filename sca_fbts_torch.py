"""
SCA-FBTS 回归器 — GPU 加速版 (Torch)
=====================================
加速项:
1. torch 批量协方差计算 (替代 pyriemann Covariances) → 10~20× 加速
2. 预滤波 (同 Fast 版)
3. 仅切空间 + 回归走 pyriemann sklearn (必须保留，无法 torch 化)
"""

import numpy as np
import pickle
from scipy import signal
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from pyriemann.tangentspace import TangentSpace


# ── 频段预设 ──────────────────────────────────────────────
VIGILANCE_BANDS_5  = [(1,4),(4,8),(8,14),(14,31),(31,50)]
VIGILANCE_BANDS_8  = [(1,4),(4,6),(6,8),(8,10),(10,12),(12,14),(14,20),(20,30)]
VIGILANCE_BANDS_25 = [(i,i+2) for i in range(0,50,2)]
VIGILANCE_BANDS_FINE = [
    (1,4),(4,6),(6,8),(8,10),(10,12),(12,14),(14,16),(16,18),
    (18,20),(20,24),(24,28),(28,32),(32,36),(36,40),(40,45),(45,50),
]

FREQ_BAND_PRESETS = {
    '5band': VIGILANCE_BANDS_5, '8band': VIGILANCE_BANDS_8,
    '25band': VIGILANCE_BANDS_25, 'fine': VIGILANCE_BANDS_FINE,
}


def apply_bandpass_filter(data, low_freq, high_freq, fs):
    """4阶 Butterworth 零相位滤波。"""
    nyquist = 0.5 * fs
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    return signal.filtfilt(b, a, data, axis=-1)


# ── Torch 批量协方差 ─────────────────────────────────────

_DEVICE = None

def _get_device():
    global _DEVICE
    if _DEVICE is None:
        try:
            import torch
            _DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            _DEVICE = 'cpu'
    return _DEVICE


def batch_covariance(epochs, estimator='scm', regularization=0.01):
    """批量计算协方差矩阵 (torch 加速)。

    对于 (n, ch, time) 的 epochs，pyriemann 逐样本计算协方差 O(n) 循环。
    torch 可以一次 bmm 全部完成，GPU 上可达 20~50× 加速。

    Args:
        epochs: (n_epochs, n_channels, n_times) — numpy 或 torch
        estimator: 'scm' (样本协方差) 或 'lwf' (Ledoit-Wolf 近似)
        regularization: 对角加载系数 (提升数值稳定性)

    Returns:
        covs: (n_epochs, n_channels, n_channels) numpy float64
    """
    import torch

    device = _get_device()

    if isinstance(epochs, np.ndarray):
        X = torch.from_numpy(epochs).float().to(device)
    else:
        X = epochs.float().to(device)

    n, ch, t = X.shape

    # 去均值
    X = X - X.mean(dim=-1, keepdim=True)

    # 批量协方差: (n, ch, ch)
    covs = torch.bmm(X, X.transpose(1, 2)) / (t - 1)

    if estimator == 'lwf':
        # Ledoit-Wolf 近似收缩 (简化版)
        # 对每个矩阵做 trace-based shrinkage
        trace_sum = torch.diagonal(covs, dim1=1, dim2=2).sum(dim=1)  # (n,)
        I = torch.eye(ch, device=device, dtype=torch.float32).unsqueeze(0)
        target = (trace_sum / ch).view(-1, 1, 1) * I
        # 使用固定收缩率 (简化; 精确 L-W 需要迭代)
        shrinkage = 0.1
        covs = (1 - shrinkage) * covs + shrinkage * target

    # 对角加载 (数值稳定性)
    if regularization > 0:
        I = torch.eye(ch, device=device, dtype=torch.float32).unsqueeze(0)
        covs = covs + regularization * (torch.diagonal(covs, dim1=1, dim2=2).mean(dim=1).view(-1, 1, 1)) * I

    return covs.cpu().numpy().astype(np.float64)


class SCAFBTSRegressorTorch:
    """SCA-FBTS 回归器 — Torch 加速版。

    用法:
        clf = SCAFBTSRegressorTorch(freq_bands='5band', metric='riemann')
        clf.precompute(raw_eeg, fs=200)      # 预滤波 + torch 批量协方差
        clf.fit(train_indices, y_train)       # 切空间 + 回归
        y_pred = clf.predict(test_indices)    # 预测
    """

    def __init__(self, freq_bands='5band', estimator='scm', metric='riemann',
                 regressor='svr', n_features=100, fs=200,
                 temporal_smoothing=True, smoothing_window=3, scaler=True):
        self.fs = fs
        self.estimator = estimator
        self.metric = metric
        self.regressor_name = regressor
        self.n_features = n_features
        self.temporal_smoothing = temporal_smoothing
        self.smoothing_window = smoothing_window
        self.scaler = scaler

        if isinstance(freq_bands, str):
            self.freq_bands = FREQ_BAND_PRESETS[freq_bands]
        else:
            self.freq_bands = freq_bands

        self._precomputed = False
        self._epochs_cov = {}
        self._n_epochs = 0
        self._n_channels = 0
        self._epoch_len = 0

        self.ts_transformers = []
        self.feature_selector = None
        self.regressor = None
        self._scaler = None

    def precompute(self, raw_eeg, fs=None):
        """预计算: 滤波 + torch 批量协方差。"""
        if fs is not None:
            self.fs = fs
        epoch_len = int(self.fs * 8)
        n_total = raw_eeg.shape[0]
        self._n_epochs = n_total // epoch_len
        self._n_channels = raw_eeg.shape[1]
        self._epoch_len = epoch_len
        raw_T = raw_eeg.T.astype(np.float32)

        self._epochs_cov = {}

        for low, high in self.freq_bands:
            print(f"    [{low}-{high}] Hz: filtering + torch cov...", end=" ", flush=True)
            import time as _time
            t0 = _time.time()

            # 一次性滤波
            filtered = apply_bandpass_filter(raw_T, low, high, self.fs)

            # 切分 epoch
            epochs = np.array([
                filtered[:, i * epoch_len:(i + 1) * epoch_len]
                for i in range(self._n_epochs)
            ], dtype=np.float32)

            # torch 批量协方差
            covs = batch_covariance(epochs, estimator=self.estimator)
            self._epochs_cov[(low, high)] = covs

            print(f"{_time.time()-t0:.1f}s")

        self._precomputed = True
        print(f"    Precomputed {self._n_epochs} epochs x "
              f"{len(self.freq_bands)} bands ({self._n_channels}ch)")

    def fit(self, indices, y):
        """从预计算缓存训练 (仅切空间 + 特征选择 + 回归)。"""
        if not self._precomputed:
            raise RuntimeError("请先调用 precompute()")

        features_list = []
        self.ts_transformers = []

        for low, high in self.freq_bands:
            covs = self._epochs_cov[(low, high)][indices]
            ts = TangentSpace(metric=self.metric)
            ts_feats = ts.fit_transform(covs, y)
            self.ts_transformers.append(ts)
            features_list.append(ts_feats)

        X_combined = np.hstack(features_list)

        if self.n_features is not None:
            n_select = min(self.n_features, X_combined.shape[1])
            self.feature_selector = SelectKBest(
                score_func=f_regression, k=n_select
            )
            X_selected = self.feature_selector.fit_transform(X_combined, y)
        else:
            self.feature_selector = None
            X_selected = X_combined

        if self.scaler:
            self._scaler = StandardScaler()
            X_selected = self._scaler.fit_transform(X_selected)

        self.regressor = self._make_regressor()
        self.regressor.fit(X_selected, y)
        return self

    def predict(self, indices):
        """从预计算缓存预测。"""
        features_list = []

        for i, (low, high) in enumerate(self.freq_bands):
            covs = self._epochs_cov[(low, high)][indices]
            ts_feats = self.ts_transformers[i].transform(covs)
            features_list.append(ts_feats)

        X_combined = np.hstack(features_list)

        if self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_combined)
        else:
            X_selected = X_combined

        if self._scaler is not None:
            X_selected = self._scaler.transform(X_selected)

        y_pred = self.regressor.predict(X_selected)

        if self.temporal_smoothing and len(y_pred) > 1:
            from scipy.ndimage import uniform_filter1d
            y_pred = uniform_filter1d(y_pred, size=self.smoothing_window)

        return y_pred

    def _make_regressor(self):
        if self.regressor_name == 'svr':
            return SVR(kernel='rbf', C=1.0, gamma='scale')
        elif self.regressor_name == 'ridge':
            return Ridge(alpha=1.0)
        elif self.regressor_name == 'rfr':
            return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        raise ValueError(f"Unknown regressor: {self.regressor_name}")

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)
