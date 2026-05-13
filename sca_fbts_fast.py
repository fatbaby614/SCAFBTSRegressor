"""
SCA-FBTS 回归器 — 优化版 (Fast)
================================
加速策略:
1. 预滤波: 全数据一次性滤波，fold 间复用 → 消除 4/5 的滤波开销
2. 预计算协方差: 885 个 epoch 的协方差矩阵计算一次，fold 间复用
3. 仅切空间 + 特征选择 + 回归按 fold 执行 (不可避免)

预期加速比: 3~5× (5band 单被试从 ~55s 降到 ~15s)
"""

import numpy as np
import pickle
from scipy import signal
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace


# ── 频段预设 (同原版) ──────────────────────────────────────
VIGILANCE_BANDS_5 = [(1,4),(4,8),(8,14),(14,31),(31,50)]
VIGILANCE_BANDS_8 = [(1,4),(4,6),(6,8),(8,10),(10,12),(12,14),(14,20),(20,30)]
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
    """4阶 Butterworth 带通滤波 (零相位)。

    Args:
        data: (n_times,) 或 (n_channels, n_times)
    """
    nyquist = 0.5 * fs
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    return signal.filtfilt(b, a, data, axis=-1)


class SCAFBTSRegressorFast:
    """SCA-FBTS 回归器 — 优化版。

    核心改动: 将滤波和协方差计算移到 fit() 之前一次性完成 (precompute)，
    fit() 只负责切空间 + 特征选择 + 回归。

    用法:
        clf = SCAFBTSRegressorFast(freq_bands='5band', ...)
        clf.precompute(raw_eeg, fs)              # 一次性预计算
        results = evaluate(clf, perclos_labels)   # 评测用 precomputed 数据
    """

    def __init__(self, freq_bands='5band', estimator='oas', metric='riemann',
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

        # 预计算缓存
        self._precomputed = False
        self._epochs_filtered = None   # dict: band → (n_epochs, ch, time)
        self._epochs_cov = None        # dict: band → (n_epochs, ch, ch) SPD matrices
        self._n_epochs = 0
        self._n_channels = 0
        self._epoch_len = 0

        # fit 后填充
        self.ts_transformers = []
        self.feature_selector = None
        self.regressor = None
        self._scaler = None

    def precompute(self, data, fs=None):
        """一次性预计算: 全数据滤波 + 所有 epoch 的协方差矩阵。

        自动检测输入格式:
          - (n_samples, n_channels): 原始数据，内部切分 epoch
          - (n_epochs, n_channels, n_times): 已切好的 epochs

        Args:
            data: 原始 EEG 或已切好的 epochs
            fs: 采样率 (默认用 __init__ 中的值)
        """
        if fs is not None:
            self.fs = fs

        # 自动检测输入格式
        if data.ndim == 2:
            # 原始数据 (n_samples, n_channels) → 切分 epoch
            epoch_len = int(self.fs * 8)
            n_total = data.shape[0]
            self._n_epochs = n_total // epoch_len
            self._n_channels = data.shape[1]
            self._epoch_len = epoch_len
            raw_T = data.T.astype(np.float64)

            self._epochs_filtered = {}
            self._epochs_cov = {}
            cov_estimator = Covariances(estimator=self.estimator)

            for low, high in self.freq_bands:
                print(f"    Precomputing band [{low}-{high}] Hz...")
                filtered = apply_bandpass_filter(raw_T, low, high, self.fs)
                epochs = np.array([
                    filtered[:, i * epoch_len:(i + 1) * epoch_len]
                    for i in range(self._n_epochs)
                ], dtype=np.float64)
                self._epochs_filtered[(low, high)] = epochs
                covs = cov_estimator.transform(epochs)
                self._epochs_cov[(low, high)] = covs

        elif data.ndim == 3:
            # 已切好的 epochs (n_epochs, n_channels, n_times)
            self._n_epochs = data.shape[0]
            self._n_channels = data.shape[1]
            self._epoch_len = data.shape[2]

            self._epochs_filtered = {}
            self._epochs_cov = {}
            cov_estimator = Covariances(estimator=self.estimator)
            X_f64 = data.astype(np.float64)

            for low, high in self.freq_bands:
                print(f"    Precomputing band [{low}-{high}] Hz...")
                # 逐 epoch 滤波 (已切分情况下无法全量滤波)
                epochs = np.array([
                    apply_bandpass_filter(e, low, high, self.fs)
                    for e in X_f64
                ], dtype=np.float64)
                self._epochs_filtered[(low, high)] = epochs
                covs = cov_estimator.transform(epochs)
                self._epochs_cov[(low, high)] = covs

        else:
            raise ValueError(f"Expected 2D or 3D input, got shape {data.shape}")

        self._precomputed = True
        print(f"    Precomputed {self._n_epochs} epochs × "
              f"{len(self.freq_bands)} bands ({self._n_channels}ch)")

    def fit(self, indices_or_X, y=None):
        """训练回归器。

        两种调用方式:
          1. fit(epoch_indices, y) — 使用预计算数据 (推荐)
          2. fit(X, y)           — 传统方式 (X 为 (n, ch, time))

        Args:
            indices_or_X: 整数数组(epoch索引) 或 (n, ch, time) 数组
            y: (n,) PERCLOS 标签
        """
        if y is None:
            raise ValueError("需要提供标签 y")

        # 判断输入类型
        if isinstance(indices_or_X, np.ndarray) and indices_or_X.ndim == 1 and \
           indices_or_X.dtype in (np.int32, np.int64, int):
            # 预计算模式: indices
            return self._fit_from_precomputed(indices_or_X, y)
        else:
            # 传统模式: 完整 X
            return self._fit_traditional(indices_or_X, y)

    def _fit_from_precomputed(self, indices, y):
        """从预计算缓存训练 (核心加速路径)。"""
        if not self._precomputed:
            raise RuntimeError("请先调用 precompute()")

        features_list = []
        self.ts_transformers = []

        for low, high in self.freq_bands:
            # 从缓存取协方差矩阵 (无需滤波!)
            covs = self._epochs_cov[(low, high)][indices]

            # 切空间投影
            ts = TangentSpace(metric=self.metric)
            ts_feats = ts.fit_transform(covs, y)
            self.ts_transformers.append(ts)
            features_list.append(ts_feats)

        X_combined = np.hstack(features_list)

        # 特征选择
        if self.n_features is not None:
            n_select = min(self.n_features, X_combined.shape[1])
            self.feature_selector = SelectKBest(
                score_func=f_regression, k=n_select
            )
            X_selected = self.feature_selector.fit_transform(X_combined, y)
        else:
            self.feature_selector = None
            X_selected = X_combined

        # 标准化
        if self.scaler:
            self._scaler = StandardScaler()
            X_selected = self._scaler.fit_transform(X_selected)

        # 回归器
        self.regressor = self._make_regressor()
        self.regressor.fit(X_selected, y)
        return self

    def _fit_traditional(self, X, y):
        """传统路径 (无预计算，与旧版兼容)。"""
        features_list = []
        self.ts_transformers = []

        for low, high in self.freq_bands:
            X_band = np.array([apply_bandpass_filter(trial, low, high, self.fs)
                               for trial in X])
            cov_estimator = Covariances(estimator=self.estimator)
            covs = cov_estimator.fit_transform(X_band)
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

    def predict(self, indices_or_X):
        """预测。

        Args:
            indices_or_X: epoch 索引数组 或 (n, ch, time) 数组
        """
        if isinstance(indices_or_X, np.ndarray) and indices_or_X.ndim == 1 and \
           indices_or_X.dtype in (np.int32, np.int64, int):
            return self._predict_from_precomputed(indices_or_X)
        else:
            return self._predict_traditional(indices_or_X)

    def _predict_from_precomputed(self, indices):
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

    def _predict_traditional(self, X):
        """传统预测路径。"""
        features_list = []

        for i, (low, high) in enumerate(self.freq_bands):
            X_band = np.array([apply_bandpass_filter(trial, low, high, self.fs)
                               for trial in X])
            covs = Covariances(estimator=self.estimator).transform(X_band)
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
        elif self.regressor_name == 'ridgecv':
            return RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        elif self.regressor_name == 'rfr':
            return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            raise ValueError(f"Unknown regressor: {self.regressor_name}")

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)
