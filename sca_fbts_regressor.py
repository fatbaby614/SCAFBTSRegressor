"""
SCA-FBTS 回归器 — 从睡眠分期到连续警觉度估计
==============================================
基于 D:/EEG/TanHuangWork/Sleep-algorithms/sca_fbts.py 改造:

核心改动:
1. 分类器 → 回归器 (SVC→SVR, LDA→Ridge, RF→RFR)
2. 移除 N1 保护逻辑
3. 时序平滑仅做普通滑动平均
4. 添加 SEED-VIG 适配的频段配置
5. 保留 Filter Bank + Riemannian Tangent Space + Feature Selection 管线

依赖: numpy, scipy, sklearn, pyriemann
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


# ── 频段配置 ─────────────────────────────────────────────────

# SEED-VIG 标准5频段 (用于公平对比 DE/PSD 特征)
VIGILANCE_BANDS_5 = [
    (1, 4),     # Delta
    (4, 8),     # Theta
    (8, 14),    # Alpha
    (14, 31),   # Beta
    (31, 50),   # Gamma
]

# 扩展8频段 (你的睡眠频段基础上调整)
VIGILANCE_BANDS_8 = [
    (1, 4),     # Delta
    (4, 6),     # Low Theta
    (6, 8),     # High Theta
    (8, 10),    # Low Alpha
    (10, 12),   # High Alpha
    (12, 14),   # Sigma (spindles)
    (14, 20),   # Low Beta
    (20, 30),   # High Beta
]

# 25频段 (2Hz 分辨率，对齐 EEG_Feature_2Hz)
VIGILANCE_BANDS_25 = [(i, i+2) for i in range(0, 50, 2)]

# 超精细频段 (1Hz 分辨率，部分覆盖)
VIGILANCE_BANDS_FINE = [
    (1, 4), (4, 6), (6, 8), (8, 10), (10, 12),
    (12, 14), (14, 16), (16, 18), (18, 20),
    (20, 24), (24, 28), (28, 32), (32, 36),
    (36, 40), (40, 45), (45, 50),
]

FREQ_BAND_PRESETS = {
    '5band': VIGILANCE_BANDS_5,
    '8band': VIGILANCE_BANDS_8,
    '25band': VIGILANCE_BANDS_25,
    'fine': VIGILANCE_BANDS_FINE,
}


# ── 带通滤波 (与原版相同) ───────────────────────────────────

def apply_bandpass_filter(data, low_freq, high_freq, fs):
    """4阶 Butterworth 带通滤波 (零相位)。

    Args:
        data: (n_times,) 或 (n_channels, n_times)
        low_freq: 低频截止
        high_freq: 高频截止
        fs: 采样率

    Returns:
        filtered_data: 同 shape
    """
    nyquist = 0.5 * fs
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, data, axis=-1)
    return filtered_data


# ── SCA-FBTS 回归器 ─────────────────────────────────────────

class SCAFBTSRegressor:
    """SCA-FBTS Regressor: Filter Bank Tangent Space for continuous vigilance estimation.

    将原始 SCA_FBTS 的分类管线改造为回归管线:

    1. Filter Bank: 多频段带通滤波
    2. Covariance: 计算 SPD 协方差矩阵
    3. Tangent Space: 黎曼切空间投影
    4. Feature Selection: f_regression 选最优特征
    5. Regression: SVR / Ridge / RandomForest

    Parameters
    ----------
    freq_bands : list of (low, high) tuples, or str preset
        频段配置。可用 preset: '5band', '8band', '25band', 'fine'
    estimator : str
        协方差估计器: 'oas', 'lwf', 'scm', 'cov', 'corr'
    metric : str
        切空间度量: 'riemann', 'euclid', 'logeuclid', 'stein'
    regressor : str
        回归器: 'svr', 'ridge', 'rfr' (random forest)
    n_features : int or None
        特征选择数量 (None=全部保留)
    fs : int
        采样率 (SEED-VIG EEG 为 200)
    temporal_smoothing : bool
        是否对预测做时间平滑
    smoothing_window : int
        平滑窗口大小 (奇数)
    scaler : bool
        是否在回归前做 StandardScaler
    """

    def __init__(self, freq_bands='5band', estimator='oas', metric='riemann',
                 regressor='svr', n_features=100, fs=200,
                 temporal_smoothing=True, smoothing_window=3,
                 scaler=True):
        self.fs = fs
        self.estimator = estimator
        self.metric = metric
        self.regressor_name = regressor
        self.n_features = n_features
        self.temporal_smoothing = temporal_smoothing
        self.smoothing_window = smoothing_window
        self.scaler = scaler

        # 解析频段
        if isinstance(freq_bands, str):
            if freq_bands not in FREQ_BAND_PRESETS:
                raise ValueError(f"未知频段预设: {freq_bands}. "
                               f"可选: {list(FREQ_BAND_PRESETS.keys())}")
            self.freq_bands = FREQ_BAND_PRESETS[freq_bands]
        else:
            self.freq_bands = freq_bands

        # 初始化内部对象 (fit 时填充)
        self.cov_estimators = []
        self.ts_transformers = []
        self.feature_selector = None
        self.regressor = None
        self._scaler = None

    def fit(self, X, y):
        """训练回归器。

        Args:
            X: (n_samples, n_channels, n_times) float64
            y: (n_samples,) float64 — PERCLOS 标签

        Returns:
            self
        """
        features_list = []
        self.cov_estimators = []
        self.ts_transformers = []

        for low, high in self.freq_bands:
            print(f"    Band [{low}-{high}] Hz...")

            # Step 1: Filter Bank
            X_band = np.array([apply_bandpass_filter(trial, low, high, self.fs)
                               for trial in X])

            # Step 2: Covariance matrices
            cov_estimator = Covariances(estimator=self.estimator)
            cov_matrices = cov_estimator.fit_transform(X_band)
            self.cov_estimators.append(cov_estimator)

            # Step 3: Tangent space projection
            ts_transformer = TangentSpace(metric=self.metric)
            ts_features = ts_transformer.fit_transform(cov_matrices, y)
            self.ts_transformers.append(ts_transformer)

            features_list.append(ts_features)

        # Step 4: Concatenate
        X_combined = np.hstack(features_list)
        print(f"    Combined features: {X_combined.shape[1]}")

        # Step 5: Feature selection (改用 f_regression)
        if self.n_features is not None:
            n_select = min(self.n_features, X_combined.shape[1])
            print(f"    Selecting top {n_select} features...")
            self.feature_selector = SelectKBest(
                score_func=f_regression, k=n_select
            )
            X_selected = self.feature_selector.fit_transform(X_combined, y)
            print(f"    Selected: {X_selected.shape[1]}")
        else:
            self.feature_selector = None
            X_selected = X_combined

        # Step 6: Optional scaling
        if self.scaler:
            self._scaler = StandardScaler()
            X_selected = self._scaler.fit_transform(X_selected)

        # Step 7: Train regressor
        if self.regressor_name == 'svr':
            self.regressor = SVR(kernel='rbf', C=1.0, gamma='scale')
        elif self.regressor_name == 'ridge':
            self.regressor = Ridge(alpha=1.0)
        elif self.regressor_name == 'ridgecv':
            self.regressor = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        elif self.regressor_name == 'rfr':
            self.regressor = RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown regressor: {self.regressor_name}")

        self.regressor.fit(X_selected, y)
        return self

    def predict(self, X):
        """预测连续警觉度值。

        Args:
            X: (n_samples, n_channels, n_times)

        Returns:
            y_pred: (n_samples,) float64
        """
        features_list = []

        for i, (low, high) in enumerate(self.freq_bands):
            # Filter Bank
            X_band = np.array([apply_bandpass_filter(trial, low, high, self.fs)
                               for trial in X])

            # Covariance
            cov_matrices = self.cov_estimators[i].transform(X_band)

            # Tangent Space
            ts_features = self.ts_transformers[i].transform(cov_matrices)
            features_list.append(ts_features)

        # Concatenate
        X_combined = np.hstack(features_list)

        # Feature selection
        if self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_combined)
        else:
            X_selected = X_combined

        # Scaling
        if self._scaler is not None:
            X_selected = self._scaler.transform(X_selected)

        # Predict
        y_pred = self.regressor.predict(X_selected)

        # Temporal smoothing (simple moving average, no N1 hack)
        if self.temporal_smoothing and len(y_pred) > 1:
            from scipy.ndimage import uniform_filter1d
            y_pred = uniform_filter1d(y_pred, size=self.smoothing_window)

        return y_pred

    def save(self, path):
        """保存模型到文件。"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """从文件加载模型。"""
        with open(path, 'rb') as f:
            return pickle.load(f)


# ── 快速测试 ─────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    sys.path.insert(0, r'D:\EEG\TanHuangWork\SCAFBTSRegressor')
    from data_loader import load_raw_eeg, load_perclos, build_epochs_from_raw

    data_root = r'D:\EEG\datasets\SEED-VIG'
    subj = '10_20151125_noon'

    print(f"Loading {subj}...")
    raw, sr = load_raw_eeg(data_root, subj)
    y = load_perclos(data_root, subj)
    X, y = build_epochs_from_raw(raw, y, fs=sr)
    print(f"X: {X.shape}, y: {y.shape}")

    # 快速测试 (只用 5band 少特征，跑个 5-fold)
    print("\n=== Riemannian SVR (5band, riemann) ===")
    from utils import evaluate_regressor
    clf = SCAFBTSRegressor(
        freq_bands='5band', estimator='oas', metric='riemann',
        regressor='svr', n_features=50, fs=sr,
        temporal_smoothing=True, smoothing_window=3,
    )
    results = evaluate_regressor(clf, X, y, n_folds=5, verbose=True)
