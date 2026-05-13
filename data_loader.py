"""
SEED-VIG 数据加载与 epoch 构建模块
=====================================
将 Raw_Data 中的原始 EEG 信号切分为 (n_epochs, n_channels, n_times) 格式，
并与 perclos_labels 对齐。

支持:
- 17通道全脑 EEG (Raw_Data)
- 4通道前额 EEG (通过 Raw_Data EOG 字段提取)
- 已提取特征 (EEG_Feature_5Bands, EEG_Feature_2Hz, EOG_Feature)
"""

import numpy as np
import scipy.io as sio
import os
from pathlib import Path


# ── 通道名称列表 ──────────────────────────────────────────────
EEG_17CH_NAMES = [
    'FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8',       # 颞区 1-6
    'CP1', 'CPZ', 'CP2', 'P1', 'PZ', 'P2',         # 后部 7-12
    'PO3', 'POZ', 'PO4', 'O1', 'OZ', 'O2'          # 后部 13-17(注: CPZ=8)
]

# 根据 Huo 2016，去掉 CPZ (靠近参考电极易短路) → 11通道
EEG_11CH_INDICES = [0,1,2,3,4,5, 6,8,9,10,11, 12,13,14,15,16]  # 实际16ch
# 准确的后部11ch索引 (排除 CPZ=7):
POSTERIOR_11CH = [0,1,2,3,4,5, 6,8,9,10,11,12,13,14,15,16]  # 全部除CPZ

# 颞区6ch
TEMPORAL_6CH = [0, 1, 2, 3, 4, 5]

# 前额4ch (Forehead EEG: Nos.4-7 → FT7, FT8, T7, T8 的近似)
FOREHEAD_4CH = [0, 1, 2, 3]

# 5频段名称
BAND_NAMES = ['delta', 'theta', 'alpha', 'beta', 'gamma']
BAND_RANGES = [(1,4), (4,8), (8,14), (14,31), (31,50)]


def list_subjects(data_root):
    """列出所有被试文件名（不含扩展名）。

    Args:
        data_root: SEED-VIG 根目录

    Returns:
        subjects: 排序后的被试 ID 列表
    """
    raw_dir = os.path.join(data_root, 'Raw_Data')
    files = sorted([f for f in os.listdir(raw_dir) if f.endswith('.mat')])
    return [f.replace('.mat', '') for f in files]


def load_raw_eeg(data_root, subject_id):
    """加载单个被试的原始 EEG 数据。

    Args:
        data_root: SEED-VIG 根目录
        subject_id: 如 '10_20151125_noon'

    Returns:
        eeg_data: (n_samples, 17) float64
        sample_rate: int
    """
    mat = sio.loadmat(os.path.join(data_root, 'Raw_Data', f'{subject_id}.mat'))
    eeg_struct = mat['EEG'][0, 0]
    # data 字段直接是 (n_samples, n_channels) 数组
    eeg_data = eeg_struct['data']
    if eeg_data.dtype == np.dtype('O'):
        eeg_data = eeg_data[0, 0]  # 兼容包装在 object array 中的情况
    sample_rate = int(eeg_struct['sample_rate'][0, 0])
    return eeg_data, sample_rate


def load_perclos(data_root, subject_id):
    """加载单个被试的 PERCLOS 标签。

    Args:
        data_root: SEED-VIG 根目录
        subject_id: 如 '10_20151125_noon'

    Returns:
        perclos: (885,) float64
    """
    mat = sio.loadmat(os.path.join(data_root, 'perclos_labels', f'{subject_id}.mat'))
    return mat['perclos'].flatten()


def load_eeg_features(data_root, subject_id, feature_dir='EEG_Feature_5Bands',
                      feature_type='de_LDS'):
    """加载已提取的 EEG 特征。

    Args:
        data_root: SEED-VIG 根目录
        subject_id: 如 '10_20151125_noon'
        feature_dir: 'EEG_Feature_5Bands' 或 'EEG_Feature_2Hz'
        feature_type: 'de_movingAve', 'de_LDS', 'psd_movingAve', 'psd_LDS'

    Returns:
        features: (n_channels, 885, n_bands) 或 (n_channels, 885, 25)
    """
    mat = sio.loadmat(os.path.join(data_root, feature_dir, f'{subject_id}.mat'))
    return mat[feature_type]  # (17, 885, n_bands)


def load_eog_features(data_root, subject_id, method='features_table_ica'):
    """加载 EOG 特征。

    Args:
        data_root: SEED-VIG 根目录
        subject_id: 如 '10_20151125_noon'
        method: 'features_table_ica', 'features_table_minus', 'features_table_icav_minh'

    Returns:
        features: (885, 36) float64
    """
    mat = sio.loadmat(os.path.join(data_root, 'EOG_Feature', f'{subject_id}.mat'))
    return mat[method]


def build_epochs_from_raw(raw_eeg, perclos, fs=200, epoch_sec=8):
    """从原始 EEG 构建 epochs。

    原始 EEG 被等分为 885 个 8 秒窗口 (每次 1600 采样点)。

    Args:
        raw_eeg: (n_samples, n_channels) 原始 EEG
        perclos: (885,) 标签
        fs: 采样率 (default 200Hz)
        epoch_sec: epoch 长度 (default 8s)

    Returns:
        X: (885, n_channels, epoch_len) float64
        y: (885,) float64
    """
    epoch_len = int(fs * epoch_sec)  # 1600
    n_epochs = perclos.shape[0]      # 885
    n_channels = raw_eeg.shape[1]

    # 安全检查
    total_needed = n_epochs * epoch_len
    assert total_needed <= raw_eeg.shape[0], \
        f"原始数据长度 {raw_eeg.shape[0]} 不足以切 {n_epochs} 个 epochs ({total_needed} 点)"

    X = np.zeros((n_epochs, n_channels, epoch_len), dtype=np.float64)
    for i in range(n_epochs):
        start = i * epoch_len
        X[i] = raw_eeg[start:start + epoch_len].T  # (ch, time)

    return X, perclos.copy()


def build_de_features_for_baseline(data_root, subject_id,
                                   channels='all',
                                   feature_type='de_LDS'):
    """构建 DE 特征矩阵用于基线对比。

    从 EEG_Feature_5Bands 加载 DE 特征，
    展平为 (885, n_channels * 5) 的二维矩阵。

    Args:
        data_root: SEED-VIG 根目录
        subject_id: 被试 ID
        channels: 'all' (17ch), 'temporal' (6ch), 'posterior' (11ch),
                  'forehead' (4ch), 或 channel indices list
        feature_type: 特征类型

    Returns:
        X: (885, n_features) float64
        y: (885,) float64
        ch_names: list of channel names used
    """
    feats = load_eeg_features(data_root, subject_id,
                              feature_dir='EEG_Feature_5Bands',
                              feature_type=feature_type)  # (17, 885, 5)

    # 选择通道
    if channels == 'all':
        ch_idx = list(range(17))
    elif channels == 'temporal':
        ch_idx = TEMPORAL_6CH
    elif channels == 'posterior':
        # 后部通道去掉 CPZ (index 7)
        ch_idx = [6, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    elif channels == 'forehead':
        ch_idx = FOREHEAD_4CH
    else:
        ch_idx = channels

    ch_names = [EEG_17CH_NAMES[i] for i in ch_idx]
    selected = feats[ch_idx]  # (n_ch, 885, 5)

    # 转置并展平: (885, n_ch, 5) → (885, n_ch * 5)
    X = selected.transpose(1, 0, 2).reshape(885, -1)
    y = load_perclos(data_root, subject_id)

    return X, y, ch_names


# ── 快速测试 ─────────────────────────────────────────────────
if __name__ == '__main__':
    data_root = r'D:\EEG\datasets\SEED-VIG'
    subjects = list_subjects(data_root)
    print(f"找到 {len(subjects)} 个被试")
    print(f"前3个: {subjects[:3]}")

    # 测试加载第一个被试
    subj = subjects[0]
    print(f"\n测试被试: {subj}")

    # Raw EEG
    raw, sr = load_raw_eeg(data_root, subj)
    print(f"Raw EEG: {raw.shape}, fs={sr}")

    # Epochs
    y = load_perclos(data_root, subj)
    X, y = build_epochs_from_raw(raw, y, fs=sr)
    print(f"Epochs: {X.shape}, labels: {y.shape}")
    print(f"PERCLOS 范围: [{y.min():.4f}, {y.max():.4f}]")

    # DE features
    X_de, y_de, chs = build_de_features_for_baseline(data_root, subj)
    print(f"DE features: {X_de.shape}, channels: {chs}")

    # EOG features
    eog = load_eog_features(data_root, subj)
    print(f"EOG features: {eog.shape}")
