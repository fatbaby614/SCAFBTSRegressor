"""
DROZY 数据集加载器
===================
EDF 格式, 512 Hz, 5 EEG 通道 (Fz,Cz,C3,C4,Pz)
标签: KSS (1-9) 或 PVT 反应时间
每测试约 10 分钟 = 600s
"""

import os
import numpy as np
import mne

# DROZY EEG 通道 (5个标准通道, 排除 Oz/Cam-Sync/PVT/EOG/EMG/ECG)
DROZY_EEG_CHANNELS = ['Fz', 'Cz', 'C3', 'C4', 'Pz']

# KSS 值 (14 subjects × 3 tests, 0 = 缺失)
KSS_VALUES = [
    [3, 6, 7],
    [3, 7, 6],
    [2, 3, 4],
    [4, 8, 9],
    [3, 7, 8],
    [2, 3, 7],
    [0, 4, 9],  # subj 7, test 1 missing
    [2, 6, 8],
    [2, 6, 8],  # subj 9, test 1 missing; tests 9-2, 9-3 present
    [3, 6, 7],
    [4, 7, 7],
    [2, 5, 6],
    [6, 3, 7],
    [5, 7, 8],
]


def list_drozy_tests(psg_dir):
    """列出所有可用的 EDF 测试文件。

    Returns:
        list of (subject_id, test_id, edf_path)
    """
    tests = []
    for fname in sorted(os.listdir(psg_dir)):
        if fname.endswith('.edf'):
            base = fname.replace('.edf', '')
            parts = base.split('-')
            if len(parts) == 2:
                subj, test = int(parts[0]), int(parts[1])
                tests.append((subj, test, os.path.join(psg_dir, fname)))
    return tests


def load_edf_eeg(edf_path, target_fs=200, epoch_sec=8):
    """加载单个 EDF 文件的 EEG 数据并切分为 epochs。

    Args:
        edf_path: EDF 文件路径
        target_fs: 目标采样率 (默认 200 Hz, 与 SEED-VIG 对齐)
        epoch_sec: epoch 长度 (默认 8s)

    Returns:
        epochs: (n_epochs, n_channels, epoch_len) float32
        raw_data: (n_times, n_channels) 降采样后的原始数据
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # 提取 EEG 通道
    eeg_indices = []
    eeg_names = []
    for ch in DROZY_EEG_CHANNELS:
        if ch in raw.ch_names:
            eeg_indices.append(raw.ch_names.index(ch))
            eeg_names.append(ch)

    if len(eeg_indices) == 0:
        raise ValueError(f"No EEG channels found in {edf_path}")

    data = raw.get_data()[eeg_indices]  # (n_ch, n_times)

    # 降采样
    if target_fs != raw.info['sfreq']:
        data_resampled = mne.filter.resample(
            data, down=raw.info['sfreq'] / target_fs, axis=-1
        )
    else:
        data_resampled = data

    n_ch = len(eeg_names)
    n_total = data_resampled.shape[1]
    epoch_len = int(target_fs * epoch_sec)
    n_epochs = n_total // epoch_len

    # 切分 epochs: (n_epochs, ch, time)
    epochs = np.zeros((n_epochs, n_ch, epoch_len), dtype=np.float32)
    for i in range(n_epochs):
        start = i * epoch_len
        epochs[i] = data_resampled[:, start:start + epoch_len]

    # raw_data 以 (n_times, n_ch) 格式返回 (兼容 SEED-VIG loader)
    raw_data = data_resampled.T.copy().astype(np.float32)

    return epochs, raw_data, eeg_names


def get_kss_label(subject_id, test_id):
    """获取 KSS 标签。

    二值化: KSS ≤ 5 → 0 (alert), KSS ≥ 6 → 1 (drowsy)
    缺失 (0) 也返回 0

    Returns:
        kss_raw: 原始 KSS 值 (1-9 或 0)
        kss_binary: 二值化标签
    """
    if subject_id < 1 or subject_id > 14:
        return 0, 0
    if test_id < 1 or test_id > 3:
        return 0, 0

    kss = KSS_VALUES[subject_id - 1][test_id - 1]
    binary = 1 if kss >= 6 else 0
    return kss, binary


def load_drozy_subject(psg_dir, subject_id, target_fs=200, epoch_sec=8):
    """加载单个被试的所有测试数据 (合并)。

    Returns:
        all_epochs: (n_total_epochs, 5, epoch_len) 或 None
        all_kss: (n_total_epochs,) KSS 标签 (每 epoch 重复同一 KSS)
        all_binary: (n_total_epochs,) 二值标签
        test_ids: (n_total_epochs,) 每 epoch 属于哪个 test
        eeg_names: channel name list
    """
    all_epochs = []
    all_kss = []
    all_binary = []
    all_test_ids = []
    eeg_names = None

    for test_id in [1, 2, 3]:
        fname = f'{subject_id}-{test_id}.edf'
        edf_path = os.path.join(psg_dir, fname)
        if not os.path.exists(edf_path):
            continue

        kss, binary = get_kss_label(subject_id, test_id)
        if kss == 0:
            continue  # 缺失测试

        try:
            epochs, _, names = load_edf_eeg(edf_path, target_fs, epoch_sec)
        except Exception as e:
            print(f"  WARNING: Failed to load {fname}: {e}")
            continue

        if eeg_names is None:
            eeg_names = names

        n_epochs = len(epochs)
        all_epochs.append(epochs)
        all_kss.extend([kss] * n_epochs)
        all_binary.extend([binary] * n_epochs)
        all_test_ids.extend([test_id] * n_epochs)

    if len(all_epochs) == 0:
        return None, None, None, None, eeg_names

    X = np.concatenate(all_epochs, axis=0)
    y_kss = np.array(all_kss, dtype=np.float32)
    y_bin = np.array(all_binary, dtype=np.int32)
    t_ids = np.array(all_test_ids, dtype=np.int32)

    return X, y_kss, y_bin, t_ids, eeg_names


def load_pvt_rt(rt_dir, subject_id, test_id, epoch_sec=8):
    """加载 PVT 反应时间并聚合为 epoch 级标签。

    PVT 文件每行: stimulus_time;reaction_time
    计算每个 8s epoch 内的平均/中位反应时间。

    Returns:
        rt_per_epoch: (n_epochs,) 每 epoch 的平均 RT
    """
    fname = f'{subject_id}-{test_id}.csv'
    path = os.path.join(rt_dir, fname)
    if not os.path.exists(path):
        return None

    rts = []
    with open(path) as f:
        header = f.readline()  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(';')
            if len(parts) == 2:
                try:
                    stim = float(parts[0])
                    react = float(parts[1])
                    rts.append(react - stim)
                except:
                    pass

    if len(rts) == 0:
        return None

    # 假设 PVT 持续约 600s, 分 epoch
    n_epochs = 600 // epoch_sec
    rt_per_epoch = np.zeros(n_epochs)
    # 按时间分桶 (简化: 等间距分配)
    rts_arr = np.array(rts)
    chunk_size = max(1, len(rts_arr) // n_epochs)
    for i in range(n_epochs):
        start = i * chunk_size
        end = min(start + chunk_size, len(rts_arr))
        if end > start:
            rt_per_epoch[i] = np.mean(rts_arr[start:end])
        else:
            rt_per_epoch[i] = rt_per_epoch[i-1] if i > 0 else 0

    return rt_per_epoch


# ── DROZY DE 特征提取 ────────────────────────────────────────

DROZY_DE_BANDS = [(1, 4), (4, 8), (8, 14), (14, 31), (31, 50)]


def extract_de_features(epochs, fs=200):
    """从 DROZY EEG epochs 提取 DE 特征。

    对每个 epoch:
      1. 5 频段 Butterworth 带通滤波
      2. 计算各通道各频段的信号方差 σ²
      3. DE = 0.5 × log(2πeσ²)

    Args:
        epochs: (n_epochs, n_channels, n_times) float32 — 8s epochs
        fs: 采样率 (default 200 Hz)

    Returns:
        X_de: (n_epochs, n_channels * 5) float64 DE 特征矩阵
    """
    from scipy import signal

    n_epochs, n_ch, n_times = epochs.shape
    n_bands = len(DROZY_DE_BANDS)

    X_de = np.zeros((n_epochs, n_ch * n_bands), dtype=np.float64)

    for i in range(n_epochs):
        epoch = epochs[i].astype(np.float64)  # (n_ch, n_times)
        for j, (low, high) in enumerate(DROZY_DE_BANDS):
            nyquist = 0.5 * fs
            b, a = signal.butter(4, [low / nyquist, high / nyquist], btype='band')
            filtered = signal.filtfilt(b, a, epoch, axis=-1)
            var = np.var(filtered, axis=-1)  # (n_ch,)
            de = 0.5 * np.log(2 * np.pi * np.e * var + 1e-10)
            X_de[i, j * n_ch:(j + 1) * n_ch] = de

    return X_de


# ── 快速测试 ─────────────────────────────────────────────────
if __name__ == '__main__':
    psg_dir = r'D:\EEG\datasets\DROZY\psg'
    rt_dir = r'D:\EEG\datasets\DROZY\pvt-rt'

    # 测试列表
    tests = list_drozy_tests(psg_dir)
    print(f"DROZY tests: {len(tests)}")
    for s, t, p in tests[:5]:
        kss, bin_ = get_kss_label(s, t)
        print(f"  subj={s}, test={t}, KSS={kss}, drowsy={bin_}")

    # 加载一个被试
    print("\n--- Loading subject 1 ---")
    X, y_kss, y_bin, t_ids, names = load_drozy_subject(psg_dir, 1)
    if X is not None:
        print(f"Epochs: {X.shape}, channels: {names}")
        print(f"KSS labels: {np.unique(y_kss)}, binary: {np.unique(y_bin)}")
        print(f"Tests: {np.unique(t_ids)}")

    # 测试 PVT RT
    print("\n--- PVT RT ---")
    rt = load_pvt_rt(rt_dir, 1, 1)
    if rt is not None:
        print(f"RT epochs: {len(rt)}, range: [{rt.min():.3f}, {rt.max():.3f}]")
