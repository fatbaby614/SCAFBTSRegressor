# 数据集详细说明

> 本文档对 SCAFBTSRegressor 项目中使用的三个 EEG 数据集进行全面技术描述，包括数据格式、目录结构、通道配置、标签类型、加载流程及已知注意事项。

---

## 目录

1. [SEED-VIG](#1-seed-vig)
2. [DROZY](#2-drozy)
3. [SEED](#3-seed)
4. [数据集对比总表](#4-数据集对比总表)

---

## 1. SEED-VIG

### 1.1 概述

| 属性 | 值 |
|---|---|
| 全称 | SJTU Emotion EEG Dataset — Vigilance |
| 任务 | 模拟驾驶连续警觉度估计 |
| 被试数 | 23（平均年龄 23.3，12 名女性） |
| 实验时长 | 约 2 小时单调驾驶任务 |
| 采样率 | **200 Hz** |
| EEG 通道 | **17 通道**（颞区 6 + 后部 11） |
| 额外模态 | 前额 EOG（7 电极）、眼动追踪 |
| 标签类型 | **PERCLOS**（眼睑闭合百分比），连续值 [0, 1] |
| 数据格式 | `.mat`（MATLAB v7） |
| 获取方式 | 需向 [BCML 实验室](http://bcmi.sjtu.edu.cn/~seed/) 申请 |
| 引用文献 | Zheng & Lu, *J. Neural Eng.*, 2017 |

### 1.2 目录结构

```
D:\EEG\datasets\SEED-VIG\
├── Raw_Data/                  # 原始 EEG 信号
│   ├── 1_20151124_noon.mat
│   ├── 2_20151124_noon.mat
│   ├── ...
│   └── 23_20151125_night.mat
│
├── perclos_labels/            # PERCLOS 标签（每被试一个文件）
│   ├── 1_20151124_noon.mat
│   ├── ...
│   └── 23_20151125_night.mat
│
├── EEG_Feature_5Bands/        # 5 频段 DE/PSD 特征（预提取）
│   ├── 1_20151124_noon.mat
│   ├── ...
│   └── 23_20151125_night.mat
│
├── EEG_Feature_2Hz/           # 25 频段特征（2Hz 分辨率）
│   ├── 1_20151124_noon.mat
│   └── ...
│
├── EOG_Feature/               # 前额 EOG 特征（预提取）
│   ├── 1_20151124_noon.mat
│   └── ...
│
└── (其他文件)
```

**被试文件命名规则**：`{subject_id}_{date}_{session}.mat`
- 示例：`10_20151125_noon` = 10 号被试，2015年11月25日，中午时段

### 1.3 原始 EEG 数据格式

#### load_raw_eeg() 加载内容

```python
mat = sio.loadmat(f'{data_root}/Raw_Data/{subject_id}.mat')
eeg_struct = mat['EEG'][0, 0]
eeg_data = eeg_struct['data']       # (n_samples, 17)  float64
sample_rate = eeg_struct['sample_rate']  # 200
```

**数据形状**：`(n_samples, 17)` — 每列为一个通道的连续时间序列。

**原始时长**：`n_samples / 200 Hz` — 约 7080 秒（约 118 分钟），不同被试可能略有差异。

**注意事项**：
- 部分 `.mat` 文件的 `data` 字段被包装在 object array 中，代码通过 `if eeg_data.dtype == np.dtype('O')` 做了兼容处理。
- `EEG_17CH_NAMES` 列表中 CPZ（索引 7）在文献中因靠近参考电极易短路被排除出后部通道子集。

### 1.4 通道配置

#### 17 通道全列表（索引 0–16）

```
颞区 (Temporal, 索引 0–5):
  0: FT7    1: FT8    2: T7     3: T8     4: TP7    5: TP8

后部 (Posterior, 索引 6–16):
  6: CP1    7: CPZ    8: CP2    9: P1    10: PZ    11: P2
 12: PO3   13: POZ   14: PO4   15: O1    16: OZ    17: O2
```

#### 通道子集定义（data_loader.py）

| 子集标识 | 索引 | 通道数 | 说明 |
|---|---|---|---|
| `all` | 0–16 | 17 | 全通道 |
| `temporal` | 0–5 | 6 | 颞区：FT7, FT8, T7, T8, TP7, TP8 |
| `posterior` | 6,8,9,10,11,12,13,14,15,16 | 10 | 后部（排除 CPZ） |
| `forehead` | 0–3 | 4 | 前额近似：FT7, FT8, T7, T8 |

**代码中的常量和变量名**：
- `TEMPORAL_6CH = [0, 1, 2, 3, 4, 5]`
- `FOREHEAD_4CH = [0, 1, 2, 3]`
- `POSTERIOR_11CH` 在代码中被误标为 16 个元素（实际包含了全部除 CPZ 的通道）

### 1.5 标签

#### PERCLOS（Percentage of Eye Closure）

- 来源：SMI 眼动追踪眼镜
- 范围：`[0, 1]`，连续值
- 形状：`(885,)` — 每个被试固定 885 个 epoch
- 语义：0 = 眼睛完全睁开（高警觉），1 = 眼睛完全闭合（低警觉/疲劳）

**加载方式**：
```python
mat = sio.loadmat(f'{data_root}/perclos_labels/{subject_id}.mat')
perclos = mat['perclos'].flatten()  # (885,)  float64
```

### 1.6 Epoch 构建

```
build_epochs_from_raw(raw_eeg, perclos, fs=200, epoch_sec=8)
```

| 参数 | 值 |
|---|---|
| epoch 长度 | **8 秒** |
| 采样点数/epoch | **1600** (200 × 8) |
| Epoch 总数 | **885** |
| 输出形状 | `(885, n_channels, 1600)` float64 |
| 切分方式 | 等距连续切分（非重叠 Hamming 窗） |

**对齐关系**：原始 EEG 从头开始等分 885 段，每段 8 秒。标签数组也是 885 个 PERCLOS 值，按顺序一一对应。

**安全断言**：代码检查 `total_needed <= raw_eeg.shape[0]`，确保数据长度足够切分。

### 1.7 预提取特征

#### EEG 特征：EEG_Feature_5Bands

```python
load_eeg_features(data_root, subject_id,
                  feature_dir='EEG_Feature_5Bands',
                  feature_type='de_LDS')
# 返回：(17, 885, 5)  float64
```

| 维度 | 含义 |
|---|---|
| 轴 0（17） | 通道 |
| 轴 1（885） | epoch |
| 轴 2（5） | 频段：delta(1–4), theta(4–8), alpha(8–14), beta(14–31), gamma(31–50) |

**可用特征类型**：

| feature_type | 说明 |
|---|---|
| `de_LDS` | 差分熵 + LDS 平滑（**主要基线**） |
| `de_movingAve` | 差分熵 + 滑动平均 |
| `psd_LDS` | 功率谱密度 + LDS 平滑 |
| `psd_movingAve` | 功率谱密度 + 滑动平均 |

**提取方法**（据论文）：短时傅里叶变换（STFT），8 秒非重叠汉明窗。

#### EEG 特征：EEG_Feature_2Hz

- 同样格式 `(17, 885, 25)`，25 个 2Hz 分辨率频段（0–50 Hz）。
- 特征类型同上四种。

#### EOG 特征：EOG_Feature

```python
load_eog_features(data_root, subject_id, method='features_table_ica')
# 返回：(885, 36)  float64
```

| method 参数 | 说明 |
|---|---|
| `features_table_ica` | ICA 分离的眨眼/扫视特征（**主用，36 维**） |
| `features_table_minus` | 差分方法提取 |
| `features_table_icav_minh` | ICA 变体 |

EOG 特征来源：7 个前额电极，小波变换检测眨眼和扫视。

#### DE 特征基线矩阵构建

`build_de_features_for_baseline()` 将预提取特征展平为 2D 矩阵：

```
(17, 885, 5) → 选择通道子集 → 转置 → (885, n_ch * 5)
```

- 17ch 全通道 DE：`(885, 85)` — 85 维特征向量
- 6ch 颞区 DE：`(885, 30)` — 30 维
- 4ch 前额 DE：`(885, 20)` — 20 维

### 1.8 在项目中的使用

| 实验脚本 | 用途 |
|---|---|
| `run_all.py` | 主实验：DE 基线 + Riemannian FBTS（5/8 band，17/6/4 ch） |
| `run_final.py` | 全量 23 被试并行，DE + Riemannian 所有配置 |
| `run_fusion.py` | 单/双/三模态融合实验 |
| `run_loso.py` | Leave-One-Subject-Out 跨被试泛化 |
| `generate_figures.py` | 论文图表生成 |

### 1.9 评测协议

- **交叉验证**：5-fold 时序保持（不能打乱顺序）
- **划分方式**：885 epoch 按时序 5 等分，每份 177 epoch
- **训练/测试**：4 份训练（708 epoch），1 份测试（177 epoch）
- **指标**：COR（Pearson 相关系数，主要指标）、RMSE、MAE
- **显著性检验**：配对双尾 t-test（df=22），α=0.05

---

## 2. DROZY

### 2.1 概述

| 属性 | 值 |
|---|---|
| 全称 | ULg Multimodality Drowsiness Database |
| 任务 | 精神运动警觉性测试（PVT）嗜睡检测 |
| 被试数 | **14** |
| 测试数/被试 | **3** 次（不同日期/时间） |
| 实验时长 | 每次测试约 10 分钟 |
| 原始采样率 | **512 Hz** → 项目降采样至 **200 Hz** |
| EEG 通道 | **5**（Fz, Cz, C3, C4, Pz） |
| 额外模态 | EOG, EMG, ECG, 视频（项目未使用） |
| 标签类型 | **KSS**（Karolinska Sleepiness Scale, 1–9）+ PVT 反应时间 |
| 数据格式 | `.edf`（EEG/PSG）+ `.csv`（PVT 反应时间） |
| 获取方式 | 需签署许可协议 |
| 引用文献 | Massoz et al., *IEEE WACV*, 2016 |

### 2.2 目录结构

```
D:\EEG\datasets\DROZY\
├── psg/                       # 多导睡眠图（EDF 文件）
│   ├── 1-1.edf               # 被试1, 测试1
│   ├── 1-2.edf
│   ├── 1-3.edf
│   ├── 2-1.edf
│   ├── ...
│   └── 14-3.edf
│
├── pvt-rt/                    # PVT 反应时间（CSV 文件）
│   ├── 1-1.csv
│   ├── 1-2.csv
│   ├── ...
│   └── 14-3.csv
│
└── (其他原始文件，项目未使用)
```

**文件命名规则**：`{subject_id}-{test_id}.edf` / `{subject_id}-{test_id}.csv`

### 2.3 EEG 数据格式

#### EDF 文件内容

EDF 文件包含多模态信号，项目仅提取 5 个 EEG 通道：

```python
DROZY_EEG_CHANNELS = ['Fz', 'Cz', 'C3', 'C4', 'Pz']
```

**排除的通道**：Oz, Cam-Sync, PVT, EOG, EMG, ECG（这些信号存在于 EDF 中但未被加载）。

#### 加载流程（load_edf_eeg）

```
EDF 文件 (512 Hz)
  ↓ mne.io.read_raw_edf(preload=True)
  ↓ 提取 5 个 EEG 通道 → (5, n_times)
  ↓ mne.filter.resample(down=512/200) → 降采样至 200 Hz
  ↓ 切分为 8s epochs → (n_epochs, 5, 1600)
```

**关键参数**：
- 目标采样率：200 Hz（与 SEED-VIG 对齐）
- epoch 长度：8 秒
- 数据类型：float32（不同于 SEED-VIG 的 float64）

### 2.4 标签

#### KSS（Karolinska Sleepiness Scale）

KSS 是 **per-test** 标签（不是 per-epoch），即每次 10 分钟测试只有一个 KSS 值。

| KSS 值 | 含义 |
|---|---|
| 1 | 极度警觉 |
| 3 | 警觉 |
| 5 | 既不警觉也不困倦 |
| 7 | 困倦，但无需努力保持清醒 |
| 9 | 极度困倦，努力对抗睡眠 |

**二值化规则**（代码中）：
- KSS ≤ 5 → `0`（alert / 警觉）
- KSS ≥ 6 → `1`（drowsy / 嗜睡）
- KSS = 0 → 缺失测试，跳过

**硬编码的 KSS 值**（drozy_loader.py 中 `KSS_VALUES`）：

```
被试: test1, test2, test3
  1:  3, 6, 7
  2:  3, 7, 6
  3:  2, 3, 4
  4:  4, 8, 9
  5:  3, 7, 8
  6:  2, 3, 7
  7:  0, 4, 9    ← test1 缺失
  8:  2, 6, 8
  9:  2, 6, 8    ← test1 缺失 (代码注释说 test 9-1 missing)
 10:  3, 6, 7
 11:  4, 7, 7
 12:  2, 5, 6
 13:  6, 3, 7
 14:  5, 7, 8
```

**KSS 标签分布特征**：
- 被试间跨度：KSS 2–9
- 被试 3 始终处于低嗜睡状态（2,3,4）
- 被试 4 包含极高嗜睡状态（8,9）
- 被试 7 和 9 各有一个测试缺失

#### PVT 反应时间

```python
load_pvt_rt(rt_dir, subject_id, test_id, epoch_sec=8)
```

- 数据格式：CSV，每行 `stimulus_time;reaction_time`
- 聚合方式：按 epoch 等间距分桶，每 epoch 取平均反应时间
- PVT RT 在项目中**未实际用于主实验**，仅为可选的另一种标签模式

### 2.5 标签与 epoch 的对应关系

**关键设计差异**：与 SEED-VIG 不同，DROZY 的 KSS 是 per-test（每次 10 分钟测试仅一个值），但数据被切成了多个 8 秒 epoch。

```
一次测试 (约 10 分钟 = 600s) → ~75 个 epoch（每个 8s）
所有 75 个 epoch 共享同一个 KSS 标签
```

加载一个被试时（`load_drozy_subject`）：
- 合并被试的所有可用 test 的 epoch
- 每个 epoch 复制对应 test 的 KSS 值
- 额外返回 `test_ids` 数组标记每个 epoch 属于哪个 test

**举例**（被试 1，3 个 test 都可用）：
- test1 (KSS=3): 75 epoch，标签全为 3
- test2 (KSS=6): 75 epoch，标签全为 6
- test3 (KSS=7): 75 epoch，标签全为 7
- 总计：225 epoch，KSS 标签含 3 个不同值

### 2.6 Leave-One-Test-Out (LOTO) 评测

由于 KSS 是 per-test 标签，DROZY 不能像 SEED-VIG 那样做 5-fold epoch 级交叉验证。项目采用了专门的 **LOTO** 策略：

```
对每个被试:
  for 每个 test 作为 held-out:
    用其余 test 的 epoch 训练 SVR 模型
    对 held-out test 的所有 epoch 预测
    取预测均值作为该 test 的预测 KSS
    与真实 KSS 对比
```

**评测指标**：
- **回归**：COR、RMSE（基于 per-test 的 KSS 预测 vs 真实）
- **分类**：将预测均值二值化（阈值 5.5），计算 Accuracy、F1

**限制**：
- 大多数被试只有 3 个 test，LOTO 仅产生 3 个评测点，统计稳定性有限
- 如果被试只有 1–2 个 test（如被试 7、9），无法进行 LOTO

### 2.7 在项目中的使用

| 实验脚本 | 用途 |
|---|---|
| `run_all.py` | DROZY FBTS LOTO 实验（5band/8band） |
| 状态 | DE 基线在 DROZY 上标记为 "DE-N/A"——未实现 |

**DROZY 在项目中的定位**：次要验证数据集。论文中提到了 DROZY 的挑战（稀疏 per-test 标签），将其列为未来工作方向。

---

## 3. SEED

### 3.1 概述

| 属性 | 值 |
|---|---|
| 全称 | SJTU Emotion EEG Dataset |
| 任务 | 情绪识别（三分类） |
| 被试数 | **15** |
| Session 数 | 每被试 **3** 个 session（共 45 个 session） |
| 每 session 刺激 | **15** 个电影片段（trial） |
| 采样率 | **200 Hz** |
| EEG 通道 | **62** 通道（全脑 ESI NeuroScan 系统） |
| 标签类型 | 情绪类别：**positive (+1) / neutral (0) / negative (−1)** |
| 数据格式 | `.mat`（MATLAB v7） |
| 获取方式 | 需向 [BCML 实验室](http://bcmi.sjtu.edu.cn/~seed/) 申请 |
| 引用文献 | Zheng & Lu, *IEEE Trans. Auton. Ment. Dev.*, 2015 |

### 3.2 目录结构

```
D:\EEG\datasets\SEED\
├── ExtractedFeatures/         # 预提取的 DE/PSD 特征（62ch, 5频段）
│   ├── 1_20131027.mat
│   ├── 1_20131030.mat
│   ├── 1_20131107.mat
│   ├── ...
│   ├── 15_20140112.mat
│   ├── label.mat             # 情绪标签定义
│   └── readme.txt
│
├── Preprocessed_EEG/          # 预处理后的原始 EEG
│   ├── 1_20131027.mat
│   ├── ...
│   └── 15_20140112.mat
│
├── subject-id-gender-seed.txt # 被试人口学信息
│
└── (其他文件)
```

**文件命名规则**：`{subject_id}_{date}.mat`
- 每被试 3 个文件，对应 3 个不同日期的 session

### 3.3 情绪标签

#### Trial 标签（15 个 trial 的固定序列）

```python
TRIAL_LABELS = np.array([1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1])
```

| 标签值 | 含义 | Trial 编号 |
|---|---|---|
| +1 | 正性情绪 | 1, 6, 9, 10, 14 |
| 0 | 中性情绪 | 2, 5, 8, 11, 13 |
| −1 | 负性情绪 | 3, 4, 7, 12, 15 |

每个情绪类别恰好 5 个 trial，三类均衡。

**注意**：TRIAL_LABELS 是硬编码在 `run_seed.py` 中的常量。论文/数据集说明中可能定义了不同的标签映射，此处的 ±1/0 映射为项目内部约定。

### 3.4 特征数据格式

#### 预提取 DE/PSD 特征（ExtractedFeatures）

每个 session 的 `.mat` 文件包含 15 个 trial 的特征：

```python
data = sio.loadmat(filepath)
# 键名: 'de_LDS1', 'de_LDS2', ..., 'de_LDS15'
# 每个: (62, n_segments, 5)  float64
```

| 维度 | 含义 |
|---|---|
| 轴 0（62） | 通道 |
| 轴 1（n_segments） | 时间片段（4 秒非重叠窗） |
| 轴 2（5） | 频段：delta(1–4), theta(4–8), alpha(8–14), beta(14–31), gamma(31–50) |

**加载后展平**（load_seed_de_features）：
```
(62, n_segments, 5) → transpose → (n_segments, 62, 5) → reshape → (n_segments, 310)
```

每个 4 秒片段对应一个 310 维特征向量（62 通道 × 5 频段）。

#### 预处理 EEG（Preprocessed_EEG）

```python
data = sio.loadmat(eeg_path)
# 键名: 'ww_eeg1', 'ww_eeg2', ..., 'ww_eeg15'
# 每个: (62, n_times)  float64
```

- 每个 trial 的 EEG 数据为 `(62, n_times)` 连续信号
- epoch 长度：**4 秒**（与 SEED-VIG 的 8 秒不同）
- epoch 切分：`n_times // (fs * 4)` 个 epoch

### 3.5 交叉验证策略

**Session 内 Trial 级 5-fold CV**（与 SEED-VIG 的 epoch 级 CV 不同）：

```
对每个 session:
  将 15 个 trial 分为 5 折（每折 3 个 trial）
  按 trial 归属划分训练/测试（同一 trial 的所有片段不能跨折）
  4 折训练（12 trial）→ 1 折测试（3 trial）
```

这样设计是为了**避免数据泄露**：同一 trial 内的连续时间片段高度相关，如果随机打散到不同折会造成过拟合估计。

### 3.6 通道信息

SEED 使用 62 通道 ESI NeuroScan 系统。项目代码中未硬编码完整通道名称列表，仅在 `run_seed_fast.py` 中定义了一个 10 通道认知相关子集：

```python
CH_10 = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ']
```

这 10 个通道覆盖前额叶和额叶区域，用于降通道快速实验。全局 62 通道的完整映射需要参考原始数据集文档。

### 3.7 在项目中的使用

| 实验脚本 | 用途 | 状态 |
|---|---|---|
| `run_seed.py` | 完整流程：DE/PSD 基线 + Riemannian FBTS（62ch 计算量较大） | ⬜ 计算受限 |
| `run_seed_fast.py` | 快速版：DE/PSD 基线（62ch），供交叉验证参考 | ✅ 已运行 |
| `run_seed_fbts.py` | 快速版：Riemannian FBTS（10ch 前额），5/8band 对比 | ✅ 已运行 |
| `_explore_seed.py` | 数据集探索脚本，打印目录结构和文件内容 | — |

**SEED 在项目中的定位**：跨任务验证数据集。论文中的核心结论为：
- DE + SVC 在 SEED 上达到 **86.1%** 三分类准确率（F1=0.846），证明 DE 跨任务泛化能力
- FBTS 10ch 前额达到 **55.8%**（5band）和 **52.9%**（8band）——低于 DE 基线，但这是 10ch vs 62ch 的降维结果
- 完整 62ch Riemannian FBTS 因计算资源限制被标注为未来工作

### 3.8 计算资源注意事项

- 62 通道 × 5 频段 = 每 epoch 的 SPD 矩阵为 `(62, 62)`，切空间特征维度为 62×63/2 = **1953 维/频段**
- 5 频段拼接后 = **9765 维**（8 频段 = 15624 维）
- 45 个 session，每个 session 数百个 epoch → 全量计算对内存和算力要求极高

---

## 4. 数据集对比总表

| 维度 | SEED-VIG | DROZY | SEED |
|---|---|---|---|
| **任务** | 驾驶警觉度回归 | PVT 嗜睡检测 | 情绪识别分类 |
| **被试数** | 23 | 14 | 15 |
| **Session/测试** | 每被试 1 次（2h） | 每被试 3 次（各 10min） | 每被试 3 session |
| **EEG 通道** | 17 | 5 | 62 |
| **采样率** | 200 Hz | 512→200 Hz | 200 Hz |
| **Epoch 长度** | 8 秒（1600 点） | 8 秒（1600 点） | 4 秒（800 点） |
| **Epoch 数/单元** | 885（固定） | ~75/test | 因 trial 而异 |
| **标签类型** | PERCLOS 连续 [0,1] | KSS 离散 1–9 | 情绪类别 ±1/0 |
| **标签粒度** | per-epoch | per-test | per-trial |
| **额外模态** | EOG（36维特征），眼动追踪 | EOG, EMG, ECG, 视频 | 无 |
| **数据格式** | .mat | .edf + .csv | .mat |
| **预提取特征** | DE/PSD（5频段，2Hz），EOG | 无 | DE/PSD（5频段） |
| **CV 策略** | 5-fold 时序 | LOTO（per-test） | Trial 级 5-fold |
| **主要指标** | COR, RMSE | COR, RMSE, ACC, F1 | ACC, F1 |
| **项目状态** | ✅ 主要实验 | ✅ 次要验证 | ⬜ 部分完成 |
| **Riemannian FBTS** | ✅ 完整运行 | ✅ 完整运行 | ✅ 10ch 前额已运行 |
| **DE 基线** | ✅ 完整运行 | ❌ 未实现 | ✅ 完整运行 |

---

## 附录 A：项目中的数据路径常量

> **v3.1.0 更新**：所有路径已集中到 `config.py`，自动适配操作系统（Linux/Windows）。修改路径只需编辑 `config.py` 一个文件。

### config.py 中的路径定义

```python
# Linux 工作站
SEED_VIG_ROOT = '/mnt/data1/home/tanhuang/datasets/SEED-VIG'
DROZY_ROOT    = '/mnt/data1/home/tanhuang/datasets/DROZY'
SEED_ROOT     = '/mnt/data1/home/tanhuang/datasets/SEED'

# Windows (自动回退)
SEED_VIG_ROOT = r'D:\EEG\datasets\SEED-VIG'
DROZY_ROOT    = r'D:\EEG\datasets\DROZY'
SEED_ROOT     = r'D:\EEG\datasets\SEED'
```

### 路径对照表

| 数据集 | 子目录 | Windows | Linux |
|---|---|---|---|
| SEED-VIG | Raw_Data/ | `D:\EEG\datasets\SEED-VIG\Raw_Data\` | `/mnt/data1/home/tanhuang/datasets/SEED-VIG/Raw_Data/` |
| SEED-VIG | perclos_labels/ | `D:\EEG\datasets\SEED-VIG\perclos_labels\` | `/mnt/data1/home/tanhuang/datasets/SEED-VIG/perclos_labels/` |
| SEED-VIG | EEG_Feature_5Bands/ | `D:\EEG\datasets\SEED-VIG\EEG_Feature_5Bands\` | `/mnt/data1/home/tanhuang/datasets/SEED-VIG/EEG_Feature_5Bands/` |
| SEED-VIG | EOG_Feature/ | `D:\EEG\datasets\SEED-VIG\EOG_Feature\` | `/mnt/data1/home/tanhuang/datasets/SEED-VIG/EOG_Feature/` |
| DROZY | psg/ | `D:\EEG\datasets\DROZY\psg\` | `/mnt/data1/home/tanhuang/datasets/DROZY/psg/` |
| DROZY | pvt-rt/ | `D:\EEG\datasets\DROZY\pvt-rt\` | `/mnt/data1/home/tanhuang/datasets/DROZY/pvt-rt/` |
| SEED | ExtractedFeatures/ | `D:\EEG\datasets\SEED\ExtractedFeatures\` | `/mnt/data1/home/tanhuang/datasets/SEED/ExtractedFeatures/` |
| SEED | Preprocessed_EEG/ | `D:\EEG\datasets\SEED\Preprocessed_EEG\` | `/mnt/data1/home/tanhuang/datasets/SEED/Preprocessed_EEG/` |

## 附录 B：数据集获取

| 数据集 | 获取地址 |
|---|---|
| SEED-VIG | <http://bcmi.sjtu.edu.cn/~seed/> |
| SEED | <http://bcmi.sjtu.edu.cn/~seed/> |
| DROZY | <http://www.drozy.ulg.ac.be/>（需联系作者签署许可协议） |

SEED-VIG 和 SEED 由上海交通大学 BCML 实验室维护，需在线填写申请表。DROZY 由列日大学（ULg）维护。
