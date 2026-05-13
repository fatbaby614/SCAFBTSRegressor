# SCAFBTSRegressor 项目深度理解文档

> 生成日期：2026-05-12
> 基于对全部源代码、论文草稿、数据文档和实验结果的完整阅读

---

## 一、项目概览

### 1.1 一句话定义

**将黎曼流形上的 Filter Bank Tangent Space（FBTS）特征首次系统性应用于 EEG 连续警觉度回归，发现在多模态融合和跨被试泛化场景下显著优于传统差分熵（DE）特征。**

### 1.2 核心信息

| 属性 | 值 |
|---|---|
| 投稿目标 | *Journal of Neural Engineering* (JNE), IOP Publishing |
| 论文 | 12 页 · 5 张表格 · 3 张图（正文）+ 补充图 · 32 篇参考文献 |
| 作者 | Xiangzhu Li, Huang Tan, Li Zhang |
| 数据集 | SEED-VIG（主要）、DROZY（验证）、SEED（跨任务） |
| 核心结果 | FBTS+EOG > DE+EOG：COR 0.618 vs 0.586, p=0.044, d=0.45 |

### 1.3 工程问题

主流 EEG 警觉度估计使用**差分熵（DE）**特征 + SVR。DE 的致命缺陷是**逐通道独立计算**——对每个通道分别求带通滤波后的信号方差然后取对数，完全不考虑通道间的协方差结构。然而，从警觉到嗜睡的神经生理转换（丘脑-皮层失联、α 波前移、半球间相干性改变）恰恰表现为**通道间功能连接的重组**。FBTS 通过 SPD 协方差矩阵天然编码这种连接组学信息，填补了这一鸿沟。

---

## 二、核心算法管线

### 2.1 五步管线

```
Step 1          Step 2              Step 3            Step 4          Step 5
原始EEG  →  Butterworth滤波器组  →  SPD协方差矩阵  →  黎曼切空间投影  →  特征选择  →  SVR回归
           4阶零相位               OAS收缩估计        仿射不变度量      f_regression  rbf核
```

### 2.2 各步骤详解

**Step 1 — 带通滤波器组**

- 实现：`scipy.signal.butter(4, [low, high], btype='band')` + `filtfilt`（零相位）
- 频段配置：5band（δ/θ/α/β/γ）、8band（细分 α/β）、25band（2Hz 分辨率）、fine（1Hz 分辨率）
- 关键实现细节：`apply_bandpass_filter()` 在 axis=-1 上滤波，支持 `(n_channels, n_times)` 和 `(n_times,)` 两种输入

**Step 2 — SPD 协方差矩阵**

- 实现：`pyriemann.estimation.Covariances(estimator='oas')`
- OAS（Oracle Approximating Shrinkage）收缩估计：在样本协方差和单位矩阵之间做最优收缩，比 Ledoit-Wolf 在小样本下更稳健
- 矩阵维度：`(n_epochs, n_channels, n_channels)`，每矩阵对称正定
- 神经生理含义：**每个 SPD 矩阵编码一个 epoch 内所有通道对之间的瞬时功能连接强度**

**Step 3 — 黎曼切空间投影**

- 实现：`pyriemann.tangentspace.TangentSpace(metric='riemann')`
- 数学本质：对数映射 `Log_P(C) = P^{1/2} log(P^{-1/2} C P^{-1/2}) P^{1/2}`（以参考点 P 为中心）
- 参考点 P 取全体训练样本的黎曼均值（仿射不变度量下的 Fréchet 平均）
- 输出维度：每频段 `C(C+1)/2`（SPD 矩阵的上三角向量化）
- 度量选项：`riemann`（仿射不变）、`euclid`（欧氏空间作为消融对照）、`logeuclid`、`stein`

**Step 4 — 特征选择**

- 实现：`sklearn.feature_selection.SelectKBest(score_func=f_regression, k=100)`
- 回归任务用 `f_regression`（ANOVA F 值），分类任务用 `f_classif`
- 默认 k=100：从 765 维（17ch×5band）中选前 100 维

**Step 5 — 回归器**

- 默认：`SVR(kernel='rbf', C=1.0, gamma='scale')`
- 可选：`Ridge(alpha=1.0)` / `RidgeCV` / `RandomForestRegressor(n_estimators=100)`
- 预测后做时序平滑：`uniform_filter1d(y_pred, size=3)`（滑动平均）

### 2.3 特征维度公式

```
D = B × C×(C+1)/2

B = 频段数, C = 通道数

示例:
  17ch × 5band  = 5 × 153  = 765 维
   6ch × 5band  = 5 × 21   = 105 维
   4ch × 5band  = 5 × 10   =  50 维
  62ch × 5band  = 5 × 1953 = 9765 维
  17ch × 8band  = 8 × 153  = 1224 维
```

---

## 三、三层架构设计（核心工程创新）

### 3.1 版本对比

| 版本 | 文件 | 加速策略 | 瓶颈 | 使用场景 |
|---|---|---|---|---|
| **基础版** | `sca_fbts_regressor.py` | 无 | 每 fold 重复滤波+协方差 | 参考实现、小样本调试 |
| **Fast 版** ★ | `sca_fbts_fast.py` | 全量数据一次滤波+协方差存入缓存 | 切空间+特征选择仍按 fold | 主力实验（6.5× 加速） |
| **Torch 版** | `sca_fbts_torch.py` | `torch.bmm` 批量协方差（GPU 20~50×） | 切空间仍需 pyriemann（CPU） | 协方差密集场景 |

### 3.2 Fast 版核心设计

```python
# 预计算阶段（一次）
clf.precompute(raw_eeg, fs=200)
    → 对每个频段: 全量滤波 → 切 885 个 epoch → 批量协方差
    → 存入 self._epochs_cov[(low, high)]  # (885, ch, ch)

# 训练（每 fold 一次）
clf.fit(train_indices, y[train_indices])
    → 从缓存取 covs[train_indices] → 切空间 → 特征选择 → SVR.fit()

# 预测（每 fold 一次）
clf.predict(test_indices)
    → 从缓存取 covs[test_indices] → 切空间.transform() → 预测
```

**关键洞察**：滤波和协方差计算占总时间的 80% 以上，但它们在 fold 间完全独立。预计算一次后，5-fold CV 仅需 5 次切空间+回归，消除了 4/5 的重复计算。

### 3.3 输入格式自动检测

Fast 版的 `precompute()` 自动检测两种输入：

- **2D `(n_samples, n_channels)`**：原始连续 EEG，内部按 `fs × 8s` 切分 epoch
- **3D `(n_epochs, n_channels, n_times)`**：已切好的 epochs，逐 epoch 滤波

这使得同一 API 可以无缝支持 SEED-VIG 的原始 EEG 和 DROZY 的预切分 epochs。

---

## 四、数据集全景

### 4.1 三数据集对比

| 维度 | SEED-VIG | DROZY | SEED |
|---|---|---|---|
| **定位** | ★ 主实验 | 次要验证 | 跨任务泛化 |
| **任务** | 模拟驾驶警觉度回归 | PVT 嗜睡检测 | 情绪识别分类 |
| **被试数** | 23 | 14 | 15 × 3 sessions |
| **EEG 通道** | 17（颞 6 + 后 11） | 5（Fz, Cz, C3, C4, Pz） | 62（NeuroScan） |
| **采样率** | 200 Hz | 200 Hz（512→200） | 200 Hz |
| **Epoch 长度** | 8 秒（1600 点） | 8 秒（1600 点） | 4 秒（800 点） |
| **Epoch 数/单元** | 885（固定，等距切分） | ~75/test | 因 trial 可变 |
| **标签类型** | PERCLOS [0, 1] 连续 | KSS 1–9 离散 per-test | 情绪 ±1/0 |
| **标签粒度** | **per-epoch**（885 个） | **per-test**（每 10 分钟测试一个值） | per-trial（15 个 trial） |
| **额外模态** | 前额 EOG（36 维 ICA 特征） | EOG, EMG, ECG, 视频 | 无 |
| **预提取特征** | DE/PSD（5band/25band）+ EOG | 无，需自提取 | DE/PSD（5band） |
| **数据格式** | `.mat`（MATLAB v7） | `.edf` + `.csv` | `.mat` |
| **CV 策略** | 5-fold 时序（epoch 级） | LOTO（per-test） | Trial 级 5-fold |

### 4.2 SEED-VIG 通道配置

```
颞区 (Temporal, 0–5):
  FT7(0), FT8(1), T7(2), T8(3), TP7(4), TP8(5)

后部 (Posterior, 6–16):
  CP1(6), CPZ(7), CP2(8), P1(9), PZ(10), P2(11),
  PO3(12), POZ(13), PO4(14), O1(15), OZ(16)

通道子集:
  all      → 0–16 (17ch)
  temporal → 0–5  (6ch,  颞区)
  forehead → 0–3  (4ch,  前额近似)
  posterior→ 6,8,9,10,11,12,13,14,15,16 (10ch, 排除 CPZ)
```

**注意**：CPZ（索引 7）因靠近参考电极易短路，在后部子集中被排除。

### 4.3 DROZY 特殊性

DROZY 的标签粒度是 **per-test**（每个 10 分钟测试一个 KSS 值），但 epochs 是 8 秒粒度（每 test 约 75 个 epochs）。这意味着模型被要求用 8 秒的 EEG 片段预测一个 10 分钟级的标签——这种粒度不匹配导致所有方法 COR 均为强负值（−0.89 到 −0.95），论文中将其作为方法局限性的坦诚验证。

### 4.4 SEED 通道选择

SEED 的 62 通道全量 Riemannian FBTS 因计算资源限制（9765 维/epoch）未完整运行。快速版（`run_seed_fbts.py`）仅用前 10 个前额/额叶通道：

```
CH_10 = [FP1, FPZ, FP2, AF3, AF4, F7, F5, F3, F1, FZ]  # 索引 0–9
```

---

## 五、实验设计详解

### 5.1 实验矩阵

| # | 实验 | 脚本 | CV | 方法 | 指标 | 论文位置 |
|---|---|---|---|---|---|---|
| 1 | 被试内回归 | `run_final.py` | 5-fold 时序 | DE/PSD/Riemannian × 通道 × 频段 | COR, RMSE, MAE | Table 1 |
| 2 | 多模态融合 | `run_fusion.py` | 5-fold 时序 | ~40 配置：单/双/三模态 | COR, RMSE | Table 2 |
| 3 | 黎曼度量消融 | `run_fusion.py` | 5-fold 时序 | riemann vs euclid | COR, RMSE | Table 3 |
| 4 | LOSO 跨被试 | `run_loso.py` | 22→1 | DE vs Riemannian + EOG | COR, RMSE | Table 4 |
| 5 | PERCLOS 二值分类 | `run_binary.py` | 5-fold 时序 | FBTS+EOG, 排除中段 0.4–0.6 | ACC, F1, BAC | — |
| 6 | DROZY LOTO | `run_all.py --dataset DROZY` | 留一测试 | DE + FBTS | COR, ACC, F1 | Table 5 |
| 7 | SEED 跨任务 | `run_seed_fbts.py` / `run_seed_fast.py` | Trial 级 5-fold | DE/PSD 62ch + FBTS 10ch | ACC, F1 | — |

### 5.2 交叉验证策略差异

- **SEED-VIG 5-fold 时序**：885 个 epoch 按时序分为 5 段（每段 177），1 段测试 4 段训练。关键：**不打乱时序**，因为连续 epoch 高度相关。
- **SEED Trial 级 CV**：15 个 trial 分 5 折（每折 3 个 trial）。关键：同一 trial 的连续片段**不能跨折**，否则数据泄露。
- **DROZY LOTO**：Leave-One-Test-Out。3 个 10 分钟测试，每次用 2 个训练 → 1 个测试。

### 5.3 融合管线

```
       DE特征    Riemannian特征    EOG特征
         │            │             │
         └────────────┼─────────────┘
                      │
              np.hstack 拼接
                      │
              SelectKBest (f_regression)
                      │
              StandardScaler
                      │
                  SVR(rbf)
```

- 单模态：DE / PSD / Riemannian 单独
- 双模态：DE+EOG / FBTS+EOG / DE+FBTS
- 三模态：DE+FBTS+EOG
- 特征选择：双模态 k=150–200，三模态 k=200
- **每 fold 独立做特征选择**，避免 fold 间信息泄露

---

## 六、核心发现与统计证据

### 6.1 主要结果

| # | 发现 | 数值 | 统计 | 意义 |
|---|---|---|---|---|
| 1 | 单模态 FBTS ≈ DE | COR 0.520 vs 0.521 | p=0.984 | 协方差信息在单模态不体现优势 |
| 2 | **FBTS+EOG > DE+EOG** ★ | COR 0.618 vs 0.586 | **p=0.044**, d=0.45 | **核心结果**：融合场景下连接组学信息被释放 |
| 3 | 三模态融合冗余 | 0.611 vs 0.618 | p=0.446 | DE 加入无额外增益 |
| 4 | LOSO 6ch FBTS 超被试内 DE | COR 0.623 vs 0.521 | — | 跨被试泛化是 FBTS 的强项 |
| 5 | 4ch 前额 ≈ 17ch（99%） | COR 0.613 vs 0.618 | — | 可穿戴 EEG 前景 |
| 6 | 二值分类极强 | ACC=0.993, F1=0.981 | — | 极端的 alert/drowsy 区分 |
| 7 | DROZY 全部负 COR | −0.89~−0.95 | — | 稀疏标签不适配的坦诚验证 |
| 8 | SEED 跨任务 86.1% | ACC=0.861, F1=0.846 | — | 跨任务泛化验证 |

### 6.2 统计方法

`generate_figures.py` 中的 `compute_statistical_tests()`：

- **配对 t 检验**（`scipy.stats.ttest_rel`）：比较同一被试的两种方法
- **Cohen's d** 效应量：`mean(diff) / std(diff, ddof=1)`
- **Wilcoxon 符号秩检验**（非参数，`scipy.stats.wilcoxon`）：作为 t 检验的补充
- 显著性标注：`***` (p<0.001), `**` (p<0.01), `*` (p<0.05), `n.s.` (不显著)

### 6.3 论文图表

| 图 | 内容 | 数据来源 |
|---|---|---|
| Fig 1 | 最佳/中位/最差 3 被试预测 vs 真实 PERCLOS 时间曲线 | FBTS+EOG 5-fold |
| Fig 2 | 全局散点图（23 被试 × 885 = 20,355 点） | FBTS+EOG 5-fold |
| Fig 3 | 特征维度消融（n_features 对 COR 影响） | 5 被试 × 10 个 n_features |
| 热力图 | 通道×频段 DE 特征：alert vs drowsy 差异 | 23 被试平均 DE |
| 统计检验.txt | 4 组配对比较的 t-test + Cohen's d + Wilcoxon | 融合结果 JSON |

---

## 七、代码组织与依赖

### 7.1 核心依赖

```
numpy, scipy         — 数值计算、信号处理（滤波）
scikit-learn         — SVR/SVC/Ridge/SelectKBest/StandardScaler
pyriemann            — Covariances（SPD 估计）+ TangentSpace（黎曼切空间投影）
mne                  — DROZY 的 EDF 读取 + 降采样
joblib               — 多被试并行
matplotlib           — 论文图表
torch (可选)         — sca_fbts_torch.py 的 GPU 协方差加速
```

### 7.2 模块职责

```
┌─────────────────────────────────────────────────┐
│                  核心引擎层                       │
│  sca_fbts_regressor.py     基础回归器（参考）      │
│  sca_fbts_fast.py          ★ 预计算加速版          │
│  sca_fbts_torch.py         GPU 协方差加速          │
├─────────────────────────────────────────────────┤
│                  数据层                           │
│  data_loader.py             SEED-VIG 加载          │
│  drozy_loader.py            DROZY 加载 + DE 提取  │
│  utils.py                   COR/RMSE/MAE + 5-fold │
├─────────────────────────────────────────────────┤
│                  实验编排层                        │
│  run_all.py                 ★ 一键多数据集          │
│  run_final.py               SEED-VIG 全量回归      │
│  run_fusion.py              ~40 融合配置           │
│  run_loso.py                LOSO 跨被试             │
│  run_binary.py              PERCLOS 二值分类        │
│  run_seed.py / run_seed_fast.py / run_seed_fbts.py │
│  run_parallel.py            Torch + Joblib 并行     │
├─────────────────────────────────────────────────┤
│                  论文输出层                        │
│  generate_figures.py        图表 + 统计检验         │
│  _compute_figures.py        图表数据预计算          │
│  paper/paper_jne.tex        JNE 稿件               │
│  paper/figures/cache/       预计算缓存（.npz）      │
│  results/                   实验结果（JSON）        │
└─────────────────────────────────────────────────┘
```

### 7.3 数据路径

所有数据路径硬编码在脚本中：

| 数据集 | 路径 |
|---|---|
| SEED-VIG | `D:\EEG\datasets\SEED-VIG\`（含 Raw_Data/, perclos_labels/, EEG_Feature_5Bands/ 等） |
| DROZY | `D:\EEG\datasets\DROZY\`（含 psg/, pvt-rt/） |
| SEED | `D:\EEG\datasets\SEED\`（含 ExtractedFeatures/, Preprocessed_EEG/） |

---

## 八、关键设计决策与洞察

### 8.1 为什么 FBTS 单模态 ≈ DE？

DE 特征本质上是各通道各频段的对数方差，FBTS 协方差矩阵的对角线元素也有类似含义。方差信息主导了被试内回归任务（同一人的 EEG 模式相对稳定），协方差（连接）信息在单模态下贡献有限。这解释了 **p=0.984 的完全等效**。

### 8.2 为什么 FBTS+EOG > DE+EOG？

EOG 特征提供了眼动信息（眨眼频率、扫视幅度等），是警觉度的强直接信号。当 EOG 与 EEG 融合时：

- DE 的逐通道谱信息与 EOG 存在信息冗余（警觉度变化同时反映在谱功率和眼动上）
- FBTS 的通道间协方差信息与 EOG **正交互补**：EOG 捕获行为层面，协方差捕获神经连接重组层面

这解释了为什么融合场景下 FBTS 的优势才被"释放"出来（**p=0.044, d=0.45**）。

### 8.3 为什么 LOSO 跨被试表现更强？

黎曼切空间投影的参考点（黎曼均值）在跨被试池化后更稳定，切空间坐标的对齐效果更好。此外，切空间特征本身具有仿射不变性——对电极阻抗差异、脑容积传导效应等被试间变异因素有一定鲁棒性。

### 8.4 DROZY 负结果的诚实价值

DROZY 的 per-test 标签粒度（每 10 分钟一个 KSS 值对应约 75 个 8 秒 epoch）与 epoch 级别的特征粒度严重不匹配。所有方法 COR 均为强负值，这并非方法失败，而是**标签稀疏性的问题**。论文将其坦诚呈现为局限性，展示了科学严谨性。

### 8.5 可穿戴场景的 4ch 前额设计

仅用 FT7/FT8/T7/T8 四个前额通道（可集成在发带/头戴式设备中），FBTS+EOG 达到 COR 0.613，为全通道 17ch 的 99%（0.618）。这一发现直接指向实际可穿戴 EEG 警觉度监测产品的可行性。

---

## 九、运行指南

### 9.1 快速验证

```bash
python run_all.py --quick              # 2 被试/数据集
python run_fusion.py --quick           # 3 被试融合
python run_binary.py --quick           # 5 被试二值分类
```

### 9.2 全量实验

```bash
python run_all.py --n-jobs 4           # 全量多数据集
python run_final.py --n-jobs 4         # 仅 SEED-VIG 回归
python run_fusion.py --n-jobs 4        # 融合实验
python run_loso.py                     # LOSO 跨被试（串行）
python run_binary.py                   # 二值分类
python run_seed_fbts.py                # SEED Riemannian 10ch
```

### 9.3 论文图表生成

```bash
python _compute_figures.py             # 预计算缓存（跑一次）
python generate_figures.py             # 从缓存秒出图表
# 或一步到位：
python generate_figures.py --recompute
```

### 9.4 论文编译

```bash
cd paper
pdflatex -interaction=nonstopmode paper_jne.tex
pdflatex -interaction=nonstopmode paper_jne.tex   # 二遍解析交叉引用
```

---

## 十、贡献总结

| 维度 | 贡献 |
|---|---|
| **方法学** | 首次将黎曼 FBTS 特征系统性用于 EEG 连续警觉度回归 |
| **实证发现** | FBTS+EOG 融合显著超越 DE+EOG（p=0.044），且 4ch 前额可穿戴方案可行 |
| **工程创新** | 预计算架构实现 6.5× 加速，torch 批量协方差拓展 GPU 路径 |
| **跨任务验证** | 同一特征管线在情绪分类（SEED）上达 86.1%，证明通用性 |
| **科学严谨** | DROZY 负结果坦诚呈现，统计检验完整（t-test + Cohen's d + Wilcoxon） |
| **临床转化** | 从神经生理学（丘脑-皮层失联）到可穿戴设备的完整论述链 |

---

## 十一、局限性

1. **SEED-VIG 被试量中等**（n=23），统计检验力有限（d=0.45 为中等效应量）
2. **DROZY 标签粒度不匹配**导致无法验证方法在该数据集的泛化
3. **SEED 62ch 全量 Riemannian FBTS 未完成**（计算资源限制），仅完成了 10ch 前额版本
4. **尚未与深度学习基线对比**（Transformer、GNN 等），文中仅与传统 ML 方法对比
5. **仅离线分析**，未验证实时推理性能
6. **PERCLOS 作为金标准有自身局限**（对光照敏感、眨眼检测误差等）

---

## 十二、未来方向

- 62 通道全量 Riemannian FBTS 在 SEED 上的完整评估
- 与 EEG 深度学习方法的系统性对比
- 实时推理管线（结合 torch 协方差加速）
- 更大规模数据集验证（如公开的疲劳驾驶 EEG 数据集）
- 临床应用探索：OSA 患者、ADHD、轮班工作障碍等警觉度缺陷人群
