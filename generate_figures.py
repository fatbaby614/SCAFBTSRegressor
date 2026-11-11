"""
论文图表绘制（从缓存加载，秒级完成）
====================================
依赖: precompute_figures.py 预先生成的 paper/figures/cache/*.npz
运行: python generate_figures.py [--recompute]
"""

import sys, os, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from numpy.polynomial.polynomial import polyfit
from collections import defaultdict

from config import FIGURES_DIR as OUTPUT, CACHE_DIR as CACHE, PROJECT_ROOT as ROOT

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import cor

os.makedirs(OUTPUT, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

CH_NAMES = ['FT7','FT8','T7','T8','TP7','TP8',
            'CP1','CP2','P1','PZ','P2',
            'PO3','POZ','PO4','O1','OZ','O2']

BAND_LABELS = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']


# ═══════════════════════════════════════════════════
# 1. 预测 vs 真实时间曲线
# ═══════════════════════════════════════════════════

def draw_prediction_curves():
    """从 cors_data.npz 加载，画最佳/中位/最差被试预测曲线。"""
    print("Drawing prediction curves...")
    data = np.load(os.path.join(CACHE, 'cors_data.npz'), allow_pickle=True)
    subjects = data['subjects']
    cors = data['cors']
    y_true_all = data['y_true']
    y_pred_all = data['y_pred']

    # 排序
    order = np.argsort(cors)
    idx_best, idx_med, idx_worst = order[-1], order[len(order)//2], order[0]

    selected = [
        (idx_best, 'Best', 'green'),
        (idx_med, 'Median', 'blue'),
        (idx_worst, 'Worst', 'red'),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    for ax, (idx, label, color) in zip(axes, selected):
        y_t = np.asarray(y_true_all[idx], dtype=float)
        y_p = np.asarray(y_pred_all[idx], dtype=float)
        y_p_smooth = y_p  # 缓存中已做过因果平滑 (precompute_figures.py)
        c_val = cors[idx]

        ax.plot(y_t, 'k-', alpha=0.4, linewidth=0.8, label='Ground Truth (PERCLOS)')
        ax.plot(y_p_smooth, color=color, linewidth=1.2, label=f'Predicted (COR={c_val:.3f})')
        ax.set_ylabel('PERCLOS')
        ax.set_title(f'{label}: Subject {subjects[idx][:15]} (COR={c_val:.3f})', pad=25)
        ax.legend(loc='upper left', bbox_to_anchor=(0, 0.98), ncol=2,
                  frameon=False, borderaxespad=0)
        ax.set_ylim(-0.1, 1.1)

    axes[-1].set_xlabel('Epoch (8s windows)')
    fig.suptitle('FBTS+EOG: Predicted vs Ground Truth Vigilance', fontsize=14, y=0.99)
    plt.subplots_adjust(hspace=0.6)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    path = os.path.join(OUTPUT, 'fig_prediction_curves.png')
    fig.savefig(path, format='png')
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════
# 2. 散点图
# ═══════════════════════════════════════════════════

def draw_scatter():
    """全局真值 vs 预测值散点图 (所有被试拼接，含回归线)。

    统计量口径 (与论文 Table 2 的 ``Global COR'' 列一致):
      - 每个被试: 把该被试 5 折的留出预测拼成一条时间序列, 与真值算 COR,
        记为该被试的 global COR;
      - 标题报告的是 23 个被试 global COR 的均值;
      - **不**报告把 23 名被试全部拼起来算的单一大样本 COR (本图为 0.880),
        因为它把被试间的 PERCLOS 分布差异计入, 系统性偏高, 且与正文/表格不一致。
    """
    print("Drawing scatter plot...")
    data = np.load(os.path.join(CACHE, 'cors_data.npz'), allow_pickle=True)
    yt_all = np.concatenate([np.asarray(arr, dtype=float) for arr in data['y_true']])
    yp_all = np.concatenate([np.asarray(arr, dtype=float) for arr in data['y_pred']])
    subj_cors = np.asarray(data['cors'], dtype=float)
    mean_subj_cor = float(np.nanmean(subj_cors))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(yt_all, yp_all, s=1, alpha=0.3, c='steelblue', rasterized=True)

    mask = ~np.isnan(yt_all) & ~np.isnan(yp_all)
    b, m = polyfit(yt_all[mask], yp_all[mask], 1)
    x_line = np.array([0, 1])
    ax.plot(x_line, b + m * x_line, 'r--', linewidth=1.5, label=f'y={m:.2f}x+{b:.2f}')
    ax.plot([0, 1], [0, 1], 'k-', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('Ground Truth PERCLOS')
    ax.set_ylabel('Predicted PERCLOS')
    ax.set_title('FBTS+EOG (17ch, 5-band) Within-Subject 5-Fold CV\n'
                 f'Mean subject-wise global COR = {mean_subj_cor:.4f} '
                 f'(N = {len(subj_cors)} subjects)')
    ax.legend()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')

    plt.tight_layout()
    path = os.path.join(OUTPUT, 'fig_scatter.png')
    fig.savefig(path, format='png')
    plt.close()
    print(f"  Saved: {path}  (mean subject-wise COR={mean_subj_cor:.4f})")


# ═══════════════════════════════════════════════════
# 3. 特征热力图
# ═══════════════════════════════════════════════════

def draw_heatmap():
    """从 heatmap_data.npz 加载，画通道×频段 DE 特征热力图。"""
    print("Drawing heatmap...")
    data = np.load(os.path.join(CACHE, 'heatmap_data.npz'), allow_pickle=True)
    de_alert = data['de_alert']
    de_drowsy = data['de_drowsy']
    de_diff = de_drowsy - de_alert

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, arr, title in zip(axes,
                               [de_alert, de_drowsy, de_diff],
                               ['Alert (DE)', 'Drowsy (DE)', 'Difference (Drowsy-Alert)']):
        im = ax.imshow(arr, aspect='auto', cmap='RdBu_r' if 'Diff' in title else 'viridis')
        ax.set_xticks(range(5))
        ax.set_xticklabels(BAND_LABELS, rotation=45)
        ax.set_yticks(range(17))
        ax.set_yticklabels(CH_NAMES, fontsize=8)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle('Channel x Frequency Band DE Features: Alert vs Drowsy', fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUTPUT, 'fig_heatmap.png')
    fig.savefig(path, format='png')
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════
# 4. 统计检验（无需缓存，读 JSON 秒出）
# ═══════════════════════════════════════════════════

def compute_statistical_tests():
    """从已有 JSON 结果中提取逐被试 COR，做配对 t-test 和 Cohen's d。"""
    print("Computing statistical tests...")
    files = sorted(glob.glob(os.path.join(ROOT, 'results', 'results_fusion_*.json')))
    if not files:
        print("  WARNING: No fusion results found")
        return

    data = json.load(open(files[-1], encoding='utf-8'))
    exp_cors = defaultdict(list)
    for r in data:
        if r.get('cor_mean') is not None:
            exp_cors[r['exp_name']].append((r['subject'], r['cor_mean']))

    comparisons = [
        ('FBTS+EOG vs DE+EOG (17ch)',
         'Riem+EOG_5band_all', 'DE+EOG_all'),
        ('FBTS+EOG vs DE+EOG (6ch temporal)',
         'Riem+EOG_5band_tempor', 'DE+EOG_tempor'),
        ('FBTS vs DE (17ch single)',
         'Riem_5band_all', 'DE_all'),
        ('FBTS+EOG vs DE+FBTS+EOG (17ch)',
         'Riem+EOG_5band_all', 'DE+Riem+EOG_5band_all'),
    ]

    results = []
    for desc, exp_a, exp_b in comparisons:
        subs_a = {s: c for s, c in exp_cors.get(exp_a, [])}
        subs_b = {s: c for s, c in exp_cors.get(exp_b, [])}
        common = sorted(set(subs_a.keys()) & set(subs_b.keys()))
        cors_a = np.array([subs_a[s] for s in common])
        cors_b = np.array([subs_b[s] for s in common])

        if len(cors_a) < 3:
            results.append(f"{desc}: insufficient data")
            continue

        t_stat, p_val = stats.ttest_rel(cors_a, cors_b)
        diff = cors_a - cors_b
        d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
        try:
            w_stat, w_p = stats.wilcoxon(cors_a, cors_b)
        except ValueError:
            w_stat, w_p = np.nan, np.nan

        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
        result = (
            f"\n{desc}:\n"
            f"  {exp_a}: {np.mean(cors_a):.4f} +/- {np.std(cors_a):.4f}\n"
            f"  {exp_b}: {np.mean(cors_b):.4f} +/- {np.std(cors_b):.4f}\n"
            f"  Paired t({len(common)-1}) = {t_stat:.4f}, p = {p_val:.6f} {sig}\n"
            f"  Cohen's d = {d:.4f}\n"
            f"  Wilcoxon W = {w_stat:.1f}, p = {w_p:.6f}"
        )
        results.append(result)
        print(result)

    path = os.path.join(OUTPUT, 'statistical_tests.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(results))
    print(f"\n  Saved: {path}")
    return results


# ═══════════════════════════════════════════════════
# 5. 特征维度消融
# ═══════════════════════════════════════════════════

def draw_ablation():
    """从 ablation_data.npz 加载，画 n_features vs COR 误差棒图。"""
    print("Drawing feature ablation...")
    data = np.load(os.path.join(CACHE, 'ablation_data.npz'), allow_pickle=True)
    labels = data['labels']
    means = data['means']
    stds = data['stds']

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.errorbar(range(len(labels)), means, yerr=stds, fmt='o-',
                capsize=4, color='steelblue', markersize=6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel('Number of Selected Features')
    ax.set_ylabel('COR')
    ax.set_title('FBTS Feature Dimension Ablation (23 subjects)', pad=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
    path = os.path.join(OUTPUT, 'fig_ablation_nfeatures.png')
    fig.savefig(path, format='png')
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════
# 6. COR 分布箱线图（Fig. 1）
# ═══════════════════════════════════════════════════

def draw_cor_distribution():
    """从最新 fusion JSON 中加载逐被试 COR，画六种方法的箱线图。"""
    print("Drawing COR distribution...")
    files = sorted(glob.glob(os.path.join(ROOT, 'results', 'results_fusion_*.json')))
    if not files:
        print("  ERROR: No fusion results found")
        return

    data = json.load(open(files[-1], encoding='utf-8'))
    exp_cors = defaultdict(list)
    for r in data:
        if r.get('cor_mean') is not None:
            exp_cors[r['exp_name']].append(r['cor_mean'])

    # 六种代表方法（按论文顺序）
    methods = [
        ('DE_all',              'DE 17ch'),
        ('Riem_5band_all',      'FBTS 17ch'),
        ('DE+EOG_all',          'DE+EOG 17ch'),
        ('Riem+EOG_5band_all',  'FBTS+EOG 17ch'),
        ('Riem+EOG_5band_tempor', 'FBTS+EOG 6ch'),
        ('DE+Riem+EOG_5band_all', 'DE+FBTS+EOG 17ch'),
    ]

    labels = [m[1] for m in methods]
    data_list = []
    for key, _ in methods:
        vals = exp_cors.get(key, [])
        data_list.append(vals)

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data_list, tick_labels=labels, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=6),
                    widths=0.5, patch_artist=True)
    colors = ['#5DADE2', '#48C9B0', '#F5B041', '#E74C3C', '#AF7AC5', '#58D68D']
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)

    # 标注均值
    for i, vals in enumerate(data_list):
        if not vals:
            continue
        m = np.mean(vals)
        ax.annotate(f'{m:.3f}', xy=(i+1, m), fontsize=9,
                     ha='center', va='bottom', color='red', fontweight='bold')

    # 叠加每个被试的原始数据点 (透明散点), 便于读者看到个体分布
    rng = np.random.default_rng(20260911)
    for i, vals in enumerate(data_list):
        if not vals:
            continue
        v = np.asarray(vals, dtype=float)
        jitter = rng.uniform(-0.13, 0.13, size=len(v))
        ax.scatter(i + 1 + jitter, v, s=14, alpha=0.55, c='k',
                   edgecolors='none', zorder=3)

    ax.set_ylabel('COR (Pearson r, subject-wise)')
    ax.set_xlabel('Method')
    ax.set_title('Per-Subject COR Distribution (23 subjects, 5-fold CV)\n'
                 'Black dots = individual subjects; red diamond = mean; box = IQR')
    ax.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=15)

    plt.tight_layout()
    path = os.path.join(OUTPUT, 'fig_cor_distribution.png')
    fig.savefig(path, format='png')
    plt.close()
    print(f"  Saved: {path}")



# =================================================
# 7. FBTS 通道对可解释性图
# =================================================

def draw_fbts_connectivity():
    """从 fbts_connectivity.npz 加载, 画 5x17x17 通道对重要性热图。"""
    print("Drawing FBTS connectivity...")
    fpath = os.path.join(CACHE, 'fbts_connectivity.npz')
    if not os.path.exists(fpath):
        print("  SKIP: run python precompute_figures.py --skip-prediction --skip-heatmap --skip-ablation")
        return
    data = np.load(fpath, allow_pickle=True)
    imp = data['importance']
    bp = data['band_pct']
    band_names = data['band_names']

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()
    for bi in range(5):
        ax = axes[bi]
        mat_sym = (imp[bi] + imp[bi].T) / 2
        im = ax.imshow(mat_sym, aspect='equal', cmap='YlOrRd', interpolation='nearest', vmin=0)
        ax.set_xticks(range(17)); ax.set_xticklabels(CH_NAMES, rotation=90, fontsize=6)
        ax.set_yticks(range(17)); ax.set_yticklabels(CH_NAMES, fontsize=6)
        ax.set_title(f'{band_names[bi]} ({bp[bi]:.1f}%)', fontsize=12)
        plt.colorbar(im, ax=ax, shrink=0.8, label='Selection freq.')
    ax = axes[5]
    bars = ax.bar(range(5), bp, color=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd'])
    ax.set_xticks(range(5)); ax.set_xticklabels(band_names, rotation=30)
    ax.set_ylabel('% of top-100 features')
    ax.set_title('Top-100 Feature Band Distribution')
    for bar, pct in zip(bars, bp):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f'{pct:.1f}%', ha='center', fontsize=9)
    fig.suptitle('Channel-Pair Selection Frequency: Top-100 Features Mapped to Bands and Electrode Pairs', fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT, 'figure8_feature_selection_frequency.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT}/figure8_feature_selection_frequency.png")


# ═══════════════════════════════════════════════════
# 8. 被试级配对图 (LOSO + PCA 消融, Editor: individual data points)
# ═══════════════════════════════════════════════════

LOSO_FILE = 'results_loso_matched_20261107_005039.json'
PCA_ABLATION_FILE = 'results_pca_ablation_20261109_123730.json'


def draw_subject_paired():
    """(a) LOSO 匹配协议六配置逐被试 COR; (b) PCA 消融维度匹配配对斜线图。"""
    print("Drawing subject-level paired plots (LOSO + PCA)...")
    loso_path = os.path.join(ROOT, 'results', LOSO_FILE)
    pca_path = os.path.join(ROOT, 'results', PCA_ABLATION_FILE)
    if not (os.path.exists(loso_path) and os.path.exists(pca_path)):
        print("  SKIP: LOSO/PCA results not found")
        return

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5),
                                   gridspec_kw={'width_ratios': [1.4, 1.0]})

    # ---- (a) LOSO matched protocol, 17ch 5-band ----
    loso = json.load(open(loso_path, encoding='utf-8'))
    configs = [
        ('eog_all_5band',         'EOG only'),
        ('de_all_5band',          'DE'),
        ('fbts_all_5band',        'FBTS'),
        ('de_eog_all_5band',      'DE+EOG'),
        ('fbts_eog_all_5band',    'FBTS+EOG'),
        ('de_fbts_eog_all_5band', 'DE+FBTS+EOG'),
    ]
    labels = [c[1] for c in configs]
    data_list = [np.asarray(list(loso[c[0]]['cor_per_subject'].values()), dtype=float)
                 for c in configs]

    rng = np.random.default_rng(20261109)
    colors = ['#5DADE2', '#48C9B0', '#F5B041', '#E74C3C', '#AF7AC5', '#58D68D']
    for i, (v, c) in enumerate(zip(data_list, colors)):
        jitter = rng.uniform(-0.12, 0.12, size=len(v))
        axL.scatter(i + 1 + jitter, v, s=18, alpha=0.65, c='k', zorder=3)
        axL.hlines(np.mean(v), i + 0.72, i + 1.28, color=c, lw=3, zorder=4)
        axL.annotate(f'{np.mean(v):.3f}', xy=(i + 1.3, np.mean(v)),
                     fontsize=9, va='center', color=c, fontweight='bold')
    axL.set_xticks(range(1, len(labels) + 1))
    axL.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    axL.set_ylabel('LOSO COR (Pearson r)')
    axL.set_xlabel('Modality configuration (17 channels, 5 bands)')
    axL.set_title('(a) LOSO, individual subjects (N=23)\nblack dot = one subject; bar = mean')
    axL.grid(True, axis='y', alpha=0.3)

    t0, p0 = stats.ttest_rel(data_list[4], data_list[3])
    d0 = np.mean(data_list[4] - data_list[3])
    print(f"  LOSO FBTS+EOG vs DE+EOG (17ch): mean diff {d0:+.3f}, t(22)={t0:.2f}, p={p0:.3f}")

    # ---- (b) PCA ablation paired: DE+EOG vs FBTS-PCA85+EOG (both 121 dims) ----
    pca = json.load(open(pca_path, encoding='utf-8'))

    def pivot(exp):
        return {r['subject']: r['cor_mean'] for r in pca['per_subject']
                if r['exp_name'] == exp}

    de_eog = pivot('DE+EOG')
    fb85 = pivot('FBTS-PCA85+EOG')
    subs = sorted(set(de_eog) & set(fb85))
    x0 = np.array([de_eog[s] for s in subs])
    x1 = np.array([fb85[s] for s in subs])

    for a, b in zip(x0, x1):
        axR.plot([0, 1], [a, b], color='grey', alpha=0.45, lw=0.9, zorder=1)
    axR.scatter(np.zeros(len(x0)), x0, s=30, c='#5DADE2', zorder=3, label='DE+EOG (121 input / 100 in-model)')
    axR.scatter(np.ones(len(x1)), x1, s=30, c='#AF7AC5', zorder=3,
                label='FBTS-PCA85+EOG (121 input / 121 in-model)')
    axR.hlines(np.mean(x0), -0.09, 0.09, color='#1f77b4', lw=3, zorder=4)
    axR.hlines(np.mean(x1), 0.91, 1.09, color='#6c3483', lw=3, zorder=4)
    axR.annotate(f'{np.mean(x0):.3f}', xy=(-0.13, np.mean(x0)), fontsize=9,
                 va='center', ha='right', fontweight='bold')
    axR.annotate(f'{np.mean(x1):.3f}', xy=(1.13, np.mean(x1)), fontsize=9,
                 va='center', ha='left', fontweight='bold')
    axR.set_xlim(-0.55, 1.55)
    axR.set_xticks([0, 1])
    axR.set_xticklabels(['DE+EOG', 'FBTS-PCA85+EOG'], fontsize=10)
    axR.set_ylabel('Within-subject COR (Pearson r)')
    axR.set_xlabel('Matched theoretical input dimensionality (121 features)')
    d1 = x1 - x0
    t1, p1 = stats.ttest_rel(x1, x0)
    axR.set_title(f'(b) PCA ablation, paired subjects (N={len(subs)})\n'
                  f'grey line = one subject; $\\Delta$={np.mean(d1):+.3f}, p={p1:.2f}')
    axR.grid(True, axis='y', alpha=0.3)
    axR.legend(loc='lower center', fontsize=8, frameon=False)

    print(f"  PCA FBTS-PCA85+EOG vs DE+EOG: mean diff {np.mean(d1):+.3f}, t(22)={t1:.2f}, p={p1:.3f}")

    plt.tight_layout()
    path = os.path.join(OUTPUT, 'fig_subject_paired.png')
    fig.savefig(path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--recompute', action='store_true',
                   help='Run precompute_figures.py first to regenerate cache')
    args = p.parse_args()

    if args.recompute:
        print("Recomputing cache...")
        import subprocess
        subprocess.run([sys.executable,
                        os.path.join(ROOT, 'precompute_figures.py')],
                       check=True)

    # 检查缓存是否存在
    required = ['cors_data.npz', 'heatmap_data.npz', 'ablation_data.npz', 'fbts_connectivity.npz']
    missing = [f for f in required
               if not os.path.exists(os.path.join(CACHE, f))]
    if missing:
        print(f"ERROR: Cache files missing: {missing}")
        print(f"Run first: python precompute_figures.py")
        print(f"Or:       python generate_figures.py --recompute")
        sys.exit(1)

    print("=" * 60)
    print("Paper Figure Generation (from cache)")
    print("=" * 60)

    draw_cor_distribution()
    draw_prediction_curves()
    draw_scatter()
    draw_heatmap()
    compute_statistical_tests()
    draw_fbts_connectivity()
    draw_ablation()
    draw_subject_paired()

    print(f"\nDone. Figures saved to: {OUTPUT}")