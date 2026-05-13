"""
论文图表绘制（从缓存加载，秒级完成）
====================================
依赖: _compute_figures.py 预先生成的 paper/figures/cache/*.npz
运行: python generate_figures.py [--recompute]
"""

import sys, os, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import uniform_filter1d
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
    data = np.load(os.path.join(CACHE, 'cors_data.npz'))
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
        y_t = y_true_all[idx]
        y_p = y_pred_all[idx]
        y_p_smooth = uniform_filter1d(y_p, size=3)
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
    """全局真值 vs 预测值散点图 (所有被试拼接，含回归线)。"""
    print("Drawing scatter plot...")
    data = np.load(os.path.join(CACHE, 'cors_data.npz'))
    yt_all = data['y_true'].ravel()
    yp_all = data['y_pred'].ravel()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(yt_all, yp_all, s=1, alpha=0.3, c='steelblue', rasterized=True)

    mask = ~np.isnan(yt_all) & ~np.isnan(yp_all)
    b, m = polyfit(yt_all[mask], yp_all[mask], 1)
    x_line = np.array([0, 1])
    ax.plot(x_line, b + m * x_line, 'r--', linewidth=1.5, label=f'y={m:.2f}x+{b:.2f}')
    ax.plot([0, 1], [0, 1], 'k-', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('Ground Truth PERCLOS')
    ax.set_ylabel('Predicted PERCLOS')
    ax.set_title(f'All Subjects, Global COR={cor(yt_all, yp_all):.4f}')
    ax.legend()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')

    plt.tight_layout()
    path = os.path.join(OUTPUT, 'fig_scatter.png')
    fig.savefig(path, format='png')
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════
# 3. 特征热力图
# ═══════════════════════════════════════════════════

def draw_heatmap():
    """从 heatmap_data.npz 加载，画通道×频段 DE 特征热力图。"""
    print("Drawing heatmap...")
    data = np.load(os.path.join(CACHE, 'heatmap_data.npz'))
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
    data = np.load(os.path.join(CACHE, 'ablation_data.npz'))
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
    ax.set_title('FBTS Feature Dimension Ablation (5 subjects)', pad=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
    path = os.path.join(OUTPUT, 'fig_ablation_nfeatures.png')
    fig.savefig(path, format='png')
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--recompute', action='store_true',
                   help='Run _compute_figures.py first to regenerate cache')
    args = p.parse_args()

    if args.recompute:
        print("Recomputing cache...")
        import subprocess
        subprocess.run([sys.executable,
                        os.path.join(ROOT, '_compute_figures.py')],
                       check=True)

    # 检查缓存是否存在
    required = ['cors_data.npz', 'heatmap_data.npz', 'ablation_data.npz']
    missing = [f for f in required
               if not os.path.exists(os.path.join(CACHE, f))]
    if missing:
        print(f"ERROR: Cache files missing: {missing}")
        print(f"Run first: python _compute_figures.py")
        print(f"Or:       python generate_figures.py --recompute")
        sys.exit(1)

    print("=" * 60)
    print("Paper Figure Generation (from cache)")
    print("=" * 60)

    draw_prediction_curves()
    draw_scatter()
    draw_heatmap()
    compute_statistical_tests()
    draw_ablation()

    print(f"\nDone. Figures saved to: {OUTPUT}")