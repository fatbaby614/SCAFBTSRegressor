"""
被试级统计检验 (JNE 修回 Experiment E)
======================================
从实验结果 JSON 中提取逐被试指标, 做配对统计, 并输出论文可直接使用的
统计表 (Comparison / Metric / A / B / Paired diff (95% CI) / Effect size /
p / p-FDR)。

原则:
  - 统计单位是被试 (或独立 test), 不把 epoch 当独立样本;
  - 优先 paired t-test, 同时给 Wilcoxon signed-rank 作为稳健性检查;
  - 效应量: Cohen's d_z (配对) 与 rank-biserial correlation (Wilcoxon);
  - 多重比较: Benjamini-Hochberg FDR (在全部 primary comparisons 家族内校正);
  - pooled COR 与 subject-wise mean COR 分开报告.

用法:
  python run_stats.py                          # 用最新结果 JSON
  python run_stats.py --fusion results/results_fusion_20261106_120000.json
  python run_stats.py --loso   results/results_loso_matched_20261106_120000.json
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scipy import stats

from config import RESULTS_DIR, PAPER_DIR


# ── Benjamini-Hochberg FDR ───────────────────────────────────

def bh_fdr(pvalues):
    """Benjamini-Hochberg FDR 校正。返回与输入同序的校正后 p 值。"""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    # 强制单调 (从大到小取累积最小)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


# ── 单次配对比较 ─────────────────────────────────────────────

def paired_compare(a, b, metric='COR'):
    """对两组逐被试指标做配对统计。

    Returns:
        dict with mean_diff, ci95, t_stat, p_t, w_stat, p_w,
        cohens_dz, rank_biserial, n
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    diff = a - b
    mean_diff = float(diff.mean())
    sd = float(diff.std(ddof=1)) if n > 1 else 0.0

    # 95% CI (t 分布)
    if n > 1 and sd > 0:
        se = sd / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, n - 1)
        ci = (mean_diff - t_crit * se, mean_diff + t_crit * se)
    else:
        ci = (float('nan'), float('nan'))

    t_stat, p_t = stats.ttest_rel(a, b)
    try:
        w_stat, p_w = stats.wilcoxon(a, b)
    except ValueError:
        w_stat, p_w = float('nan'), float('nan')

    dz = mean_diff / sd if sd > 0 else 0.0

    # rank-biserial correlation (基于 Wilcoxon 符号秩)
    diff_nz = diff[diff != 0]
    if len(diff_nz) > 0:
        ranks = stats.rankdata(np.abs(diff_nz))
        r_pos = ranks[diff_nz > 0].sum()
        r_neg = ranks[diff_nz < 0].sum()
        rb = (r_pos - r_neg) / (r_pos + r_neg)
    else:
        rb = 0.0

    return {
        'metric': metric,
        'n': int(n),
        'mean_a': float(a.mean()),
        'mean_b': float(b.mean()),
        'std_a': float(a.std(ddof=1)) if n > 1 else 0.0,
        'std_b': float(b.std(ddof=1)) if n > 1 else 0.0,
        'mean_diff': mean_diff,
        'ci95_low': float(ci[0]),
        'ci95_high': float(ci[1]),
        't_stat': float(t_stat),
        'p_ttest': float(p_t),
        'w_stat': float(w_stat),
        'p_wilcoxon': float(p_w),
        'cohens_dz': float(dz),
        'rank_biserial': float(rb),
    }


# ── 结果载入 ─────────────────────────────────────────────────

def load_latest(pattern):
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, pattern)))
    return files[-1] if files else None


def load_fusion_per_subject(path):
    """results_fusion_*.json -> {exp_name: {subject: cor}}"""
    data = json.load(open(path, encoding='utf-8'))
    out = {}
    for r in data:
        if r.get('cor_mean') is None:
            continue
        out.setdefault(r['exp_name'], {})[r['subject']] = r['cor_mean']
    return out


def load_loso_per_subject(path):
    """results_loso_matched_*.json -> {exp_name: {subject: cor}}"""
    data = json.load(open(path, encoding='utf-8'))
    out = {}
    for name, v in data.items():
        if isinstance(v, dict) and 'cor_per_subject' in v:
            out[name] = v['cor_per_subject']
    return out


# ── 预定义的 primary comparisons ─────────────────────────────
#
# within_subject family 的比较集合必须与论文中报告的比较**完全一致**,
# 因为 BH-FDR 校正依赖 family 内的比较总数。少列一个比较会改变所有 q 值。

FUSION_PAIRS = [
    ('Within-subject: FBTS vs DE (17ch/5band)',
     'Riem_5band_all', 'DE_all'),
    ('Within-subject: FBTS+EOG vs DE+EOG (17ch/5band)',
     'Riem+EOG_5band_all', 'DE+EOG_all'),
    ('Within-subject: FBTS+EOG vs DE+EOG (17ch/8band)',
     'Riem+EOG_8band_all', 'DE+EOG_all'),
    ('Within-subject: FBTS+EOG vs DE+EOG (6ch temp/5band)',
     'Riem+EOG_5band_tempor', 'DE+EOG_tempor'),
    ('Within-subject: FBTS+EOG vs FBTS (17ch/5band)',
     'Riem+EOG_5band_all', 'Riem_5band_all'),
    ('Within-subject: FBTS+EOG vs FBTS (6ch temp/5band)',
     'Riem+EOG_5band_tempor', 'Riem_5band_tempor'),
    ('Within-subject: DE+EOG vs DE (17ch)',
     'DE+EOG_all', 'DE_all'),
    ('Within-subject: DE+FBTS+EOG vs FBTS+EOG (17ch)',
     'DE+Riem+EOG_5band_all', 'Riem+EOG_5band_all'),
    ('Within-subject: DE+FBTS+EOG vs DE+EOG (17ch)',
     'DE+Riem+EOG_5band_all', 'DE+EOG_all'),
    ('Within-subject: FBTS 8band vs 5band (17ch)',
     'Riem_8band_all', 'Riem_5band_all'),
    ('Within-subject: PSD vs DE (17ch)',
     'PSD_all', 'DE_all'),
    ('Within-subject: FBTS 4ch anterior vs DE 6ch temporal',
     'Riem_5band_forehe', 'DE_tempor'),
    ('Within-subject: DE+FBTS+EOG 4ch anterior vs DE+EOG (17ch)',
     'DE+Riem+EOG_5band_4ch_fh', 'DE+EOG_all'),
]

LOSO_PAIRS = [
    ('LOSO: FBTS vs DE', 'fbts', 'de'),
    ('LOSO: FBTS+EOG vs DE+EOG', 'fbts_eog', 'de_eog'),
    ('LOSO: FBTS+EOG vs DE+FBTS+EOG', 'fbts_eog', 'de_fbts_eog'),
    ('LOSO: EOG-only vs DE', 'eog', 'de'),
]


def match_groups(groups, group_a, group_b):
    """在 LOSO 结果中找出同时包含 group_a / group_b 的配置名对。"""
    out = []
    for name in groups:
        base = name.rsplit('_', 2)
        if len(base) != 3:
            continue
        group, ch, band = base
        if group != group_a:
            continue
        other = f'{group_b}_{ch}_{band}'
        if other in groups:
            out.append((name, other, f'{group} vs {group_b} [{ch}, {band}]'))
    return out


def main():
    p = argparse.ArgumentParser(description='Subject-level statistics (Exp E)')
    p.add_argument('--fusion', default=None,
                   help='results_fusion_*.json (默认取最新)')
    p.add_argument('--loso', default=None,
                   help='results_loso_matched_*.json (默认取最新)')
    p.add_argument('--out', default=None,
                   help='统计表输出路径 (默认写 results/)')
    args = p.parse_args()

    fusion_path = args.fusion or load_latest('results_fusion_*.json')
    loso_path = args.loso or load_latest('results_loso_matched_*.json')

    rows = []

    if fusion_path and os.path.exists(fusion_path):
        groups = load_fusion_per_subject(fusion_path)
        print(f'Fusion results: {os.path.basename(fusion_path)} '
              f'({len(groups)} experiment groups)')
        for desc, a, b in FUSION_PAIRS:
            if a not in groups or b not in groups:
                print(f'  SKIP (missing): {desc}')
                continue
            common = sorted(set(groups[a]) & set(groups[b]))
            if len(common) < 3:
                print(f'  SKIP (n<3): {desc}')
                continue
            row = paired_compare([groups[a][s] for s in common],
                                 [groups[b][s] for s in common])
            row.update({'comparison': desc, 'family': 'within_subject',
                        'method_a': a, 'method_b': b, 'subjects': common})
            rows.append(row)
    else:
        print('No fusion results found')

    if loso_path and os.path.exists(loso_path):
        groups = load_loso_per_subject(loso_path)
        print(f'LOSO results: {os.path.basename(loso_path)} '
              f'({len(groups)} experiment groups)')
        for desc, ga, gb in LOSO_PAIRS:
            for na, nb, label in match_groups(groups, ga, gb):
                common = sorted(set(groups[na]) & set(groups[nb]))
                if len(common) < 3:
                    continue
                row = paired_compare([groups[na][s] for s in common],
                                     [groups[nb][s] for s in common])
                row.update({'comparison': f'LOSO: {label}',
                            'family': 'loso',
                            'method_a': na, 'method_b': nb, 'subjects': common})
                rows.append(row)
    else:
        print('No LOSO results found')

    if not rows:
        print('\nNo valid comparisons — nothing to report.')
        return

    # ── FDR (在每个 family 内校正) ──
    for family in {r['family'] for r in rows}:
        idx = [i for i, r in enumerate(rows) if r['family'] == family]
        adj = bh_fdr([rows[i]['p_ttest'] for i in idx])
        for i, a in zip(idx, adj):
            rows[i]['p_fdr'] = float(a)

    # ── 输出表 ──
    print(f"\n{'=' * 118}")
    print(f"{'Comparison':<48s} {'A':>7s} {'B':>7s} {'diff':>7s} "
          f"{'95% CI':>17s} {'t':>7s} {'p':>8s} {'p-FDR':>8s} {'d_z':>6s}")
    print('-' * 118)
    for r in rows:
        ci = f"[{r['ci95_low']:+.3f},{r['ci95_high']:+.3f}]"
        print(f"{r['comparison']:<48s} {r['mean_a']:>7.4f} {r['mean_b']:>7.4f} "
              f"{r['mean_diff']:>+7.4f} {ci:>17s} {r['t_stat']:>7.3f} "
              f"{r['p_ttest']:>8.4f} {r['p_fdr']:>8.4f} {r['cohens_dz']:>6.3f}")

    print(f"\n{'=' * 118}")
    print(f"{'Comparison':<48s} {'W':>8s} {'p-Wilcoxon':>11s} {'r_b':>7s}")
    print('-' * 118)
    for r in rows:
        print(f"{r['comparison']:<48s} {r['w_stat']:>8.1f} "
              f"{r['p_wilcoxon']:>11.4f} {r['rank_biserial']:>7.3f}")

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = args.out or os.path.join(RESULTS_DIR,
                                        f'results_stats_{ts}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'comparisons': rows,
                   'sources': {'fusion': fusion_path, 'loso': loso_path}},
                  f, indent=2, ensure_ascii=False, default=str)
    print(f'\nSaved: {out_path}')

    # ── LaTeX 片段 (直接贴进论文) ──
    tex_path = os.path.join(PAPER_DIR, f'stats_table_{ts}.tex')
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write('% Generated by run_stats.py -- do not edit manually\n')
        f.write('\\begin{tabular}{llrrlrr}\n\\toprule\n')
        f.write('\\textbf{Comparison} & \\textbf{Metric} & \\textbf{A} & '
                '\\textbf{B} & \\textbf{Paired diff (95\\% CI)} & '
                '\\textbf{$p$} & \\textbf{$p$-FDR} \\\\\n\\midrule\n')
        for r in rows:
            ci = (f"{r['mean_diff']:+.3f} [{r['ci95_low']:+.3f}, "
                  f"{r['ci95_high']:+.3f}]")
            f.write(f"{r['comparison']} & COR & {r['mean_a']:.3f} & "
                    f"{r['mean_b']:.3f} & {ci} & {r['p_ttest']:.4f} & "
                    f"{r['p_fdr']:.4f} \\\\\n")
        f.write('\\bottomrule\n\\end{tabular}\n')
    print(f'Saved: {tex_path}')


if __name__ == '__main__':
    main()
