import json, glob, numpy as np
from collections import defaultdict
from config import RESULTS_DIR

# Find latest fusion results
files = sorted(glob.glob(os.path.join(RESULTS_DIR, 'results_fusion_*.json')))
if not files:
    print("No fusion results found")
    exit()

f = files[-1]
data = json.load(open(f, encoding='utf-8'))
print(f"Fusion Results: {len(data)} entries from {f.split(chr(92))[-1]}")

agg = defaultdict(lambda: {'cor': [], 'rmse': [], 'gcor': [], 'time': []})
for r in data:
    if r.get('cor_mean') is not None:
        agg[r['exp_name']]['cor'].append(r['cor_mean'])
        agg[r['exp_name']]['rmse'].append(r.get('rmse_mean', 0))
        agg[r['exp_name']]['gcor'].append(r.get('global_cor', 0))
        agg[r['exp_name']]['time'].append(r.get('time_sec', 0))

# Sort by COR descending
sorted_names = sorted(agg.keys(), key=lambda n: -np.mean(agg[n]['cor']))

print(f"\n{'Rank':<5s} {'Experiment':<30s} {'N':>4s} {'COR':>8s} {'CORstd':>8s} "
      f"{'RMSE':>8s} {'G.COR':>8s} {'Time':>8s}")
print("-" * 90)
for i, name in enumerate(sorted_names, 1):
    s = agg[name]
    n = len(s['cor'])
    if n == 0:
        continue
    marker = " ★" if i <= 3 else ""
    print(f"{i:<5d} {name:<30s} {n:>4d} {np.mean(s['cor']):>8.4f} "
          f"{np.std(s['cor']):>8.4f} {np.mean(s['rmse']):>8.4f} "
          f"{np.mean(s['gcor']):>8.4f} {np.mean(s['time']):>7.1f}s{marker}")