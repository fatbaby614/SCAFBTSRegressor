import json, numpy as np
from collections import defaultdict

from config import RESULTS_DIR
# LOSO results
with open(glob.glob(os.path.join(RESULTS_DIR, 'results_loso_*.json'))[-1], encoding='utf-8') as f:
    data = json.load(f)

print(f"LOSO entries: {len(data)}")
agg = defaultdict(lambda: {'cor': [], 'rmse': [], 'time': []})
for r in data:
    if r.get('cor_mean') is not None:
        agg[r['exp_name']]['cor'].append(r['cor_mean'])
        agg[r['exp_name']]['rmse'].append(r.get('rmse_mean', 0))
        agg[r['exp_name']]['time'].append(r.get('time_sec', 0))

print(f"\n{'Exp':<35s} {'N':>3s} {'COR':>8s} {'CORstd':>8s} {'RMSE':>8s}")
print("-" * 70)
for n, s in sorted(agg.items()):
    print(f"{n:<35s} {len(s['cor']):>3d} {np.mean(s['cor']):>8.4f} "
          f"{np.std(s['cor']):>8.4f} {np.mean(s['rmse']):>8.4f}")