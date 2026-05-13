import json, glob, numpy as np
from collections import defaultdict
from config import RESULTS_DIR

files = sorted(glob.glob(os.path.join(RESULTS_DIR, 'results_all_*.json')))
f = files[-1]
data = json.load(open(f, encoding='utf-8'))
print(f"Results: {len(data)} entries from {f}")

# Errors
errors = [r for r in data if r.get('error')]
if errors:
    print(f"\nErrors ({len(errors)}):")
    for e in errors[:5]:
        print(f"  {e['dataset']}/{e['subject']}/{e['exp_name']}: {e['error'][:80]}")

# Aggregate
agg = defaultdict(lambda: {'cor': [], 'rmse': [], 'gcor': [], 'time': []})
for r in data:
    if r.get('cor_mean') is not None:
        agg[r['exp_name']]['cor'].append(r['cor_mean'])
        agg[r['exp_name']]['rmse'].append(r.get('rmse_mean', 0))
        agg[r['exp_name']]['gcor'].append(r.get('global_cor', 0))
        agg[r['exp_name']]['time'].append(r.get('time_sec', 0))

print(f"\n{'Exp':<30s} {'N':>4s} {'COR':>8s} {'CORstd':>8s} {'RMSE':>8s} {'G.COR':>8s} {'Time':>8s}")
print("-" * 85)
for n, s in sorted(agg.items()):
    print(f"{n:<30s} {len(s['cor']):>4d} {np.mean(s['cor']):>8.4f} "
          f"{np.std(s['cor']):>8.4f} {np.mean(s['rmse']):>8.4f} "
          f"{np.mean(s['gcor']):>8.4f} {np.mean(s['time']):>7.1f}s")