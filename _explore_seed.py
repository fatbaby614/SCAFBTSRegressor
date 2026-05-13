import scipy.io as sio
import numpy as np
import os
from config import SEED_ROOT as base

# 1. README
for d in ['ExtractedFeatures', 'Preprocessed_EEG']:
    readme = os.path.join(base, d, 'readme.txt')
    if os.path.exists(readme):
        with open(readme, encoding='utf-8') as f:
            content = f.read()
        print(f"=== {d}/readme.txt ===")
        print(content[:2000])
        print()

# 2. Label file
label = sio.loadmat(os.path.join(base, 'ExtractedFeatures', 'label.mat'))
print("Label keys:", list(label.keys()))
for k in label.keys():
    if not k.startswith('__'):
        v = label[k]
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        if v.size < 50:
            print(f"    values: {v.flatten()[:20]}")

# 3. Sample feature file
feat_dir = os.path.join(base, 'ExtractedFeatures')
files = sorted([f for f in os.listdir(feat_dir) if f.endswith('.mat') and f != 'label.mat' and f != 'readme.txt'])
f = files[0]
d = sio.loadmat(os.path.join(feat_dir, f))
print(f"\n=== {f} ===")
print("Keys:", [k for k in d.keys() if not k.startswith('__')])
for k in d.keys():
    if not k.startswith('__'):
        print(f"  {k}: shape={d[k].shape}, dtype={d[k].dtype}")

# 4. Sample Preprocessed_EEG
eeg_dir = os.path.join(base, 'Preprocessed_EEG')
eeg_files = sorted([f for f in os.listdir(eeg_dir) if f.endswith('.mat') and f != 'label.mat' and f != 'readme.txt'])
f = eeg_files[0]
d = sio.loadmat(os.path.join(eeg_dir, f))
print(f"\n=== Preprocessed_EEG/{f} ===")
print("Keys:", [k for k in d.keys() if not k.startswith('__')])
for k in d.keys():
    if not k.startswith('__'):
        v = d[k]
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        if v.ndim <= 2 and v.size <= 100:
            print(f"    values: {v.flatten()[:20]}")

# 5. Subject info
subj_file = os.path.join(base, 'subject-id-gender-seed.txt')
if os.path.exists(subj_file):
    with open(subj_file) as f:
        print(f"\n=== subject info ===")
        print(f.read()[:500])