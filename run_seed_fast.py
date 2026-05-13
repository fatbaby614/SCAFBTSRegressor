"""SEED 快速版 — DE + 降通道 Riemannian"""
import os, sys, json, time, numpy as np
import scipy.io as sio
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import get_5fold_splits
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score

from config import SEED_ROOT as SEED, RESULTS_DIR as OUT

TRIAL_LABELS = np.array([1,0,-1,-1,0,1,-1,0,1,1,0,-1,0,1,-1])

# 选 10 个认知相关通道 (缩减 62→10)
CH_10 = ['FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ']

def list_sessions():
    d = os.path.join(SEED, 'ExtractedFeatures')
    return sorted([os.path.join(d, f) for f in os.listdir(d)
                   if f.endswith('.mat') and f not in ('label.mat','readme.txt')])

def run_de(sessions):
    accs, f1s = [], []
    for fp in sessions:
        data = sio.loadmat(fp)
        Xl, yl = [], []
        for t in range(1,16):
            k = f'de_LDS{t}'
            if k not in data: continue
            f = data[k].transpose(1,0,2).reshape(data[k].shape[1],-1)
            Xl.append(f); yl.extend([TRIAL_LABELS[t-1]]*f.shape[0])
        X = np.concatenate(Xl); y = np.array(yl)

        scaler = StandardScaler(); X = scaler.fit_transform(X)
        sel = SelectKBest(f_classif, k=200); X = sel.fit_transform(X, y)

        sp = len(y)//15
        splits = get_5fold_splits(15,5)
        af, ff = [], []
        for tr, te in splits:
            tm = np.zeros(len(y),bool); em = np.zeros(len(y),bool)
            for t in tr: tm[t*sp:(t+1)*sp] = True
            for t in te: em[t*sp:(t+1)*sp] = True
            clf = SVC(kernel='rbf',C=1.0,gamma='scale',class_weight='balanced')
            clf.fit(X[tm], y[tm])
            yp = clf.predict(X[em])
            af.append(accuracy_score(y[em],yp))
            ff.append(f1_score(y[em],yp,average='macro',zero_division=0))
        accs.append(np.mean(af)); f1s.append(np.mean(ff))
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(f1s))

print(f"SEED Fast: {len(list_sessions())} sessions")

t0 = time.time()
acc_m, acc_s, f1_m = run_de(list_sessions())
print(f"\nDE LDS: ACC={acc_m:.4f}+/-{acc_s:.4f}, F1={f1_m:.4f}, {time.time()-t0:.0f}s")

# PSD
t0 = time.time()

# Quick PSD by modifying the function inline
def run_psd(sessions):
    accs, f1s = [], []
    for fp in sessions:
        data = sio.loadmat(fp)
        Xl, yl = [], []
        for t in range(1,16):
            k = f'psd_LDS{t}'
            if k not in data: continue
            f = data[k].transpose(1,0,2).reshape(data[k].shape[1],-1)
            Xl.append(f); yl.extend([TRIAL_LABELS[t-1]]*f.shape[0])
        X = np.concatenate(Xl); y = np.array(yl)
        scaler = StandardScaler(); X = scaler.fit_transform(X)
        sel = SelectKBest(f_classif, k=200); X = sel.fit_transform(X, y)
        sp = len(y)//15
        splits = get_5fold_splits(15,5)
        af, ff = [], []
        for tr, te in splits:
            tm = np.zeros(len(y),bool); em = np.zeros(len(y),bool)
            for t in tr: tm[t*sp:(t+1)*sp] = True
            for t in te: em[t*sp:(t+1)*sp] = True
            clf = SVC(kernel='rbf',C=1.0,gamma='scale',class_weight='balanced')
            clf.fit(X[tm], y[tm])
            yp = clf.predict(X[em])
            af.append(accuracy_score(y[em],yp))
            ff.append(f1_score(y[em],yp,average='macro',zero_division=0))
        accs.append(np.mean(af)); f1s.append(np.mean(ff))
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(f1s))

acc_m2, acc_s2, f1_m2 = run_psd(list_sessions())
print(f"PSD LDS: ACC={acc_m2:.4f}+/-{acc_s2:.4f}, F1={f1_m2:.4f}, {time.time()-t0:.0f}s")

# Save
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
path = os.path.join(OUT, f'results_seed_fast_{ts}.json')
results = [
    dict(exp_name='SEED_DE', acc_mean=acc_m, acc_std=acc_s, f1_mean=f1_m),
    dict(exp_name='SEED_PSD', acc_mean=acc_m2, acc_std=acc_s2, f1_mean=f1_m2),
]
with open(path,'w') as f: json.dump(results, f, indent=2)
print(f"\nSaved: {path}")
print(f"\n=== Cross-Dataset Summary ===")
print(f"SEED-VIG (vigilance regression): FBTS+EOG COR=0.618")
print(f"SEED (emotion classification):   DE ACC={acc_m:.4f}")