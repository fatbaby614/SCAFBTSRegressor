"""
计算延迟与实时性测量 (JNE 修回 Experiment G)
============================================
现稿称 "FBTS 特征提取 4.2 ms/epoch, 端到端延迟未测量"。审稿人要求补充
硬件/软件信息、重复次数, 并区分各阶段延迟。

本脚本分阶段测量 (每阶段独立重复 N 次, 报 mean±SD / median / P95):
  1  band-pass filtering         单 epoch 单频段
  2  covariance + OAS            单 epoch 单频段 (含滤波)
  3  tangent-space mapping       单 epoch 单频段 (含滤波+协方差)
  4  F-test / PCA (训练阶段)     一次性拟合
  5  SVR / Ridge inference       单 epoch
  6  full pipeline               单 epoch 端到端 (滤波→预测)

同时报告:
  - 8 s 信号采集窗口
  - 3 点因果平滑引入的延迟 (2 epoch = 16 s)
  - 端到端响应延迟 = 采集窗口 + 平滑延迟 + 推理延迟
  - CPU / RAM / OS / 库版本

用法:
  python run_latency_benchmark.py               # 默认 50 次重复
  python run_latency_benchmark.py --repeats 200
"""

import os
import sys
import json
import time
import platform
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from sca_fbts_fast import apply_bandpass_filter

from config import SEED_VIG_ROOT, RESULTS_DIR, BANDS_5


FS = 200
EPOCH_SEC = 8
EPOCH_LEN = FS * EPOCH_SEC
N_CH = 17
N_SUBJECTS = 3


def environment_info():
    """硬件与软件环境。"""
    info = {
        'platform': platform.platform(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'python': platform.python_version(),
        'cpu_count': os.cpu_count(),
    }
    try:
        import psutil
        info['ram_gb'] = round(psutil.virtual_memory().total / 1e9, 1)
        info['cpu_freq_mhz'] = round(psutil.cpu_freq().max, 0)
    except Exception:
        info['ram_gb'] = None
    for mod in ['numpy', 'scipy', 'sklearn', 'pyriemann']:
        try:
            info[f'{mod}_version'] = __import__(mod).__version__
        except Exception:
            info[f'{mod}_version'] = None
    try:
        import torch
        info['torch_version'] = torch.__version__
        info['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'
    except Exception:
        info['torch_version'] = None
        info['gpu'] = 'not available'
    return info


def summarize(samples):
    """mean / std / median / P95。"""
    a = np.asarray(samples, dtype=float)
    return {
        'mean_ms': float(a.mean() * 1000),
        'std_ms': float(a.std(ddof=1) * 1000) if len(a) > 1 else 0.0,
        'median_ms': float(np.median(a) * 1000),
        'p95_ms': float(np.percentile(a, 95) * 1000),
        'min_ms': float(a.min() * 1000),
        'n_repeats': int(len(a)),
    }


def timed(fn, repeats, warmup=3):
    """预热后重复计时。"""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return samples


def build_epoch(n_ch=N_CH):
    """构造一个真实的 EEG epoch (带生理性 1/f 谱 + alpha 节律)。"""
    rng = np.random.default_rng(42)
    t = np.arange(EPOCH_LEN) / FS
    sig = rng.standard_normal((n_ch, EPOCH_LEN)) * 20.0
    # 10 Hz alpha 节律 (后部通道更强)
    for ch in range(n_ch):
        amp = 5.0 if ch >= 8 else 2.0
        sig[ch] += amp * np.sin(2 * np.pi * 10 * t + rng.random())
    # 每通道独立增益 (模拟电极差异)
    sig *= rng.uniform(0.5, 1.5, size=(n_ch, 1))
    return sig


def benchmark_pipeline(n_ch, repeats, bands=('bandpass', 'covariance',
                                             'tangent')):
    """对单 epoch 测量各阶段延迟 (秒)。"""
    epoch = build_epoch(n_ch)
    low, high = BANDS_5[1]
    cov_est = Covariances(estimator='oas')

    results = {}

    # 1. band-pass filtering
    results['filter'] = summarize(timed(
        lambda: apply_bandpass_filter(epoch, low, high, FS), repeats))

    # 2. covariance + OAS (含滤波)
    def _cov():
        filtered = apply_bandpass_filter(epoch, low, high, FS)
        cov_est.transform(np.expand_dims(filtered, axis=0))
    results['covariance'] = summarize(timed(_cov, repeats))

    # 3. tangent-space mapping (含滤波 + 协方差)
    ts = TangentSpace(metric='riemann')
    cov_ref = cov_est.transform(np.expand_dims(
        apply_bandpass_filter(epoch, low, high, FS), axis=0))

    def _tangent():
        filtered = apply_bandpass_filter(epoch, low, high, FS)
        c = cov_est.transform(np.expand_dims(filtered, axis=0))
        ts.transform(c)
    results['tangent'] = summarize(timed(_tangent, repeats))

    return results


def benchmark_train_stage(n_ch, repeats):
    """训练阶段: F-test / PCA 拟合耗时。"""
    n_samples = 708
    n_bands = len(BANDS_5)
    dim_band = n_ch * (n_ch + 1) // 2
    n_feat = n_bands * dim_band
    X = np.random.default_rng(0).standard_normal((n_samples, n_feat))
    y = np.random.default_rng(1).standard_normal(n_samples)

    # PCA 成分数不能超过 min(n_samples, n_features)
    n_comp = int(min(100, n_feat, n_samples - 1))

    results = {}

    results['ftest'] = summarize(timed(
        lambda: SelectKBest(f_regression, k=min(100, n_feat)).fit(X, y),
        max(3, repeats // 10), warmup=1))

    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(X)
    results['pca'] = summarize(timed(
        lambda: PCA(n_components=n_comp, random_state=42).fit(X),
        max(3, repeats // 10), warmup=1))
    results['pca']['n_components'] = n_comp
    results['pca']['explained_variance'] = float(
        pca.explained_variance_ratio_.sum())

    return results


def benchmark_inference(repeats, n_ch=N_CH, n_features=100):
    """SVR 推理耗时 (单 epoch)。"""
    X = np.random.default_rng(2).standard_normal((1000, n_features))
    y = np.random.default_rng(3).standard_normal(1000)
    scaler = StandardScaler().fit(X)
    model = SVR(kernel='rbf', C=1.0, gamma='scale').fit(scaler.transform(X), y)
    Xs = scaler.transform(X[:1])

    return {'svr_inference': summarize(
        timed(lambda: model.predict(Xs), repeats, warmup=5))}


def benchmark_full_pipeline(n_ch, repeats, n_bands=5, n_features=100):
    """端到端: 单 epoch 全部 5 频段滤波 + 协方差 + 切空间 + 推理。"""
    epoch = build_epoch(n_ch)
    bands = BANDS_5[:n_bands]
    cov_est = Covariances(estimator='oas')
    dim_band = n_ch * (n_ch + 1) // 2

    # 用随机投影模型代表真实推理步骤 (切空间维度大, 避免把 O(d^2) 误当推理耗时)
    rng = np.random.default_rng(4)
    W = rng.standard_normal((n_bands * dim_band, n_features)) / np.sqrt(dim_band)
    model = SVR(kernel='rbf', C=1.0, gamma='scale')
    model.fit(rng.standard_normal((200, n_features)),
              rng.standard_normal(200))

    ts_list = []
    for band in bands:
        c = cov_est.transform(np.expand_dims(
            apply_bandpass_filter(epoch, band[0], band[1], FS), axis=0))
        ts = TangentSpace(metric='riemann')
        ts.fit(c)
        ts_list.append(ts)

    def _full():
        feats = []
        for band, ts in zip(bands, ts_list):
            filtered = apply_bandpass_filter(epoch, band[0], band[1], FS)
            c = cov_est.transform(np.expand_dims(filtered, axis=0))
            feats.append(ts.transform(c)[0])
        x = np.concatenate(feats) @ W
        return model.predict(x.reshape(1, -1))

    return {'full_pipeline': summarize(timed(_full, repeats))}


def main():
    import argparse
    p = argparse.ArgumentParser(description='Staged latency benchmark (Exp G)')
    p.add_argument('--repeats', type=int, default=50)
    p.add_argument('--channels', type=int, nargs='+', default=[17, 6, 4])
    args = p.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    env = environment_info()
    print('=' * 78)
    print('Latency benchmark (Exp G)')
    print('=' * 78)
    print(f"  Platform : {env['platform']}")
    print(f"  CPU      : {env.get('processor') or env['machine']} "
          f"({env['cpu_count']} logical cores)")
    print(f"  RAM      : {env.get('ram_gb')} GB")
    print(f"  GPU      : {env['gpu']}")
    print(f"  Python   : {env['python']}  numpy {env['numpy_version']}  "
          f"scipy {env['scipy_version']}  sklearn {env['sklearn_version']}  "
          f"pyriemann {env['pyriemann_version']}")
    print(f"  Epoch    : {N_CH} ch / {FS} Hz / {EPOCH_SEC} s")

    all_results = {'environment': env, 'per_config': []}

    for n_ch in args.channels:
        t0 = time.time()
        print(f"\n--- {n_ch} channels ({args.repeats} repeats) ---")
        entry = {'n_channels': n_ch, 'n_repeats': args.repeats}

        entry.update(benchmark_pipeline(n_ch, args.repeats))
        entry.update(benchmark_train_stage(n_ch, args.repeats))
        entry.update(benchmark_inference(args.repeats, n_ch=n_ch))
        entry.update(benchmark_full_pipeline(n_ch, args.repeats))

        for stage in ['filter', 'covariance', 'tangent', 'ftest', 'pca',
                      'svr_inference', 'full_pipeline']:
            s = entry[stage]
            print(f"  {stage:<18s} mean={s['mean_ms']:>9.3f} ms  "
                  f"std={s['std_ms']:>7.3f}  median={s['median_ms']:>9.3f}  "
                  f"P95={s['p95_ms']:>9.3f}")

        # 端到端响应延迟
        acq_s = EPOCH_SEC
        smooth_delay_s = 2 * EPOCH_SEC   # 3 点因果平滑: 当前 epoch 用到 2 个过去 epoch
        infer_s = entry['full_pipeline']['mean_ms'] / 1000.0
        entry['acquisition_window_s'] = acq_s
        entry['smoothing_delay_s'] = smooth_delay_s
        entry['end_to_end_latency_s'] = acq_s + smooth_delay_s + infer_s
        print(f"  acquisition window   : {acq_s} s")
        print(f"  causal smoothing delay: {smooth_delay_s} s "
              f"(3-point MA uses 2 preceding epochs)")
        print(f"  end-to-end latency   : {entry['end_to_end_latency_s']:.2f} s "
              f"(acquisition + smoothing + inference)")

        entry['wall_time_sec'] = time.time() - t0
        all_results['per_config'].append(entry)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(RESULTS_DIR, f'results_latency_{ts}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f'\nSaved: {path}')


if __name__ == '__main__':
    main()
