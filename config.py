#!/usr/bin/env python3
"""
SCAFBTSRegressor Global Configuration
Centralized paths for Linux workstation (and Windows fallback).
All hardcoded paths live here — edit once, work everywhere.
"""
import os, platform

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Dataset paths
if platform.system() == 'Linux':
    SEED_VIG_ROOT = '/mnt/data1/home/tanhuang/datasets/SEED-VIG'
    DROZY_ROOT    = '/mnt/data1/home/tanhuang/datasets/DROZY'
    SEED_ROOT     = '/mnt/data1/home/tanhuang/datasets/SEED'
else:
    SEED_VIG_ROOT = r'D:\EEG\datasets\SEED-VIG'
    DROZY_ROOT    = r'D:\EEG\datasets\DROZY'
    SEED_ROOT     = r'D:\EEG\datasets\SEED'

DROZY_PSG_DIR = os.path.join(DROZY_ROOT, 'psg')
DROZY_RT_DIR  = os.path.join(DROZY_ROOT, 'pvt-rt')
SEED_FEAT_DIR = os.path.join(SEED_ROOT, 'ExtractedFeatures')
SEED_EEG_DIR  = os.path.join(SEED_ROOT, 'Preprocessed_EEG')

# Output
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'paper', 'figures')
CACHE_DIR   = os.path.join(FIGURES_DIR, 'cache')
PAPER_DIR   = os.path.join(PROJECT_ROOT, 'paper')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Frequency bands
BANDS_5  = [(1,4),(4,8),(8,14),(14,31),(31,50)]
BANDS_8  = [(1,4),(4,6),(6,8),(8,10),(10,12),(12,14),(14,20),(20,30)]
BAND_NAMES_5 = ['Delta(1-4Hz)','Theta(4-8Hz)','Alpha(8-14Hz)','Beta(14-31Hz)','Gamma(31-50Hz)']

# Channel configs
TEMPORAL_6CH = [0,1,2,3,4,5]
FOREHEAD_4CH = [0,1,2,3]
ALL_17CH = list(range(17))

# Options
ESTIMATORS = ['oas','lwf','scm','cov','corr']
REGRESSORS = ['svr','ridge','ridgecv','rfr']
METRICS = ['riemann','euclid','logeuclid']

# Defaults
DEFAULT_FS = 200
DEFAULT_N_FEATURES = 100
DEFAULT_N_JOBS = -1
DEFAULT_QUICK_N = 5
DEFAULT_N_FOLDS = 5
