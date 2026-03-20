#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot SLA violations specifically per network slice to verify isolation and SLA satisfaction.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scenario_creator import scenarios

def movingaverage(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

if __name__=='__main__':
    scenario = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    # You can specify the algorithm you want to inspect (defaults to KBRL_99)
    algo = sys.argv[2] if len(sys.argv) > 2 else 'KBRL_99'
    
    WINDOW = 100

    path = './results/scenario_{}/{}/'.format(scenario, algo)
    if not os.path.exists(path):
        print(f"Directory {path} not found. Please run experiments first.")
        sys.exit(1)

    all_violations = []

    for filename in os.listdir(path):
        if filename.endswith(".npz"):
            histories = np.load(path + filename, allow_pickle=True)
            # histories['violation'] is a list of arrays (e.g. [array([0,1,0]), ...])
            # We convert it into a 2D numpy array of shape (steps, n_slices)
            _violations = np.array(list(histories['violation']))
            all_violations.append(_violations)
    
    if not all_violations:
        print(f"No .npz files found in {path}")
        sys.exit(1)

    # Align all runs to the minimum length in case they stopped at different steps
    min_len = min([len(v) for v in all_violations])
    trimmed_violations = np.array([v[:min_len] for v in all_violations]) # shape: (runs, steps, slices)
    
    # Average across all experiment runs
    mean_violations = np.mean(trimmed_violations, axis=0) # shape: (steps, slices)
    n_slices = mean_violations.shape[1]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    steps = np.arange(min_len - WINDOW + 1)
    
    # Dynamically generate slice labels based on the chosen scenario
    scenario_cfg = scenarios[scenario]
    n_embb = scenario_cfg['n_embb']
    n_mmtc = scenario_cfg['n_mmtc']
    
    slice_labels = {}
    idx = 0
    for _ in range(n_embb):
        slice_labels[idx] = f'Slice {idx} (eMBB)'
        idx += 1
    for _ in range(n_mmtc):
        slice_labels[idx] = f'Slice {idx} (mMTC)'
        idx += 1
    
    for s in range(n_slices):
        smoothed = movingaverage(mean_violations[:, s], WINDOW)
        label = slice_labels.get(s, f'Slice {s}')
        ax.plot(steps, smoothed, label=label, linewidth=2)
    
    ax.set_title(f'SLA Violations Per Individual Slice\n(Scenario {scenario}, Algorithm: {algo})', fontsize=16)
    ax.set_xlabel('Step', fontsize=14)
    ax.set_ylabel('SLA Violations (100-step Moving Average)', fontsize=14)
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True)
    
    out_dir = './figures/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    out_path = os.path.join(out_dir, f'slices_violations_{algo}_sc{scenario}.png')
    fig.tight_layout()
    fig.savefig(out_path, format='png', dpi=300)
    
    print("--------------------------------------------------")
    print(f"Successfully generated per-slice visualization!")
    print(f"Target Algorithm: {algo}")
    print(f"Output saved to : {out_path}")
    print("--------------------------------------------------")
