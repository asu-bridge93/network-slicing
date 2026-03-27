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
    WINDOW = 100

    base_path = f'./results/scenario_{scenario}/'
    if not os.path.exists(base_path):
        print(f"Directory {base_path} not found. Please run experiments first.")
        sys.exit(1)

    if len(sys.argv) > 2:
        algos = [sys.argv[2]]
    else:
        algos = [name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name))]
        order_map = {'KBRL': 0, 'QR_nocost': 1, 'QR_cost': 2}
        algos.sort(key=lambda x: order_map.get(x, 99))
    
    if not algos:
        print(f"No algorithm directories found in {base_path}")
        sys.exit(1)

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

    all_algo_data = {}

    for algo in algos:
        path = os.path.join(base_path, algo + '/')
        all_violations = []
    
        if not os.path.exists(path):
            continue

        for filename in os.listdir(path):
            if filename.endswith(".npz"):
                histories = np.load(path + filename, allow_pickle=True)
                # histories['violation'] is a list of arrays (e.g. [array([0,1,0]), ...])
                # We convert it into a 2D numpy array of shape (steps, n_slices)
                _violations = np.array(list(histories['violation']))
                all_violations.append(_violations)
        
        if not all_violations:
            print(f"No .npz files found in {path}")
            continue
    
        # Align all runs to the minimum length in case they stopped at different steps
        min_len = min([len(v) for v in all_violations])
        trimmed_violations = np.array([v[:min_len] for v in all_violations]) # shape: (runs, steps, slices)
        
        # Average across all experiment runs
        mean_violations = np.mean(trimmed_violations, axis=0) # shape: (steps, slices)
        n_slices = mean_violations.shape[1]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        steps = np.arange(min_len - WINDOW + 1)
        
        smoothed_slices = []
        for s in range(n_slices):
            smoothed = movingaverage(mean_violations[:, s], WINDOW)
            smoothed_slices.append(smoothed)
            label = slice_labels.get(s, f'Slice {s}')
            ax.plot(steps, smoothed, label=label, linewidth=2)
            
        all_algo_data[algo] = {
            'smoothed': smoothed_slices,
            'steps': steps
        }
        
        ax.set_title(f'SLA Violations Per Individual Slice\n(Scenario {scenario}, Algorithm: {algo})', fontsize=16)
        ax.set_xlabel('Step', fontsize=14)
        ax.set_ylabel('SLA Violation Rate\n(100-step Moving Average)', fontsize=14)
        ax.set_ylim(0, 0.01)
        ax.legend(loc='upper right', fontsize=12)
        ax.grid(True)
        
        out_dir = './figures/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        out_path = os.path.join(out_dir, f'slices_violations_{algo}_sc{scenario}.png')
        fig.tight_layout()
        fig.savefig(out_path, format='png', dpi=300)
        plt.close(fig)
        
        print("--------------------------------------------------")
        print(f"Successfully generated per-slice visualization!")
        print(f"Target Algorithm: {algo}")
        print(f"Output saved to : {out_path}")
        print("--------------------------------------------------")
        
    # Generate combined plot if multiple algorithms were found
    if len(all_algo_data) > 1:
        first_algo = list(all_algo_data.keys())[0]
        n_slices_comb = len(all_algo_data[first_algo]['smoothed'])
        
        fig_comb, axes_comb = plt.subplots(nrows=n_slices_comb, ncols=1, figsize=(12, 4 * n_slices_comb), sharex=True)
        if n_slices_comb == 1:
            axes_comb = [axes_comb]
            
        color_list = plt.cm.tab10.colors
        
        for s in range(n_slices_comb):
            ax = axes_comb[s]
            label = slice_labels.get(s, f'Slice {s}')
            ax.set_title(f'{label} SLA Violations (Scenario {scenario})', fontsize=16)
            
            for i, algo in enumerate(all_algo_data.keys()):
                algo_color = color_list[i % len(color_list)]
                data = all_algo_data[algo]
                
                # Format the label similarly to plot_results.py
                label_algo = algo.split('_')[0] + '_' + algo.split('_')[-1]
                if label_algo == 'QR_nocost': label_algo = 'SQR w/o Cost'
                if label_algo == 'QR_cost': label_algo = 'SQR w/ Cost'
                if algo == 'KBRL': label_algo = 'KBRL'
                if algo == 'KBRL_99': label_algo = 'KBRL'
                
                # if there is not enough length, plot as much as available
                if s < len(data['smoothed']):
                    ax.plot(data['steps'], data['smoothed'][s], label=label_algo, linewidth=2, color=algo_color)
                
            ax.set_ylabel('SLA Violation Rate', fontsize=14)
            ax.set_ylim(0, 0.01)
            ax.legend(loc='upper right', fontsize=12)
            ax.grid(True)
            
        axes_comb[-1].set_xlabel('Step', fontsize=14)
        
        out_path_comb = os.path.join(out_dir, f'slices_violations_combined_sc{scenario}.png')
        fig_comb.tight_layout()
        fig_comb.savefig(out_path_comb, format='png', dpi=300)
        plt.close(fig_comb)
        
        print(f"Successfully generated combined slices visualization!")
        print(f"Combined Output saved to : {out_path_comb}")
        print("--------------------------------------------------")
