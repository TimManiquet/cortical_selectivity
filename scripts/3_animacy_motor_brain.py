#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Figures and stats for analyses about *animacy* motor task x brain

Author Tim Maniquet
Created 2025/11/04 13:09:37
'''

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from src import *
from src.utils import *

## Data
animacy_motor_df = pd.concat([
    pd.read_pickle(os.path.join(animacy_motor_data_dir, "preprocessed_data_part1.pkl")),
    pd.read_pickle(os.path.join(animacy_motor_data_dir, "preprocessed_data_part2.pkl"))],
ignore_index=True)
brain_data = pd.read_table(os.path.join(brain_data_dir, "average_roi_activations.tsv"))
animacy_trajectories = np.concatenate([
    np.load(os.path.join(animacy_motor_data_dir, f"target-animacy_img_trajectories_part1.npy")),
    np.load(os.path.join(animacy_motor_data_dir, f"target-animacy_img_trajectories_part2.npy"))
])

## Figure 4: motor movements & correlation with brain

# figure 4a: all trajectories of example participant

for ex_ppt in [10]:
    
    ex_data = animacy_motor_df[animacy_motor_df['ppt_number'] == ex_ppt]
    
    plt.figure(figsize=(2.0, 1.6))
    ax = plt.gca()
    for i, row in ex_data.iterrows():
        img = row['image']
        img_category = row['animacy']
        x = row['scaled_x'][:max_t]
        y = row['scaled_y'][:max_t]
        ax.plot(x, y, color=palette[img_category], alpha=0.2, clip_on=False)
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('X coordinate (scaled)')
    ax.set_ylabel('Y coordinate (scaled)')
    ax.spines['left'].set_position(('outward', 15))
    ax.spines['bottom'].set_position(('outward', 15))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(os.path.join(figure_dir, f'figure_4a_ex-ppt-{ex_ppt}_trajectories.{figure_ext}'), bbox_inches='tight', dpi=300)
    # plt.show()
    plt.close()

# figure 4b: average x coordinates across time

random_img_indices = np.random.choice(len(images), size=185, replace=False)

avg_trajectories = np.nanmean(animacy_trajectories, axis=1)

f, ax = plt.subplots(figsize=(2.0, 1.5))
# f, ax = plt.subplots(figsize=(2.2, 1.5))
for random_img_idx in random_img_indices:
    trajectory = avg_trajectories[random_img_idx, :max_t] # without smoothing
    trajectory = pd.Series(trajectory).rolling(window=10, min_periods=1, center=True).mean().values
    img_animacy = category_to_animacy[[c for c in categories if c in images[random_img_idx]][0]]
    ax.plot(range(max_t), trajectory, color=palette[img_animacy], alpha=0.15)
plt.savefig(os.path.join(figure_dir, f'figure_4b_avg_trajectories_animacy.{figure_ext}'), bbox_inches='tight', dpi=300)
# plt.show()
plt.close()

# Run the correlation between brain activations and motor trajectories

roi_corrs = {roi:{} for roi in rois}
for roi in rois:
    
    preferred_category = roi_to_category[roi]
    roi_data = brain_data[roi]
    
    img_indices = {
        'face': [idx for idx, img in enumerate(images) if 'face' in img],
        'body': [idx for idx, img in enumerate(images) if 'body' in img],
        'object': [idx for idx, img in enumerate(images) if 'object' in img],
        'scene': [idx for idx, img in enumerate(images) if 'scene' in img],
        'non-preferred': [idx for idx, img in enumerate(images) if preferred_category not in img]
    }
    correlations = {}
    
    for category, img_indices in img_indices.items():
        corrs = []
        pvalues = []
        for t in range(max_t):
            traj_values = np.nanmean(animacy_trajectories[img_indices, :, t], axis=1)
            brain_values = roi_data.iloc[img_indices].values            
            corr, pvalue = pearsonr(traj_values, brain_values)
            corrs.append(corr)
            pvalues.append(pvalue)
        roi_corrs[roi][category] = {'corrs': corrs, 'pvalues': pvalues}
        
        # Print some results
        peak_idx = np.nanargmax(np.abs(corrs))
        peak_corr = corrs[peak_idx]
        peak_pvalue = pvalues[peak_idx]
        # Find periods of significance (pvalue < 0.05)
        sig_mask = np.array(pvalues) < 0.05
        periods = []
        in_period = False
        for idx, sig in enumerate(sig_mask):
            if sig and not in_period:
                start = idx
                in_period = True
            elif not sig and in_period:
                end = idx - 1
                periods.append((start, end))
                in_period = False
        if in_period:
            periods.append((start, max_t - 1))
        print(f"{roi.upper()} Category: {category}")
        print(f"    Significant periods (p < 0.05): {periods}")
        print(f"    Peak correlation: {peak_corr:.3f} at t={peak_idx} (pvalue={peak_pvalue:.3e})\n")


# Figure 4c: time-resolved correlations for each category

for roi in rois:
    
    f, ax = plt.subplots(figsize=(1.9, 1.2))
    # f, ax = plt.subplots(figsize=(2.2, 1.5))
    for category in roi_corrs[roi].keys():
        ax.plot(
            range(max_t),
            roi_corrs[roi][category]['corrs'],
            color=[palette[category] if category in categories else 'gray'][0], 
            alpha=0.5,
            clip_on = False
        )
    plt.ylim(-0.6, 0.6)
    plt.axhline(0, color='black', linewidth=1)  # x bar at 0
    # Remove the top x-axis line (spine)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xticks([])
    plt.xlabel('')
    
    dot_y_base = -0.55
    if roi == 'ppa':
        dot_y_base = 0.45
        
    dot_y_step = 0.05
    dot_size = 1

    for i, category in enumerate(roi_corrs[roi].keys()):
        pvals = np.array(roi_corrs[roi][category]['pvalues'])
        sig_mask = pvals < 0.05
        dot_y = dot_y_base + i * dot_y_step
        ax.scatter(
            np.where(sig_mask)[0],
            np.full(sig_mask.sum(), dot_y),
            color=[palette[category] if category in categories else 'gray'][0],
            s=dot_size,
            marker='*',
            alpha=0.2,
            zorder=10,
            clip_on=False
        )
    plt.savefig(os.path.join(figure_dir, f'figure_4c_{roi}_trajectory_brain_correlation.{figure_ext}'), bbox_inches='tight', dpi=300)
    # plt.show()
    plt.close()

# Figure 4d: time-resolved correlations for animate and inanimate images

roi_corrs = {roi:{} for roi in rois}
for roi in rois:
    
    roi_data = brain_data[roi]
    
    img_indices = {
        'animate': [idx for idx, img in enumerate(images) if category_to_animacy[[c for c in categories if c in img][0]]=='animate'],
        'inanimate': [idx for idx, img in enumerate(images) if category_to_animacy[[c for c in categories if c in img][0]]=='inanimate']
    }
    correlations = {}
    
    for animacy, img_indices in img_indices.items():
        corrs = []
        pvalues = []
        for t in range(max_t):
            traj_values = np.nanmean(animacy_trajectories[img_indices, :, t], axis=1)
            brain_values = roi_data.iloc[img_indices].values
            corr, pvalue = pearsonr(traj_values, brain_values)
            corrs.append(corr)
            pvalues.append(pvalue)
        roi_corrs[roi][animacy] = {'corrs': corrs, 'pvalues': pvalues}
        
        # Print some results
        peak_idx = np.nanargmax(np.abs(corrs))
        peak_corr = corrs[peak_idx]
        peak_pvalue = pvalues[peak_idx]
        # Find periods of significance (pvalue < 0.05)
        sig_mask = np.array(pvalues) < 0.05
        periods = []
        in_period = False
        for idx, sig in enumerate(sig_mask):
            if sig and not in_period:
                start = idx
                in_period = True
            elif not sig and in_period:
                end = idx - 1
                periods.append((start, end))
                in_period = False
        if in_period:
            periods.append((start, max_t - 1))
        print(f"{roi.upper()} Category: {animacy}")
        print(f"    Significant periods (p < 0.05): {periods}")
        print(f"    Peak correlation: {peak_corr:.3f} at t={peak_idx} (pvalue={peak_pvalue:.3e})\n")

for roi in rois:
    
    f, ax = plt.subplots(figsize=(1.5, 0.7))
    # f, ax = plt.subplots(figsize=(2.2, 1.5))
    for animacy in roi_corrs[roi].keys():
        ax.plot(
            range(max_t),
            roi_corrs[roi][animacy]['corrs'],
            color=palette[animacy],
            alpha=0.5,
            clip_on = False
        )
    plt.ylim(-0.6, 0.6)
    plt.axhline(0, color='black', linewidth=1)  # x bar at 0
    # Remove the top x-axis line (spine)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xticks([])
    plt.xlabel('')
    
    dot_y_base = -0.55
    if roi == 'ppa':
        dot_y_base = 0.55
        
    dot_y_step = 0.05
    dot_size = 1

    for i, animacy in enumerate(roi_corrs[roi].keys()):
        pvals = np.array(roi_corrs[roi][animacy]['pvalues'])
        sig_mask = pvals < 0.05
        dot_y = dot_y_base + i * dot_y_step
        ax.scatter(
            np.where(sig_mask)[0],
            np.full(sig_mask.sum(), dot_y),
            color=palette[animacy],
            s=dot_size,
            marker='*',
            alpha=0.2,
            zorder=10,
            clip_on=False
        )
    plt.savefig(os.path.join(figure_dir, f'figure_4d_{roi}_trajectory_brain_correlation_animacy.{figure_ext}'), bbox_inches='tight', dpi=300)
    # plt.show()
    plt.close()