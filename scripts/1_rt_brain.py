#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Figures and stats for the results concerning RT x brain

Author Tim Maniquet
Created 2025/11/04 13:09:37
'''

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr
import matplotlib as mpl
import matplotlib.colors as mcolors 

from src import *
from src.utils import *

## Data
basic_rt_df = pd.read_table(os.path.join(basic_rt_data_dir, "preprocessed_data.tsv"))
brain_data = pd.read_table(os.path.join(brain_data_dir, "average_roi_activations.tsv"))

## Figure 1B showing brain activity

# create a reording index from images to images_category_ordered
reordering_index = [images.index(img) for img in images_category_ordered]

for roi in rois:
    
    # Extract brain data
    roi_data = brain_data[roi]
    
    # New way: show bar plots + individual dots for each image activation per category
    scale = 0.5
    f, ax = plt.subplots(figsize=(2.0*scale, 1.3*scale))
    avg_activity_per_category = []
    sem_activity_per_category = []
    for category_idx, category in enumerate(categories):
        category_indices = [i for i, img in enumerate(images) if category in img]
        category_activations = roi_data.values[category_indices]
        avg_activity_per_category.append(np.mean(category_activations))
        sem_activity_per_category.append(np.std(category_activations) / np.sqrt(len(category_activations)))
        
        # Add individual points for each image
        ax.scatter(
            category_activations, [len(categories) - 1 - category_idx]*len(category_activations),
            color='None', alpha=0.4, s=10, zorder=2, clip_on=False, edgecolors=palette[category]
        )
    
    ax.barh(categories[::-1], list(reversed(avg_activity_per_category)), xerr=list(reversed(sem_activity_per_category)), zorder=3,
           color=[palette[c] for c in categories[::-1]], alpha=1.0, edgecolor='none', linewidth=0)
    ax.set_xlim(-2.5, 2.5)
    # Turn off clipping for the entire axes
    ax.set_clip_on(False)
        
    plt.savefig(os.path.join(figure_dir, f'figure_1B_{roi}_activations.{figure_ext}'), bbox_inches='tight', dpi=300)
    # plt.show()
    plt.close()
    

## Figure 2: RT x ROI activations

rt_means = (
    basic_rt_df
    .groupby(['image', 'target', 'accuracy', 'category'], as_index=False)
    .agg(rt=('reaction_time', 'mean'))
)
# Select only accurate trials
acc_rt_means = rt_means[rt_means['accuracy']==1]

## Figure 2a: RT distributions across tasks

f, ax = plt.subplots(figsize=(2.3,1.3))
for target in ['face', 'body', 'scene']:
    data = acc_rt_means.loc[(acc_rt_means['target'] == target), 'rt'].dropna()
    # smooth KDE line with filled area under the curve
    sns.kdeplot(data, bw_adjust=0.8, color=palette[target], linewidth=2,
                fill=True, alpha=0.25, label=target, ax=ax, legend=False)

ax.set_xlabel('Mean reaction time (s)')
plt.tight_layout()
plt.savefig(os.path.join(figure_dir, f'figure_2a_basic_rt_task_histogram.{figure_ext}'), bbox_inches='tight', dpi=300)
# plt.show()
plt.close()

## Figure 2a, part II: RT distrubutions per category per task
for task in ['face', 'body', 'scene']:
    task_data = acc_rt_means.loc[(acc_rt_means['target'] == task)]
    f, ax = plt.subplots(figsize=(1.3,0.7))
    avg_rts = [task_data.loc[(task_data['category'] == c), 'rt'].mean() for c in categories]
    sem_rts = [task_data.loc[(task_data['category'] == c), 'rt'].sem() for c in categories]
    ax.bar(categories, avg_rts, yerr=sem_rts, color=[palette[c] for c in categories],
           edgecolor='none', linewidth=0, alpha=1.0)
    # ax.set_xlabel('Mean reaction time (s)')
    # ax.set_yticks([])
    ax.set_ylabel('')
    ax.set_ylim(0.3, 0.8)
    # ax.spines['left'].set_visible(False)
    # ax.set_xlim(0.29, 0.72)
    ax.set_xlabel('')
    ax.set_xticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, f'figure_2a_basic_rt_task_bars_{task}.{figure_ext}'), bbox_inches='tight', dpi=300)
    # plt.show()
    plt.close()

# Figure 2c: scatter plots of RTs vs ROI activations & bar plot of correlations
for roi in rois:
    
    preferred_category = roi_to_category[roi]
    target_rt_data = acc_rt_means.loc[acc_rt_means['target']==preferred_category]
    roi_data = brain_data[roi]
    
    plt.figure(figsize=(1.6,1.6))
    for img in images:
        category = [c for c in categories if c in img][0]
        rt = target_rt_data.loc[target_rt_data['image']==img, 'rt'].values
        roi_data_value = roi_data.loc[brain_data['image']==img].values[0]
        plt.scatter(rt, roi_data_value, color=palette[category], alpha=0.5, s=20)
    plt.xlabel('Mean reaction time (s)')
    plt.ylabel(f'{roi.upper()} activation')
    plt.savefig(os.path.join(figure_dir, f'figure_2b_{roi}_x_rt_activation.{figure_ext}'), bbox_inches='tight', dpi=300)
    # plt.show()
    plt.close()

    # Calculate correlations for each category + all non preferred categories
    correlations = {}
    pvals = {}
    for category in categories:
        category_images = [img for img in images if category in img]
        rt_values = [target_rt_data.loc[target_rt_data['image']==img, 'rt'].values[0] for img in category_images]
        roi_values = [roi_data.loc[brain_data['image']==img].values[0] for img in category_images]
        corr, pvalue = pearsonr(rt_values, roi_values)
        correlations[category] = corr
        pvals[category] = pvalue
    non_preferred_images = [img for img in images if preferred_category not in img]
    rt_values = [target_rt_data.loc[target_rt_data['image']==img, 'rt'].values[0] for img in non_preferred_images]
    roi_values = [roi_data.loc[brain_data['image']==img].values[0] for img in non_preferred_images]
    corr, pvalue = pearsonr(rt_values, roi_values)
    correlations['non_preferred'] = corr
    pvals['non_preferred'] = pvalue
    
    # Make a bar plot of the correlations
    plt.figure(figsize=(1.8,0.7))
    # draw filled bars with per-face alpha (no global alpha) and no edge
    categories_to_plot = [*categories, 'non_preferred']
    bars = plt.bar(
        categories_to_plot,
        [correlations[c] for c in categories_to_plot],
        color=[mcolors.to_rgba(palette.get(c, 'gray'), alpha=0.5) for c in categories_to_plot],
        edgecolor='none',
        linewidth=0,
    )
    # add an inner, fully opaque outline slightly narrower than the filled bar
    ax = plt.gca()
    for rect, c in zip(bars, categories_to_plot):
        x, y = rect.get_x(), rect.get_y()
        w, h = rect.get_width(), rect.get_height()
        shrink = w * 0.06  # adjust how much smaller the outline is (20% of bar width)
        inner = mpl.patches.Rectangle(
            (x + shrink / 2, y + (0 if h >= 0 else 0)),  # keep vertical same, shrink horizontally
            w - shrink,
            h,
            fill=False,
            edgecolor=mcolors.to_rgba(palette.get(c, 'gray'), alpha=1.0),
            linewidth=1.25,
            joinstyle='round',
            # zorder=3,
        )
        ax.add_patch(inner)

    ylim = (-1, 1)
    plt.ylim(*ylim)
    yrange = ylim[1] - ylim[0]
    offset = yrange * 0.05  # distance of star from bar (5% of axis range)
    for rect, c in zip(bars, categories_to_plot):
        pval = pvals.get(c, None)
        if pval is None:
            continue
        star = p_to_star(pval)
        if not star:
            continue
        x, y = rect.get_x(), rect.get_y()
        w, h = rect.get_width(), rect.get_height()
        # determine top and bottom of the bar in data coords
        bar_top = max(y, y + h)
        bar_bottom = min(y, y + h)
        if correlations[c] >= 0:
            star_y = bar_top + offset
            va = 'bottom'
        else:
            star_y = bar_bottom - offset
            va = 'top'
        ax.text(x + w / 2, star_y, star, ha='center', va=va,
                color='black', fontsize=6, zorder=4)
        
    plt.ylim(-1, 1)
    plt.axhline(0, color='black', linewidth=1)  # x bar at 0
    # Remove the top x-axis line (spine)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xticks([])
    plt.xlabel('')
    
    plt.savefig(os.path.join(figure_dir, f'figure_2c_{roi}_rt_activation_correlation.{figure_ext}'), bbox_inches='tight', dpi=300)
    # plt.show()
    plt.close()


    # Print the correlation results
    print("Pearson's R correlations between ROIs and RTs in their preferred category task\n")
    print(f"{roi.upper()} Correlation results:")
    for category, corr in correlations.items():
        pval = pvals[category]
        star = p_to_star(pval)
        print(f"  {category}: r = {corr:.3f}, p = {pval:.4f} {star}")