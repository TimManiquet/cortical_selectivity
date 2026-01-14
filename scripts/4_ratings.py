#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Plots and stats for analyses about image ratings

Author Tim Maniquet
Created 2025/11/07 08:37:12
'''

import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import matplotlib as mpl
import matplotlib.colors as mcolors 

from src import *
from src.utils import *

## Data
ratings = pd.read_table(os.path.join(ratings_data_dir, "ratings.tsv"))
basic_rt_df = pd.read_table(os.path.join(basic_rt_data_dir, "preprocessed_data.tsv"))
brain_data = pd.read_table(os.path.join(brain_data_dir, "average_roi_activations.tsv"))

## Correlate the ratings for each category with the RTs from the basic rt task

rt_means = (
    basic_rt_df
    .groupby(['image', 'target', 'accuracy'], as_index=False)
    .agg(rt=('reaction_time', 'mean'))
)
acc_rt_means = rt_means[rt_means['accuracy']==1]

# Figure 5a: ratings x RTs

for category in ['face', 'body', 'scene']:
    
    category_indices = [img_idx for img_idx, img in enumerate(images) if category in img]
    non_category_indices = [img_idx for img_idx, img in enumerate(images) if category not in img]
    
    category_ratings = ratings[category]
    category_category_ratings = category_ratings[category_indices]
    non_category_ratings = category_ratings[non_category_indices]
    
    task_rts = acc_rt_means.loc[acc_rt_means['target'] == category].set_index('image').reindex(images)['rt'].values
    category_task_rts = task_rts[category_indices]
    non_category_task_rts = task_rts[non_category_indices]

    pref_corr, pref_pvalue = pearsonr(category_category_ratings, category_task_rts)
    print(f"R {category} ratings - RTs for {category} targets: r={pref_corr:.3f}, p={pref_pvalue:.3e}")
    non_pref_corr, non_pref_pvalue = pearsonr(non_category_ratings, non_category_task_rts)
    print(f"R {category} ratings - RTs for all other targets: r={non_pref_corr:.3f}, p={non_pref_pvalue:.3e}")

    # Make a bar plot
    
    plt.figure(figsize=(1.5, 1.2))
    bars = plt.bar(
        x=['non-preferred', 'preferred'],
        height=[non_pref_corr, pref_corr],
        color=['gray', palette[category]],
        alpha=0.5
    )
    
    ax = plt.gca()
    for rect, c in zip(bars, ['non-preferred', category]):
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
    for rect, pval, corr in zip(bars, [non_pref_pvalue, pref_pvalue], [non_pref_corr, pref_corr]):
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
        if corr >= 0:
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
    plt.savefig(os.path.join(figure_dir, f"figure_5a_ratings_x_{category}-task.{figure_ext}"), dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()

## Correlate the ratings for each category with the brain data for each ROI

for roi in rois:
    preferred_category = roi_to_category[roi]
    
    preferred_indices = [img_idx for img_idx, img in enumerate(images) if preferred_category in img]
    non_preferred_indices = [img_idx for img_idx, img in enumerate(images) if preferred_category not in img]
    
    category_ratings = ratings[preferred_category]
    preferred_ratings = category_ratings[preferred_indices]
    preferred_roi_activations = brain_data[roi].values[preferred_indices]

    pref_corr, pref_pvalue = pearsonr(preferred_ratings, preferred_roi_activations)
    print(f"R {preferred_category} ratings - {roi} activations for {preferred_category}: r={pref_corr:.3f}, p={pref_pvalue:.3e}")

    non_preferred_ratings = category_ratings[non_preferred_indices]
    non_preferred_roi_activations = brain_data[roi].values[non_preferred_indices]

    non_pref_corr, non_pref_pvalue = pearsonr(non_preferred_ratings, non_preferred_roi_activations)
    print(f"R {preferred_category} ratings - {roi} activations for non-preferred categories: r={non_pref_corr:.3f}, p={non_pref_pvalue:.3e}")
    
    # Figure 5b: ratings x ROI activations
    
    plt.figure(figsize=(1.5, 1.2))
    bars = plt.bar(
        x=['non-preferred', 'preferred'],
        height=[non_pref_corr, pref_corr],
        color=['gray', palette[preferred_category]],
        alpha=0.5
    )
    
    ax = plt.gca()
    for rect, c in zip(bars, ['non-preferred', preferred_category]):
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
    for rect, pval, corr in zip(bars, [non_pref_pvalue, pref_pvalue], [non_pref_corr, pref_corr]):
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
        if corr >= 0:
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
    plt.savefig(os.path.join(figure_dir, f"figure_5b_ratings_x_{roi}.{figure_ext}"), dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()