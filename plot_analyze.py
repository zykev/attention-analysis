# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 16:26:49 2022

@author: Administrator
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from plotnine import ggplot, aes, geom_boxplot
import seaborn as sns

atten_prob_path1 = "E:/Project/child_eyetrace/atten_result/group1"
atten_prob_path2 = "E:/Project/child_eyetrace/atten_result2/group1"

save_dir = "E:/Project/child_eyetrace/analyze"
 
list_group0 = ['fj', 'lcz', 'qzy', 'yxy', 'yzq', 'zf', 'hzx', 'lmz', 'lyl', 'nrx', 'yry']
list_group1 = ['dhn', 'lmh', 'lsc', 'lyr', 'lyx', 'wzy', 'gsx', 'wzt', 'xzc', 'yzh']


# # load analysis data
# atten_emo1 = pd.read_csv(os.path.join(atten_prob_path1, 'agg_atten_emo.csv'))
# atten_emo2 = pd.read_csv(os.path.join(atten_prob_path2, 'agg_atten_emo.csv'))

# atten_emo = pd.concat([atten_emo1,atten_emo2],axis=0,ignore_index=True)

# atten_emo.to_csv(os.path.join(save_dir, 'agg_atten_emo_group1.csv'), index=False, sep=',')

# load analysis data
atten_emo_group0 = pd.read_csv(os.path.join(save_dir, 'agg_atten_emo_group0.csv'))
atten_emo_group1 = pd.read_csv(os.path.join(save_dir, 'agg_atten_emo_group1.csv'))

# statistical description on emotion
agg_res1_group0 = pd.pivot_table(atten_emo_group0,index='vid_id',
                          values=['valence', 'arousal'], 
                          aggfunc={'valence': ['mean', 'std'], 'arousal': ['mean', 'std']})
agg_res_group0 = pd.pivot_table(atten_emo_group0,index='vid_id', columns='class', 
                         values=['valence', 'arousal'], 
                         aggfunc={'valence': ['mean', 'std'], 'arousal': ['mean', 'std']})
agg_res1_group1 = pd.pivot_table(atten_emo_group1,index='vid_id',
                          values=['valence', 'arousal'], 
                          aggfunc={'valence': ['mean', 'std'], 'arousal': ['mean', 'std']})
agg_res_group1 = pd.pivot_table(atten_emo_group1,index='vid_id', columns='class', 
                         values=['valence', 'arousal'], 
                         aggfunc={'valence': ['mean', 'std'], 'arousal': ['mean', 'std']})
agg_res1_group0.columns = ['arousal_mean', 'arousal_std', 'valence_mean', 'valence_std']
agg_res1_group1.columns = ['arousal_mean', 'arousal_std', 'valence_mean', 'valence_std']
agg_res1_group0 = agg_res1_group0.reset_index()
agg_res1_group1 = agg_res1_group1.reset_index()
agg_res1_group0['class'] = 4
agg_res1_group1['class'] = 4
agg_res1_group0['condition'] = 'group0'
agg_res1_group1['condition'] = 'group1'
agg_res1_group0 = agg_res1_group0[['vid_id', 'arousal_mean', 'valence_mean', 'arousal_std', 'valence_std', 'class', 'condition']]
agg_res1_group1 = agg_res1_group1[['vid_id', 'arousal_mean', 'valence_mean', 'arousal_std', 'valence_std', 'class', 'condition']]

# box plot on mean and std of valence and arousal
va_agg = pd.DataFrame()
for i in range(3):
    tmp_1 = agg_res_group0.loc[:, ('arousal', 'mean', i)].to_list()
    tmp_2 = agg_res_group0.loc[:, ('valence', 'mean', i)].to_list()
    tmp_3 = agg_res_group0.loc[:, ('arousal', 'std', i)].to_list()
    tmp_4 = agg_res_group0.loc[:, ('valence', 'std', i)].to_list()
    tmp_df = pd.DataFrame({'vid_id': agg_res_group0.index.to_list(), 
                           'arousal_mean': tmp_1, 'valence_mean': tmp_2,
                           'arousal_std': tmp_3, 'valence_std': tmp_4,
                           'class': i+1})
    va_agg = va_agg.append(tmp_df, ignore_index=True)

va_agg['condition'] = 'group0'


va_agg2 = pd.DataFrame()
for i in range(3):
    tmp_1 = agg_res_group1.loc[:, ('arousal', 'mean', i)].to_list()
    tmp_2 = agg_res_group1.loc[:, ('valence', 'mean', i)].to_list()
    tmp_3 = agg_res_group1.loc[:, ('arousal', 'std', i)].to_list()
    tmp_4 = agg_res_group1.loc[:, ('valence', 'std', i)].to_list()
    tmp_df = pd.DataFrame({'vid_id': agg_res_group1.index.to_list(), 
                           'arousal_mean': tmp_1, 'valence_mean': tmp_2,
                           'arousal_std': tmp_3, 'valence_std': tmp_4,
                           'class': i+1})
    va_agg2 = va_agg2.append(tmp_df, ignore_index=True)

va_agg2['condition'] = 'group1'

va_agg_whole = pd.concat([va_agg, va_agg2, agg_res1_group0, agg_res1_group1], axis=0, ignore_index=True)
va_agg_whole = va_agg_whole.sort_values(by=['class','condition'])

# box plot on class
plt.figure(figsize=(10,12))
plt.subplot(2,2,1)
sns.boxplot(data = va_agg_whole[va_agg_whole['class'] != 4], x = 'class', y = 'arousal_mean', hue = 'condition')
plt.subplot(2,2,2)
sns.boxplot(data = va_agg_whole[va_agg_whole['class'] != 4], x = 'class', y = 'valence_mean', hue = 'condition')
plt.subplot(2,2,3)
sns.boxplot(data = va_agg_whole[va_agg_whole['class'] != 4], x = 'class', y = 'arousal_std', hue = 'condition')
plt.subplot(2,2,4)
sns.boxplot(data = va_agg_whole[va_agg_whole['class'] != 4], x = 'class', y = 'valence_std', hue = 'condition')
plt.savefig(os.path.join(save_dir, 'vaplot.jpg'))

plt.figure(figsize=(10,12))
plt.subplot(2,2,1)
sns.boxplot(data = va_agg_whole[va_agg_whole['class'] == 4], x = 'class', y = 'arousal_mean', hue = 'condition')
plt.subplot(2,2,2)
sns.boxplot(data = va_agg_whole[va_agg_whole['class'] == 4], x = 'class', y = 'valence_mean', hue = 'condition')
plt.savefig(os.path.join(save_dir, 'vaplot2.jpg'))
plt.subplot(2,2,3)
sns.boxplot(data = va_agg_whole[va_agg_whole['class'] == 4], x = 'class', y = 'arousal_std', hue = 'condition')
plt.subplot(2,2,4)
sns.boxplot(data = va_agg_whole[va_agg_whole['class'] == 4], x = 'class', y = 'valence_std', hue = 'condition')
plt.savefig(os.path.join(save_dir, 'vaplot2.jpg'))


# plot on attention
atten_group0 = pd.read_excel(os.path.join(save_dir, 'gaze_analyze2.xlsx'), sheet_name='group0_atten')
atten_group0['condition'] = 'group0'
atten_group1 = pd.read_excel(os.path.join(save_dir, 'gaze_analyze2.xlsx'), sheet_name='group1_atten')
atten_group1['condition'] = 'group1'
atten_group = pd.concat([atten_group0, atten_group1], axis=0, ignore_index=True)
atten_group = atten_group.set_index('vid')
atten_group_sub1 = atten_group[['class', 'total_time', 'dwell_time', 'dwell_freq', 'condition']]
atten_group_sub2 = atten_group[['class', 'first_fix_time', 'first_fix_duration', 'first_fix_dur_std', 'condition']]
atten_group_sub1 = atten_group_sub1[atten_group_sub1.ne('None').all(1)]
atten_group_sub2 = atten_group_sub2[atten_group_sub2.ne('None').all(1)]
atten_group_sub1 = atten_group_sub1.astype({'class': float, 'total_time': float, 'dwell_time': float, 'dwell_freq': float, 'condition': str})
atten_group_sub2 = atten_group_sub2.astype({'class': float, 'first_fix_time': float, 'first_fix_duration': float, 'first_fix_dur_std': float, 'condition': str})


# box plot on class
plt.figure(figsize=(14,12))
plt.subplot(2,2,1)
sns.boxplot(data = atten_group_sub1[atten_group_sub1['class'] != 4], x = 'class', y = 'dwell_time', hue = 'condition')
plt.subplot(2,2,2)
sns.boxplot(data = atten_group_sub1[atten_group_sub1['class'] != 4], x = 'class', y = 'dwell_freq', hue = 'condition')
plt.subplot(2,2,3)
sns.boxplot(data = atten_group_sub2[atten_group_sub2['class'] != 4], x = 'class', y = 'first_fix_time', hue = 'condition')
plt.subplot(2,2,4)
sns.boxplot(data = atten_group_sub2[atten_group_sub2['class'] != 4], x = 'class', y = 'first_fix_duration', hue = 'condition')
plt.savefig(os.path.join(save_dir, 'attenplot.jpg'))

# box plot on total sample
plt.figure(figsize=(8,10))
plt.subplot(2,2,1)
sns.boxplot(data = atten_group_sub1[atten_group_sub1['class'] == 4], x = 'class', y = 'dwell_time', hue = 'condition')
plt.subplot(2,2,2)
sns.boxplot(data = atten_group_sub1[atten_group_sub1['class'] == 4], x = 'class', y = 'dwell_freq', hue = 'condition')
plt.subplot(2,2,3)
sns.boxplot(data = atten_group_sub2[atten_group_sub2['class'] == 4], x = 'class', y = 'first_fix_time', hue = 'condition')
plt.subplot(2,2,4)
sns.boxplot(data = atten_group_sub2[atten_group_sub2['class'] == 4], x = 'class', y = 'first_fix_duration', hue = 'condition')
plt.savefig(os.path.join(save_dir, 'attenplot2.jpg'))


# sample distribution
for i in list_group0:
    plt.figure()
    hist_data = atten_emo_group0[atten_emo_group0['vid_id'] == i]
    hist_factor = hist_data[['arousal', 'class']]
    sns.distplot()
    
# correlation scatter
save_corr = "E:/Project/child_eyetrace/analyze/group0"
for i in list_group0:
    corr_data = atten_emo_group0[atten_emo_group0['vid_id'] == 'fj']
    corr_factor = corr_data[['angle_std', 'dist_3d_std', 'dist_2d_std', 'prob', 'valence', 'arousal']]
    corr_plot = sns.pairplot(corr_factor)
    corr_fig = corr_plot.get_figure()
    corr_fig.savefig(os.path.join(save_corr, 'corr_'+i+'.jpg'))
    
tmp_data = atten_group_sub1[atten_group_sub1['class'] == 3]
tmp_data_0 = tmp_data[tmp_data['condition'] == 'group1']
tmp_data_0.mean()
