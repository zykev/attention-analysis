# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 22:25:42 2021

@author: dell
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np


def find_img(points, img_ls):
    
    img_ls = np.array(img_ls)
    idx_ls = []
    for i in points:
        idx = np.argwhere(img_ls == str(i) + '.jpg')
        while len(idx) == 0:
            i = i - 1
            idx = np.argwhere(img_ls == str(i) + '.jpg')
        idx_ls.append(idx.T.squeeze(0)[0])
    return idx_ls
        

atten_prob_path = "E:/Project/child_eyetrace/atten_result2/group0"
emo_path = "E:/Project/child_eyetrace/emo_result2/group0"
save_dir = "E:/Project/child_eyetrace/atten_result2/group0"
 
list1_group0 = ['fj', 'lcz', 'qzy', 'yxy', 'yzq', 'zf']
list1_group1 = ['dhn', 'lmh', 'lsc', 'lyr', 'lyx', 'wzy']
list2_group0 = ['hzx', 'lmz', 'lyl', 'nrx', 'yry']
list2_group1 = ['gsx', 'wzt', 'xzc', 'yzh']


# params = {'fj': [718, 718+975],
#           'lcz': [205, 205+825],
#           'qzy': [750],
#           'yxy': [550, 550+875],
#           'yzq': [200, 200+875],
#           'zf': [299, 288+800],
#           'dhn': [530, 530+900],
#           'lmh': [650, 650+650],
#           'lsc': [1025, 1025+825],
#           'lyr': [425, 425+825],
#           'lyx': [525, 525+825],
#           'wzy': [900, 900+1125],
#           }
params = {'hzx': [600, 600+1205],
          'lmz': [825, 825+925],
          'lyl': [525, 525+750],
          'nrx': [725, 725+1050],
          'yry': [655, 655+1025],
          'ftx': [650, 650+1075],
          'gsx': [1100, 1100+650],
          'wzt': [493, 493+675],
          'xzc': [1200, 1200+675],
          'yzh': [600, 600+925]}

# concatenate analysis data together
# agg_atten_emo = pd.DataFrame()

# for img_name in list2_group0:

#     atten_prob = pd.read_csv(os.path.join(atten_prob_path, img_name + '_prob.csv'), sep=',')
#     emo = pd.read_csv(os.path.join(emo_path, img_name + '.csv'), sep=' ')
    
#     atten_emo = pd.merge(atten_prob, emo, how='inner', on='img_id')
#     atten_emo['vid_id'] = img_name
    
#     idx_ls = find_img(params[img_name], atten_emo['img_id'])
#     if len(idx_ls) == 2:
#         time_ls = [0, idx_ls[0], idx_ls[1], len(atten_emo)]
#     else:
#         time_ls = [0, idx_ls[0], len(atten_emo)]
    
#     atten_emo['class'] = 0
#     for i in range(len(atten_emo)):
#         if i >= time_ls[0] and i < time_ls[1]:
#             atten_emo.loc[i, 'class'] = 0
#         elif i >= time_ls[1] and i < time_ls[2]:
#             atten_emo.loc[i, 'class'] = 1
#         else:
#             atten_emo.loc[i, 'class'] = 2
            
#     atten_emo = atten_emo[['vid_id', 'img_id', 'class', 
#                            'angle','angle_std','dist_3d','dist_3d_std','dist_2d','dist_2d_std',
#                            'prob', 'valence', 'arousal']]       
#     agg_atten_emo = agg_atten_emo.append(atten_emo, ignore_index=True)

# agg_atten_emo.to_csv(os.path.join(save_dir, 'agg_atten_emo.csv'), index=False, sep=',')


# load analysis data
atten_emo = pd.read_csv(os.path.join(atten_prob_path, 'agg_atten_emo.csv'))

# statistical description
agg_res1 = pd.pivot_table(atten_emo,index='vid_id',
                          values=['valence', 'arousal'], 
                          aggfunc={'valence': ['mean', 'std'], 'arousal': ['mean', 'std']})
agg_res = pd.pivot_table(atten_emo,index='vid_id', columns='class', 
                         values=['valence', 'arousal'], 
                         aggfunc={'valence': ['mean', 'std'], 'arousal': ['mean', 'std']})


        

# emotion type description
for vid_name in list2_group1:
    atten_emo_sub = atten_emo[atten_emo['vid_id'] == vid_name]  
    idx_ls = find_img(params[vid_name], atten_emo_sub['img_id'])
    
    if len(idx_ls) == 2:
        time_ls = [0, idx_ls[0], idx_ls[1], len(atten_emo_sub)]
    else:
        time_ls = [0, idx_ls[0], len(atten_emo_sub)]
    for count, i in enumerate(range(len(time_ls))):
        dist1 = dist2 = dist3 = dist4 = 0
        if i != len(time_ls)-1:
            atten_emo_tmp = atten_emo_sub.iloc[time_ls[i]:time_ls[i+1], :]
        else:
            atten_emo_tmp = atten_emo_sub
        for i in range(len(atten_emo_tmp)): 
                if atten_emo_tmp.iloc[i, 10] > 0 and atten_emo_tmp.iloc[i, 11] > 0:
                    dist1 += 1
                elif atten_emo_tmp.iloc[i, 10] < 0 and atten_emo_tmp.iloc[i, 11] > 0:
                    dist2 += 1
                elif atten_emo_tmp.iloc[i, 10] < 0 and atten_emo_tmp.iloc[i, 11] < 0:
                    dist3 += 1
                elif atten_emo_tmp.iloc[i, 10] > 0 and atten_emo_tmp.iloc[i, 11] < 0:
                    dist4 += 1
        with open(os.path.join(save_dir, 'a_emocount_result.txt'), 'a+') as f:
            f.write(vid_name + ' ' + str(count) + ' ' +
                    str(dist1) + ' ' + str(dist2) + ' ' +
                    str(dist3) + ' ' + str(dist4) + ' ' +
                    str(len(atten_emo_tmp)))
            f.write('\n')
        
# correlation between attention prob and emotion value
for vid_name in list2_group1:
    atten_emo_sub = atten_emo[atten_emo['vid_id'] == vid_name]  
    for class_id in range(4):
        if class_id != 3:
            atten_emo_sub2  = atten_emo_sub[atten_emo_sub['class'] == class_id]
            corr_val_prob = atten_emo_sub2.loc[:, 'prob'].corr(atten_emo_sub2.loc[:, 'valence'])
            corr_ar_prob = atten_emo_sub2.loc[:, 'prob'].corr(atten_emo_sub2.loc[:, 'arousal'])
        else:
            corr_val_prob = atten_emo_sub.loc[:, 'prob'].corr(atten_emo_sub.loc[:, 'valence'])
            corr_ar_prob = atten_emo_sub.loc[:, 'prob'].corr(atten_emo_sub.loc[:, 'arousal'])
       
        with open(os.path.join(save_dir, 'a_corr_result.txt'), 'a+') as f:
            f.write(vid_name + ' ' + str(class_id) + ' ' +
                    str(corr_val_prob) + ' ' + str(corr_ar_prob))
            
            f.write('\n')

for vid_name in list2_group1:
    for img_type in ['prob', 'valence', 'arousal']:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        a = atten_emo[atten_emo['vid_id'] == vid_name][img_type]
        ax.scatter(range(len(a)),a)
        ax.set_xlabel('time frame')
        ax.set_ylabel(img_type)
        ax.set_title('atten_emo on ' + vid_name)
        fig.savefig(os.path.join('./atten_emo_plot2/group0', vid_name + '_' + img_type + '.jpg'))
        
# visualize emotion valence and arousal series along time
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111)
# ax.scatter(atten_emo['prob'], atten_emo['valence'], s=10, c='royalblue')
# ax.set_title('corr: ' + str(corr_val_prob))
# ax.set_xlabel('atten prob')
# ax.set_ylabel('emotion valence')
# plt.show()

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111)
# ax.scatter(atten_emo['prob'], atten_emo['arousal'], s=10, c='royalblue')
# ax.set_title('corr: ' + str(corr_ar_prob))
# ax.set_xlabel('atten prob')
# ax.set_ylabel('emotion arousal')
# plt.show()