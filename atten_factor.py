# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 17:13:00 2021

@author: dell
"""
import os
import cv2
import numpy as np
from gaze_analyze import *
from gaze_generate import coord_trans_c2w
import pandas as pd


img_name = 'qzy'
img_dir = os.path.join('D:/Project/child_eyetrace/Data/img_group0', img_name)
result_dir = 'D:/Project/child_eyetrace/gaze_result/group0'
save_dir = 'D:/Project/child_eyetrace/atten_result/group0'



# get gaze vector
gazev_dir = os.path.join(result_dir, img_name + '.npz')
gazev = np.load(gazev_dir)
gazev_c = gazev['gazev_c']
gazev_w = gazev['gazev_w']
gazev_2d = gazev['gazev_2d']
eye_center_3d = gazev['eye_center_3d']
eye_center = gazev['eye_center']
rot_all = gazev['rot_all']
trans_all = gazev['trans_all']

# get image index list
vector_path = os.path.join(result_dir, img_name + '.txt')
gaze_vector = np.loadtxt(vector_path, delimiter=' ', dtype=str)
select_img = np.array(gaze_vector[:, 0]).repeat(2)

# # estimate new sample
# im = cv2.imread(os.path.join(img_dir, select_img[200])) 
# # target_3d = estimate_target_3d(im, 139, 0.53, eye_center_3d, gazev_c) 
# target_3d = give_target_3d(im, [0.01531588, 0.08479741, 0.24073954]) 
# print(target_3d)

params = {'child': ([]), #1634
          'fj': ([0.02689303, 0.07233836, 0.22342839], [718, 718+975]),
          'lcz': ([-0.02400704, 0.08636678, 0.20595967], [205, 205+825]),
          'qzy': ([0.01484843, 0.08025278, 0.2442926], [750]),
          'yxy': ([-0.01007869, 0.08823072, 0.22666836], [550, 550+875]),
          'yzq': ([0.01531588, 0.08479741, 0.24073954], [200, 200+875]),
          'zf': ([0.02476592, 0.08243534, 0.26315121], [299, 288+800]),
          'dhn': ([-0.0835438, 0.09610221, 0.24519613], [530, 530+900]),
          'lmh': ([-0.0128928, 0.08060925, 0.29178958], [650, 650+650]),
          'lsc': ([-0.01501477, 0.09031127, 0.24619808], [1025, 1025+825]),
          'lyr': ([-0.00531588, 0.08879741, 0.26073954], [425, 425+82]),
          'lyx': ([-0.00531588, 0.08879741, 0.26073954], [525, 525+825]),
          'wzy': ([-0.04876592, 0.06243534, 0.17315121], [900, 900+1125])
          }


# get sample image
im = cv2.imread(os.path.join(img_dir, select_img[0])) 

# find a proper target 3d in camera coordinate
target_3d = np.array(params[img_name][0])

# target_3d in world coordinate
target_3d_w = []
for i in range(len(rot_all)):
    target_3d_w.append(coord_trans_c2w(target_3d, rot_all[i, :], trans_all[i, :]))
target_3d_w = np.vstack(target_3d_w)
target_3d_w = np.repeat(target_3d_w, 2, axis=0)

# distance between target and eye center in world coordinate
target_length_w = np.linalg.norm(target_3d_w, axis=1)

# distance between target and eye center in camera coordinate
target_length_c = np.linalg.norm(eye_center_3d - target_3d , axis=1)

# target attention probability
count_list, prob = atten_prob(im, gazev_w, gazev_c, target_3d_w, target_3d, 
                              eye_center_3d, target_length_c, 0.8)
save_data = pd.DataFrame({'img_id': select_img, 'prob': prob})
save_path = os.path.join(save_dir, img_name + '_prob.csv')
save_data.to_csv(save_path, sep=' ', index=0)

# plot gaze figure
plot_3dgaze(gazev_w, count_list, img_name, save_dir)
plot_2dgaze(gazev_c, eye_center_3d, count_list, target_length_c, im, img_name, save_dir)

# # calculate gaze attention factor
# total_time, dwell_time, dwell_freq, fix_time, fix_duration, fix_duration_std = atten_cal(count_list, len(gazev_w))
# print(total_time, dwell_time, dwell_freq, fix_time, fix_duration, fix_duration_std)

# calculate gaze attention factor on different time lines
idx_ls = find_img(params[img_name][1], select_img)
if len(idx_ls) == 2:
    time_ls = [0, idx_ls[0], idx_ls[1], len(select_img)]
else:
    time_ls = [0, idx_ls[0], len(select_img)]


for count, i in enumerate(range(len(time_ls))):
    if i != len(time_ls)-1:
        count_list, prob = atten_prob(im, gazev_w[time_ls[i]:time_ls[i+1]], 
                                      gazev_c[time_ls[i]:time_ls[i+1]], 
                                      target_3d_w[time_ls[i]:time_ls[i+1]], 
                                      target_3d, 
                                      eye_center_3d[time_ls[i]:time_ls[i+1]], 
                                      target_length_c[time_ls[i]:time_ls[i+1]], 0.8)
                                                      
        total_time, dwell_time, dwell_freq, fix_time, fix_duration, fix_duration_std = atten_cal(count_list, 
                                                                                                 len(gazev_c[time_ls[i]:time_ls[i+1]]))
    else:
        count_list, prob = atten_prob(im, gazev_w, gazev_c, target_3d_w, target_3d, 
                                      eye_center_3d, target_length_c, 0.8)
                                                      
        total_time, dwell_time, dwell_freq, fix_time, fix_duration, fix_duration_std = atten_cal(count_list, len(gazev_w))
   
    with open(os.path.join(save_dir, img_name + '_atten_factor.txt'), 'a+') as f:
        if fix_time is not None:
            f.write(img_name + ' ' + str(count+1) + ' ' +
                    str(round(total_time, 4)) + ' ' +
                    str(round(dwell_time, 4)) + ' ' + 
                    str(round(dwell_freq, 4)) + ' ' +
                    str(round(fix_time, 4)) + ' ' +
                    str(round(fix_duration, 4)) + ' ' +
                    str(round(fix_duration_std, 4)))
        else:
            f.write(img_name + ' ' + str(count+1) + ' ' +
                    str(round(total_time, 4)) + ' ' +
                    str(round(dwell_time, 4)) + ' ' + 
                    str(round(dwell_freq, 4)) + ' ' +
                    str(None) + ' ' +
                    str(None) + ' ' +
                    str(None))
        f.write('\n')


# get gaze_data 2d in specific image
# id_num = 90
# im = cv2.imread(os.path.join(img_dir, select_img[id_num]))
# gaze_data = eye_center_3d[id_num, :] + target_length_c[id_num] * gazev_c[id_num, :]
# gaze_data2d_u, gaze_data2d_v = coord_trans_c(im, gaze_data.reshape(1,3))
# u, v = coord_trans_w(im, target_3d_w[id_num, :].reshape(1, 3), rot_all[id_num // 2, :], trans_all[id_num // 2, :])
# ecenter_u, ecenter_v = coord_trans_c(im, eye_center_3d[id_num, :].reshape(1,3))
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111)
# ax.imshow(im[:, :, ::-1])
# ax.scatter(gaze_data2d_u, gaze_data2d_v, s=5, c='royalblue')
# ax.scatter(u, v, s=5, c='red')
# ax.scatter(ecenter_u, ecenter_v, s=5, c='orange')
# ax.set_axis_off()