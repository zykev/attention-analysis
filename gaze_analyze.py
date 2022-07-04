# -*- coding: utf-8 -*-
"""
Created on Thu May 13 14:03:45 2021

@author: admin
"""

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

from gaze_generate import coord_trans_c



# visualize 3d gaze vector in world coordinate

def plot_3dgaze(gazev_w, count_list, img_name, save_dir):
    fig = plt.figure(figsize=(8, 6))
    ax = Axes3D(fig)
   
    if count_list is not None:
        ax.scatter(gazev_w[count_list, 0], gazev_w[count_list, 1], gazev_w[count_list, 2], c='gold')
        else_list = list(set(list(range(len(gazev_w)))) - set(count_list))
        ax.scatter(gazev_w[else_list, 0], gazev_w[else_list, 1], gazev_w[else_list, 2], c='royalblue', alpha=0.1)
    else:
         ax.scatter(gazev_w[:, 0], gazev_w[:, 1], gazev_w[:, 2], c='royalblue', alpha=0.2)
    ax.scatter(0, 0, 0, c='r', marker='o')
    
    line_x = np.vstack([np.zeros((1,len(gazev_w))), gazev_w[:, 0]]).T
    line_y = np.vstack([np.zeros((1,len(gazev_w))), gazev_w[:, 1]]).T
    line_z = np.vstack([np.zeros((1,len(gazev_w))), gazev_w[:, 2]]).T
    
    for i in range(len(line_x)):
        ax.plot(line_x[i], line_y[i], line_z[i], c='gray', linestyle='--', alpha=0.2, lw=0.5)
    
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()
    save_path = os.path.join(save_dir, img_name + '_3d.jpg')
    fig.savefig(save_path)


# visualize 2d gaze vector in image pixel coordinate

def plot_2dgaze(gazev_c, eye_center_3d, count_list, length, im, img_name, save_dir):
    
    ecenter_u, ecenter_v = coord_trans_c(im, eye_center_3d)
    ecenter = np.vstack((ecenter_u, ecenter_v)).T
    
    gaze_point = eye_center_3d + gazev_c * length[:, np.newaxis]
    gaze2d_u, gaze2d_v = coord_trans_c(im, gaze_point)
    gaze2d = np.vstack((gaze2d_u, gaze2d_v)).T
       
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
       
    ax.scatter(gaze2d[:, 0], gaze2d[:, 1], s=5, c='royalblue', alpha=0.2)
    if count_list is not None:
        ax.scatter(gaze2d[count_list, 0], gaze2d[count_list, 1], s=5, c='green')
    ax.scatter(ecenter[:, 0], ecenter[:, 1], c='orange', marker='o', s=2, alpha=0.2)

    ax.imshow(im[:, :, ::-1])   
    ax.set_axis_off() 
    save_path = os.path.join(save_dir, img_name + '_2d.jpg')
    fig.savefig(save_path)


def estimate_target_3d(im, vec_id, length, eye_center_3d, gazev_c):
    gaze_data = eye_center_3d[vec_id, :] + length * gazev_c[vec_id, :]
    gaze_data2d_u, gaze_data2d_v = coord_trans_c(im, gaze_data.reshape(1,3))
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.scatter(gaze_data2d_u, gaze_data2d_v, s=10, c='red')
    ax.imshow(im[:, :, ::-1])
    ax.set_axis_off() 
    
    return gaze_data

def give_target_3d(im, gaze_data):
    gaze_data = np.array(gaze_data)
    gaze_data2d_u, gaze_data2d_v = coord_trans_c(im, gaze_data.reshape(1,3))
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.scatter(gaze_data2d_u, gaze_data2d_v, s=10, c='red')
    ax.imshow(im[:, :, ::-1])
    ax.set_axis_off() 
    
    return gaze_data

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


def atten_angle(gazev, target_3d, eye_center_3d, thred, plot_angle=False):
    gaze_target = target_3d - eye_center_3d
    if gaze_target.shape[0] == 1:
        gaze_target = np.tile(gaze_target, (len(gazev), 1))
    angle = np.arccos(np.diagonal(np.dot(gazev, gaze_target.T)) / (np.linalg.norm(gazev, axis=1) * np.linalg.norm(gaze_target, axis=1)))
    angle = np.rad2deg(angle)
    
    # count angle within a range
    # count_list = np.argwhere(angle <= thred).T.squeeze(0) 
    count_list = []
    for i in range(0, len(angle)-1, 2):
        if angle[i] <= thred and angle[i+1] <= thred:
            count_list.append(i)
            count_list.append(i+1)
            
    # plot angle list 
    if plot_angle:
        fig = plt.figure(figsize=(10,3))
        ax = fig.add_subplot(111)
        ax.plot(angle)
        ax.set_xlabel('frame')
        ax.set_ylabel('angle between target and gaze')
        fig.show()
        # fig.savefig(os.path.join(save_dir, img_name + '_angle.jpg'))
        
    return count_list

def atten_area(im, eye_center_3d, gazev_c, length, target_box, box_bound=[]):
    
    count_list = []
    box_bound.insert(0,0)
    box_bound.append(len(gazev_c))
    target_box = np.array(target_box)
    
    gaze_point = eye_center_3d + gazev_c * length[:, np.newaxis]
    gaze2d_u, gaze2d_v = coord_trans_c(im, gaze_point)
    gaze2d = np.vstack((gaze2d_u, gaze2d_v)).T
    
    for j in range(len(target_box)):
        for i in range(box_bound[j], box_bound[j+1], 2):
            if gaze2d[i,0] >= target_box[j,0] and gaze2d[i,0] <= target_box[j,2] and gaze2d[i,1] >= target_box[j,1] and gaze2d[i,1] <= target_box[j,3]:
                if gaze2d[i+1,0] >= target_box[j,0] and gaze2d[i+1,0] <= target_box[j,2] and gaze2d[i+1,1] >= target_box[j,1] and gaze2d[i+1,1] <= target_box[j,3]:
                    count_list.append(i)
                    count_list.append(i+1)
    
    return count_list

def atten_anglearea(im, eye_center_c, gazev_w, gazev_c, gazev_length, angle_thred, target_w, target_box, box_bound=[]):
    gaze_angle = np.arccos(np.diagonal(np.dot(gazev_w, target_w.T)) / (np.linalg.norm(gazev_w, axis=1) * np.linalg.norm(target_w, axis=1)))
    gaze_angle = np.rad2deg(gaze_angle)
    
    # count angle within a range
    angle_list = []
    for i in range(0, len(gaze_angle)-1, 2):
        if gaze_angle[i] <= angle_thred and gaze_angle[i+1] <= angle_thred:
            angle_list.append(i)
            angle_list.append(i+1)
    
    area_list = []
    box_bound.insert(0,0)
    box_bound.append(len(gazev_c))
    target_box = np.array(target_box)
    
    gaze_point = eye_center_c + gazev_c * gazev_length[:, np.newaxis]
    gaze2d_u, gaze2d_v = coord_trans_c(im, gaze_point)
    gaze2d = np.vstack((gaze2d_u, gaze2d_v)).T
    
    for j in range(len(target_box)):
        for i in range(box_bound[j], box_bound[j+1], 2):
            if gaze2d[i,0] >= target_box[j,0] and gaze2d[i,0] <= target_box[j,2] and gaze2d[i,1] >= target_box[j,1] and gaze2d[i,1] <= target_box[j,3]:
                if gaze2d[i+1,0] >= target_box[j,0] and gaze2d[i+1,0] <= target_box[j,2] and gaze2d[i+1,1] >= target_box[j,1] and gaze2d[i+1,1] <= target_box[j,3]:
                    area_list.append(i)
                    area_list.append(i+1)
                    
    # count_list = list(set(angle_list).intersection(set(area_list)))
    count_list = list(set(angle_list)|set(area_list))
    count_list.sort()
    
    return count_list
    
def area_judge(ax, ay, px, py, x1, y1, x2, y2):
    
  ax, ay, px, py = float(ax), float(ay), float(px), float(py)
  x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

  top_x = (y1 - ay) * (px - ax) / (py - ay) + ax
  if top_x >= x1 and top_x <= x2:
    return True

  bottom_x = (y2 - ay) * (px - ax) / (py - ay) + ax
  if bottom_x >= x1 and bottom_x <= x2:
    return True

  left_y = (x1 - ax) * (py - ay) / (px - ax) + ay
  if left_y >= y1 and left_y <= y2:
    return True

  right_y = (x2 - ax) * (py - ay) / (px - ax) + ay
  if right_y <= y1 and right_y >= y2:
    return True
  return False

def atten_area2(im, eye_center_3d, gazev_2d, target_box, box_bound=[]):
    
    count_list = []
    box_bound.insert(0,0)
    box_bound.append(len(gazev_2d))
    target_box = np.array(target_box)
    
    ecenter_u, ecenter_v = coord_trans_c(im, eye_center_3d)
    ecenter = np.vstack((ecenter_u, ecenter_v)).T
    gaze_point = gazev_2d + ecenter
    
    for j in range(len(target_box)):
        for i in range(box_bound[j], box_bound[j+1], 2):
            if area_judge(ecenter[i,0], ecenter[i,1], gaze_point[i,0], gaze_point[i,1],
                          target_box[j,0], target_box[j,1], target_box[j,2], target_box[j,3]):
                if area_judge(ecenter[i+1,0], ecenter[i+1,1], gaze_point[i+1,0], gaze_point[i+1,1],
                          target_box[j,0], target_box[j,1], target_box[j,2], target_box[j,3]):
                    count_list.append(i)
                    count_list.append(i+1)
                    
    return count_list


def atten_prob(im, gazev_w, gazev_c, target_w, target_c, eye_center_c, gazev_length, thred):
    
    # calculate angle between target and gaze point
    angle = np.arccos(np.diagonal(np.dot(gazev_w, target_w.T)) / (np.linalg.norm(gazev_w, axis=1) * np.linalg.norm(target_w, axis=1)))
    angle = np.rad2deg(angle)
    angle_std = (angle - min(angle)) / (max(angle) - min(angle))
    
    # calculate 3d distance between target and gaze point in camera coordinate
    gaze_point = eye_center_c + gazev_c * gazev_length[:, np.newaxis]
    dist_3d = np.linalg.norm(gaze_point - target_c, axis=1) ** 2
    dist_3d_std = (dist_3d - min(dist_3d)) / (max(dist_3d) - min(dist_3d))
    
    # caculate 2d distance between target adn gaze point in image coordinate
    gaze2d_u, gaze2d_v = coord_trans_c(im, gaze_point)
    gaze2d = np.vstack((gaze2d_u, gaze2d_v)).T
    target_2d_u, target_2d_v = coord_trans_c(im, target_c.reshape(1,3))
    target_2d = np.vstack((target_2d_u, target_2d_v)).T
    
    dist_2d = np.linalg.norm(gaze2d - target_2d, axis=1) ** 2
    dist_2d_std = (dist_2d - min(dist_2d)) / (max(dist_2d) - min(dist_2d))
    
    # calculate attention probability based on angle and distance
    prob = 1 - (0.5 * angle_std + 0.25 * dist_3d_std + 0.25 * dist_2d_std)
    # count_list = np.argwhere(prob >= thred).T.squeeze(0)
    count_list = []
    for i in range(0, len(prob), 2):
        if prob[i] >= thred and prob[i+1] >= thred:
            count_list.append(i)
            count_list.append(i+1)
    
    return count_list, prob

    
def get_dwell_point(count_list, thred):
    count_dif = np.diff(np.array(count_list))
    count_dif = np.insert(count_dif,0,0)
    dwell_point_idx = np.argwhere(count_dif > thred).T.squeeze(0)
    dwell_point = [count_list[0]]
    for i in dwell_point_idx:
        dwell_point.append(count_list[i-1])
        dwell_point.append(count_list[i])
    dwell_point.append(count_list[-1])
    
    return dwell_point
    

def atten_cal(count_list, num_frames, fqs=30, thred=10):
    
    # total time of the full video series
    total_time = num_frames / (2 * fqs)
    
    # dwell time and freq
    dwell_point = get_dwell_point(count_list, thred)
    dwell_count = 0
    for i in range(0, len(dwell_point), 2):
        dwell_count += dwell_point[i+1] - dwell_point[i] + 1
    dwell_time = dwell_count / (2 * fqs)
    dwell_freq = dwell_count / num_frames
    
    # first fixation time and duration
    fix_point = -1
    for i in range(0, len(dwell_point), 2):
        if dwell_point[i+1] - dwell_point[i] >= 30:
            fix_point = dwell_point[i]
            fix_len = dwell_point[i+1] - dwell_point[i] + 1
            break
    if fix_point != -1:
        fix_time = fix_point / (2 * fqs)
        
       
        fix_duration = fix_len / (2 * fqs)
        fix_duration_std = fix_len / num_frames
    else:
        fix_time = None
        fix_duration = None
        fix_duration_std = None
    
    return total_time, dwell_time, dwell_freq, fix_time, fix_duration, fix_duration_std





