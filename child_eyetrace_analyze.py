# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 14:15:09 2021

@author: admin
"""
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter
import cv2
from scipy.spatial.transform import Rotation

from head_pose import head_pose

ffmpegpath = os.path.abspath("D:/ffmpeg/bin/ffmpeg.exe")
matplotlib.rcParams["animation.ffmpeg_path"] = ffmpegpath
save_dir = 'E:/pytorch_mpiigaze_demo/child_eyetrace/result'
file_path = 'E:/pytorch_mpiigaze_demo/child_eyetrace/process_csv/fj.csv'


data = pd.read_csv(file_path)
data.columns = ['frame','face_id','timestamp','confidence','success',
                                     'gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z',
                                     'gaze_angle_x', 'gaze_angle_y']



fig = plt.figure(figsize=(8,6))  # 设置画布大小
ax = Axes3D(fig)
ax.scatter(data['gaze_0_x'], data['gaze_0_y'], data['gaze_0_z'])  # 三个数组对应三个维度（三个数组中的数一一对应）
ax.scatter(0,0,0,c='r',marker='o')

line_x = np.vstack([np.zeros((1,len(data))), data['gaze_0_x']]).T
line_y = np.vstack([np.zeros((1,len(data))), data['gaze_0_y']]).T
line_z = np.vstack([np.zeros((1,len(data))), data['gaze_0_z']]).T

for i in range(len(line_x)):
    ax.plot(line_x[i], line_y[i], line_z[i], c='gray', linestyle='--', alpha=0.2, lw=0.5)


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
save_path = os.path.join(save_dir, file_path.split('/')[-1] + '_3d.jpg')
plt.savefig(save_path)

# projection to 2D plane
image_path = 'E:/pytorch_mpiigaze_demo/Data/images/fj/100.jpg'
# Read Image
im = cv2.imread(image_path)
estimator = head_pose(im)
boxes, image_points = estimator.face_estimator()
camera_matrix, rotation_vector, translation_vector, dist_coeffs = estimator.head_estimator(image_points)

points_3d = np.array(data[['gaze_0_x', 'gaze_0_y', 'gaze_0_z']])
(points_2d, jacobian) = cv2.projectPoints(points_3d,
                                 rotation_vector,
                                 translation_vector,
                                 camera_matrix,
                                 dist_coeffs)
points_2d = np.squeeze(points_2d, axis=1)
plt.figure(figsize=(8,6))
plt.scatter(points_2d[:,0],points_2d[:,1])
plt.show()
save_path = os.path.join(save_dir, file_path.split('/')[-1] + '_2d.jpg')
plt.savefig(save_path)


# plot points_2d on orig image
plt.figure(figsize=(12,8))
im = im[:, :, ::-1]
plt.imshow(im)
plt.scatter(points_2d[:,0], points_2d[:,1])
plt.show()
        
        
# animation
#2d
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot((111))



def init():
    ax.set_xlim(min(points_2d[:,0]) - 0.03, max(points_2d[:,0]) + 0.03)
    ax.set_ylim(min(points_2d[:,1]) - 0.03, max(points_2d[:,1]) + 0.03)
    
def update(frame):
    graph.set_offsets(points_2d[0:frame,:])
    
    return graph

graph = ax.scatter(points_2d[0,0], points_2d[0,1], c='blue', marker='o')
ani = FuncAnimation(fig, update, frames=len(points_2d),
                    init_func=init, interval=1, blit=False)
# plt.show()

writer = FFMpegWriter()
save_path = os.path.join(save_dir, file_path.split('/')[-1] + '_2d_ani.mp4')
ani.save(save_path, writer = writer)

#3d
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot((111), projection='3d')

data_draw = data[['gaze_0_x', 'gaze_0_y', 'gaze_0_z']]
data_draw = data_draw.reset_index()

def init():
    
    ax.scatter(0, 0, 0, c = 'red', marker = 'o')
    ax.set_xlim3d(min(data_draw['gaze_0_x']) - 0.02, max(data_draw['gaze_0_x']) + 0.02)
    ax.set_ylim3d(min(data_draw['gaze_0_y']) - 0.02, max(data_draw['gaze_0_y']) + 0.02)
    ax.set_zlim3d(min(data_draw['gaze_0_z']) - 0.02, max(max(data_draw['gaze_0_z']) + 0.02, 0))
    
def update(frame, graph2):
    plot_data = data_draw.iloc[0:frame,:]
    plot_data = np.array(plot_data)
    if plot_data.ndim == 1:
        plot_data = np.expand_dims(plot_data, 1).T

    graph._offsets3d = (plot_data[:,1], plot_data[:,2], plot_data[:,3])
    

    line_data = np.vstack([np.zeros((1,4)), data_draw.iloc[frame,:]])
    for line in graph2:
        line.set_data(line_data[:,1:3].T)
        line.set_3d_properties(line_data[:,3])
    
    return graph, graph2

plot_data = data_draw.iloc[0,:]
plot_data = np.array(plot_data)
if plot_data.ndim == 1:
    plot_data = np.expand_dims(plot_data, 1).T
    
line_data = np.vstack([np.zeros((1,4)), data_draw.iloc[0,:]])
graph = ax.scatter(plot_data[:,1], plot_data[:,2], plot_data[:,3], c = 'blue', marker = 'o')
graph2 = [ax.plot(line_data[:,1], line_data[:,2], line_data[:,3], 
                 c = 'gray', linestyle = '--', lw = 1)]


ani = FuncAnimation(fig, update, frames=len(data), init_func=init(), fargs=(graph2), interval=1, blit=False)
# plt.show()

writer = FFMpegWriter()
save_path = os.path.join(save_dir, file_path.split('/')[-1] + '_3d_ani.mp4')
ani.save(save_path, writer = writer)
