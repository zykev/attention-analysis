# -*- coding: utf-8 -*-
"""
Created on Fri May  7 11:32:09 2021

@author: admin
"""

# 3d gaze points to 2d projection on camera image

import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import dlib
from facenet_pytorch import MTCNN


# define coordinate transform to transform world coordinates to image pixel coordinates
def coord_trans_w(im, w_coords, rot, trans):
    # first transform world coordinate to camera coordinate
    rot = Rotation.from_rotvec(rot)
    rot = rot.as_matrix()
    c_coords = w_coords @ rot.T + trans.T
    # then transform camera coordinate to image pixel coordinate
    size = im.shape
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    u = np.zeros((len(w_coords), 1))
    u = np.zeros((len(w_coords), 1))
    u = focal_length * c_coords[:, 0] / c_coords[:, 2] + center[0]
    v = focal_length * c_coords[:, 1] / c_coords[:, 2] + center[1]

    return u, v


# define coordinate transform to transform camera coordinates to image pixel coordinates
def coord_trans_c(im, c_coords):
    # transform camera coordinate to image pixel coordinate
    size = im.shape
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    u = np.zeros((len(c_coords), 1))
    u = np.zeros((len(c_coords), 1))
    u = focal_length * c_coords[:, 0] / c_coords[:, 2] + center[0]
    v = focal_length * c_coords[:, 1] / c_coords[:, 2] + center[1]

    return u, v


def coord_trans_c2w(c_coords, rot, trans):
    rot = Rotation.from_rotvec(rot)
    rot = rot.as_matrix()
    w_coords = (c_coords - trans.T) @ np.linalg.inv(rot.T)

    return w_coords


def coord_trans_w2c(w_coords, rot, trans):
    rot = Rotation.from_rotvec(rot)
    rot = rot.as_matrix()
    c_coords = w_coords @ rot.T + trans.T

    return c_coords


def head_pose_estimate(camera_matrix, face_landmarks):
    points_3d = np.array([
        [-0.07141807, -0.02827123, 0.08114384],
        [-0.07067417, -0.00961522, 0.08035654],
        [-0.06844646, 0.00895837, 0.08046731],
        [-0.06474301, 0.02708319, 0.08045689],
        [-0.05778475, 0.04384917, 0.07802191],
        [-0.04673809, 0.05812865, 0.07192291],
        [-0.03293922, 0.06962711, 0.06106274],
        [-0.01744018, 0.07850638, 0.04752971],
        [0., 0.08105961, 0.0425195],
        [0.01744018, 0.07850638, 0.04752971],
        [0.03293922, 0.06962711, 0.06106274],
        [0.04673809, 0.05812865, 0.07192291],
        [0.05778475, 0.04384917, 0.07802191],
        [0.06474301, 0.02708319, 0.08045689],
        [0.06844646, 0.00895837, 0.08046731],
        [0.07067417, -0.00961522, 0.08035654],
        [0.07141807, -0.02827123, 0.08114384],
        [-0.05977758, -0.0447858, 0.04562813],
        [-0.05055506, -0.05334294, 0.03834846],
        [-0.0375633, -0.05609241, 0.03158344],
        [-0.02423648, -0.05463779, 0.02510117],
        [-0.01168798, -0.04986641, 0.02050337],
        [0.01168798, -0.04986641, 0.02050337],
        [0.02423648, -0.05463779, 0.02510117],
        [0.0375633, -0.05609241, 0.03158344],
        [0.05055506, -0.05334294, 0.03834846],
        [0.05977758, -0.0447858, 0.04562813],
        [0., -0.03515768, 0.02038099],
        [0., -0.02350421, 0.01366667],
        [0., -0.01196914, 0.00658284],
        [0., 0., 0.],
        [-0.01479319, 0.00949072, 0.01708772],
        [-0.00762319, 0.01179908, 0.01419133],
        [0., 0.01381676, 0.01205559],
        [0.00762319, 0.01179908, 0.01419133],
        [0.01479319, 0.00949072, 0.01708772],
        [-0.045, -0.032415, 0.03976718],
        [-0.0370546, -0.0371723, 0.03579593],
        [-0.0275166, -0.03714814, 0.03425518],
        [-0.01919724, -0.03101962, 0.03359268],
        [-0.02813814, -0.0294397, 0.03345652],
        [-0.03763013, -0.02948442, 0.03497732],
        [0.01919724, -0.03101962, 0.03359268],
        [0.0275166, -0.03714814, 0.03425518],
        [0.0370546, -0.0371723, 0.03579593],
        [0.045, -0.032415, 0.03976718],
        [0.03763013, -0.02948442, 0.03497732],
        [0.02813814, -0.0294397, 0.03345652],
        [-0.02847002, 0.03331642, 0.03667993],
        [-0.01796181, 0.02843251, 0.02335485],
        [-0.00742947, 0.0258057, 0.01630812],
        [0., 0.0275555, 0.01538404],
        [0.00742947, 0.0258057, 0.01630812],
        [0.01796181, 0.02843251, 0.02335485],
        [0.02847002, 0.03331642, 0.03667993],
        [0.0183606, 0.0423393, 0.02523355],
        [0.00808323, 0.04614537, 0.01820142],
        [0., 0.04688623, 0.01716318],
        [-0.00808323, 0.04614537, 0.01820142],
        [-0.0183606, 0.0423393, 0.02523355],
        [-0.02409981, 0.03367606, 0.03421466],
        [-0.00756874, 0.03192644, 0.01851247],
        [0., 0.03263345, 0.01732347],
        [0.00756874, 0.03192644, 0.01851247],
        [0.02409981, 0.03367606, 0.03421466],
        [0.00771924, 0.03711846, 0.01940396],
        [0., 0.03791103, 0.0180805],
        [-0.00771924, 0.03711846, 0.01940396],
    ], dtype=np.float)

    # Assuming no lens distortion
    dist_coeffs = np.zeros((5, 1))

    rot_vec = np.zeros(3, dtype=np.float)
    trans_vec = np.array([0, 0, 1], dtype=np.float)
    success, rot_vec, trans_vec = cv2.solvePnP(points_3d,
                                               face_landmarks,
                                               camera_matrix,
                                               dist_coeffs,
                                               rot_vec,
                                               trans_vec,
                                               useExtrinsicGuess=True,
                                               flags=cv2.SOLVEPNP_ITERATIVE)

    # eye_center_3d in camera coordinate
    ecenter_r_3d_w = points_3d[[36, 39], :].mean(axis=0)
    ecenter_l_3d_w = points_3d[[42, 45], :].mean(axis=0)
    ecenter_3d_w = np.vstack((ecenter_r_3d_w, ecenter_l_3d_w))

    ecenter_3d_c = coord_trans_w2c(ecenter_3d_w, rot_vec, trans_vec)

    return rot_vec, trans_vec, ecenter_3d_c


def get_gaze_data(img_dir, vector_path):
    gaze_vector = np.loadtxt(vector_path, delimiter=' ', dtype=str)
    predictor_model_path = 'D:/Project/stylegan/face align/detector/shape_predictor_68_face_landmarks.dat'

    detector = MTCNN(select_largest=False, post_process=False)
    shape_predictor = dlib.shape_predictor(predictor_model_path)

    tracking_state = False

    img_list = os.listdir(img_dir)
    img_list.sort(key=lambda x: int(x.split('.')[0]))  # sort files by ascending

    # Camera internals
    im = cv2.imread(os.path.join(img_dir, img_list[0]))
    size = im.shape
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double")

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    gazev_2d = []
    gazev_c = []
    gazev_w = []
    eye_center = []
    eye_center_3d = []
    rot_all = []
    trans_all = []

    for img_id, img_name in enumerate(img_list):
        img_path = os.path.join(img_dir, img_name)
        frame = cv2.imread(img_path)
        frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
        dets, _ = detector.detect(frame[:, :, ::-1])

        landmarks = None
        if tracking_state is False:
            if len(dets) > 0:
                for box in dets:
                    tracking_state = True
                    last_box_center = [np.mean([box[0], box[2]]), np.mean([box[1], box[3]])]
                    bbox = dlib.rectangle(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                    predictions = shape_predictor(frame[:, :, ::-1], bbox)
                    landmarks = np.array([(pt.x, pt.y) for pt in predictions.parts()],
                                         dtype=np.float)
                    rotv, transv, ecenter_3d_c = head_pose_estimate(camera_matrix, landmarks)



        else:
            if len(dets) > 0:
                min_center = 5000
                min_center_id = -1
                for i, box in enumerate(dets):
                    box_center = [np.mean([box[0], box[2]]), np.mean([box[1], box[3]])]
                    box_center_dist = np.linalg.norm(np.array(box_center) - np.array(last_box_center)) ** 2
                    if box_center_dist < min_center:
                        min_center = box_center_dist
                        min_center_id = i
                if min_center_id != -1:
                    last_box = dets[min_center_id]
                    last_box_center = [np.mean([last_box[0], last_box[2]]), np.mean([last_box[1], last_box[3]])]
                    bbox = dlib.rectangle(int(last_box[0]), int(last_box[1]), int(last_box[2]), int(last_box[3]))
                    predictions = shape_predictor(frame[:, :, ::-1], bbox)
                    landmarks = np.array([(pt.x, pt.y) for pt in predictions.parts()],
                                         dtype=np.float)

                    rotv, transv, ecenter_3d_c = head_pose_estimate(camera_matrix, landmarks)

        if landmarks is not None:
            gazev = gaze_vector[gaze_vector[:, 0] == img_name][0]
            gazev = np.array(gazev[2:]).astype(np.float)
            gazev_c.append(gazev.reshape((2, 3)))

            gaze_point = ecenter_3d_c + 0.5 * gazev.reshape((2, 3))
            gaze2d_u, gaze2d_v = coord_trans_c(im, gaze_point)
            gaze2d = np.vstack((gaze2d_u, gaze2d_v)).T
            ecenter2d_u, ecenter2d_v = coord_trans_c(im, ecenter_3d_c)
            ecenter2d = np.vstack((ecenter2d_u, ecenter2d_v)).T
            gazev_2d.append(gaze2d - ecenter2d)

            gazev_w.append(coord_trans_c2w(gazev.reshape((2, 3)), rotv, transv))

            rot_all.append(rotv)
            trans_all.append(transv)

            ecenter_left = landmarks[[36, 39], :].mean(axis=0)
            ecenter_right = landmarks[[42, 45], :].mean(axis=0)
            eye_center.append(np.vstack((ecenter_left, ecenter_right)))

            eye_center_3d.append(ecenter_3d_c)

    gazev_2d = np.vstack(gazev_2d)
    gazev_c = np.vstack(gazev_c)
    gazev_w = np.vstack(gazev_w)
    eye_center = np.vstack(eye_center)
    eye_center_3d = np.vstack(eye_center_3d)

    rot_all = np.vstack(rot_all)
    trans_all = np.vstack(trans_all)

    return gazev_2d, gazev_c, gazev_w, eye_center, eye_center_3d, rot_all, trans_all


def get_gazev_c(vector_path):
    gaze_vector = np.loadtxt(vector_path, delimiter=' ', dtype=str)
    gazev_right = np.array(gaze_vector[:, 2:5]).astype(np.float)
    gazev_left = np.array(gaze_vector[:, 5:]).astype(np.float)
    gazev_c = np.zeros((len(gaze_vector) * 2, 3))
    gazev_c[::2] = gazev_right
    gazev_c[1::2] = gazev_left

    return gazev_c


def get_gazev_2d(im, eye_center_3d, gazev_c):
    gaze_point = eye_center_3d + 0.5 * gazev_c
    gaze2d_u, gaze2d_v = coord_trans_c(im, gaze_point)
    gaze2d = np.vstack((gaze2d_u, gaze2d_v)).T

    ecenter2d_u, ecenter2d_v = coord_trans_c(im, eye_center_3d)
    ecenter2d = np.vstack((ecenter2d_u, ecenter2d_v)).T

    gazev_2d = gaze2d - ecenter2d

    return gazev_2d


def main():
    whole_img_dir = 'D:/Project/child_eyetrace/Data/img_group0'
    save_dir = 'D:/Project/child_eyetrace/gaze_result/group0'
    dir_list = os.listdir(whole_img_dir)

    for id, i in enumerate(dir_list):
        print('Process', i)
        img_dir = os.path.join(whole_img_dir, i)
        vector_path = os.path.join(save_dir, i + '.txt')

        gazev_2d, gazev_c, gazev_w, eye_center, eye_center_3d, rot_all, trans_all = get_gaze_data(img_dir,
                                                                                                  vector_path)  # right eye [1::2]  left eye[::2]

        # save variables
        print('=== saving variables')
        np.savez(os.path.join(save_dir, i + '.npz'), gazev_2d=gazev_2d, gazev_w=gazev_w,
                 eye_center=eye_center, eye_center_3d=eye_center_3d, gazev_c=gazev_c, rot_all=rot_all,
                 trans_all=trans_all)


if __name__ == '__main__':
    main()

    # img_dir = 'E:/pytorch_mpiigaze_demo/Data/poss_imgs2/dhn'
    # vector_path = 'E:/pytorch_mpiigaze_demo/child_eyetrace/pygaze_result/dhn.txt'
    # save_dir = 'E:/pytorch_mpiigaze_demo/child_eyetrace/result/gazev'
    #
    # gazev_2d, gazev_c, gazev_w, eye_center, eye_center_3d, rot_all, trans_all = get_gaze_data(img_dir, vector_path)
    # np.savez(os.path.join(save_dir, 'dhn' + '.npz'), gazev_2d=gazev_2d, gazev_w=gazev_w,
    #          eye_center=eye_center, eye_center_3d=eye_center_3d, gazev_c=gazev_c, rot_all=rot_all, trans_all=trans_all)
    # # right eye [1::2]  left eye[::2]
    # # gazev_c_a = get_gazev_c(vector_path)
