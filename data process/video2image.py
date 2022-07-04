# -*- coding: utf-8 -*-

# This file is created by Zeyu Chen, BIAI, Inc.
# Title           :video2image.py
# Version         :1.0
# Email           :k83110835@126.com
# Copyright       :BIAI, Inc.
# ==============================================================================


import os
import cv2
import shutil
from PIL import Image, ImageFilter
import numpy as np


def is_video(file_name):
    """
    This function will detect whether a file is a video.
    """
    video_ext = ['mp4', 'mov', 'mpg', 'avi']
    ext = file_name[file_name.rfind('.') + 1:].lower()
    return ext in video_ext


def save_image(image, addr, num):
    """
    Define the images to be saved.
    Args:
        image: the name of the saving image
        addr:  the picture directory address and the first part of the picture name
        num:   int dtype, the id number in the image filename
    """
    address = os.path.join(addr, str(num + 1) + '.jpg')
    cv2.imwrite(address, image)


class GaussianBlur(ImageFilter.Filter):

    def __init__(self, radius=2, bounds=None):
        self.radius = radius
        self.bounds = bounds

    def filter(self, image):
        if self.bounds:
            clips = image.crop(self.bounds).gaussian_blur(self.radius)
            image.paste(clips, self.bounds)
            return image
        else:
            return image.gaussian_blur(self.radius)

def blur_image(image, radius, bounds):
    image = Image.fromarray(image)
    image = image.filter(GaussianBlur(radius=radius, bounds=bounds))
    image = np.array(image)
    return image

def readvideo2image(output_dir, video_path):

    # single video file
    if is_video(video_path):
        video_name = video_path.split('/')[-1].split('.')

        output_image_path = os.path.join(output_dir, video_name[0])

        if os.path.exists(output_image_path):
            shutil.rmtree(output_image_path)

        os.makedirs(output_image_path)

        videoCapture = cv2.VideoCapture(video_path)

        success, frame = videoCapture.read()
        i = 0
        while success:
            if i == 0:
                frame = blur_image(frame, radius=30, bounds=(280, 90, 330, 170))
            save_image(frame, output_image_path, i)
            i += 1
            success, frame = videoCapture.read()

    # directory consisted of videos
    elif os.path.isdir(video_path):
        ls = os.listdir(video_path)
        for i, file_name in enumerate(sorted(ls)):
            if is_video(file_name):
                try:
                    print('Loading video {}'.format(file_name))
                    sub_video_name = file_name.split('.')

                    output_image_path = os.path.join(output_dir, sub_video_name[0])

                    if os.path.exists(output_image_path):
                        shutil.rmtree(output_image_path)

                    os.makedirs(output_image_path)

                    file_path = os.path.join(video_path, file_name)
                    videoCapture = cv2.VideoCapture(file_path)


                    success, frame = videoCapture.read()
                    i = 0
                    while success:
                        # if frame.shape[0] < frame.shape[1]:
                        #     frame = cv2.transpose(frame)
                        #     frame = cv2.flip(frame, -1)
                        save_image(frame, output_image_path, i)
                        i += 1
                        success, frame = videoCapture.read()
                except:
                    print('Processing video {} failed'.format(file_name))

# single frame image blur
def imageblur(imagedir):

    frame = cv2.imread(imagedir)
    frame = blur_image(frame, radius=20, bounds=(204, 215, 259, 279))
    # frame = blur_image(frame, radius=30, bounds=(348,362,385,404))
    # frame = blur_image(frame, radius=30, bounds=(322,560,367,611))
    cv2.imwrite(imagedir, frame)


if __name__ == '__main__':

    # outdir = 'E:/Project/child_eyetrace/Data/image3_group0'
    #
    # whole_dir = 'E:/Project/child_eyetrace/Data/video3_group0'
    #
    # readvideo2image(outdir, whole_dir)

    imagedir = 'C:/Users/Administrator/Desktop/pic1.png'
    imageblur(imagedir)

