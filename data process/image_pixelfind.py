# -*- coding: utf-8 -*-
"""
Created on Sat May  8 15:16:33 2021

@author: admin
"""


import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


#图片路径
img = cv2.imread('C:/Users/Administrator/Desktop/pic1.png')
a = []
b = []
def on_EVENT_LBUTTONDOWN(event, x, y,flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        a.append(x)
        b.append(y)
        cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 255, 0), thickness=1)
        cv2.imshow("image", img)

cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
cv2.imshow("image", img)
cv2.waitKey(0)
print(a[0],b[0])
