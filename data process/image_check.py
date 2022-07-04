import os
import cv2
from facenet_pytorch import MTCNN

img_path = 'E:/Project/child_eyetrace/Data/image3_group0/czh_3/1.jpg'
all_boxes = []
detector = MTCNN(select_largest=False, post_process=False)

frame = cv2.imread(img_path)
dets, _ = detector.detect(frame[:, :, ::-1])
if len(dets) == 1:
    for box in dets:
        cv2.rectangle(frame, (int(box[0]), int(box[1])),
                      (int(box[2]), int(box[3])),
                      (0, 255, 0), 1)

cv2.imshow('frame', frame)
cv2.waitKey(0)