from facenet_pytorch import MTCNN
import numpy as np
import cv2
from PIL import Image
import dlib
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import os

#https://www.pyimagesearch.com/2020/05/18/image-inpainting-with-opencv-and-python/
#https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/

def align(img, landmarks, desiredLeftEye=(0.35, 0.35), desiredFaceWidth=224, desiredFaceHeight=None):
    if desiredFaceHeight is None:
        desiredFaceHeight=desiredFaceWidth

    leftEyeCenter = landmarks[0]
    rightEyeCenter = landmarks[1]

    # compute the angle between the eye centroids
    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    angle = np.degrees(np.arctan2(dY, dX))

    # compute the desired right eye x-coordinate based on the
    # desired x-coordinate of the left eye
    desiredRightEyeX = 1.0 - desiredLeftEye[0]
    # determine the scale of the new resulting image by taking
    # the ratio of the distance between eyes in the *current*
    # image to the ratio of distance between eyes in the
    # *desired* image
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desiredDist = (desiredRightEyeX - desiredLeftEye[0])
    desiredDist *= desiredFaceWidth
    scale = desiredDist / dist

    # compute center (x, y)-coordinates (i.e., the median point)
    # between the two eyes in the input image
    eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                (leftEyeCenter[1] + rightEyeCenter[1]) // 2)
    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
    # update the translation component of the matrix
    tX = desiredFaceWidth * 0.5
    tY = desiredFaceHeight * desiredLeftEye[1]
    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])

    # apply the affine transformation
    (w, h) = (desiredFaceWidth, desiredFaceHeight)
    output = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)

    return output

def border_fill(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[gray == 0] = 255

    output = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    return output


def is_image(file_name):
    """
    This function will detect whether a file is a video.
    """
    img_ext = ['jpg', 'png']
    ext = file_name[file_name.rfind('.') + 1:].lower()
    return ext in img_ext

def face_alignment_on_img_dir(img_dir, align_dir):
    """
        This function will conduct face alignment on an image dir.
        """
    image_ls = os.listdir(img_dir)
    assert is_image(image_ls[0])
    image_ls.sort(key=lambda x: int(x.split('.')[0]))  # sort files by ascending
    detector = MTCNN(select_largest=False, post_process=False)
    tracking_state = False
    poss_img = []
    savedir_path = os.path.join(align_dir, img_dir.split('/')[-1])
    if not os.path.exists(savedir_path):
        os.makedirs(savedir_path)

    for image_id, image_name in enumerate(image_ls):

        image_path = os.path.join(img_dir, image_name)
        img = cv2.imread(image_path)

        img = cv2.copyMakeBorder(img, 200, 200, 200, 200, cv2.BORDER_CONSTANT, value=0)  # 上下左右边缘扩充200个像素点
        batch_boxes, batch_probs, batch_points = detector.detect(img[:, :, ::-1], landmarks=True)

        if tracking_state is False:
            if len(batch_boxes) == 1:
                for box in batch_boxes:
                    tracking_state = True
                    last_box_center = [np.mean([box[0], box[2]]), np.mean([box[1], box[3]])]
                    outface = align(img[:, :, ::-1], batch_points[0])
                    out = border_fill(outface)
                    cv2.imwrite(os.path.join(savedir_path, image_name), out[:, :, ::-1])
                    poss_img.append(image_id)

        else:
            if len(batch_boxes) > 0:
                min_center = 5000
                min_center_id = -1
                for i, box in enumerate(batch_boxes):
                    box_center = [np.mean([box[0], box[2]]), np.mean([box[1], box[3]])]
                    box_center_dist = np.linalg.norm(np.array(box_center) - np.array(last_box_center)) ** 2
                    if box_center_dist < min_center:
                        min_center = box_center_dist
                        min_center_id = i
                if min_center_id != -1:
                    last_box = batch_boxes[min_center_id]
                    last_box_center = [np.mean([last_box[0], last_box[2]]), np.mean([last_box[1], last_box[3]])]
                    outface = align(img[:, :, ::-1], batch_points[min_center_id])
                    out = border_fill(outface)
                    cv2.imwrite(os.path.join(savedir_path, image_name), out[:, :, ::-1])
                    poss_img.append(image_id)

    return poss_img

if __name__ == "__main__":

    img_dir = 'D:/Project/child_eyetrace/Data/img_group0'
    align_dir = 'D:/Project/child_eyetrace/Data/align_group0'
    log_name = 'D:/Project/child_eyetrace/Data/align_group0/align_log.txt'
    # process one image dir
    # poss_img = face_alignment_on_img_dir(img_dir, align_dir)

    # process multi image dirs
    for i, subdir_name in enumerate(os.listdir(img_dir)):
        print('Processing ', subdir_name)
        subdir_path = os.path.join(img_dir, subdir_name)
        image_ls = os.listdir(subdir_path)
        image_ls.sort(key=lambda x: int(x.split('.')[0]))  # sort files by ascending
        savedir_path = os.path.join(align_dir, subdir_name)
        if not os.path.exists(savedir_path):
            os.makedirs(savedir_path)

        detector = MTCNN(select_largest=False, post_process=False)
        tracking_state = False
        poss_img = []

        for image_id, image_name in enumerate(image_ls):

            image_path = os.path.join(subdir_path, image_name)
            img = cv2.imread(image_path)
            img = cv2.copyMakeBorder(img, 200, 200, 200, 200, cv2.BORDER_CONSTANT, value=0)  # 上下左右边缘扩充200个像素点
            batch_boxes, batch_probs, batch_points = detector.detect(img[:, :, ::-1], landmarks=True)

            if tracking_state is False:
                if len(batch_boxes) == 1:
                    for box in batch_boxes:
                        tracking_state = True
                        last_box_center = [np.mean([box[0], box[2]]), np.mean([box[1], box[3]])]
                        outface = align(img[:, :, ::-1], batch_points[0])
                        out = border_fill(outface)
                        cv2.imwrite(os.path.join(savedir_path, image_name), out[:, :, ::-1])
                        poss_img.append(image_id)

            else:
                if len(batch_boxes) > 0:
                    min_center = 5000
                    min_center_id = -1
                    for i, box in enumerate(batch_boxes):
                        box_center = [np.mean([box[0], box[2]]), np.mean([box[1], box[3]])]
                        box_center_dist = np.linalg.norm(np.array(box_center) - np.array(last_box_center)) ** 2
                        if box_center_dist < min_center:
                            min_center = box_center_dist
                            min_center_id = i
                    if min_center_id != -1:
                        last_box = batch_boxes[min_center_id]
                        last_box_center = [np.mean([last_box[0], last_box[2]]), np.mean([last_box[1], last_box[3]])]
                        outface = align(img[:, :, ::-1], batch_points[min_center_id])
                        out = border_fill(outface)
                        cv2.imwrite(os.path.join(savedir_path, image_name), out[:, :, ::-1])
                        poss_img.append(image_id)


        with open(log_name, "a+") as f:
            f.write(str(subdir_name) + ' ')
            for i in poss_img:
                f.write(str(i) + " ")
            f.write("\n")



    # predictor_model_path = 'E:/onlyfat_selfa3D_2/face_detect_align/detector/shape_predictor_68_face_landmarks.dat'
    # shape_predictor = dlib.shape_predictor(predictor_model_path)
    # det = dlib.rectangle(int(boxes[0]), int(boxes[1]), int(boxes[2]), int(boxes[3]))
    # face_landmarks = [(item.x, item.y) for item in shape_predictor(img, det).parts()]
    # face_landmarks = np.array(face_landmarks)

    # plt.figure(figsize=(12,8))
    # img = np.asarray(img)
    # plt.imshow(img)
    # currentAxis=plt.gca()
    # rect=patches.Rectangle((boxes[0], boxes[1]),boxes[2]-boxes[0],boxes[3]-boxes[1],linewidth=1,edgecolor='r',facecolor='none')
    # currentAxis.add_patch(rect)
    # plt.scatter(landmarks[:,0], landmarks[:,1])
    # plt.show()

