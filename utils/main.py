import library as lb
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import copy
import SimpleITK as sitk
import cv2


def normalization(data):
    range = np.max(data) - np.min(data)
    return (data - np.min(data)) / range


def inputimage(image_path):
    image = Image.open(image_path)

    image = np.array(image)

    segmentation = lb.separation(lb.gullyDetect(image))[0]

    joint_up = copy.deepcopy(image)
    joint_up[segmentation == 0] = 0
    joint_down = copy.deepcopy(image)
    joint_down[segmentation == 1] = 0

    return joint_up, joint_down


if __name__ == '__main__':
    image_path1 = 'Data/finger_joint/4_1468_L/0_0/jMAC.1.2.392.200036.9107.307.16322.20170425.101019.1013622_L.bmp'
    image_path2 = 'Data/finger_joint/4_1468_L/0_0/jMAC.1.2.392.200036.9107.500.304.2046.20130125.95544.102046_L.bmp'

    img1 = inputimage(image_path1)[0]
    img2 = inputimage(image_path2)[0]

    # Initiate AKAZE detector
    akaze = cv2.AKAZE_create()
    # Find the keypoints and descriptors with SIFT
    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])

    # Draw matches
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.imshow(img3)
    #plt.imshow(inputimage(image_path2)[1], alpha=0.15)
    plt.show()


    # Select good matched keypoints
    ref_matched_kpts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches])
    sensed_matched_kpts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches])

    # Compute homography
    H, status = cv2.findHomography(sensed_matched_kpts, ref_matched_kpts, cv2.RANSAC, 5.0)

    # Warp image
    warped_image = cv2.warpPerspective(img2, H, (img2.shape[1], img2.shape[0]))



