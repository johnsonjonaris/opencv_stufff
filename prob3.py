__author__ = 'Johnson'

import os, sys, shutil
import cv2
import numpy as np
from utilities import my_imshow
import matplotlib.pyplot as plt
import matplotlib.cm as cm

dir = "./prob3/"
files = os.listdir(dir)

images = []
for file in files:
    filename = os.path.join(dir, file)
    image = cv2.imread(filename)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary_img = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    binary_img = cv2.erode(binary_img, kernel=np.ones((2,2), 'uint8'), iterations=2)
    images.append(binary_img)

def plot_images():
    i = 1
    for image in images:
        plt.subplot(3,3,i)
        my_imshow(image, cmap=cm.Greys_r)
        i += 1

if 1:
    plot_images()
# create a simple blob feature detector and select filtering by circularity
params = cv2.SimpleBlobDetector_Params()
# Change thresholds
params.minThreshold = 10
params.maxThreshold = 255
# Filter by Area.
params.filterByArea = False
params.minArea = 150
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.89
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.985
# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.5
# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

i = 1
f1 = plt.figure()
for image in images:
    contours = cv2.findContours(image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)[1]
    # we care about the longest contour
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    image2 = np.zeros_like(image)
    cv2.drawContours(image2, contours=contours, contourIdx=0, color=(255,), thickness=1)
    print("Found %d contours for image %d" % (len(contours), i))
    keypoints = detector.detect(image2)
    print keypoints
    image3 = cv2.drawKeypoints((image2), keypoints, np.array([]),
                               (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # get circles
    # circles = cv2.HoughCircles(image, method=cv2.HOUGH_GRADIENT, dp=1, minDist=20,
    #                            param1=100, param2=30)
    plt.subplot(3,3,i)
    my_imshow(image3)
    if len(keypoints) == 0:
        if len(contours) == 1:
            plt.title("Bad - Broken")
        else:
            plt.title("Bad")
    if len(contours) > 2:
        plt.title("Bad")

    i += 1

plt.show()

# i = 0
# f = plt.figure()
# for image in proc_images:
#     circles = cv2.HoughCircles(image, method=cv2.HOUGH_GRADIENT, dp=1, minDist=20,
#                                param1=30, param2=50, minRadius=0,maxRadius=0)
#     bw_image = np.zeros_like(images[0])
#     if circles is not None:
#         for circle in circles[0,:]:
#             cv2.circle(bw_image, (circle[0], circle[1]), circle[2], rand_color(), 1)
#     plt.subplot(3,3,i+1)
#     my_imshow(bw_image)
#     i += 1