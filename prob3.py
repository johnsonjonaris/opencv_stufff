__author__ = 'Johnson'

import os, sys, shutil
import cv2
import numpy as np
from utilities import my_imshow
import matplotlib.pyplot as plt
import matplotlib.cm as cm

dir = "./prob3/"
files = os.listdir(dir)

def rand_color():
    return np.random.randint(0,255,(1,3))[0]

images = []
for file in files:
    filename = os.path.join(dir, file)
    image = cv2.imread(filename)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary_img = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY_INV)[1]
    images.append(binary_img)

# class1, class2, class3 = images[0:3], images[3:6], images[6:]

def plot_images():
    i = 1
    for image in images:
        plt.subplot(3,3,i)
        my_imshow(image, cmap=cm.Greys_r)
        i += 1

if 0: plot_images()
# plt.show()

# f2 = plt.figure(2)
i = 1
proc_images = []
for image in images:
    if i > 10:
        break
    f1 = plt.figure()
    (_, contours, hierarchy) = cv2.findContours(image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    # cnts = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
    j = 1
    for c in contours:
        if len(c) > 10:
            cv2.drawContours(image, contours=c, contourIdx=-1, color=rand_color(), thickness=3)
        j += 5
    image = cv2.erode(image, kernel=np.ones([2,2]))
    # get circles
    circles = cv2.HoughCircles(image, method=cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=100, param2=30)
    # plt.figure(1)
    # plt.subplot(3,3,i)
    plt.subplot(1,2,1)
    if circles is None:
        print "No circles found for %i" % i
    else:
        for circle in circles[0,:]:
            center = tuple(circle[0:2])
            radius = circle[2]
            print center, radius
            cv2.circle(image, center, radius, (255,0,0), 1)
            cv2.circle(image, center, 0, (255,0,0), 1)
            # print radius
            for c in contours:
                if len(c) > 10:
                    c = np.squeeze(c, axis=1)
                    p = c.copy()
                    for l in (0,1):
                        p[:,l] -= center[l]
                        p[:,l] *= p[:,l]
                    pp = np.sqrt(p[:,0] + p[:,1])
                    plt.hist(pp)
                    print np.mean(pp)
                    # print pp
    # plt.figure(2)
    # plt.subplot(3,3,i)
    plt.subplot(1,2,2)
    my_imshow(image)

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
plt.show()
