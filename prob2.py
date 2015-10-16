__author__ = 'Johnson'

import cv2
import numpy as np
from utilities import my_imshow
import matplotlib.pyplot as plt
import matplotlib.cm as cm

filename = "./prob2/bottles.bmp"
bgr_img = cv2.imread(filename)
(w, h, na) = bgr_img.shape
gray_img = np.zeros([w,h])
gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
# my_imshow(gray_img)
xstops = [0, 105, 205, 305, 405, h]
ystops = [0, 235, 275, 510, 550, w]
images = []
i = 0
# f = plt.figure()
for y in range(3):
    for x in range(len(xstops)-1):
        sub_img = bgr_img[ystops[y*2]:ystops[y*2+1], xstops[x]:xstops[x+1],:]
        ax = plt.subplot(3, 5, i+1)
        images.append(sub_img)
        my_imshow(sub_img)
        i += 1
plt.show()

# apply edge detector to images

edge_images = []
i = 0
for image in images:
    # image[image < 45] = 0
    # filter_image = cv2.blur(image, ksize=(5,5))
    # edge_image = cv2.Canny(filter_image, threshold1=0, threshold2=15, apertureSize=3)
    # edge_image = cv2.erode(edge_image, kernel=np.ones((1,1),'uint8'))
    # edge_image = cv2.dilate(edge_image, kernel=np.ones((3,3),'uint8'))
    # edge_image2 = cv2.erode(edge_image, kernel=np.ones((2,2),'uint8'))
    # edge_image = edge_image - edge_image2
    # edge_image[edge_image > 0] = 255
    # edge_images.append(edge_image.copy())
    i += 1

i = 0
f = plt.figure()
for image in images:
    plt.subplot(1, 2, 1)
    my_imshow(image, cmap=cm.Greys_r)
    plt.subplot(1, 2, 2)
    my_imshow(edge_images[i], cmap=cm.Greys_r)
    i += 1
    plt.show()
#
# f = plt.figure()
# i = 0
# for edge_image in edge_images:
#     if i == 0:
#         print np.unique(edge_image)
#         plt.subplot(1,2,1)
#         image = edge_image.copy()
#         image[image > 0] = 1
#         my_imshow(image, cmap=cm.Greys_r)
#         contours = cv2.findContours(image, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)[0]
#     print contours
#     if i == 0:
#         plt.subplot(1,2,2)
#         my_imshow(image, show=True, cmap=cm.Greys_r)
#         color = np.random.randint(0,255,[1,3])[0]
#         img = images[i].copy()
#         print type(contours)
#         cv2.drawContours(img, contours=contours, contourIdx=1, color=color)
#
#     # plt.subplot(3, 5, i+1)
#     # for contour in contours:
#     #     plt.plot(contours)
#     i += 1
#
#
#
#
