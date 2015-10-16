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
        sub_img = gray_img[ystops[y*2]:ystops[y*2+1], xstops[x]:xstops[x+1]]
        ax = plt.subplot(3, 10, i+1)
        images.append(sub_img)
        my_imshow(sub_img, cmap=cm.Greys_r)
        plt.subplot(3, 10, i+2)
        plt.hist(sub_img.flatten())
        i += 2
# plt.show()

# apply edge detector to images
edge_images = []
mask_images = []
seed = (50,50)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
for image in images:
    # image[image < 55] = 0
    filter_image = cv2.blur(image, ksize=(5,5))
    edge_image = cv2.Canny(filter_image, threshold1=0, threshold2=60, apertureSize=3)
    edge_image = cv2.erode(edge_image, kernel=np.ones((1,1),'uint8'))
    edge_image = cv2.dilate(edge_image, kernel=np.ones((5,5),'uint8'))
    edge_image = cv2.erode(edge_image, kernel=np.ones((3,3),'uint8'))
    # edge_image = edge_image - edge_image2
    edge_image[edge_image > 0] = 255
    edge_images.append(edge_image.copy())
    # fill the holes as much as possible
    cv2.floodFill(edge_image, mask=None, seedPoint=seed, newVal=255)
    # find the largest contour assuming it will be the outer shape of the bottle
    contours = cv2.findContours(edge_image, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)[0]
    print contours
    max_contour = 0
    idx = 0
    for i in range(len(contours)):
        if len(contours[i]) > max_contour:
            max_contour = len(contours[i])
            idx = i
    mask = np.zeros_like(image)
    cv2.drawContours(mask, contours=contours, contourIdx=idx, color=255)
    mask_images.append(mask.copy())

i = 0
f = plt.figure()
for (mask_image,edge_image) in zip(mask_images,edge_images):
    plt.subplot(3, 10, i+1)
    my_imshow(edge_image, cmap=cm.Greys_r)
    plt.subplot(3, 10, i+2)
    my_imshow(mask_image, cmap=cm.Greys_r)
    i += 2
# plt.show()

lines_images = []
i = 0
for mask_image in mask_images:
    lines = cv2.HoughLinesP(mask_image, rho=1, theta=np.pi/180.0, threshold=10, minLineLength=0, maxLineGap=0)
    print lines
    mask = np.zeros_like(mask_image)
    for line in lines:
        cv2.drawContours(mask, contours=lines, contourIdx=-1, color=255)
    plt.subplot(3,5,i+1)
    my_imshow(mask, cmap=cm.Greys_r)
    i += 1

plt.show()
