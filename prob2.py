__author__ = 'Johnson'

import cv2
import numpy as np
from utilities import my_imshow, getIntersection
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
        images.append(sub_img)

        # ax = plt.subplot(3, 10, i+1)
        # my_imshow(sub_img, cmap=cm.Greys_r)
        # plt.subplot(3, 10, i+2)
        # plt.hist(sub_img.flatten())
        # i += 2
# plt.show()

# apply edge detector to images
seed = (50,50)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
i = 0
for image in images:
    # image[image < 55] = 0
    filter_image = cv2.blur(image, ksize=(5,5))
    edge_image = cv2.Canny(filter_image, threshold1=0, threshold2=60, apertureSize=3)
    edge_image = cv2.erode(edge_image, kernel=np.ones((1,1),'uint8'))
    edge_image = cv2.dilate(edge_image, kernel=np.ones((5,5),'uint8'))
    edge_image = cv2.erode(edge_image, kernel=np.ones((3,3),'uint8'))
    edge_image[edge_image > 0] = 255
    # fill the holes as much as possible
    cv2.floodFill(edge_image, mask=None, seedPoint=seed, newVal=255)
    # find the largest contour assuming it will be the outer shape of the bottle
    contours = cv2.findContours(edge_image, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)[1]
    max_contour = 0
    idx = 0
    for l in range(len(contours)):
        if len(contours[l]) > max_contour:
            max_contour = len(contours[l])
            idx = l
    mask_image = np.zeros_like(image)
    cv2.drawContours(mask_image, contours=contours, contourIdx=idx, color=255)
    lines = cv2.HoughLinesP(mask_image, rho=1, theta=np.pi/180.0, threshold=30, minLineLength=30, maxLineGap=100)
    out_lines = []
    for line in lines:
        pt1 = tuple(line[0][0:2])
        pt2 = tuple(line[0][2:])
        # exclude oblique lines
        v1 = (line[0][0:2] - line[0][2:])
        v1 = v1/np.linalg.norm(v1)
        v2 = np.array([0.0,1.0])
        angle = np.arccos(np.dot(v1,v2))*180.0/np.pi
        tol = 3.75
        test = angle < tol or abs(angle - 90) < tol or abs(angle - 180) < tol
        if test:
            out_lines.append(line[0])
    # discover lines intersection
    nLines = len(out_lines)
    points = []
    size = mask_image.shape
    for j in range(nLines):
        for k in range(j+1,nLines):
            pt = getIntersection(out_lines[j], out_lines[k])
            test = pt is not None and np.all(np.greater(pt,0)) > 0 and pt[0] < size[1] and pt[1] < size[0]
            if test:
                points.append(pt)
    x1 = min(points[0][0], points[1][0])
    x2 = max(points[0][0], points[1][0])
    y2 = min(points[0][1], points[1][1])
    y1 = y2 - 100
    label = image[y1:y2,x1:x2]
    plt.subplot(3,10,i+1)
    my_imshow(label, cmap=cm.Greys_r)
    plt.subplot(3,10,i+2)
    plt.hist(label.flatten())
    hist_item = cv2.calcHist([label], channels=[0], mask=None, histSize=[5], ranges=[0,256])
    print(i/2)
    print hist_item[3]
    i += 2

plt.show()
