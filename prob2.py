__author__ = 'Johnson'

import cv2
import numpy as np
from utilities import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def lineIntersectsSquare(line, left, top, right, bottom):
    x0 = line[0]
    y0 = line[1]
    x1 = line[2]
    y1 = line[3]

    def calcY(xval):
        if x1 == x0: return float('nan')
        return y0 + (xval - x0)*(y1 - y0)/(x1 - x0)
    def calcX(yval):
        if x1 == x0: return float('nan')
        return x0 + (yval - y0)*(y1 - y0)/(x1 - x0)

    if (calcX(bottom) < right and calcX(bottom) > left): return True
    if (calcX(top) < right and calcX(top) > left): return True
    if (calcY(left) < top and calcY(left) > bottom): return True
    if (calcY(right) < top and calcY(right) > bottom): return True
    return False

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
    # get edges of the image
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
    # discover the lines in the outer shape of the bottle
    lines = cv2.HoughLinesP(mask_image, rho=1, theta=np.pi/180.0, threshold=30, minLineLength=30, maxLineGap=100)
    # get the label area from the bottle
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
    # extract label area from image
    label = image[y1:y2,x1:x2]
    # analyze label area
    filter_image = cv2.blur(label, ksize=(3,3))
    edge_image = cv2.Canny(filter_image, threshold1=5, threshold2=30, apertureSize=3)
    # edge_image = cv2.erode(edge_image, kernel=np.ones((1,1),'uint8'))
    # edge_image = cv2.dilate(edge_image, kernel=np.ones((3,3),'uint8'))
    # edge_image = cv2.erode(edge_image, kernel=np.ones((1,1),'uint8'))
    edge_image[edge_image > 0] = 25500
    # test for label existence
    label_exist = True
    if np.sum(edge_image[20:-20,20:-20]) == 0:
        label_exist = False
    if label_exist:
        # fill the holes as much as possible
        # cv2.floodFill(edge_image, mask=None, seedPoint=seed, newVal=255)
        # discover the lines in the outer shape of the bottle
        lines = cv2.HoughLinesP(edge_image, rho=1, theta=np.pi/180.0, threshold=30, minLineLength=30, maxLineGap=100)
        edge_image = np.zeros_like(label)
        (w, h) = edge_image.shape
        for line in lines:
            pt1 = tuple(line[0][0:2])
            pt2 = tuple(line[0][2:])
            # exclude oblique lines
            v1 = (line[0][0:2] - line[0][2:])
            v1 = v1/np.linalg.norm(v1)
            v2 = np.array([0.0,1.0])
            angle = np.arccos(np.dot(v1,v2))*180.0/np.pi
            tol = 3.75
            test1 = angle < tol or abs(angle - 90) < tol or abs(angle - 180) < tol
            test2 = False
            # http://stackoverflow.com/questions/99353/how-to-test-if-a-line-segment-intersects-an-axis-aligned-rectange-in-2d
            if test1:
                test2 = LineIntersectsRect(Line(line[0]), Rectangle(Point(20, 20), width=w-20, height=h-20))
            if test1 and not test2:
                cv2.line(edge_image, pt1, pt2, color=(255,))

    plt.subplot(3,10,i+1)
    my_imshow(label, cmap=cm.Greys_r)
    plt.subplot(3,10,i+2)
    my_imshow(edge_image, cmap=cm.Greys_r)
    plt.title("%d" % label_exist)
    i += 2

plt.show()



