__author__ = 'Johnson'

import cv2
import numpy as np
from utilities import *
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
margin = 20
labels = []
method = 2
for image in images:
    print("Processing image # %d" % (i/2+1))
    # get edges of the image
    filter_image = cv2.blur(image, ksize=(5,5))
    edge_image = cv2.Canny(filter_image, threshold1=0, threshold2=60, apertureSize=3)
    edge_image = cv2.erode(edge_image, kernel=np.ones((1,1), 'uint8'))
    edge_image = cv2.dilate(edge_image, kernel=np.ones((5,5), 'uint8'))
    edge_image = cv2.erode(edge_image, kernel=np.ones((3,3), 'uint8'))
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
    lines = cv2.HoughLinesP(mask_image, rho=1, theta=np.pi/180.0,
                            threshold=30, minLineLength=30, maxLineGap=100)
    # get the label area from the bottle
    out_lines = []
    for line in lines:
        # exclude oblique lines
        angle = Line.AngleBetweenLines(Line(line[0]), Line([0,0,1,0]))
        tol = 3.75
        test = angle < tol or abs(angle - 90) < tol or abs(angle - 180) < tol
        if test:
            out_lines.append(line[0])
    # discover lines intersection
    nLines = len(out_lines)
    points = []
    h, w = mask_image.shape
    frame = Rectangle(Point(0,0), w, h)
    for j in range(nLines):
        for k in range(j+1,nLines):
            pt = Line.GetLinesIntersection(Line(out_lines[j]), Line(out_lines[k]))
            if pt is not None and frame.contains(pt):
                points.append(pt)
    x1 = min(points[0].x, points[1].x)
    x2 = max(points[0].x, points[1].x)
    y2 = min(points[0].y, points[1].y)
    y1 = y2 - 100
    # extract label area from image
    label = image[y1:y2,x1:x2]
    labels.append(label)
    if method == 2:
        continue
    # analyze label area
    filter_image = cv2.blur(label, ksize=(3,3))
    edge_image = cv2.Canny(filter_image, threshold1=5, threshold2=30,
                           apertureSize=3, L2gradient=True)
    edge_image[edge_image > 0] = 255
    # test for label existence
    sum = np.sum(edge_image[margin:-margin, margin:-margin])
    label_exist = not (sum == 0)
    correct_label = label_exist
    if label_exist:
        # discover the lines in the outer shape of the bottle
        lines = cv2.HoughLinesP(edge_image, rho=1, theta=np.pi/180.0,
                                threshold=30, minLineLength=30, maxLineGap=100)
        edge_image = np.zeros_like(label)
        (h, w) = edge_image.shape
        rect = Rectangle(Point(margin, margin), w-2*margin, h-2*margin)
        frame = Rectangle(Point(0,0), w, h)
        side_lines = []
        for line in lines:
            # exclude oblique lines
            angle = Line.AngleBetweenLines(Line(line[0]), Line([0, 0, 1, 0]))
            tol = 3.75
            test1 = angle < tol or abs(angle - 90) < tol or abs(angle - 180) < tol
            if test1 and not Line.LineIntersectsRect(Line(line[0]), rect):
                pt1 = tuple(line[0][0:2])
                pt2 = tuple(line[0][2:])
                cv2.line(edge_image, pt1, pt2, color=(255,))
                side_lines.append(line[0])
        # intersect side lines to find the rectangle
        nLines = len(side_lines)
        intersection_pts = []
        for j in range(nLines):
            for k in range(j+1, nLines):
                pt = Line.GetLinesIntersection(Line(side_lines[j]), Line(side_lines[k]))
                # if the point is not within the label image, then it is incorrect
                if pt is not None and frame.contains(pt):
                    intersection_pts.append(pt)
        # infer corner points
        bottom_left = []
        bottom_right = []
        top_left = []
        top_right = []
        for pt in intersection_pts:
            if pt.x < w/2 and pt.y < h/2:
                bottom_left.append(pt)
            elif pt.x < w/2:
                bottom_right.append(pt)
            elif pt.x > w/2 and pt.y < h/2:
                top_left.append(pt)
            else:
                top_right.append(pt)
        # if we couldn't find intersection points in all four corners then the label is damaged
        correct_label = len(bottom_left) > 0 and \
                        len(bottom_right) > 0 and \
                        len(top_left) > 0 and \
                        len(top_right) > 0
    plt.subplot(3,10,i+1)
    my_imshow(label, cmap=cm.Greys_r)
    plt.subplot(3,10,i+2)
    my_imshow(edge_image, cmap=cm.Greys_r)
    plt.title("%d - %d" % (label_exist, correct_label))
    i += 2

plt.show()

# another method
i = 0
for label in labels:
    print("Processing image # %d" % (i/2+1))
    # analyze label area
    filter_image = cv2.bilateralFilter(label, d=11, sigmaColor=17, sigmaSpace=17)
    edge_image = cv2.Canny(filter_image, threshold1=5, threshold2=30,
                           apertureSize=3, L2gradient=True)
    edge_image[edge_image > 0] = 255
    # test for label existence
    sum = np.sum(edge_image[margin:-margin, margin:-margin])
    label_exist = not (sum == 0)
    correct_label = label_exist
    if label_exist:
        contours = cv2.findContours(edge_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
        # edge_image.fill(0)
        # cv2.drawContours(edge_image, contours, -1, 255, -1)
        # contours = cv2.findContours(edge_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        idx = 0
        edge_image.fill(0)
        for contour in contours:
            # approximate the contour
            peri = cv2.arcLength(contour, closed=True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, closed=True)
            # if our approximated contour has four points, then
            # we can assume that we have found our label
            correct_label = len(approx) == 4
            idx += 1
            if correct_label:
                cv2.drawContours(edge_image, contours=contours, contourIdx=idx, color=255)
                break
        if not correct_label:
            cv2.drawContours(edge_image, contours=contours, contourIdx=-1, color=255)

    plt.subplot(3,10,i+1)
    my_imshow(label, cmap=cm.Greys_r)
    plt.subplot(3,10,i+2)
    my_imshow(edge_image, cmap=cm.Greys_r)
    plt.title("%d - %d" % (label_exist, correct_label))
    i += 2

plt.show()
