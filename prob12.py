__author__ = 'Johnson'

import cv2
import numpy as np
from utilities import my_imshow
import matplotlib.pyplot as plt
import matplotlib.cm as cm

filename = "./prob12/prob12.bmp"
image = cv2.imread(filename)
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
(w, h) = gray_img.shape

ystops = [0, 280, 580, w]
images = []
for i in range(3):
    img = gray_img[ystops[i]:ystops[i+1],:]
    img[img>100] = 0
    images.append(img)
    plt.subplot(2,1,1)
    my_imshow(img, cmap=cm.Greys_r)
    # get edges of the image
    filter_image = cv2.blur(img, ksize=(5,5))
    edge_image = cv2.Canny(filter_image, threshold1=0, threshold2=60, apertureSize=3)
    plt.subplot(2,1,2)
    my_imshow(edge_image, cmap=cm.Greys_r)
    plt.show()

