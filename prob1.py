__author__ = 'Johnson'

import cv2
import numpy as np
from utilities import my_imshow
import matplotlib.pyplot as plt

filename = "./prob1/baby_food_can.bmp"
bgr_img = cv2.imread(filename)
# cv2.imshow('Image', bgr_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

b,g,r = cv2.split(bgr_img)       # get b,g,r
rgb_img = cv2.merge([r,g,b])     # switch it to rgb

(w, h, na) = rgb_img.shape
xstops = [0, 225, 450, h]
ystops = [0, 150, w]
images = []
i = 0

for x in range(3):
    for y in range(2):
        sub_img = rgb_img[ystops[y]:ystops[y+1], xstops[x]:xstops[x+1], :]
        ax = plt.subplot(2, 3, i+1)
        images.append(sub_img)
        my_imshow(sub_img, ax=ax)
        i += 1


def plot_color_histogram(rbg_img):
    color = ('r','g','b')
    for i, col in enumerate(color):
        histr = cv2.calcHist([rbg_img], [i], None, [256], [0,256])
        plt.plot(histr, color=col)
        plt.xlim([0,256])

def test_for_red(rgb_img):
    red_boundaries = ([100, 15, 17],
                      [255, 56, 50])
    # create NumPy arrays from the boundaries
    lower = np.array(red_boundaries[0], dtype = "uint8")
    upper = np.array(red_boundaries[1], dtype = "uint8")

    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(rgb_img, lower, upper)
    mask[mask>0] = 1
    return mask
# plot color histogram for every image
# i = 1
# f = plt.figure()
# for image in images:
#     ax = plt.subplot(2, 3, i)
#     plot_color_histogram(image)
#     i += 1
# detect red
i = 1
f = plt.figure()
for image in images:
    ax = plt.subplot(2, 3, i)
    mask = test_for_red(image)
    red_density = np.sum(mask)

    if red_density == 0:
        plt.title("No spoon found")
    elif red_density < 1000:
        plt.title("found one spoon")
    else:
        plt.title("found more than one spoon")

    my_imshow(mask, ax=ax)
    i += 1

plt.show()