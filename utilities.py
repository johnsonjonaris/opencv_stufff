__author__ = 'Johnson'
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axes

def my_range(start, step, stop):
    a = np.arange(start, stop, step)
    if (a[-1] + step) == stop:
        a = np.append(a, stop)
    return a

def my_imshow(img, ax=None, show=False, **kwargs):
    if ax is None:
        ax = plt.gca()
    def format_coord(x, y):
        x = int(x + 0.5)
        y = int(y + 0.5)
        try:
            return "[%4i, %4i] = %s" % (x, y, img[y, x])
        except IndexError:
            return ""
    im = ax.imshow(img, **kwargs)
    # ax.invert_yaxis()
    ax.format_coord = format_coord
    plt.draw()
    if(show):
        plt.show()
    return im

def plot_mip(img):
    for k in range(0, 3):
        ax = plt.subplot(1, 3, k + 1)  # subplot start counting from 1
        my_imshow(img.max(k).T, ax)
    plt.tight_layout()
    plt.draw()
    plt.show()

# usage: compare_vol(img1, img2, ...)
def compare_vol(*args):
    assert (len(args) < 6), \
        "Do not support more than 5 volumes to compare."
    for img in args:
        assert (type(img) == np.ndarray) and (img.ndim == 3), \
            "All inputs should be a 3D numpy arrays."
    # init a maximized figure
    plt.figure()
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    # gather info about inputs
    cut = np.zeros([1,0])
    sz = np.zeros([1,0])
    nImgs = len(args)

    def update(img, axis, imgN):
        offset = imgN*3
        if axis == 0 or axis == -1:
            ax = plt.subplot(nImgs, 3, 1+offset)
            ax.aname = 0+offset
            im = my_imshow(img[cut[0+offset],:,:].T, ax)
            ax.set_xlabel('y')
            ax.set_ylabel('z')
            ax.set_title(str(cut[0+offset]+1) + ' out of ' + str(sz[0+offset]))
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            if axis == -1:
                plt.colorbar(im, ax=ax)

        if axis == 1 or axis == -1:
            ax = plt.subplot(nImgs, 3, 2+offset)
            ax.aname = 1+offset
            im = my_imshow(img[:,cut[1+offset],:].T, ax)
            ax.set_xlabel('x')
            ax.set_ylabel('z')
            ax.set_title(str(cut[1+offset]+1) + ' out of ' + str(sz[1+offset]))
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            if axis == -1:
                plt.colorbar(im, ax=ax)

        if axis == 2 or axis == -1:
            ax = plt.subplot(nImgs, 3, 3+offset)
            ax.aname = 2+offset
            im = my_imshow(img[:,:,cut[2+offset]].T, ax)
            ax.set_xlabel('y')
            ax.set_ylabel('x')
            ax.set_title(str(cut[2+offset]+1) + ' out of ' + str(sz[2+offset]))
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            if axis == -1:
                # PCM = ax.get_children()[2] #get the mappable, the 1st and the 2nd are the x and y axes
                plt.colorbar(im, ax=ax)

        plt.draw()

    def on_scroll(event):
        n = event.inaxes.aname
        if event.button == "up":
            cut[n] += 1
        else:
            cut[n] -= 1
        cut[n] = min(max(cut[n], 0), sz[n]-1)
        imgN = n/3
        axis = n%3
        update(args[imgN], axis, imgN)

    imgCount = 0
    for img in args:
        sz = np.append(sz, np.array(img.shape))
        cut = np.append(cut, np.array(img.shape)/2)
        update(img, -1, imgCount)
        imgCount += 1
    plt.tight_layout()
    plt.gcf().canvas.mpl_connect('scroll_event', on_scroll)
    plt.show()

# usage: compare_vol(p1, p2, ...)
def display_projections(*args):
    assert (len(args) < 6), \
        "Do not support more than 5 volumes to compare."
    vmin = 1e300
    vmax = -1e300
    for img in args:
        assert (type(img) == np.ndarray) and (img.ndim == 3), \
            "All inputs should be a 3D numpy arrays."
        vmin = min(img.min(), vmin)
        vmax = max(img.max(), vmax)
    # init a maximized figure
    plt.figure()
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    # gather info about inputs
    cut = np.zeros([1,0])
    sz = np.zeros([1,0])
    nImgs = len(args)
    def update(img, imgN, update=False):
        ax = plt.subplot(1, nImgs, 1+imgN)
        ax.aname = imgN
        im = my_imshow(img[:,:,cut[imgN]], ax) #, vmin=vmin, vmax=vmax
        ax.set_title(str(cut[imgN]+1) + ' out of ' + str(sz[imgN]))
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        if not update:
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_clim(vmin, vmax)
        plt.draw()
    # updating on scrolling
    def on_scroll(event):
        imgN = event.inaxes.aname
        if event.button == "up":
            cut[imgN] += 1
        else:
            cut[imgN] -= 1
        cut[imgN] = min(max(cut[imgN], 0), sz[imgN]-1)
        update(args[imgN], imgN, True)
    # first time
    imgCount = 0
    for img in args:
        sz = np.append(sz, np.array(img.shape[2]))
        cut = np.append(cut, np.array([0,0,0]))
        update(img, imgCount)
        imgCount += 1
    plt.tight_layout()
    # connecting the scroll event of the figure to the scroll callback
    plt.gcf().canvas.mpl_connect('scroll_event', on_scroll)
    plt.show()

def getIntersection(line1, line2):
    # https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    l1_p1 = line1[0:2]
    l1_p2 = line1[2:]
    l2_p1 = line2[0:2]
    l2_p2 = line2[2:]
    # x1y2 - y1x2
    a = (l1_p1[0]*l1_p2[1] - l1_p1[1]*l1_p2[0])
    # x3y4 - y3x4
    b = (l2_p1[0]*l2_p2[1] - l2_p1[1]*l2_p2[0])
    # (x1 - x2)(y3 - y4) - (y1 - y2)(x3 - x4)
    c = (l1_p1[0] - l1_p2[0])*(l2_p1[1] - l2_p2[1]) - (l1_p1[1] - l1_p2[1])*(l2_p1[0] - l2_p2[0])
    if (abs(c) < 1e-12):
        return None
    x = (a*(l2_p1[0] - l2_p2[0]) - (l1_p1[0] - l1_p2[0])*b)/c
    y = (a*(l2_p1[1] - l2_p2[1]) - (l1_p1[1] - l1_p2[1])*b)/c
    return np.array([x, y], dtype='float64')

class Point:

    def __init__(self, x, y):
        self.x = x
        self.y = y
    def _init__(self, xy):
        """
        :param xy: array of two elements
        """
        self.x = xy[0]
        self.y = xy[1]

class Line:

    def __init__(self, pt1, pt2):
        """
        :param pt1: first point in line
        :param pt2: second point in line
        """
        self.pt1 = pt1
        self.pt2 = pt2
    def __init__(self, line_array):
        """
        :param line_array: array of 4 elements
        """
        print line_array
        self.pt1 = Point(line_array[0:2])
        self.pt2 = Point(line_array[2:])

class Rectangle:
    """
    rectangle class
    """
    def __init__(self, corner, width, height):
        """
        :param corner: bottom left corner
        :param width: width
        :param height: height
        :return:
        """
        self.corner = corner
        self.width = width
        self.height = height

    def __init__(self, bottom_left, top_right):
        """
        :param bottom_left: bottom left corner
        :param top_right: top right corner
        :return:
        """
        self.corner = bottom_left
        self.width = top_right.x - bottom_left.x
        self.height = top_right.y - bottom_left.y

    def inRect(self, point):
        print("not implemented yet")


def LineIntersectsRect(line, rect):

    return LineIntersectsLine(line, Line(rect.corner,
                                         Point(rect.corner.x + rect.width, rect.corner.y))) or \
        LineIntersectsLine(line, Line(Point(rect.corner.x + rect.width, rect.corner.y),
                                      Point(rect.corner.x + rect.width, rect.corner.y + rect.height))) or \
        LineIntersectsLine(line, Line(Point(rect.corner.x + rect.width, rect.corner.y + rect.height),
                                      Point(rect.corner.x, rect.corner.y + rect.height))) or \
        LineIntersectsLine(line, Line(Point(rect.corner.x, rect.corner.y + rect.height),
                                      rect.corner))

def LineIntersectsLine(line1, line2):
    # parametric line to line intersection, does not assume that the
    # lines can be extended
    q = (line1.pt1.y - line2.pt1.y)*(line2.pt2.x - line2.pt1.x) - \
        (line1.pt1.x - line2.pt1.x)*(line2.pt2.y - line2.pt1.y)
    d = (line1.pt2.x - line1.pt1.x)*(line2.pt2.y - line2.pt1.y) - \
        (line1.pt2.y - line1.pt1.y)*(line2.pt2.x - line2.pt1.x)
    if d == 0 :
        return False
    r = q / d
    q = (line1.pt1.y - line2.pt1.y) * (line1.pt2.x - line1.pt1.x) - \
        (line1.x - line2.pt1.x) * (line1.pt2.y - line1.pt1.y)
    s = q / d
    if ( r < 0 or r > 1 or s < 0 or s > 1 ):
        return False
    return True
