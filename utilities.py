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

import math

class Point:

    def __init__(self, *args):
        """
        Usage: Point(x,y)
        :param x: x coordinate
        :param y: y coordinate
        Usage: Point(xy)
        :param xy: array of two scalars
        """
        nArgs = len(args)
        if nArgs == 1:
            self.x = float(args[0][0])
            self.y = float(args[0][1])
        elif nArgs == 2:
            self.x = float(args[0])
            self.y = float(args[1])
        else:
            raise ValueError("Usage: Point(x,y) or Point(xy)")

    def __add__(self, other):
        """
        adds two points
        :return: self + other
        """
        result = Point(self.x, self.y)
        result.x += other.x
        result.y += other.y
        return result

    def __sub__(self, other):
        """
        subtracts two points
        :return: self - other
        """
        result = Point(self.x, self.y)
        result.x -= other.x
        result.y -= other.y
        return result

    @property
    def length(self):
        """
        :return: norm of the point
        """
        return math.sqrt(self.x**2 + self.y**2)

    def normalize(self):
        """
        assume point is a vector and normalize it
        """
        l = self.length
        self.x /= l
        self.y /= l

    @staticmethod
    def dot(pt1, pt2):
        """
        :return dot product of the two points
        """
        return pt1.x*pt2.x + pt1.y*pt2.y

    def __str__(self):
        return "Point (%f,%f)" % (self.x, self.y)

class Rectangle:
    """
    rectangle class
    """
    def __init__(self, *args):
        """
        Rectangle(corner, width, height)
        :param corner: bottom left corner
        :param width: width
        :param height: height
        Rectangle(bottom_left, top_right)
        :param bottom_left: bottom left corner
        :param top_right: top right corner
        """
        nArgs = len(args)
        if nArgs == 3:
            self.corner = args[0]
            self.width = float(args[1])
            self.height = float(args[2])
        elif nArgs == 2:
            self.corner = args[0]
            self.width = args[1].x - args[0].x
            self.height = args[1].y - args[0].y
        else:
            raise ValueError("Constructor accepts two Points or one point and two scalars (width and height)")

    def contains(self, pt):
        """
        test if point in rectangle
        :param pt: point
        :return: True if point in rectangle
        """
        return pt.x > self.corner.x and \
               pt.x < self.corner.x + self.width and \
               pt.y > self.corner.y and \
               pt.y < self.corner.y + self.height

class Line:

    def __init__(self, *args):
        """
        Line(pt1, pt2)
        :param pt1: first point in line
        :param pt2: second point in line
        Line(line_array)
        :param line_array: array of 4 elements
        """
        nArgs = len(args)
        if nArgs == 2:
            self.pt1 = args[0]
            self.pt2 = args[1]
        elif nArgs == 1:
            self.pt1 = Point(args[0][0:2])
            self.pt2 = Point(args[0][2:])
        else:
            raise ValueError("Constructor accepts two Points or one array of 4 elements")

    @staticmethod
    def LineIntersectsLine(line1, line2):
        """
        parametric line to line intersection,
        does not assume that the lines can be extended
        :param line1: first line
        :param line2: second line
        :return: true if the line intersects each other
        """
        q1 = (line1.pt1.y - line2.pt1.y)*(line2.pt2.x - line2.pt1.x) - \
            (line1.pt1.x - line2.pt1.x)*(line2.pt2.y - line2.pt1.y)
        d = (line1.pt2.x - line1.pt1.x)*(line2.pt2.y - line2.pt1.y) - \
            (line1.pt2.y - line1.pt1.y)*(line2.pt2.x - line2.pt1.x)
        if abs(d) < 1e-12:
            return False
        r = q1 / d
        q2 = (line1.pt1.y - line2.pt1.y) * (line1.pt2.x - line1.pt1.x) - \
            (line1.pt1.x - line2.pt1.x) * (line1.pt2.y - line1.pt1.y)
        s = q2 / d
        if r < 0 or r > 1 or s < 0 or s > 1:
            return False
        return True

    @staticmethod
    def GetLinesIntersection(line1, line2):
        """
        compute the intersection points of two lines
        assumes the lines can be extended
        https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
        :param line1: first line
        :param line2: second line
        :return: intersection point
        """
        # x1y2 - y1x2
        a = line1.pt1.x*line1.pt2.y - line1.pt1.y*line1.pt2.x
        # x3y4 - y3x4
        b = line2.pt1.x*line2.pt2.y - line2.pt1.y*line2.pt2.x
        # (x1 - x2)(y3 - y4) - (y1 - y2)(x3 - x4)
        c = (line1.pt1.x - line1.pt2.x)*(line2.pt1.y - line2.pt2.y) - \
            (line1.pt1.y - line1.pt2.y)*(line2.pt1.x - line2.pt2.x)
        if (abs(c) < 1e-12):
            return None
        x = (a*(line2.pt1.x - line2.pt2.x) - (line1.pt1.x - line1.pt2.x)*b)/c
        y = (a*(line2.pt1.y - line2.pt2.y) - (line1.pt1.y - line1.pt2.y)*b)/c
        return Point(x, y)

    @staticmethod
    def AngleBetweenLines(line1, line2):
        """
        :param line1: first line
        :param line2: second line
        :return: angle between the two lines in degree
        """
        v1 = line1.pt1 - line1.pt2
        v1.normalize()
        v2 = line2.pt1 - line2.pt2
        v2.normalize()
        return math.acos(Point.dot(v1, v2))*180.0/math.pi

    @staticmethod
    def LineIntersectsRect(line, rect):
        """
        test if a line intersects a rectangle
        :param line: line
        :param rect: rectangle
        :return: true if line intersects the rectangle
        """
        return Line.LineIntersectsLine(line, Line(rect.corner,
                                             Point(rect.corner.x + rect.width,
                                                   rect.corner.y))) or \
            Line.LineIntersectsLine(line, Line(Point(rect.corner.x + rect.width,
                                                     rect.corner.y),
                                          Point(rect.corner.x + rect.width,
                                                rect.corner.y + rect.height))) or \
            Line.LineIntersectsLine(line, Line(Point(rect.corner.x + rect.width,
                                                     rect.corner.y + rect.height),
                                          Point(rect.corner.x,
                                                rect.corner.y + rect.height))) or \
            Line.LineIntersectsLine(line, Line(Point(rect.corner.x,
                                                     rect.corner.y + rect.height),
                                          rect.corner))

import math

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise
    rotation about the given axis by theta in degree.
    """
    theta = float(theta)
    axis = np.asarray(axis)
    axis = axis/np.linalg.norm(axis)
    a = math.cos(math.radians(theta/2))
    b, c, d = -axis*math.sin(math.radians(theta/2))
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.mat([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                   [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                   [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

def rot2anglesExtrinsic(R, order='zyx'):
    """
    compute the extrinsic Euler angles according to the provided order
    ref: https://en.wikipedia.org/wiki/Euler_angles
    Note that if the order is xyz, then R = Rz(a3)*Ry(a2)*Rx(a1),
    this is an extrinsic rotation about the x axis with angle a1 followed by a
    rotation about the y axis with angle a2 followed by a rotation about the z
    axis with angle a3.
    :param R: rotation matrix
    :param order: order of rotation, default is zyx
    :return: list of Euler angles in order [a1, a2, a3]
    """
    if order is 'xyz':
        """
        R = Rz(a3)Ry(a2)Rx(a1)
        """
        a1 = math.degrees(math.atan2(R[2,1], R[2,2]))
        a2 = math.degrees(math.asin(-R[2,0]))
        a3 = math.degrees(math.atan2(R[1,0], R[0,0]))
    elif order is 'zyx':
        """
        R = Rx(a3)Ry(a2)Rz(a1)
        """
        a1 = math.degrees(math.atan2(-R[0,1], R[0,0]))
        a2 = math.degrees(math.asin(R[0,2]))
        a3 = math.degrees(math.atan2(-R[1,2], R[2,2]))

    elif order is 'yxz':
        """
        R = Rz(a3)Rx(a2)Ry(a1)
        """
        a1 = math.degrees(math.atan2(-R[2,0], R[2,2]))
        a2 = math.degrees(math.asin(R[2,1]))
        a3 = math.degrees(math.atan2(-R[0,1], R[1,1]))
    elif order is 'zxy':
        """
        R = Ry(a3)Rx(a2)Rz(a1)
        """
        a1 = math.degrees(math.atan2(R[1,0], R[1,1]))
        a2 = math.degrees(math.asin(-R[1,2]))
        a3 = math.degrees(math.atan2(R[0,2], R[2,2]))
    elif order is 'xzy':
        """
        R = Ry(a3)Rz(a2)Rx(a1)
        """
        a1 = math.degrees(math.atan2(-R[1,2], R[1,1]))
        a2 = math.degrees(math.asin(R[1,0]))
        a3 = math.degrees(math.atan2(-R[2,0], R[0,0]))
    elif order is 'yzx':
        """
        R = Rx(a3)Rz(a2)Ry(a1)
        """
        a1 = math.degrees(math.atan2(R[0,2], R[0,0]))
        a2 = math.degrees(math.asin(-R[0,1]))
        a3 = math.degrees(math.atan2(R[2,1], R[1,1]))
    elif order is "xzx":
        """
        R = Rx(a3)Rz(a2)Rx(a1)
        """
        a1 = math.degrees(math.atan2(R[0,2], -R[0,1]))
        a2 = math.degrees(math.acos(R[0,0]))
        a3 = math.degrees(math.atan2(R[2,0], R[1,0]))
    elif order is "xyx":
        """
        R = Rx(a3)Ry(a2)Rx(a1)
        """
        a1 = math.degrees(math.atan2(R[0,1], R[0,2]))
        a2 = math.degrees(math.acos(R[0,0]))
        a3 = math.degrees(math.atan2(R[1,0], -R[2,0]))
    elif order is "yzy":
        """
        R = Ry(a3)Rz(a2)Ry(a1)
        """
        a1 = math.degrees(math.atan2(R[1,2], R[1,0]))
        a2 = math.degrees(math.acos(R[1,1]))
        a3 = math.degrees(math.atan2(R[2,1], -R[0,1]))
    elif order is "yxy":
        """
        R = Ry(a3)Rx(a2)Ry(a1)
        """
        a1 = math.degrees(math.atan2(R[1,0], -R[1,2]))
        a2 = math.degrees(math.acos(R[1,1]))
        a3 = math.degrees(math.atan2(R[0,1], R[2,1]))
    elif order is "zyz":
        """
        R = Rz(a3)Ry(a2)Rz(a1)
        """
        a1 = math.degrees(math.atan2(R[2,1], -R[2,0]))
        a2 = math.degrees(math.acos(R[2,2]))
        a3 = math.degrees(math.atan2(R[1,2], R[0,2]))
    elif order is 'zxz':
        """
        R = Rz(a3)Rx(a2)Rz(a1)
        """
        a1 = math.degrees(math.atan2(R[2,0], R[2,1]))
        a2 = math.degrees(math.acos(R[2,2]))
        a3 = math.degrees(math.atan2(R[0,2], -R[1,2]))
    return [a1, a2, a3]

def test_rot2anglesExtrinsic():
    x = [1.0,0,0]
    y = [0,1.0,0]
    z = [0,0,1.0]

    alpha = 48.0
    beta = 10.0
    gamma = 75.0

    Rx = rotation_matrix(x, alpha)
    Ry = rotation_matrix(y, beta)
    Rz = rotation_matrix(z, gamma)

    np.set_printoptions(precision=4, suppress=True)
    print "extrinsic rotation: " + str(rot2anglesExtrinsic(Rx*Ry*Rz,'zyx'))
    print "extrinsic rotation: " + str(rot2anglesExtrinsic(Rz*Ry*Rx,'xyz'))

    print "extrinsic rotation: " + str(rot2anglesExtrinsic(Rz*Rx*Ry,'yxz'))
    print "extrinsic rotation: " + str(rot2anglesExtrinsic(Ry*Rx*Rz,'zxy'))

    print "extrinsic rotation: " + str(rot2anglesExtrinsic(Rx*Rz*Ry,'yzx'))
    print "extrinsic rotation: " + str(rot2anglesExtrinsic(Ry*Rz*Rx,'xzy'))

    print "extrinsic rotation: " + str(rot2anglesExtrinsic(Rz*Rx*Rz,'zxz'))
    print "extrinsic rotation: " + str(rot2anglesExtrinsic(Rx*Rz*Rx,'xzx'))

    print "extrinsic rotation: " + str(rot2anglesExtrinsic(Ry*Rx*Ry,'yxy'))
    print "extrinsic rotation: " + str(rot2anglesExtrinsic(Rx*Ry*Rx,'xyx'))

    print "extrinsic rotation: " + str(rot2anglesExtrinsic(Rz*Ry*Rz,'zyz'))
    print "extrinsic rotation: " + str(rot2anglesExtrinsic(Ry*Rz*Ry,'yzy'))

def rot2anglesIntrinsic(R, order='zyx'):
    return rot2anglesExtrinsic(R, order)[::-1]


def rand_color():
    return np.random.randint(0,255,(1,3))[0]

