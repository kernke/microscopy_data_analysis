# -*- coding: utf-8 -*-
"""
@author: kernke
"""

import numpy as np
import cv2
import copy
from numba import njit
import scipy.special

from skimage.draw import circle_perimeter
from skimage.draw import line_aa
import matplotlib.pyplot as plt

from .image_processing import img_make_square



#%% folder_file
def folder_file(path_string):
    name = path_string.replace("\\", "/")
    pos = name[::-1].find("/")
    return name[:-pos], name[-pos:]

#%% assure_multiple
def assure_multiple(*x):
    res = []
    count = 0
    for i in x:
        count += 1
        if len(np.shape(i)) == 0:
            res.append([i])
        else:
            res.append(i)

    if count == 1:
        return res[0]
    else:
        return res

#%%
def rfft_circ_mask(imshape, mask_radius=680, mask_sigma=50):
    kernel_size = 7 * mask_sigma
    if kernel_size % 2 == 0:
        kernel_size += 1
    mask = make_circular_mask(
        (imshape[0] - 1) / 2, (imshape[1] - 1) / 2, mask_radius, np.zeros(imshape)
    )
    maskb = cv2.GaussianBlur(
        mask.astype(np.double), [kernel_size, kernel_size], mask_sigma
    )

    rolledmask = np.roll(maskb, -int(maskb.shape[0] / 2), axis=0)
    rolledmask = np.roll(rolledmask, -int(maskb.shape[1] / 2), axis=1)

    halfrolledmask = rolledmask[:, : rolledmask.shape[1] // 2 + 1]
    return halfrolledmask



#%% peak_com
def peak_com(y, delta=None, roi=None):
    if roi is None:
        pos = np.argmax(y)
    else:
        pos = roi[0] + np.argmax(y[roi[0] : roi[1]])
    if delta is None:
        delta = min(pos, len(y) - pos)
        print(delta)
        start = pos - delta
        end = pos + delta
    else:
        start = max(0, pos - delta)
        end = min(len(y), pos + delta)
    return np.sum(np.arange(start, end) * y[start:end]) / np.sum(y[start:end]), pos


#%% peak_com2d


def peak_com2d(data, delta=None, roi=None):

    if len(np.shape(delta)) == 0:
        delt = [delta, delta]
    else:
        delt = delta

    if roi is None:
        dat = data
        roi = [[0, data.shape[0]], [0, data.shape[1]]]
    else:
        dat = data[roi[0][0] : roi[0][1], roi[1][0] : roi[1][1]]

    dist = np.argmax(dat)
    disty = dist % dat.shape[1]
    distx = dist // dat.shape[1]

    delt[0] = min(distx, dat.shape[0] - distx)
    delt[1] = min(disty, dat.shape[1] - disty)

    xstart = distx - delt[0]
    xend = distx + delt[0]
    ystart = disty - delt[1]
    yend = disty + delt[1]
    dat2 = dat[xstart:xend, ystart:yend]

    y = np.sum(dat2, axis=0)
    x = np.sum(dat2, axis=1)

    indx = np.arange(xstart, xend)
    indy = np.arange(ystart, yend)

    mvposx = distx + roi[0][0]
    mvposy = disty + roi[1][0]

    xpos = np.sum(indx * x) / np.sum(x)
    ypos = np.sum(indy * y) / np.sum(y)

    xpos += roi[0][0]
    ypos += roi[1][0]

    return np.array([xpos, ypos]), np.array([mvposx, mvposy])


#%%
def polygon_roi(directions_deg, radius):
    directions_deg = np.array(directions_deg)
    directions_neg = directions_deg + 180

    directions_all = np.concatenate((directions_deg, directions_neg))
    directions_rad = directions_all / 180 * np.pi

    x = radius * np.cos(directions_rad)
    y = radius * np.sin(directions_rad)

    x -= np.min(x)
    y -= np.min(y)

    return x.astype(int), y.astype(int)


#%%
@njit("float64(float64,float64,float64,float64,float64,float64)")
def isleft(P0_0, P0_1, P1_0, P1_1, P2_0, P2_1):
    return (P1_0 - P0_0) * (P2_1 - P0_1) - (P2_0 - P0_0) * (P1_1 - P0_1)


def point_in_roi(xroi, yroi, xpoint, ypoint):
    signs = np.zeros(len(xroi))
    for i in range(len(xroi)):
        signs[i] = isleft(xroi[i - 1], yroi[i - 1], xroi[i], yroi[i], xpoint, ypoint)
    return np.prod(np.sign(signs)) > 0

#%% lineIntersection
def lineIntersection(a, b, c, d):
    # Line AB represented as a1x + b1y = c1
    a1 = b[1] - a[1]
    b1 = a[0] - b[0]
    c1 = a1 * (a[0]) + b1 * (a[1])

    # Line CD represented as a2x + b2y = c2
    a2 = d[1] - c[1]
    b2 = c[0] - d[0]
    c2 = a2 * (c[0]) + b2 * (c[1])

    determinant = a1 * b2 - a2 * b1

    x = (b2 * c1 - b1 * c2) / determinant
    y = (a1 * c2 - a2 * c1) / determinant
    return (x, y)


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


# Return true if line segments AB and CD intersect
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

#%%
def get_n_peaks_1d(y, x=None, delta=0, n=5, roi=None):
    """
    Obtain n maxima from y in descending order
    Calculated by consecutively finding the maximum of y
    and setting values near the maximum with distance +-delta
    to a median value of y, then repeating the process n times

    Parameters
    ----------
    y: array

    optional:
    x: array, defaults to None
    delta: float, defaults to 0
    n: int, defaults to 1
    roi: tuple, defaults to None

    Returns
    -------
    array with length n containing the positions of the n peaks
    """

    if x is None:
        x = np.arange(len(y))
    else:
        dist = 1
        while x[dist] - x[0] == 0:
            dist += 1
        if x[dist] - x[0] < 0:
            sortindex = np.argsort(x)
            x = x[sortindex]
            y = y[sortindex]

    if roi is None:
        newy = copy.deepcopy(y)
        xadd = 0
    else:
        start = np.where(x >= roi[0])[0][0]
        if roi[1] >= np.max(x):
            end = len(x)
        else:
            end = np.where(x > roi[1])[0][0]
        xadd = start
        newy = copy.deepcopy(y[start:end])

    ymaxpos = np.zeros(n, dtype=int)
    med = np.min(y)
    for i in range(n):
        pos = np.argmax(newy)
        ymaxpos[i] = pos + xadd

        delstart = np.where(x >= x[ymaxpos[i]] - delta)[0][0] - xadd
        delend = np.where(x <= x[ymaxpos[i]] + delta)[0][-1] + 1 - xadd
        if delstart < 0:
            delstart = 0
        newy[delstart:delend] = med

    return x[ymaxpos]




#%% get_angular_dist
def get_angular_dist(image, borderdist=100, centerdist=20, plotcheck=False):

    if image.shape[0] != image.shape[1]:
        print("Warning: image is cropped to square")
        img = img_make_square(image)

    fftimage = np.fft.rfft2(img)
    rffti = np.roll(fftimage, -int(fftimage.shape[0] / 2), axis=0)
    fim = np.log(np.abs(rffti))
    halfheight = fim.shape[0] // 2
    radius = halfheight - borderdist
    center = np.array([fim.shape[0] // 2, 0], dtype=int)

    rcirc, ccirc = circle_perimeter(*center, radius, shape=fim.shape)

    values = np.zeros(len(rcirc))
    angles = np.zeros(len(rcirc))

    checkcenter = np.zeros([len(rcirc), 2])
    checkcirc = np.zeros([len(rcirc), 2])

    for i in range(len(rcirc)):
        dx, dy = ccirc[i], rcirc[i] - halfheight
        angles[i] = -np.arctan2(dy, dx)
        effcenter = np.round(center + centerdist * np.array([dy, dx]) / radius).astype(
            int
        )

        checkcenter[i] = effcenter
        checkcirc[i] = rcirc[i], ccirc[i]

        rr, cc, vv = line_aa(*effcenter, rcirc[i], ccirc[i])
        values[i] = np.sum(fim[rr, cc] * vv) / np.sum(vv)

    sortindex = np.argsort(angles)
    angles = angles[sortindex]
    values = values[sortindex]
    checkcenter = checkcenter[sortindex]
    checkcirc = checkcirc[sortindex]

    angledeg = angles / np.pi * 180 + 90

    if plotcheck:

        colors = ["b", "r", "g", "c", "m", "y", "k", "gray"]
        plt.imshow(fim)

        div = len(sortindex) // 8

        for i in range(8):
            anglename = np.round(angledeg[i * div], 1)

            plt.plot(
                [checkcenter[i * div][1], checkcirc[i * div][1]],
                [checkcenter[i * div][0], checkcirc[i * div][0]],
                "-o",
                c=colors[i],
                alpha=0.4,
                label=str(anglename) + r"$\,^\circ$",
            )

        plt.title("2D real-valued discrete Fourier-Transformation")
        plt.legend(
            title="examplary\nline profiles",
            bbox_to_anchor=(1.04, 0.5),
            loc="center left",
            borderaxespad=0,
        )
        plt.show()
    return angledeg, values




#%% take_map
def take_map(mapimage, tilesize=1000, overlap=0.25):
    mis = np.array(np.shape(mapimage))
    number_of_tiles = np.round(mis / tilesize).astype(int)
    tile_size = (mis / (1.0 * number_of_tiles)).astype(int)
    over_lap = overlap * tile_size
    images = {}
    lowhigh = {}
    for i in range(number_of_tiles[0]):
        vlow = i * tile_size[0] - int(0.5 * over_lap[0]) * 1 * (i > 0)
        if i != number_of_tiles[0] - 1:
            vhigh = (i + 1) * tile_size[0] + int(0.5 * over_lap[0])
        else:
            vhigh = mis[0]

        for j in range(number_of_tiles[1]):
            hlow = j * tile_size[1] - int(0.5 * over_lap[1]) * 1 * (j > 0)
            if j != number_of_tiles[1] - 1:
                hhigh = (j + 1) * tile_size[1] + int(0.5 * over_lap[1])
            else:
                hhigh = mis[1]

            images[i, j] = mapimage[vlow:vhigh, hlow:hhigh]
            lowhigh[i, j] = [[vlow, vhigh], [hlow, hhigh]]
    return images, lowhigh


#%% circ_mask


@njit
def circ_mask(x0, y0, r, image):
    mask = np.zeros(np.shape(image), dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if (i - x0) ** 2 + (j - y0) ** 2 < r**2:
                mask[i, j] = 1
    return mask


def make_circular_mask(x0, y0, r, image):
    if x0 % 2 != 0 or y0 % 2 != 0:
        return circ_mask(x0, y0, r, image)
    else:
        mask = np.zeros(np.shape(image))
        return cv2.circle(mask, [x0, y0], r, 1, -1)


#%% get_max
def get_max(z):
    "returns the indices of the maximum value of an array MxN"
    maxpos = np.argmax(z)
    x0 = maxpos // z.shape[1]
    y0 = maxpos % z.shape[1]
    return np.array([x0, y0]).astype(int)


#%% pascal triangle
def pascal_numbers(n):
    """Returns the n-th row of Pascal's triangle'"""
    return scipy.special.comb(n, np.arange(n + 1))


#%% smoothbox_kernel


def smoothbox_kernel(kernel_size):
    """Gaussian Smoothing kernel approximated by integer-values obtained via binomial distribution"""
    r = kernel_size[0]
    c = kernel_size[1]
    sb = np.zeros([r, c])

    row = pascal_numbers(r - 1)
    col = pascal_numbers(c - 1)

    row /= np.sum(row)
    col /= np.sum(col)

    for i in range(r):
        for j in range(c):
            sb[i, j] = row[i] * col[j]

    return sb



#%% make Mask


@njit
def make_mask(rot, d=3):
    mask = rot > 0
    newmask = rot > 0
    for i in range(d, mask.shape[0] - d):
        for j in range(d, mask.shape[1] - d):
            if mask[i, j]:
                if (
                    not mask[i - d, j]
                    or not mask[i + d, j]
                    or not mask[i, j - d]
                    or not mask[i, j + d]
                ):
                    newmask[i, j] = 0

    newmask[:d, :] = 0
    newmask[:, :d] = 0
    newmask[-d:, :] = 0
    newmask[:, -d:] = 0
    return newmask




#%%


def determine_thresh(image):
    Nrows, Ncols = image.shape
    nr = Nrows // 3
    nc = Ncols // 3
    roi = image[nr : 2 * nr, nc : 2 * nc]
    i_mean, i_std = np.mean(roi), np.std(roi)
    thresh = i_mean - 2.575829 * i_std
    return thresh


