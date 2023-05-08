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
import h5py

import imageio

#%% folder_file
def folder_file(path_string):
    name = path_string.replace("\\", "/")
    pos = name[::-1].find("/")
    return name[:-pos], name[-pos:]



class h5func:

    #%% merge_h5pys
    
    
    def merge_h5pys(newh5, *h5files):
        with h5py.File(newh5, "w") as res:
            for i in h5files:
                with h5py.File(i, "r") as hf:
                    for j in hf.keys():
                        hf.copy(hf[j], res, j)
    
    
    def merge_h5files(newh5, *h5files):
        with h5py.File(newh5, "w") as res:
            for i in h5files:
                pathname, groupname = folder_file(i)
                with h5py.File(i, "r") as hf:
                    for j in hf.keys():
                        hf.copy(hf[j], res, groupname[:-3] + "/" + j)


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


def to_uint8(img):
    img -= np.min(img)
    return (img / np.max(img) * 255.5).astype(np.uint8)


#%% points_on_image
def points_on_image(image):
    global list_of_points
    global ax
    list_of_points = []

    fig, ax = plt.subplots()

    ax.imshow(image, cmap="gray")
    # ax.set_title("$")
    # ax.set_xticks([0,np.pi,2*np.pi],["0","$\pi$","$2\pi$"])
    fig.canvas.mpl_connect("button_press_event", click)
    plt.gcf().canvas.draw_idle()

    return list_of_points


#%% click
def click(event):
    global list_of_points

    if event.button == 3:  # right clicking

        x = event.xdata
        y = event.ydata

        list_of_points.append([x, y])
        ax.plot(x, y, "o")
        print(x, y)
        plt.gcf().canvas.draw()


#%% align_images


def align_image_fast1(im1, matrix1, reswidth, resheight):
    return cv2.warpPerspective(
        im1, matrix1, (reswidth, resheight), flags=cv2.INTER_CUBIC
    )


def align_image_fast2(im2, reswidth, resheight, width_shift, height_shift):
    img2Reg = np.zeros([resheight, reswidth])
    img2Reg[
        height_shift : height_shift + im2.shape[0],
        width_shift : width_shift + im2.shape[1],
    ] = im2
    return img2Reg


def align_images(im1s, im2, p1s, p2, verbose=False):
    # align p1 to p2
    # p2 higher resolution recommended

    allwidths = []
    allheights = []
    for i in range(len(im1s)):

        im1 = im1s[i]
        p1 = p1s[i]

        matrix1, mask1 = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)

        xf = np.arange(im1.shape[1] - 1).tolist()
        xf += (np.zeros(im1.shape[0] - 1) + im1.shape[1] - 1).tolist()
        xf += np.arange(1, im1.shape[1]).tolist()
        xf += np.zeros(im1.shape[0] - 1).tolist()

        yf = np.zeros(im1.shape[1] - 1).tolist()
        yf += np.arange(im1.shape[0] - 1).tolist()
        yf += (np.zeros(im1.shape[1] - 1) + im1.shape[0] - 1).tolist()
        yf += np.arange(1, im1.shape[0]).tolist()

        img_matrix = np.stack([xf, yf, np.ones(len(xf))])

        res = np.tensordot(matrix1, img_matrix, axes=1)

        allwidths.append(np.round(min(np.min(res[0]), 0)).astype(int))
        allheights.append(np.round(min(np.min(res[1]), 0)).astype(int))

        allwidths.append(
            np.round(max(np.max(res[0]), im2.shape[1])).astype(int) - allwidths[-1]
        )
        allheights.append(
            np.round(max(np.max(res[1]), im2.shape[0])).astype(int) - allheights[-1]
        )

    reswidth = np.max(allwidths)
    resheight = np.max(allheights)
    width_shift = np.abs(np.min(allwidths))
    height_shift = np.abs(np.min(allheights))
    shift = np.array([width_shift, height_shift])

    img2Reg = np.zeros([resheight, reswidth])
    img2Reg[
        height_shift : height_shift + im2.shape[0],
        width_shift : width_shift + im2.shape[1],
    ] = im2

    im1res = []

    p2a = np.zeros(p2.shape)
    for i in range(len(p2)):
        p2a[i] = p2[i] + shift

    matrices = []
    for i in range(len(im1s)):
        p1 = p1s[i]
        im1 = im1s[i]

        matrix1, mask1 = cv2.findHomography(p1, p2a, cv2.RANSAC, 5.0)
        matrices.append(matrix1)
        img1Reg = cv2.warpPerspective(
            im1, matrix1, (reswidth, resheight), flags=cv2.INTER_CUBIC
        )
        im1res.append(img1Reg)

    if verbose:
        return im1res, img2Reg, matrices, reswidth, resheight, width_shift, height_shift
    else:
        return im1res, img2Reg


#%%  plot_sortout
def plot_sortout(image, sortout, legend=True, alpha=0.5, markersize=0.5):
    plt.imshow(image, cmap="gray")
    colors = ["b", "r", "g", "c", "m", "y"]
    for j in range(len(sortout)):
        count = 0
        for i in sortout[j]:
            if count == 0:
                plt.plot(
                    i[:, 1],
                    i[:, 0],
                    "o",
                    c=colors[j],
                    alpha=alpha,
                    label=str(j),
                    markersize=markersize,
                )
            else:
                plt.plot(
                    i[:, 1],
                    i[:, 0],
                    "o",
                    c=colors[j],
                    alpha=alpha,
                    markersize=markersize,
                )
            count += 1
    if legend == True:
        plt.legend()


#%% morphLaplace
def morphLaplace(image, kernel):
    return cv2.erode(image, kernel) + cv2.dilate(image, kernel) - 2 * image - 128


#%% gammaCorrection
def gammaCorrection(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)


#%% make_scale_bar
def make_scale_bar(
    images, pixratios, lengthperpix, barlength, org, thickness=4, color=(255, 0, 0)
):

    org = np.array(org)
    for i in range(len(images)):
        pixlength = (barlength / lengthperpix) / pixratios[i]
        pixlength = np.round(pixlength).astype(int)
        pt2 = org + np.array([0, pixlength])
        cv2.line(images[i], org[::-1], pt2[::-1], color, thickness=thickness)


#%% make_square


def make_square(image, startindex=None):
    """
    crops the largest square image from the original, by default from the center
    the position of the cropped square can be specified via startindex,
    moving the frame from the upper left corner at startindex=0
    to the lower right corner at startindex=|M-N|

    Parameters
    ----------
    image: MxN array ; numpy array
    startindex: 0 <= startindex <= |M-N| ; int

    Returns:
    square_image either MxM or NxN ; numpy array

    """

    ishape = image.shape
    index_small, index_big = np.argsort(ishape)

    roi = np.zeros([2, 2], dtype=int)

    roi[index_small] = [0, ishape[index_small]]

    delta = np.abs(ishape[1] - ishape[0])

    if startindex is None:
        startindex = np.floor(delta / 2)
    else:
        if startindex > delta or startindex < 0:
            print("Error: Invalid startindex")
            print("0 <= startindex <= " + str(delta))

    roi[index_big] = startindex, startindex + ishape[index_small]

    square_image = image[roi[0, 0] : roi[0, 1], roi[1, 0] : roi[1, 1]]

    return square_image


#%% make_mp4
def make_mp4(filename, images, fps):

    with imageio.get_writer(filename, mode="I", fps=fps) as writer:
        for i in range(len(images)):
            writer.append_data(images[i])

    return True


#%% zoom


def zoom(img, zoom_center, final_height, steps, gif_resolution_to_final=1):

    iratio = img.shape[0] / img.shape[1]

    final_size = np.array([final_height, final_height / iratio])

    startpoints = np.zeros([4, 2], dtype=int)
    endpoints = np.zeros([4, 2], dtype=int)

    startpoints[2, 1] = img.shape[1] - 1
    startpoints[1, 0] = img.shape[0] - 1
    startpoints[3] = img.shape
    startpoints[3] -= 1

    endpoints[0] = np.round(zoom_center - final_size / 2).astype(int)
    endpoints[3] = np.round(zoom_center + final_size / 2).astype(int)

    tocorner = np.array([-final_size[0], final_size[1]]) / 2
    endpoints[1] = np.round(zoom_center - tocorner).astype(int)
    tocorner = np.array([final_size[0], -final_size[1]]) / 2
    endpoints[2] = np.round(zoom_center - tocorner).astype(int)

    steps += 1
    cornerpoints = np.zeros([steps, 4, 2], dtype=int)
    pixratios = np.zeros(steps)
    for i in range(4):
        for j in range(2):
            cornerpoints[:, i, j] = np.round(
                np.linspace(startpoints[i, j], endpoints[i, j], steps)
            ).astype(int)

    final_resolution = np.round(final_size * gif_resolution_to_final).astype(int)
    images = np.zeros([steps, final_resolution[0], final_resolution[1]])
    for i in range(steps):
        pixratios[i] = (
            cornerpoints[i, 1, 0] - cornerpoints[i, 0, 0]
        ) / final_resolution[0]

        roi_img = img[
            cornerpoints[i, 0, 0] : cornerpoints[i, 1, 0],
            cornerpoints[i, 0, 1] : cornerpoints[i, 2, 1],
        ]

        ratio = roi_img.shape[0] / final_resolution[0]  # size
        sigma = ratio / 4

        ksize = np.round(5 * sigma).astype(int)
        if ksize % 2 == 0:
            ksize += 1
        roi_img = cv2.GaussianBlur(roi_img, [ksize, ksize], sigma)
        images[i] = cv2.resize(roi_img, final_resolution[::-1], cv2.INTER_AREA)

    return images, pixratios


#%% get_angular_dist
def get_angular_dist(image, borderdist=100, centerdist=20, plotcheck=False):

    if image.shape[0] != image.shape[1]:
        print("Warning: image is cropped to square")
        img = make_square(image)

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


#%% MIC_tile


def MIC_tile(im, tiles=3):
    """Minimum-Image-Convention tiling"""
    s = np.array(im.shape)
    new = np.zeros(s * tiles, dtype=im.dtype)
    for i in range(tiles):
        for j in range(tiles):
            new[s[0] * i : s[0] * (i + 1), s[1] * j : s[1] * (j + 1)] = im

    oij = tiles // 2
    orig = (s[0] * oij, s[0] * (oij + 1)), (s[1] * oij, s[1] * (oij + 1))
    return new, orig


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


#%% rebin
def rebin(arr, new_shape):
    """reduce the resolution of an image MxN to mxn by taking an average,
    whereby M and N must be multiples of m and n"""
    shape = (
        new_shape[0],
        arr.shape[0] // new_shape[0],
        new_shape[1],
        arr.shape[1] // new_shape[1],
    )
    return arr.reshape(shape).mean(-1).mean(1)


#%% rotate

# this function is a modified version of the original from
# https://github.com/PyImageSearch/imutils/blob/master/imutils/convenience.py#L41
def rotate_bound(image, angle, flag="cubic", bm=1):

    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int(np.round((h * sin) + (w * cos)))
    nH = int(np.round((h * cos) + (w * sin)))

    if bm == 0:
        bm = cv2.BORDER_CONSTANT
    elif bm == 1:
        bm = cv2.BORDER_REPLICATE

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    invRotateMatrix = cv2.invertAffineTransform(M)
    log = [(h, w), invRotateMatrix]
    # perform the actual rotation and return the image
    if flag == "cubic":
        return (
            cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=bm),
            log,
        )
    else:
        return (
            cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=bm),
            log,
        )


#%% rotate back
def rotate_back(image, log, flag="cubic", bm=1):

    (h, w), invM = log

    if bm == 0:
        bm = cv2.BORDER_CONSTANT
    elif bm == 1:
        bm = cv2.BORDER_REPLICATE

    if flag == "cubic":
        return cv2.warpAffine(image, invM, (w, h), flags=cv2.INTER_CUBIC, borderMode=bm)
    else:
        return cv2.warpAffine(
            image, invM, (w, h), flags=cv2.INTER_LINEAR, borderMode=bm
        )


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


#%% asymmetric non maximum supppression


def anms(img, mask, thresh_ratio=1.5, ksize=5, asympix=0, damping=5):
    newimg = copy.deepcopy(img)
    cimg = cv2.sepFilter2D(
        img, cv2.CV_64F, np.ones(1), np.ones(ksize), borderType=cv2.BORDER_ISOLATED
    )
    rimg = cv2.sepFilter2D(
        img,
        cv2.CV_64F,
        np.ones(ksize + asympix),
        np.ones(1),
        borderType=cv2.BORDER_ISOLATED,
    )
    return aysmmetric_non_maximum_suppression(
        newimg, img, cimg, rimg, mask, thresh_ratio, ksize, asympix, damping
    )


@njit
def aysmmetric_non_maximum_suppression(
    newimg, img, cimg, rimg, mask, thresh_ratio, ksize, asympix, damping
):
    ioffs = ksize // 2
    joffs = ksize // 2 + asympix // 2

    for i in range(ioffs, img.shape[0] - ioffs):
        for j in range(joffs, img.shape[1] - joffs):
            if not mask[i, j]:
                pass
            elif (
                not mask[i - ioffs, j - joffs]
                or not mask[i + ioffs, j + joffs]
                or not mask[i - ioffs, j + joffs]
                or not mask[i + ioffs, j - joffs]
            ):
                pass
            else:
                v = max(cimg[i, j - joffs : j + joffs + 1])
                h = max(rimg[i - ioffs : i + ioffs + 1, j]) * ksize / (ksize + asympix)

                if h > v * thresh_ratio:
                    newimg[i, j] = img[i, j]
                else:
                    newimg[i, j] = (
                        img[i, j] / damping
                    )  # np.min(img[i-ioffs:i+ioffs+1,j-joffs:j+joffs+1])
    return newimg


#%% asymmetric non maximum supppression median


def anms_median(img, mask, thresh_ratio=1.5, ksize=5, asympix=0):
    newimg = copy.deepcopy(img)
    return aysmmetric_non_maximum_suppression_median(
        newimg, img, mask, thresh_ratio, ksize, asympix
    )


@njit
def aysmmetric_non_maximum_suppression_median(
    newimg, img, mask, thresh_ratio, ksize, asympix
):
    ioffs = ksize // 2  # +asympix//2
    joffs = ksize // 2 + asympix // 2
    # newimg=img#np.zeros(img.shape)
    for i in range(ioffs, img.shape[0] - ioffs):
        for j in range(joffs, img.shape[1] - joffs):
            if not mask[i, j]:
                pass
            elif (
                not mask[i - ioffs, j - joffs]
                or not mask[i + ioffs, j + joffs]
                or not mask[i - ioffs, j + joffs]
                or not mask[i + ioffs, j - joffs]
            ):
                newimg[i, j] = img[i, j]
            else:
                g = img[i - ioffs : i + ioffs + 1, j - joffs : j + joffs + 1]
                v = max(np.sum(g, axis=0))  # * ksize/(ksize+asympix)
                h = max(np.sum(g, axis=1)) * ksize / (ksize + asympix)
                if h > v * thresh_ratio:
                    newimg[i, j] = img[i, j]
                else:
                    newimg[i, j] = np.median(g)
    return newimg


#%% noise level determination from aysmmetric_non_maximum_suppression


@njit
def anms_noise(img, mask, thresh_ratio, ksize, asympix):
    ioffs = ksize // 2  # +asympix//2
    joffs = ksize // 2 + asympix // 2
    npix = ksize * (ksize + asympix)
    noisemean = []
    noisemax = []
    noisestd = []
    # newimg=img#np.zeros(img.shape)
    for i in range(ioffs, img.shape[0] - ioffs):
        for j in range(joffs, img.shape[1] - joffs):
            if not mask[i, j]:
                pass
            elif (
                not mask[i - ioffs, j - joffs]
                or not mask[i + ioffs, j + joffs]
                or not mask[i - ioffs, j + joffs]
                or not mask[i + ioffs, j - joffs]
            ):
                pass
                #    newimg[i,j]=img[i,j]
            else:
                g = img[i - ioffs : i + ioffs + 1, j - joffs : j + joffs + 1]
                v = max(np.sum(g, axis=0))  # * ksize/(ksize+asympix)
                h = max(np.sum(g, axis=1))
                if h * ksize / (ksize + asympix) > v * thresh_ratio:
                    pass
                    # newimg[i,j]=img[i,j]
                else:
                    ave = (v + h) / npix
                    g = (g - ave) ** 2
                    noisemean.append(ave)
                    noisemax.append(np.max(g))
                    std = np.sum(g)
                    noisestd.append(std)
                    # newimg[i,j]=np.min(g)
    return noisemax, noisemean, noisestd


def determine_thresh(image):
    Nrows, Ncols = image.shape
    nr = Nrows // 3
    nc = Ncols // 3
    roi = image[nr : 2 * nr, nc : 2 * nc]
    i_mean, i_std = np.mean(roi), np.std(roi)
    thresh = i_mean - 2.575829 * i_std
    return thresh


def determine_noise_threshold(img, mask, thresh_ratio, ksize, asympix):
    npix = ksize * (ksize + asympix)

    noisemax, noisemean, noisestd = anms_noise(img, mask, thresh_ratio, ksize, asympix)
    nma = np.array(noisemax)
    nme = np.array(noisemean)
    nms = np.array(noisestd)
    return np.sqrt(nma), nme, np.sqrt(nms / npix)
