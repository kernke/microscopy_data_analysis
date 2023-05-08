# -*- coding: utf-8 -*-
"""
@author: kernke
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from .general_util import *
import copy
from numba import njit
import h5py
import time

#%% go_over_data
a = 3

def go_over_data(
    newh5, oldh5, rotangles, tempnames, params, roi=None, no_output=False, timed=False
):
    rotangles, tempnames = assure_multiple(rotangles, tempnames)
    (
        size,
        overlap,
        anms_threshold,
        ksize_anms,
        ksize_erodil,
        line,
        db_dist,
        kernel,
    ) = params

    if roi is None:
        testindex = 0
    else:
        testindex = roi[0]

    with h5py.File(newh5, "w") as res, h5py.File(oldh5, "r") as hf:

        imshape = hf[tempnames[0] + "/imgs"][testindex].shape
        dummy = np.ones(imshape)
        masks = []
        for i in range(len(rotangles)):
            drot, log = rotate_bound(dummy, rotangles[i], bm=0)
            newmask = make_mask(drot, 2)
            masks.append(newmask)

        for j in range(len(tempnames)):

            print(j)
            print("________________________________")
            if timed:
                time_now = time.time()

            times = hf[tempnames[j] + "/times"][()]

            if roi is None:
                iroi = np.arange(len(times))
            else:
                iroi = roi

            time0 = times[0]

            for ccounter in range(len(rotangles)):
                name = "/check" + str(ccounter) + "_"
                res.create_dataset(
                    tempnames[j] + name,
                    shape=[len(iroi), imshape[0], imshape[1]],
                    dtype=np.uint8,
                    chunks=(1, imshape[0], imshape[1]),
                    compression="gzip",
                    compression_opts=2,
                )
            res.create_dataset(tempnames[j] + "/times", shape=(len(iroi)), dtype="f")

            for i in iroi:
                print(i)
                if timed:
                    print(time.time() - time_now)
                img = hf[tempnames[j] + "/imgs"][i]

                if no_output:
                    checkmaps = np.ones([len(rotangles), imshape[0], imshape[1]])
                else:

                    lapl = morphLaplace(img, kernel)

                    summed = np.zeros(lapl.shape, dtype=np.double)
                    summed += 255 - lapl
                    summed += img
                    copt = to_uint8(summed)

                    images, lowhigh = take_map(copt, size, overlap)
                    checkmaps = process_partial(
                        images,
                        rotangles,
                        lowhigh,
                        masks,
                        line=line,
                        anms_threshold=anms_threshold,
                        ksize_anms=ksize_anms,
                        ksize_erodil=ksize_erodil,
                        db_dist=db_dist,
                    )

                res[tempnames[j] + "/times"][i] = times[i] - time0
                for ccounter in range(len(rotangles)):
                    name = "/check" + str(ccounter) + "_"
                    cm = checkmaps[ccounter] / np.max(checkmaps[ccounter]) * 255
                    res[tempnames[j] + name][i] = cm.astype(np.uint8)
    return checkmaps


#%% align_data
def align_data(
    newh5, oldh5, refh5, rotangles, tempnames, params, roi=None, no_output=False
):
    rotangles, tempnames = assure_multiple(rotangles, tempnames)

    affxs, driftindices, acl, aclc, cl, clc = params

    if roi is None:
        testindex = 0
    else:
        testindex = roi[0]

    with h5py.File(newh5, "w") as res, h5py.File(oldh5, "r") as hf, h5py.File(
        refh5, "r"
    ) as rf:

        img = rf[tempnames[0] + "/imgs"][testindex]
        (
            im1s,
            im2,
            matrices,
            reswidth,
            resheight,
            width_shift,
            height_shift,
        ) = align_images([img, cl], clc, [affxs[0][0], acl], aclc, verbose=True)

        imshape = im2.shape
        for j in range(len(tempnames)):
            print(j)
            print("________________________________")

            times = hf[tempnames[j] + "/times"][()]

            if roi is None:
                iroi = np.arange(len(times))
            else:
                iroi = roi

            cdi = 0

            for ccounter in range(len(rotangles)):
                name = "/check" + str(ccounter) + "_"
                res.create_dataset(
                    tempnames[j] + name,
                    shape=[len(iroi), imshape[0], imshape[1]],
                    dtype=np.uint8,
                    chunks=(1, imshape[0], imshape[1]),
                    compression="gzip",
                    compression_opts=2,
                )

            res.create_dataset(
                "ref_" + tempnames[j],
                shape=[len(iroi), imshape[0], imshape[1]],
                dtype=np.uint8,
                chunks=(1, imshape[0], imshape[1]),
                compression="gzip",
                compression_opts=2,
            )

            res[tempnames[j]] = times

            for i in iroi:
                print(i)
                img = rf[tempnames[j] + "/imgs"][i]

                if i == driftindices[j][cdi]:
                    (
                        im1s,
                        im2,
                        matrices,
                        reswidth,
                        resheight,
                        width_shift,
                        height_shift,
                    ) = align_images(
                        [img, cl], clc, [affxs[j][cdi], acl], aclc, verbose=True
                    )

                    matrix = matrices[0]
                    cdi += 1

                alignedmaps = np.ones(
                    [len(rotangles), imshape[0], imshape[1]], dtype=np.uint8
                )
                if no_output:
                    refim = np.ones(imshape, dtype=np.uint8)
                else:
                    refim = align_image_fast1(img, matrix, reswidth, resheight)
                    for ccounter in range(len(rotangles)):
                        prefix = "check" + str(ccounter) + "_"
                        cm = hf[prefix + tempnames[j]][i]
                        alignedmaps[ccounter] = align_image_fast1(
                            cm, matrix, reswidth, resheight
                        )

                res["ref_" + tempnames[j]][i] = refim
                for ccounter in range(len(rotangles)):
                    prefix = "check" + str(ccounter) + "_"
                    res[prefix + tempnames[j]][i] = alignedmaps[ccounter].astype(
                        np.uint8
                    )
    return refim


#%% (enhance_and_align)


def enhance_and_align(
    newh5,
    oldh5,
    kernel,
    rotangles,
    tempnames,
    params_enhance,
    params_align,
    roi=None,
    no_output=False,
    timed=False,
):

    rotangles, tempnames = assure_multiple(rotangles, tempnames)
    (
        size,
        overlap,
        anms_threshold,
        ksize_anms,
        ksize_erodil,
        line,
        db_dist,
        kernel,
    ) = params_enhance
    affxs, driftindices, acl, aclc, cl, clc = params_align

    if roi is None:
        testindex = 0
    else:
        testindex = roi[0]

    with h5py.File(newh5, "w") as res, h5py.File(oldh5, "r") as hf:

        imshape = hf[tempnames[0] + "/imgs"][testindex].shape
        dummy = np.ones(imshape)
        masks = []
        for i in range(len(rotangles)):
            drot, log = rotate_bound(dummy, rotangles[i], bm=0)
            newmask = make_mask(drot, 2)
            masks.append(newmask)

        for j in range(len(tempnames)):

            print(j)
            print("________________________________")
            if timed:
                time_now = time.time()

            cdi = 0

            times = hf[tempnames[j] + "/times"][()]

            if roi is None:
                iroi = np.arange(len(times))
            else:
                iroi = roi

            time0 = times[0]

            # for ccounter in range(len(rotangles)):
            #    name="/check"+str(ccounter)+"_"
            #    res.create_dataset(tempnames[j]+name, shape=[len(iroi),imshape[0],imshape[1]],dtype=np.uint8,
            #        chunks=(1,imshape[0],imshape[1]),compression='gzip',compression_opts=2)
            # res.create_dataset(tempnames[j]+"/times",shape=(len(iroi)),dtype='f')

            for i in iroi:
                print(i)
                if timed:
                    print(time.time() - time_now)
                img = hf[tempnames[j] + "/imgs"][i]

                if no_output:
                    checkmaps = np.ones([len(rotangles), imshape[0], imshape[1]])
                else:

                    lapl = morphLaplace(img, kernel)

                    summed = np.zeros(lapl.shape, dtype=np.double)
                    summed += 255 - lapl
                    summed += img
                    copt = to_uint8(summed)

                    images, lowhigh = take_map(copt, size, overlap)
                    checkmaps = process_partial(
                        images,
                        rotangles,
                        lowhigh,
                        masks,
                        line=line,
                        anms_threshold=anms_threshold,
                        ksize_anms=ksize_anms,
                        ksize_erodil=ksize_erodil,
                        db_dist=db_dist,
                    )

                    if i == driftindices[j][cdi]:
                        (
                            im1s,
                            im2,
                            matrices,
                            reswidth,
                            resheight,
                            width_shift,
                            height_shift,
                        ) = align_images(
                            [img, cl], clc, [affxs[j][cdi], acl], aclc, verbose=True
                        )

                        matrix = matrices[0]
                        cdi += 1

                    alignedmaps = np.ones(
                        [len(rotangles), imshape[0], imshape[1]], dtype=np.uint8
                    )

                    refim = align_image_fast1(img, matrix, reswidth, resheight)
                    for k in range(len(checkmaps)):
                        alignedmaps[ccounter] = align_image_fast1(
                            checkmaps[k], matrix, reswidth, resheight
                        )

                res[tempnames[j] + "/times"][i] = times[i] - time0
                for ccounter in range(len(rotangles)):
                    name = "/check" + str(ccounter) + "_"
                    cm = checkmaps[ccounter] / np.max(checkmaps[ccounter]) * 255
                    res[tempnames[j] + name][i] = cm.astype(np.uint8)
    return checkmaps


#%%
def thresholding(newh5, oldh5, rotangles, tnames, inames, params, roi=None):
    houghdist, Hthreshold, Hminlength, Hmaxgap, deg_tol, im2 = params
    with h5py.File(newh5, "w") as res, h5py.File(oldh5, "r") as hf:

        for j in range(len(tnames)):

            times = hf[tnames[j]][()]

            name = "check0_" + inames[j]
            img = hf[name][0]
            imshape = img.shape

            res.create_dataset(
                "srb_" + inames[j],
                shape=[len(times), imshape[0], imshape[1]],
                dtype=np.uint8,
                chunks=(1, imshape[0], imshape[1]),
                compression="gzip",
                compression_opts=2,
            )

            for ccounter in range(len(rotangles)):
                prefix = "check" + str(ccounter) + "_"
                res.create_dataset(
                    prefix + inames[j],
                    shape=[len(times), imshape[0], imshape[1]],
                    dtype=np.uint8,
                    chunks=(1, imshape[0], imshape[1]),
                    compression="gzip",
                    compression_opts=2,
                )
            # res.create_dataset(tnames[j],shape=(len(times)),dtype='f')

            res[tnames[j]] = times

            if roi is None:
                iroi = np.arange(len(times))
            else:
                iroi = roi

            for k in iroi:
                print(k)
                # if k==1:
                #    break

                srbs = []
                for i in range(len(rotangles)):
                    name = "check" + str(i) + "_" + inames[j]

                    workimg = np.zeros(imshape, dtype=np.double)
                    workimg += hf[name][k]
                    workimg *= im2

                    (thresh, srb) = cv2.threshold(
                        to_uint8(workimg), 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
                    )
                    srb = (workimg > thresh).astype(np.uint8)

                    srbs.append(srb)
                    lines = cv2.HoughLinesP(
                        srb,  # Input edge image
                        houghdist,  # 0.5 Distance resolution in pixels
                        np.pi / 1800,  # Angle resolution in radians
                        threshold=Hthreshold,  # Min number of votes for valid line
                        minLineLength=Hminlength,  # Min allowed length of line
                        maxLineGap=Hmaxgap,  # Max allowed gap between line for joining them
                    )

                    newlines = check_angle_s(lines, rotangles[i], deg_tol)
                    # newlines=lines
                    newimg = np.zeros(srb.shape)
                    for points in newlines:
                        x1, y1, x2, y2 = points  # [0]
                        cv2.line(newimg, (x1, y1), (x2, y2), 255, 1)

                    prefix = "check" + str(i) + "_"
                    res[prefix + inames[j]][k] = newimg

                res["srb_" + inames[j]][k] = (np.sum(srbs, axis=0) > 0).astype(np.uint8)

    return np.sum(srbs, axis=0) > 0, newlines


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


#%% processing
def processing(
    image, angles, rois, filterheight, asympix=20, thresh_ratio=1.5, smoothing=11
):

    res = []

    for i in range(len(angles)):
        dark, nmask, log = enhance_lines(image, angles[i], line="dark")
        bright, nmask1, log = enhance_lines(image, angles[i], line="bright")
        smoothkernel = smoothbox_kernel([1, smoothing])[0]
        dres = cv2.sepFilter2D(dark, cv2.CV_64F, smoothkernel, np.ones(1))
        bres = cv2.sepFilter2D(bright, cv2.CV_64F, smoothkernel, np.ones(1))

        dres = anms(
            dres, nmask, thresh_ratio=thresh_ratio, ksize=filterheight, asympix=asympix
        )
        bres = anms(
            bres, nmask, thresh_ratio=thresh_ratio, ksize=filterheight, asympix=asympix
        )

        drows = np.sum(dres, axis=1)
        brows = np.sum(bres, axis=1)

        dpos = peak_com(drows, filterheight, rois[i])
        bpos = peak_com(brows, filterheight, rois[i])

        delta = int(np.round((dpos - bpos) / 2) * 2)
        nres = np.zeros(dres.shape)
        if delta < 0:
            delta = -delta
            nres[: -delta // 2] += bres[delta // 2 :]
            nres[delta // 2 :] += dres[: -delta // 2]
            nres[delta // 2 : -delta // 2] /= 2.0

        elif delta == 0:
            nres += bres
            nres += dres
            nres /= 2.0

        else:
            nres[: -delta // 2] += dres[delta // 2 :]
            nres[delta // 2 :] += bres[: -delta // 2]
            nres[delta // 2 : -delta // 2] /= 2.0

        res1 = rotate_back(nres, log)
        # rotate_bound(nres,-angles[i],flag2='back')
        offset = np.zeros(2, dtype=int)
        offset = (np.array(res1.shape) - np.array(image.shape)) / 2
        offset = offset.astype(int)
        gres = res1[offset[0] : -offset[0], offset[1] : -offset[1]]

        res.append(gres)
    return res[0] + res[1]


#%% obtain snr  for optimal rotation / deprecated
# deprecated
def obtain_snr(image, mask, line, show, minlength):

    rmeans = []
    rstds = []

    for j in range(image.shape[0]):
        roi = image[j][mask[j]]

        if len(roi) < minlength:
            pass
        else:
            rmeans.append(np.mean(roi))
            rstds.append(np.mean(np.sqrt((roi - rmeans[-1]) ** 2)) / np.sqrt(len(roi)))

    rmeans = np.array(rmeans)
    x = np.arange(len(rmeans))
    p = np.polyfit(x, rmeans, 5)
    rmeans -= np.polyval(p, x)
    rstds = np.array(rstds)

    val = rmeans / rstds
    if line == "dark":
        val = np.mean(val) - val
    elif line == "bright":
        val -= np.mean(val)

    vstd = np.std(val)

    if show:
        val2 = val - np.min(val)
        plt.plot(val2 * 1 / np.max(val2), c="r")
        plt.show()

    return np.max(val) / vstd


#%% optimal rotation /deprecated
# deprecated
def optimal_rotation(image_roi, angle, thresh=5, minlength=50, line="dark", show=False):
    # image_roi should contain lines with the respective angle, optimally not ending inside the ROI
    # angle in degree

    # optimal angle is determined by the two characteristics of a misfit line:
    # lower brightness
    # lower brightness variance

    pmangle = np.arange(-3, 4)

    dummy = np.ones(np.shape(image_roi))

    for k in range(5):
        snr = np.zeros(7)
        for i in range(7):
            rot, log = rotate_bound(image_roi, angle + pmangle[i])
            drot, log = rotate_bound(dummy, angle + pmangle[i], bm=0)
            mask = drot > 0.1
            # rot -= pr/5

            snr[i] = obtain_snr(rot, mask, line, False, minlength)

        # plt.plot(snr)
        # plt.show()

        if np.max(snr) < thresh:
            print(np.max(snr))
            print("signal to noise ratio below threshold")
            return False

        angle += pmangle[np.argmax(snr)]
        pmangle = pmangle / 3

    res = np.round(angle, 2)

    if show:
        rot, log = rotate_bound(image_roi, res)
        drot, log = rotate_bound(dummy, res, bm=0)
        mask = drot > 0.1
        # obtain_snr(rot, mask, line,True,minlength)
        plt.imshow(rot * mask)
        plt.show()

    return res


#%% enhance_lines_partial
def enhance_lines_partial(
    trot, newmask, angle, ksize=None, dist=1, iterations=2, line="dark"
):

    if line == "dark":
        trot -= np.min(trot)
        trot = np.max(trot) - trot
    elif line == "bright":
        pass

    tres = trot / np.max(trot) * 255

    res = np.copy(tres)

    for i in range(iterations):
        if ksize is None:
            srot = cv2.Sobel(tres, cv2.CV_64F, 0, 1)
        else:
            srot = cv2.Sobel(tres, cv2.CV_64F, 0, 1, ksize=ksize)

        msrot = np.ma.array(srot, mask=np.invert(newmask))

        middle = np.mean(msrot)

        t1 = srot > middle
        t2 = srot <= middle

        tres = np.zeros(srot.shape)

        tres[:-dist, :] -= t2[dist:, :] * (srot[dist:, :] - middle)
        tres[dist:, :] += t1[:-dist, :] * (srot[:-dist, :] - middle)

        res *= tres

    return res ** (1 / (iterations + 1))  # *newmask#,trot/np.max(trot)*255


#%% Enhance Lines
def enhance_lines(
    image, angle, number_of_bins=61, ksize=None, dist=1, iterations=2, line="dark"
):

    dummy = np.ones(image.shape)
    rot, log = rotate_bound(image, angle)
    drot, log = rotate_bound(dummy, angle, bm=0)
    newmask = make_mask(drot, 2)

    trot = np.clip(rot, np.min(image), np.max(image))
    if line == "dark":
        trot -= np.min(trot)
        trot = np.max(trot) - trot
    elif line == "bright":
        pass

    tres = trot / np.max(trot) * 255

    res = np.copy(tres)

    for i in range(iterations):
        if ksize is None:
            srot = cv2.Sobel(tres, cv2.CV_64F, 0, 1)
        else:
            srot = cv2.Sobel(tres, cv2.CV_64F, 0, 1, ksize=ksize)

        msrot = np.ma.array(srot, mask=np.invert(newmask))

        middle = np.mean(msrot)

        t1 = srot > middle
        t2 = srot <= middle

        tres = np.zeros(srot.shape)

        tres[:-dist, :] -= t2[dist:, :] * (srot[dist:, :] - middle)
        tres[dist:, :] += t1[:-dist, :] * (srot[:-dist, :] - middle)

        res *= tres

    return (
        res ** (1 / (iterations + 1)) * newmask,
        newmask,
        log,
    )  # ,trot/np.max(trot)*255


#%% Enhance Lines2
def enhance_lines2(
    image, angle, number_of_bins=61, ksize=None, dist=1, iterations=2, line="dark"
):

    if len(np.shape(dist)) == 0:
        dist = [dist]

    dummy = np.ones(image.shape)
    rot, log = rotate_bound(image, angle)
    drot, log = rotate_bound(dummy, angle, bm=0)
    newmask = make_mask(drot, 2)

    trot = np.clip(rot, np.min(image), np.max(image))
    if line == "dark":
        trot -= np.min(trot)
        trot = np.max(trot) - trot
    elif line == "bright":
        pass

    tres = trot / np.max(trot) * 255

    res = np.copy(tres)

    for i in range(iterations):
        if ksize is None:
            srot = cv2.Sobel(tres, cv2.CV_64F, 0, 1)
        else:
            srot = cv2.Sobel(tres, cv2.CV_64F, 0, 1, ksize=ksize)

        msrot = np.ma.array(srot, mask=np.invert(newmask))

        middle = np.mean(msrot)

        t1 = srot > middle
        t2 = srot <= middle

        tres2 = np.zeros([len(dist), srot.shape[0], srot.shape[1]])
        for j in range(len(dist)):
            # tres=np.zeros(srot.shape)
            tres2[j, : -dist[j], :] -= t2[dist[j] :, :] * (srot[dist[j] :, :] - middle)
            tres2[j, dist[j] :, :] += t1[: -dist[j], :] * (srot[: -dist[j], :] - middle)
        tres = np.sum(tres2, axis=0)
        res *= tres

    return (
        res ** (1 / (iterations + 1)) * newmask,
        newmask,
        log,
        trot / np.max(trot) * 255,
    )


#%% enhance_lines_prototype
def enhance_lines_prototype(
    image, angle, number_of_bins=61, ksize=None, dist=1, iterations=2, line="dark"
):

    dummy = np.ones(image.shape)
    rot, log = rotate_bound(image, angle)
    drot, log = rotate_bound(dummy, angle, bm=0)
    newmask = make_mask(drot, 2)

    trot = np.clip(rot, np.min(image), np.max(image))
    if line == "dark":
        trot -= np.min(trot)
        trot = np.max(trot) - trot
    elif line == "bright":
        pass

    tres = trot / np.max(trot) * 255

    # res=np.copy(tres)

    for i in range(iterations):
        if ksize is None:
            srot = cv2.Sobel(tres, cv2.CV_64F, 0, 1)
        else:
            srot = cv2.Sobel(tres, cv2.CV_64F, 0, 1, ksize=ksize)

        msrot = np.ma.array(srot, mask=np.invert(newmask))

        middle = np.mean(msrot)

        t1 = srot > middle
        t2 = srot <= middle

        tres = np.zeros(srot.shape)
        tres[:-dist, :] -= t2[dist:, :] * (srot[dist:, :] - middle)
        tres[dist:, :] += t1[:-dist, :] * (srot[:-dist, :] - middle)
        # res *= tres

    return tres * newmask, newmask, log


#%% obtain_maps
def obtain_maps(qkeys, qline_images, qsum_images, qcheck_images, lowhigh):
    xmax = lowhigh[qkeys[-1]][0][0] + qsum_images[qkeys[-1]].shape[0]
    ymax = lowhigh[qkeys[-1]][1][0] + qsum_images[qkeys[-1]].shape[1]

    nangles = len(qcheck_images[qkeys[0]])

    fullmap = np.zeros([xmax, ymax])
    singlemaps = np.zeros([nangles, xmax, ymax])
    checkmaps = np.zeros([nangles, xmax, ymax])
    # print(xmax,ymax)
    for i in qkeys:
        xmin = lowhigh[i][0][0]
        ymin = lowhigh[i][1][0]
        ishape = qsum_images[i].shape
        fullmap[xmin : xmin + ishape[0], ymin : ymin + ishape[1]] += qsum_images[i]

        for j in range(nangles):
            checkmaps[
                j, xmin : xmin + ishape[0], ymin : ymin + ishape[1]
            ] += qcheck_images[i][j]
            singlemaps[j, xmin : xmin + ishape[0], ymin : ymin + ishape[1]] += (
                qline_images[i][j] > 10
            ) * 1.0

    # singlem=copy.deepcopy(singlemaps)

    singlereturn = (singlemaps > 0) * 1

    singlemaps[singlemaps == 0] = 1.0
    # for j in range(2,6):
    #    checkmaps[singlemaps==j]=checkmaps[singlemaps==j]/j
    for i in range(len(checkmaps)):
        checkmaps[i] -= np.min(checkmaps[i])

    checkmaps /= singlemaps

    return singlereturn, (fullmap > 0) * 1.0, checkmaps  # ,singlem


#%% check_angle


def check_angle(lines, deg_tol=1.5):
    delta_ratio = np.tan(deg_tol / 180 * np.pi)
    newlines = []
    for points in lines:
        x1, y1, x2, y2 = points[0]
        denom = max(np.abs(x2 - x1), 1)
        if np.abs(y2 - y1) / denom < delta_ratio:
            newlines.append([x1, y1, x2, y2])
    return newlines


@njit
def check_angle_s(lines, rotangle, deg_tol=1.5):
    # assumes positive angles
    delta_ratios = np.zeros(2)
    dangle0 = rotangle - deg_tol
    dangle1 = rotangle + deg_tol
    delta_ratios[0] = np.tan(dangle0 / 180 * np.pi)
    delta_ratios[1] = np.tan(dangle1 / 180 * np.pi)

    newlines = []
    if dangle0 < 90 and dangle1 > 90:
        for points in lines:
            x1, y1, x2, y2 = points[0]
            denom = max(np.abs(x2 - x1), 1)
            if (
                abs(y2 - y1) / denom < delta_ratios[1]
                or abs(y2 - y1) / denom > delta_ratios[0]
            ):
                newlines.append([x1, y1, x2, y2])

    elif dangle0 < 0 and dangle1 > 0:
        for points in lines:
            x1, y1, x2, y2 = points[0]
            denom = max(np.abs(x2 - x1), 1)
            if (
                abs(y2 - y1) / denom < delta_ratios[1]
                and abs(y2 - y1) / denom > delta_ratios[0]
            ):
                newlines.append([x1, y1, x2, y2])
    else:
        delta_ratios = np.sort(np.abs(delta_ratios))

        for points in lines:
            x1, y1, x2, y2 = points[0]
            denom = max(np.abs(x2 - x1), 1)
            if (
                abs(y2 - y1) / denom < delta_ratios[1]
                and abs(y2 - y1) / denom > delta_ratios[0]
            ):
                newlines.append([x1, y1, x2, y2])

    return newlines


#%% noise_line_suppression


def noise_line_suppression(image, ksize_erodil):
    erod_img = cv2.erode(image, np.ones([1, ksize_erodil]))
    return cv2.dilate(erod_img, np.ones([1, ksize_erodil]))


#%% process

# smoothsize=35
def process(
    images,
    rotangles,
    ksize_erodil=15,
    ksize_anms=15,
    damp=10,
    smoothsize=1,
    Hthreshold=50,
    Hminlength=5,
    Hmaxgap=50,
    line="dark",
    ksize=None,
    iterations=2,
    anms_threshold=2,
    dist=1,
    houghdist=1,
    powerlaw=1,
):
    qkeys = []
    qkeys = list(images.keys())
    qsum_images = {}
    qline_images = {}
    qcheck_images = {}

    for m in range(len(qkeys)):
        line_images = []
        check_images = []
        image = images[qkeys[m]]
        sum_image = np.zeros(image.shape)

        print(str(m + 1) + " / " + str(len(qkeys)))
        print(qkeys[m])
        for k in range(len(rotangles)):
            tres, newmask, log = enhance_lines(
                image,
                rotangles[k],
                iterations=iterations,
                ksize=ksize,
                line=line,
                dist=dist,
            )

            nms = anms(
                tres,
                newmask,
                thresh_ratio=anms_threshold,
                ksize=ksize_anms,
                damping=damp,
            )
            clean = noise_line_suppression(nms, ksize_erodil)
            clean = clean / np.max(clean) * 255
            # clean[clean<1]=1
            clean = clean.astype(np.double)
            clean = clean**powerlaw
            clean = to_uint8(clean)
            # clean -= np.min(clean)
            # clean= clean/np.max(clean) *255
            # clean=clean.astype(np.uint8)

            new = clean[newmask > 0]

            (thresh, srb) = cv2.threshold(
                new.astype(np.uint8), 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
            )
            thresh = max(thresh, 255 / damp)
            # print(thresh)
            srb = (clean > thresh).astype(np.uint8)

            # lines_list =[]
            lines = cv2.HoughLinesP(
                srb,  # Input edge image
                houghdist,  # 0.5 Distance resolution in pixels
                np.pi / 1800,  # Angle resolution in radians
                threshold=Hthreshold,  # Min number of votes for valid line
                minLineLength=Hminlength,  # Min allowed length of line
                maxLineGap=Hmaxgap,  # Max allowed gap between line for joining them
            )

            if lines is None:
                newlines = []
            else:
                newlines = check_angle(lines)

            newimg = np.zeros(srb.shape)
            for points in newlines:
                x1, y1, x2, y2 = points
                cv2.line(newimg, (x1, y1), (x2, y2), 255, 1)

            check_images.append(rotate_back(clean, log))

            lines1 = rotate_back(newimg, log)
            line_images.append(lines1)
            sum_image += lines1

        qcheck_images[qkeys[m]] = check_images
        qsum_images[qkeys[m]] = sum_image > 10
        qline_images[qkeys[m]] = line_images

    return qkeys, qline_images, qcheck_images, qsum_images


#%% process_partial
def process_partial(
    images,
    rotangles,
    lowhigh,
    masks,
    ksize_erodil=15,
    ksize_anms=15,
    damp=10,
    line="dark",
    ksize=None,
    iterations=2,
    db_dist=None,
    anms_threshold=2,
    dist=1,
    powerlaw=1,
):
    qkeys = []
    qkeys = list(images.keys())

    qcheck_images = {}

    for m in range(len(qkeys)):

        check_images = []
        image = images[qkeys[m]]

        # print(str(m+1)+" / " +str(len(qkeys)))
        # print(qkeys[m])
        for k in range(len(rotangles)):

            rot, log = rotate_bound(image, rotangles[k])

            if db_dist is None:
                tres = enhance_lines_partial(
                    rot,
                    masks[k],
                    rotangles[k],
                    iterations=iterations,
                    ksize=ksize,
                    line=line,
                    dist=dist,
                )
                nms = anms(
                    tres,
                    masks[k],
                    thresh_ratio=anms_threshold,
                    ksize=ksize_anms,
                    damping=damp,
                )
                clean = noise_line_suppression(nms, ksize_erodil)

            elif db_dist == 0:
                tres = enhance_lines_partial(
                    rot,
                    masks[k],
                    rotangles[k],
                    iterations=iterations,
                    ksize=ksize,
                    line="bright",
                    dist=dist,
                )
                tres2 = enhance_lines_partial(
                    rot,
                    masks[k],
                    rotangles[k],
                    iterations=iterations,
                    ksize=ksize,
                    line="dark",
                    dist=dist,
                )
                nms = anms(
                    tres,
                    masks[k],
                    thresh_ratio=anms_threshold,
                    ksize=ksize_anms,
                    damping=damp,
                )
                nms2 = anms(
                    tres2,
                    masks[k],
                    thresh_ratio=anms_threshold,
                    ksize=ksize_anms,
                    damping=damp,
                )
                clean = noise_line_suppression(nms, ksize_erodil)
                clean2 = noise_line_suppression(nms2, ksize_erodil)
                clean += clean2
            else:

                tres = enhance_lines_partial(
                    rot,
                    masks[k],
                    rotangles[k],
                    iterations=iterations,
                    ksize=ksize,
                    line="bright",
                    dist=dist,
                )
                tres2 = enhance_lines_partial(
                    rot,
                    masks[k],
                    rotangles[k],
                    iterations=iterations,
                    ksize=ksize,
                    line="dark",
                    dist=dist,
                )
                nms = anms(
                    tres,
                    masks[k],
                    thresh_ratio=anms_threshold,
                    ksize=ksize_anms,
                    damping=damp,
                )
                nms2 = anms(
                    tres2,
                    masks[k],
                    thresh_ratio=anms_threshold,
                    ksize=ksize_anms,
                    damping=damp,
                )
                clean = noise_line_suppression(nms, ksize_erodil)
                clean2 = noise_line_suppression(nms2, ksize_erodil)

                if db_dist < 0:
                    db_dist_abs = -db_dist
                    clean[db_dist_abs:] += clean2[:-db_dist_abs]
                else:
                    clean[:-db_dist] += clean2[db_dist:]

            # clean=clean**powerlaw
            clean = to_uint8(clean)
            # clean -= np.min(clean)
            # clean= clean/np.max(clean) *255
            # clean=clean.astype(np.uint8)

            check_images.append(rotate_back(clean, log))

        qcheck_images[qkeys[m]] = check_images

    xmax = lowhigh[qkeys[-1]][0][0] + qcheck_images[qkeys[-1]][0].shape[0]
    ymax = lowhigh[qkeys[-1]][1][0] + qcheck_images[qkeys[-1]][0].shape[1]

    nangles = len(qcheck_images[qkeys[0]])

    checkmaps = np.zeros([nangles, xmax, ymax])
    # print(xmax,ymax)
    for i in qkeys:
        xmin = lowhigh[i][0][0]
        ymin = lowhigh[i][1][0]
        ishape = qcheck_images[i][0].shape

        for j in range(nangles):
            checkmaps[
                j, xmin : xmin + ishape[0], ymin : ymin + ishape[1]
            ] += qcheck_images[i][j]

    return checkmaps


#%% process2

# smoothsize=35
def process2(
    images,
    rotangles,
    ksize_erodil=15,
    ksize_anms=15,
    damp=10,
    smoothsize=1,
    Hthreshold=50,
    Hminlength=5,
    Hmaxgap=50,
    line="dark",
    ksize=None,
    iterations=2,
    anms_threshold=2,
    dist=1,
    houghdist=1,
):
    qkeys = []
    qkeys = list(images.keys())
    qsum_images = {}
    qline_images = {}
    qcheck_images = {}

    for m in range(len(qkeys)):
        line_images = []
        check_images = []
        image = images[qkeys[m]]
        sum_image = np.zeros(image.shape)

        print(str(m + 1) + " / " + str(len(qkeys)))
        print(qkeys[m])
        for k in range(len(rotangles)):
            tres, newmask, log, rotimg = enhance_lines2(
                image,
                rotangles[k],
                iterations=iterations,
                ksize=ksize,
                line=line,
                dist=dist,
            )

            nms = anms(
                tres,
                newmask,
                thresh_ratio=anms_threshold,
                ksize=ksize_anms,
                damping=damp,
            )  # ksize=19)

            clean = noise_line_suppression(nms, ksize_erodil)

            clean = clean / np.max(clean) * 255

            nclean = np.zeros(clean.shape, dtype=np.double)
            nclean += clean
            nclean += rotimg

            nclean -= np.min(nclean)
            nclean = (nclean / np.max(nclean) * 255).astype(np.uint8)

            new = nclean[newmask > 0]

            check_images.append(rotate_back(nclean, log))
            line_images.append(np.zeros(check_images[-1].shape))

        qcheck_images[qkeys[m]] = check_images
        qsum_images[qkeys[m]] = sum_image > 10
        qline_images[qkeys[m]] = line_images

    return qkeys, qline_images, qcheck_images, qsum_images
