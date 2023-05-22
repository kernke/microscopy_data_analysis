# -*- coding: utf-8 -*-
"""
@author: kernke
"""

import numpy as np
import cv2
from .general_util import make_mask
from .image_processing import img_rotate_bound,img_rotate_back,img_anms,img_noise_line_suppression,img_to_uint8
from numba import njit



#%% enhance_lines_partial
def line_enhance_partial(
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

    return res ** (1 / (iterations + 1))  


#%% Enhance Lines
def line_enhance(
    image, angle, number_of_bins=61, ksize=None, dist=1, iterations=2, line="dark"
):

    dummy = np.ones(image.shape)
    rot, log = img_rotate_bound(image, angle)
    drot, log = img_rotate_bound(dummy, angle, bm=0)
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
    )  


#%% Enhance Lines2
def line_enhance2(
    image, angle, number_of_bins=61, ksize=None, dist=1, iterations=2, line="dark"
):

    if len(np.shape(dist)) == 0:
        dist = [dist]

    dummy = np.ones(image.shape)
    rot, log = img_rotate_bound(image, angle)
    drot, log = img_rotate_bound(dummy, angle, bm=0)
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


#%% obtain_maps
def obtain_maps(qkeys, qline_images, qsum_images, qcheck_images, lowhigh):
    xmax = lowhigh[qkeys[-1]][0][0] + qsum_images[qkeys[-1]].shape[0]
    ymax = lowhigh[qkeys[-1]][1][0] + qsum_images[qkeys[-1]].shape[1]

    nangles = len(qcheck_images[qkeys[0]])

    fullmap = np.zeros([xmax, ymax])
    singlemaps = np.zeros([nangles, xmax, ymax])
    checkmaps = np.zeros([nangles, xmax, ymax])
   
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


    singlereturn = (singlemaps > 0) * 1

    singlemaps[singlemaps == 0] = 1.0

    for i in range(len(checkmaps)):
        checkmaps[i] -= np.min(checkmaps[i])

    checkmaps /= singlemaps

    return singlereturn, (fullmap > 0) * 1.0, checkmaps  # ,singlem


#%% check_angle (little deprecated)
def line_check_angle(lines, deg_tol=1.5):
    delta_ratio = np.tan(deg_tol / 180 * np.pi)
    newlines = []
    for points in lines:
        x1, y1, x2, y2 = points[0]
        denom = max(np.abs(x2 - x1), 1)
        if np.abs(y2 - y1) / denom < delta_ratio:
            newlines.append([x1, y1, x2, y2])
    return newlines

#%% check_angle

@njit
def line_check_angle_s(lines, rotangle, deg_tol=1.5):
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



#%% process

def line_process(
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
    houghdist=1,#powerlaw=1,
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
            tres, newmask, log = line_enhance(
                image,
                rotangles[k],
                iterations=iterations,
                ksize=ksize,
                line=line,
                dist=dist,
            )

            nms = img_anms(
                tres,
                newmask,
                thresh_ratio=anms_threshold,
                ksize=ksize_anms,
                damping=damp,
            )
            clean = img_noise_line_suppression(nms, ksize_erodil)
            #clean = clean / np.max(clean) * 255
            #clean = clean.astype(np.double)
            #clean = clean**powerlaw
            clean = img_to_uint8(clean)

            new = clean[newmask > 0]

            (thresh, srb) = cv2.threshold(
                new.astype(np.uint8), 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
            )
            thresh = max(thresh, 255 / damp)

            srb = (clean > thresh).astype(np.uint8)

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
                newlines = line_check_angle(lines)

            newimg = np.zeros(srb.shape)
            for points in newlines:
                x1, y1, x2, y2 = points
                cv2.line(newimg, (x1, y1), (x2, y2), 255, 1)

            check_images.append(img_rotate_back(clean, log))

            lines1 = img_rotate_back(newimg, log)
            line_images.append(lines1)
            sum_image += lines1

        qcheck_images[qkeys[m]] = check_images
        qsum_images[qkeys[m]] = sum_image > 10
        qline_images[qkeys[m]] = line_images

    return qkeys, qline_images, qcheck_images, qsum_images


#%% process_partial
def line_process_partial(
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

        for k in range(len(rotangles)):

            rot, log = img_rotate_bound(image, rotangles[k])

            if db_dist is None:
                tres = line_enhance_partial(
                    rot,
                    masks[k],
                    rotangles[k],
                    iterations=iterations,
                    ksize=ksize,
                    line=line,
                    dist=dist,
                )
                nms = img_anms(
                    tres,
                    masks[k],
                    thresh_ratio=anms_threshold,
                    ksize=ksize_anms,
                    damping=damp,
                )
                clean = img_noise_line_suppression(nms, ksize_erodil)

            elif db_dist == 0:
                tres = line_enhance_partial(
                    rot,
                    masks[k],
                    rotangles[k],
                    iterations=iterations,
                    ksize=ksize,
                    line="bright",
                    dist=dist,
                )
                tres2 = line_enhance_partial(
                    rot,
                    masks[k],
                    rotangles[k],
                    iterations=iterations,
                    ksize=ksize,
                    line="dark",
                    dist=dist,
                )
                nms = img_anms(
                    tres,
                    masks[k],
                    thresh_ratio=anms_threshold,
                    ksize=ksize_anms,
                    damping=damp,
                )
                nms2 = img_anms(
                    tres2,
                    masks[k],
                    thresh_ratio=anms_threshold,
                    ksize=ksize_anms,
                    damping=damp,
                )
                clean = img_noise_line_suppression(nms, ksize_erodil)
                clean2 = img_noise_line_suppression(nms2, ksize_erodil)
                clean += clean2
            else:

                tres = line_enhance_partial(
                    rot,
                    masks[k],
                    rotangles[k],
                    iterations=iterations,
                    ksize=ksize,
                    line="bright",
                    dist=dist,
                )
                tres2 = line_enhance_partial(
                    rot,
                    masks[k],
                    rotangles[k],
                    iterations=iterations,
                    ksize=ksize,
                    line="dark",
                    dist=dist,
                )
                nms = img_anms(
                    tres,
                    masks[k],
                    thresh_ratio=anms_threshold,
                    ksize=ksize_anms,
                    damping=damp,
                )
                nms2 = img_anms(
                    tres2,
                    masks[k],
                    thresh_ratio=anms_threshold,
                    ksize=ksize_anms,
                    damping=damp,
                )
                clean = img_noise_line_suppression(nms, ksize_erodil)
                clean2 = img_noise_line_suppression(nms2, ksize_erodil)

                if db_dist < 0:
                    db_dist_abs = -db_dist
                    clean[db_dist_abs:] += clean2[:-db_dist_abs]
                else:
                    clean[:-db_dist] += clean2[db_dist:]

            # clean=clean**powerlaw
            clean = img_to_uint8(clean)

            check_images.append(img_rotate_back(clean, log))

        qcheck_images[qkeys[m]] = check_images

    xmax = lowhigh[qkeys[-1]][0][0] + qcheck_images[qkeys[-1]][0].shape[0]
    ymax = lowhigh[qkeys[-1]][1][0] + qcheck_images[qkeys[-1]][0].shape[1]

    nangles = len(qcheck_images[qkeys[0]])

    checkmaps = np.zeros([nangles, xmax, ymax])

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

def line_process2(
    image,
    rotangles,
    ksize_erodil=15,
    ksize_anms=15,#19
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

    check_images = []

    for k in range(len(rotangles)):
        tres, newmask, log, rotimg = line_enhance2(
            image,
            rotangles[k],
            iterations=iterations,
            ksize=ksize,
            line=line,
            dist=dist,
        )

        nms = img_anms(
            tres,
            newmask,
            thresh_ratio=anms_threshold,
            ksize=ksize_anms,
            damping=damp,
        )  

        clean = img_noise_line_suppression(nms, ksize_erodil)
        clean = clean / np.max(clean) * 255

        nclean = np.zeros(clean.shape, dtype=np.double)
        nclean += clean
        nclean += rotimg

        nclean = img_to_uint8(nclean)


        check_images.append(img_rotate_back(nclean, log))


    return check_images
