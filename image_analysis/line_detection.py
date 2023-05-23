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
def line_enhance_horizontal(
    trot, mask, ksize=None, dist=1, iterations=2, line="dark"
):

    if ksize is None:
        ksize=3
    
    if line == "dark":
        trot -= np.min(trot)
        trot = np.max(trot) - trot
    elif line == "bright":
        pass

    tres = trot / np.max(trot) * 255

    res = np.copy(tres)

    for i in range(iterations):
        
        srot = cv2.Sobel(tres, cv2.CV_64F, 0, 1, ksize=ksize)

        msrot = np.ma.array(srot, mask=np.invert(mask))

        middle = np.mean(msrot)

        t1 = srot > middle
        t2 = srot <= middle

        tres = np.zeros(srot.shape)

        tres[:-dist, :] -= t2[dist:, :] * (srot[dist:, :] - middle)
        tres[dist:, :] += t1[:-dist, :] * (srot[:-dist, :] - middle)

        res *= tres
    return res ** (1 / (iterations + 1)),trot / np.max(trot) * 255  



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
    image,
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

    line_images = np.zeros([len(rotangles),*image.shape])
    check_images = np.zeros([len(rotangles),*image.shape])

    for k in range(len(rotangles)):
        
        rot, log = img_rotate_bound(image, rotangles[k])
        dummy = np.ones(image.shape)
        drot, log = img_rotate_bound(dummy, rotangles[k], bm=0)
        mask = make_mask(drot, 2)
        
        
        tres, rotimg = line_enhance_horizontal(
            rot,
            mask,
            iterations=iterations,
            ksize=ksize,
            line=line,
            dist=dist,
        )

        nms = img_anms(
            tres,
            mask,
            thresh_ratio=anms_threshold,
            ksize=ksize_anms,
            damping=damp,
        )
        clean = img_noise_line_suppression(nms, ksize_erodil)
        clean = img_to_uint8(clean)

        new = clean[mask > 0]

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

        check_images[k]=img_rotate_back(clean, log)

        line_images[k] = img_rotate_back(newimg, log)

    return check_images,(line_images > 0) * 1
#%% process_partial
def line_process_partial(
    image,
    rotangles,
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
):

    check_images = np.zeros([len(rotangles),*image.shape])
    

    for k in range(len(rotangles)):

        rot, log = img_rotate_bound(image, rotangles[k])
        #rot=np.array(rot).astype(np.double)
        if db_dist is None:
            tres,rotimg = line_enhance_horizontal(
                rot,
                masks[k],
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

        else:

            tres,rotimg = line_enhance_horizontal(
                rot,
                masks[k],
                iterations=iterations,
                ksize=ksize,
                line="bright",
                dist=dist,
            )
            tres2,rotimg = line_enhance_horizontal(
                rot,
                masks[k],
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
                
        clean = img_to_uint8(clean)

        check_images[k]=img_rotate_back(clean, log)

    return check_images


#%% process2

def line_process2(
    image,
    rotangles,
    masks,
    ksize_erodil=15,
    ksize_anms=15,#19
    damp=10,
    line="dark",
    ksize=None,
    iterations=2,
    anms_threshold=2,
    dist=1,
):

    check_images = np.zeros([len(rotangles),*image.shape])

    for k in range(len(rotangles)):
        rot, log = img_rotate_bound(image, rotangles[k])
        tres, rotimg = line_enhance_horizontal(
            rot,
            masks[k],
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
        clean = clean / np.max(clean) * 255

        nclean = np.zeros(clean.shape, dtype=np.double)
        nclean += clean
        nclean += rotimg

        nclean = img_to_uint8(nclean)
        check_images[k]=img_rotate_back(nclean, log)

    return check_images