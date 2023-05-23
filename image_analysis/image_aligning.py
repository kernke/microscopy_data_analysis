# -*- coding: utf-8 -*-
"""
@author: kernke
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from copy import deepcopy

# def phase_correlation(im1,im2):
#    return (ifftn(fftn(im1)*ifftn(im2))).real



#%% phase_correlation
def phase_correlation(a, b):
    G_a = np.fft.fft2(a)
    G_b = np.fft.fft2(b)
    conj_b = np.ma.conjugate(G_b)
    R = G_a * conj_b
    R /= np.absolute(R)
    r = np.fft.ifft2(R).real
    return r

#%% pos_from_pcm_short
def pos_from_pcm_short(roipcm):

    dist = np.argmax(roipcm)
    dist1 = dist % roipcm.shape[1]
    dist0 = dist // roipcm.shape[1]

    return np.array([dist0, dist1])


#%% stitch_auto / close ...
def stitch_auto(im1, im2, mode=0, zerogone=False, half=0):
    pcm = phase_correlation(im1, im2)
    if zerogone:
        pcm[0, 0] = np.min(pcm)
    pc = pos_from_pcm_short(pcm)
    pcs = np.array(np.shape(pcm))

    pc_copy = np.copy(pc)
    vert0 = 0
    hor0 = 0

    if mode == 0:
        if pc[0] > pcs[0] / 2.5:
            hs = int(pcs[0] / 2) - 1
            pcm2 = phase_correlation(im1[hs : 2 * hs, :], im2[:hs, :])
            if zerogone:
                pcm2[0, 0] = np.min(pcm2)
            pc2 = pos_from_pcm_short(pcm2)
            if pcm2[pc2[0], pc2[1]] < pcm[pc_copy[0], pc_copy[1]]:
                vert0 = pcs[0] - pc[0]
                pc[0] = 0

        if pc[1] > pcs[1] / 2.5:
            hs = int(pcs[1] / 2) - 1
            pcm2 = phase_correlation(im1[:, hs : 2 * hs], im2[:, :hs])
            if zerogone:
                pcm2[0, 0] = np.min(pcm2)
            pc2 = pos_from_pcm_short(pcm2)
            if pcm2[pc2[0], pc2[1]] < pcm[pc_copy[0], pc_copy[1]]:
                hor0 = pcs[1] - pc[1]
                pc[1] = 0

    if mode == 1:
        pass

    elif mode == 2:
        vert0 = pcs[0] - pc[0]
        pc[0] = 0

    elif mode == 3:
        vert0 = pcs[0] - pc[0]
        pc[0] = 0
        hor0 = pcs[1] - pc[1]
        pc[1] = 0

    elif mode == 4:
        hor0 = pcs[1] - pc[1]
        pc[1] = 0

    vh = np.array([vert0, hor0])
    sheet = np.zeros(pcs + pc + vh)
    sheetdiv = np.zeros(pcs + pc + vh)

    if half == 0:

        sheet[vert0 : vert0 + pcs[0], hor0 : hor0 + pcs[1]] += im1
        sheetdiv[vert0 : vert0 + pcs[0], hor0 : hor0 + pcs[1]] += np.ones(pcs)

        sheet[pc[0] : pc[0] + pcs[0], pc[1] : pc[1] + pcs[1]] += im2
        sheetdiv[pc[0] : pc[0] + pcs[0], pc[1] : pc[1] + pcs[1]] += np.ones(pcs)

    if half == 1:

        sheet[pc[0] : pc[0] + pcs[0], pc[1] : pc[1] + pcs[1]] += im2
        sheetdiv[pc[0] : pc[0] + pcs[0], pc[1] : pc[1] + pcs[1]] += np.ones(pcs)

    elif half == -1:

        sheet[vert0 : vert0 + pcs[0], hor0 : hor0 + pcs[1]] += im1
        sheetdiv[vert0 : vert0 + pcs[0], hor0 : hor0 + pcs[1]] += np.ones(pcs)

    sheetdiv[sheetdiv == 0] = -1

    abs_pos = np.zeros([1, 2, 2])
    abs_pos[0, 0, 0] = vert0
    abs_pos[0, 0, 1] = hor0
    abs_pos[0, 1, 0] = pc[0]
    abs_pos[0, 1, 1] = pc[1]
    return sheet / sheetdiv, abs_pos


def stitch(im1, im2):
    pcm = phase_correlation(im1, im2)
    pc = pos_from_pcm_short(pcm)
    pcs = np.array(np.shape(pcm))

    pc_copy = np.copy(pc)
    vert0 = 0
    hor0 = 0

    if pc[0] > pcs[0] / 2.5:
        hs = int(pcs[0] / 2) - 1
        pcm2 = phase_correlation(im1[hs : 2 * hs, :], im2[:hs, :])
        pc2 = pos_from_pcm_short(pcm2)
        if pcm2[pc2[0], pc2[1]] < pcm[pc_copy[0], pc_copy[1]]:
            vert0 = pcs[0] - pc[0]
            pc[0] = 0

    if pc[1] > pcs[1] / 2.5:
        hs = int(pcs[1] / 2) - 1
        pcm2 = phase_correlation(im1[:, hs : 2 * hs], im2[:, :hs])
        pc2 = pos_from_pcm_short(pcm2)
        if pcm2[pc2[0], pc2[1]] < pcm[pc_copy[0], pc_copy[1]]:
            hor0 = pcs[1] - pc[1]
            pc[1] = 0

    vh = np.array([vert0, hor0])
    sheet = np.zeros(pcs + pc + vh)
    sheetdiv = np.zeros(pcs + pc + vh)

    sheet[vert0 : vert0 + pcs[0], hor0 : hor0 + pcs[1]] += im1
    sheetdiv[vert0 : vert0 + pcs[0], hor0 : hor0 + pcs[1]] += np.ones(pcs)

    sheet[pc[0] : pc[0] + pcs[0], pc[1] : pc[1] + pcs[1]] += im2
    sheetdiv[pc[0] : pc[0] + pcs[0], pc[1] : pc[1] + pcs[1]] += np.ones(pcs)

    sheetdiv[sheetdiv == 0] = -1
    return sheet / sheetdiv


def stitch_close(im1, im2):
    pcm = phase_correlation(im1, im2)
    pc = pos_from_pcm_short(pcm)
    pcs = np.array(np.shape(pcm))

    vert0 = 0
    hor0 = 0

    if pc[0] > pcs[0] / 2.0:

        vert0 = pcs[0] - pc[0]
        pc[0] = 0

    if pc[1] > pcs[1] / 2.5:
        hor0 = pcs[1] - pc[1]
        pc[1] = 0

    vh = np.array([vert0, hor0])
    s = pcs + pc + vh
    s = s.astype(int)
    sheet = np.zeros([s[0], s[1], 2])

    sheet[vert0 : vert0 + pcs[0], hor0 : hor0 + pcs[1], 0] += im1
    # sheetdiv[vert0:vert0+pcs[0],hor0:hor0+pcs[1]]+=np.ones(pcs)

    sheet[pc[0] : pc[0] + pcs[0], pc[1] : pc[1] + pcs[1], 1] += im2

    return np.max(sheet, axis=-1)


def stitch_close2(im1, im2):
    pcm = phase_correlation(im1, im2)
    pc = pos_from_pcm_short(pcm)
    pcs = np.array(np.shape(pcm))

    vert0 = 0
    hor0 = 0

    if pc[0] > pcs[0] / 2.0:

        vert0 = pcs[0] - pc[0]
        pc[0] = 0

    if pc[1] > pcs[1] / 2.5:
        hor0 = pcs[1] - pc[1]
        pc[1] = 0

    vh = np.array([vert0, hor0])
    sheet = np.zeros(pcs + pc + vh)
    sheetdiv = np.zeros(pcs + pc + vh)

    sheet[vert0 : vert0 + pcs[0], hor0 : hor0 + pcs[1]] += im1
    sheetdiv[vert0 : vert0 + pcs[0], hor0 : hor0 + pcs[1]] += np.ones(pcs)

    sheet[pc[0] : pc[0] + pcs[0], pc[1] : pc[1] + pcs[1]] += im2
    sheetdiv[pc[0] : pc[0] + pcs[0], pc[1] : pc[1] + pcs[1]] += np.ones(pcs)

    sheetdiv[sheetdiv == 0] = -1
    return sheet / sheetdiv


#%% align
def align(im1, im2):
    pcm = phase_correlation(im1, im2)
    pc = pos_from_pcm_short(pcm)
    pcs = np.array(np.shape(pcm))

    pc_copy = np.copy(pc)

    if pc[0] > pcs[0] / 2.5:
        hs = int(pcs[0] / 2) - 1
        pcm2 = phase_correlation(im1[hs : 2 * hs, :], im2[:hs, :])
        pc2 = pos_from_pcm_short(pcm2)
        if pcm2[pc2[0], pc2[1]] < pcm[pc_copy[0], pc_copy[1]]:
            pc[0] = pc[0] - pcs[0]

    if pc[1] > pcs[1] / 2.5:
        hs = int(pcs[1] / 2) - 1
        pcm2 = phase_correlation(im1[:, hs : 2 * hs], im2[:, :hs])
        pc2 = pos_from_pcm_short(pcm2)
        if pcm2[pc2[0], pc2[1]] < pcm[pc_copy[0], pc_copy[1]]:
            pc[1] = pc[1] - pcs[1]

    return pc


#%% pos_from_pcm


def pos_from_pcm(pcm, overlap_limits, mode, tolerance, imdim, rdrift, cdrift):
    rwidth = overlap_limits[0, 1] - overlap_limits[0, 0]
    cwidth = overlap_limits[1, 1] - overlap_limits[1, 0]

    if mode == "vertical":
        rstart = imdim[0] - overlap_limits[0, 1] + rdrift
        rend = imdim[0] - overlap_limits[0, 0] + rdrift
        cstart = -cwidth // 2 + cdrift
        cend = cwidth // 2 + cdrift

    elif mode == "horizontal":
        cstart = imdim[0] - overlap_limits[1, 1] + cdrift
        cend = imdim[0] - overlap_limits[1, 0] + cdrift
        rstart = -rwidth // 2 + rdrift
        rend = rwidth // 2 + rdrift

    rows = np.arange(rstart, rend, dtype=int)
    cols = np.arange(cstart, cend, dtype=int)
    rowgrid, colgrid = np.meshgrid(rows, cols)

    roipcm = pcm[rowgrid, colgrid]

    dist = np.argmax(roipcm)
    roidist1 = dist % roipcm.shape[1]
    roidist0 = dist // roipcm.shape[1]

    dist0 = rowgrid[roidist0, roidist1]
    dist1 = colgrid[roidist0, roidist1]

    return dist0, dist1, pcm[dist0, dist1]


#%% relative_stitching_positions
def relative_stitching_positions(
    images,
    tile_dimensions,
    overlap_rows_cols=[0.25, 0.25],
    tolerance=0.1,
    ignore_montage_edges=0,
    drifts=[[0, 0], [0, 0]],
    blur=0,
):
    # images: list of images as a series of rows from top to bottom and within the row from left to right
    # tile_dimensions: tuple consisting of first number of rows and second number of columns
    # overlap: tuple of values between 0.0 and 1.0 indicating the expected relative overlap of pictures
    # tolerance: relative allowed deviation from the expected overlap
    # note: all images should have the same resolution

    imdim = images[0].shape
    overlap_limits = np.zeros([2, 2])
    overlap_limits[0, 0] = imdim[0] * (overlap_rows_cols[0] - tolerance)
    overlap_limits[0, 1] = imdim[0] * (overlap_rows_cols[0] + tolerance)
    overlap_limits[1, 0] = imdim[1] * (overlap_rows_cols[1] - tolerance)
    overlap_limits[1, 1] = imdim[1] * (overlap_rows_cols[1] + tolerance)

    mask_edgeright = np.ones(imdim)
    mask_edgeup = np.ones(imdim)
    if ignore_montage_edges != 0:
        mask_edgeright[:, -int(ignore_montage_edges * imdim[1]) :] = 0
        mask_edgeup[: int(ignore_montage_edges * imdim[0]), :] = 0

    # mask areas far from the overlap, to ensure that even if the side opposite to the stitching edge
    # looks similar, the stitching happens on the right side of the image
    maskleft = np.ones(imdim)
    maskright = np.ones(imdim)
    maskup = np.ones(imdim)
    maskdown = np.ones(imdim)
    maskup[: int(imdim[0] - 2 * overlap_rows_cols[0] * imdim[0]), :] = 0
    maskdown[int(2 * overlap_rows_cols[0] * imdim[0]) :, :] = 0
    maskleft[:, : int(imdim[1] - 2 * overlap_rows_cols[1] * imdim[1])] = 0
    maskright[:, int(2 * overlap_rows_cols[1] * imdim[1]) :] = 0

    positions = np.zeros(
        [
            tile_dimensions[0] * tile_dimensions[1],
            tile_dimensions[0] * tile_dimensions[1],
            2,
        ]
    )
    pos_pcms = np.zeros(
        [
            tile_dimensions[0] * tile_dimensions[1],
            tile_dimensions[0] * tile_dimensions[1],
        ]
    )
    neighbours = []
    # loop checks for each image the relative position of its right and bottom neighour
    # via the maximum of the phase-correlation-matrix (PCM)
    for i in range(len(images) - 1):
        neighbours.append([])
        if (i + 1) % tile_dimensions[1] == 0:  # no right neighbour at the end of a row
            # if i%tile_dimensions[1]==0:
            if (
                i < (tile_dimensions[0] - 1) * tile_dimensions[1]
            ):  # no bottom neighbours in the last row
                j = i + tile_dimensions[1]  # j is the image below
                if j < len(images):
                    neighbours[i].append(j)
                    pcm = phase_correlation(
                        images[i] * maskup * mask_edgeright,
                        images[j] * maskdown * mask_edgeright,
                    )
                    if blur != 0:
                        pcm = cv2.blur(pcm, (blur, blur))
                    dist0, dist1, pcms = pos_from_pcm(
                        pcm,
                        overlap_limits,
                        "vertical",
                        tolerance,
                        imdim,
                        drifts[1][0],
                        drifts[1][1],
                    )
                    positions[i, j] = dist0, dist1
                    positions[j, i] = dist0, dist1
                    pos_pcms[i, j] = pcms
                    pos_pcms[j, i] = pcms

        else:
            j = i + 1  # j is the image right
            if j < len(images):
                neighbours[i].append(j)
                if i < tile_dimensions[1]:
                    pcm = phase_correlation(
                        images[i] * maskleft * mask_edgeup,
                        images[j] * maskright * mask_edgeup,
                    )
                    if blur != 0:
                        pcm = cv2.blur(pcm, (blur, blur))
                else:
                    pcm = phase_correlation(images[i] * maskleft, images[j] * maskright)
                    if blur != 0:
                        pcm = cv2.blur(pcm, (blur, blur))
                dist0, dist1, pcms = pos_from_pcm(
                    pcm,
                    overlap_limits,
                    "horizontal",
                    tolerance,
                    imdim,
                    drifts[0][0],
                    drifts[0][1],
                )
                positions[i, j] = dist0, dist1
                positions[j, i] = dist0, dist1
                pos_pcms[i, j] = pcms
                pos_pcms[j, i] = pcms

            if i < (tile_dimensions[0] - 1) * tile_dimensions[1]:
                j = i + tile_dimensions[1]  # j is the image below
                if j < len(images):
                    neighbours[i].append(j)
                    pcm = phase_correlation(images[i] * maskup, images[j] * maskdown)
                    if blur != 0:
                        pcm = cv2.blur(pcm, (blur, blur))

                    dist0, dist1, pcms = pos_from_pcm(
                        pcm,
                        overlap_limits,
                        "vertical",
                        tolerance,
                        imdim,
                        drifts[1][0],
                        drifts[1][1],
                    )
                    positions[i, j] = dist0, dist1
                    positions[j, i] = dist0, dist1
                    pos_pcms[i, j] = pcms
                    pos_pcms[j, i] = pcms

    return positions, neighbours, pos_pcms


#%% absolute_stitching_positions
def absolute_stitching_positions(
    positions, neighbours, tile_dimensions, pos_pcms, conflict_sol="weighted"
):
    # in case of non-matching relative image-positions resulting from different neighbours
    # the average of the conflicting values is taken

    pos_pcms -= np.min(pos_pcms)
    pos_pcms += 0.000001

    absolute_positions = np.zeros([tile_dimensions[0], tile_dimensions[1], 2])
    weights = np.zeros([tile_dimensions[0], tile_dimensions[1], 2])
    for i in range(len(neighbours)):
        row0 = i // tile_dimensions[1]
        column0 = i % tile_dimensions[1]

        for j in neighbours[i]:
            row1 = j // tile_dimensions[1]
            column1 = j % tile_dimensions[1]

            if conflict_sol == "last":
                absolute_positions[row1, column1, 0] += (
                    absolute_positions[row0, column0, 0] + positions[i, j, 0]
                )
                absolute_positions[row1, column1, 1] += (
                    absolute_positions[row0, column0, 1] + positions[i, j, 1]
                )
                break

            if conflict_sol == "weighted":
                if np.sum(absolute_positions[row0, column0]) == 0:
                    absolute_positions[row1, column1, 0] += (
                        absolute_positions[row0, column0, 0] + positions[i, j, 0]
                    )
                    absolute_positions[row1, column1, 1] += (
                        absolute_positions[row0, column0, 1] + positions[i, j, 1]
                    )
                    weights[row1, column1, 0] += pos_pcms[i, j]
                    weights[row1, column1, 1] += pos_pcms[i, j]
                else:
                    absolute_positions[row1, column1, 0] = (
                        absolute_positions[row1, column1, 0] * weights[row1, column1, 0]
                        + (absolute_positions[row0, column0, 0] + positions[i, j, 0])
                        * pos_pcms[i, j]
                    )
                    absolute_positions[row1, column1, 1] = (
                        absolute_positions[row1, column1, 1] * weights[row1, column1, 1]
                        + (absolute_positions[row0, column0, 1] + positions[i, j, 1])
                        * pos_pcms[i, j]
                    )
                    weights[row1, column1, 0] += pos_pcms[i, j]
                    weights[row1, column1, 1] += pos_pcms[i, j]
                    absolute_positions[row1, column1] /= weights[row1, column1]

            else:
                if sum(absolute_positions[row1, column1]) == 0:
                    average_division = 1
                else:
                    average_division = 2

                absolute_positions[row1, column1, 0] += (
                    absolute_positions[row0, column0, 0] + positions[i, j, 0]
                )
                absolute_positions[row1, column1, 1] += (
                    absolute_positions[row0, column0, 1] + positions[i, j, 1]
                )

                absolute_positions[row1, column1] /= average_division

    # shift to have only positive positions
    absolute_positions[:, :, 0] -= np.min(absolute_positions[:, :, 0])
    absolute_positions[:, :, 1] -= np.min(absolute_positions[:, :, 1])

    return absolute_positions.astype(int)


#%% contrast_correction
def contrast_correction(images_c):
    # normalize the brightness of all pictures in the series, by multiplying each image
    # with a factor, so that the maximum of the pixel-value-histogram of every image is
    # at the same position
    imdim = images_c[0].shape
    hists = []
    for i in range(len(images_c)):
        flatim = np.reshape(images_c[i], imdim[0] * imdim[1])
        vals, bins = np.histogram(flatim, 100)
        maxpos = np.argmax(vals[5:]) + 5
        brightness = (bins[maxpos] + bins[maxpos + 1]) / 2
        hists.append(brightness)

    ref = np.mean(hists)
    images_cc = []
    for i in range(len(images_c)):
        images_cc.append(images_c[i] * ref / hists[i])
    return images_cc


#%% drift_correction
def drift_correction(images, tile_dimensions, overlap_rows_cols, tolerance=0.1):
    # images: list of images as a series of rows from top to bottom and within the row from left to right
    # tile_dimensions: tuple consisting of first number of rows and second number of columns
    # overlap: tuple of values between 0.0 and 1.0 indicating the expected relative overlap of pictures
    # tolerance: relative allowed deviation from the expected overlap
    # note: all images should have the same resolution

    imdim = images[0].shape

    overlap_limits = np.zeros([2, 2])
    overlap_limits[0, 0] = imdim[0] * (overlap_rows_cols[0] - tolerance)
    overlap_limits[0, 1] = imdim[0] * (overlap_rows_cols[0] + tolerance)
    overlap_limits[1, 0] = imdim[1] * (overlap_rows_cols[1] - tolerance)
    overlap_limits[1, 1] = imdim[1] * (overlap_rows_cols[1] + tolerance)

    positions = np.zeros(
        [
            tile_dimensions[0] * tile_dimensions[1],
            tile_dimensions[0] * tile_dimensions[1],
            2,
        ]
    )
    pos_pcms = np.zeros(
        [
            tile_dimensions[0] * tile_dimensions[1],
            tile_dimensions[0] * tile_dimensions[1],
        ]
    )

    # loop checks for each image the relative position of its right and bottom neighour
    # via the maximum of the phase-correlation-matrix (PCM)
    for i in range(len(images) - 1):

        if (i + 1) % tile_dimensions[1] == 0:  # no right neighbour at the end of a row
            # if i%tile_dimensions[1]==0:
            if (
                i < (tile_dimensions[0] - 1) * tile_dimensions[1]
            ):  # no bottom neighbours in the last row
                j = i + tile_dimensions[1]  # j is the image below
                if j < len(images):
                    pcm = phase_correlation(images[i], images[j])

                    dist0, dist1, pcms = pos_from_pcm(
                        pcm, overlap_limits, "vertical", tolerance, imdim, 0, 0
                    )
                    positions[i, j] = dist0, dist1
                    positions[j, i] = dist0, dist1
                    pos_pcms[i, j] = pcms
                    pos_pcms[j, i] = pcms

        else:
            j = i + 1  # j is the image right
            if j < len(images):
                if i < tile_dimensions[1]:
                    pcm = phase_correlation(images[i], images[j])
                else:
                    pcm = phase_correlation(images[i], images[j])
                dist0, dist1, pcms = pos_from_pcm(
                    pcm, overlap_limits, "horizontal", tolerance, imdim, 0, 0
                )
                positions[i, j] = dist0, dist1
                positions[j, i] = dist0, dist1
                pos_pcms[i, j] = pcms
                pos_pcms[j, i] = pcms

            if i < (tile_dimensions[0] - 1) * tile_dimensions[1]:
                j = i + tile_dimensions[1]  # j is the image below
                if j < len(images):
                    pcm = phase_correlation(images[i], images[j])

                    dist0, dist1, pcms = pos_from_pcm(
                        pcm, overlap_limits, "vertical", tolerance, imdim, 0, 0
                    )
                    positions[i, j] = dist0, dist1
                    positions[j, i] = dist0, dist1
                    pos_pcms[i, j] = pcms
                    pos_pcms[j, i] = pcms

    rightmoves = np.diag(pos_pcms, 1)
    downmoves = np.diag(pos_pcms, tile_dimensions[1])
    right0 = np.argmax(rightmoves)
    down0 = np.argmax(downmoves)
    right1 = right0 + 1
    down1 = down0 + tile_dimensions[1]

    drift_down, drift_right = np.zeros(2), np.zeros(2)

    expected_row_pos = imdim[0] - imdim[0] * overlap_rows_cols[0]
    expected_col_pos = imdim[1] - imdim[1] * overlap_rows_cols[1]
    drift_down[0] = positions[down0, down1, 0] - expected_row_pos
    drift_down[1] = positions[down0, down1, 1]

    drift_right[0] = positions[right0, right1, 0]
    drift_right[1] = positions[right0, right1, 1] - expected_col_pos

    drifts = []
    drifts.append(drift_right)
    drifts.append(drift_down)

    alldrifts_right = []
    alldrifts_down = []
    for i in range(len(rightmoves)):
        if rightmoves[i] == 0:
            pass
        else:
            adr0 = positions[i, i + 1, 0]
            adr1 = positions[i, i + 1, 1] - expected_row_pos
            alldrifts_right.append([adr0, adr1])
    for i in range(len(downmoves)):
        add0 = positions[i, i + tile_dimensions[1], 0] - expected_col_pos
        add1 = positions[i, i + tile_dimensions[1], 1]
        alldrifts_down.append([add0, add1])

    return drifts, alldrifts_right, alldrifts_down


#%% stitch_grid
def stitch_grid(images, absolute_positions, tile_dimensions, mask):
    # to ensure a smooth transition between two pictures, a weighted sum in the overlap-region is executed
    # the weights are given by mask

    imdim = images[0].shape
    vmax = np.max(absolute_positions[:, :, 0]) + imdim[0]
    hmax = np.max(absolute_positions[:, :, 1]) + imdim[1]

    division = np.zeros([vmax, hmax])  # ,dtype=np.double)
    montage = np.zeros([vmax, hmax])  # ,dtype=np.uint16)

    for i in range(len(images)):
        row = i // tile_dimensions[1]
        column = i % tile_dimensions[1]
        v0 = absolute_positions[row, column, 0]
        v1 = absolute_positions[row, column, 0] + imdim[0]
        h0, h1 = (
            absolute_positions[row, column, 1],
            absolute_positions[row, column, 1] + imdim[1],
        )
        montage[v0:v1, h0:h1] += images[i] * mask
        division[v0:v1, h0:h1] += mask

    division[division == 0] = 1.0
    montage /= division
    montage -= np.min(montage)
    return montage / np.max(montage)


#%% optimize_images
def optimize_images(images, background_division="mask"):
    # to achieve homogeneous brightness at non-optimal lightning,
    # the images are normalized (divided by) mask, a blurred median-image of the series
    # weighting areas differently so,
    # when later overlapping image-regions are summed, mask ensures that
    # the influence of a well illuminated area is bigger, than a poorly illuminated area

    immed = np.median(images, axis=0)

    images_mad = np.abs(images - immed)
    mad = np.median(images_mad, axis=0)
    madnorm = mad / np.max(mad)

    mask = cv2.blur(immed, (11, 11))
    mask = cv2.blur(mask, (31, 31))
    mask = cv2.blur(mask, (51, 51))
    mask = cv2.blur(mask, (71, 71))

    if background_division == "mask":
        images_c = images / mask
    elif background_division == "median":
        images_c = np.clip((images / immed / 2), 0, 1) * 255
        images_c = images_c.astype(np.uint8)

    return images_c, immed, madnorm, mask


#%% two_imshow
def two_imshow(images, absolute_positions, tile_dimensions, i, mask, zoom):
    row = i // tile_dimensions[1]
    column = i % tile_dimensions[1]
    imshape = np.array(images[0].shape)

    if i < tile_dimensions[1]:  # first row
        j = i - 1
        ims = [i, j]
        positions = np.zeros([2, 2], dtype=int)
        positions[0] = absolute_positions[row, column]
        positions[1] = absolute_positions[row, column - 1]

    else:
        if i % tile_dimensions[1] == 0:  # first element of a row
            j = i - tile_dimensions[1]
            ims = [i, j]
            positions = np.zeros([2, 2], dtype=int)
            positions[0] = absolute_positions[row, column]
            positions[1] = absolute_positions[row - 1, column]
        else:

            j = i - tile_dimensions[1]
            k = i - 1
            m = j - 1
            ims = [i, j, k, m]
            positions = np.zeros([4, 2], dtype=int)
            positions[0] = absolute_positions[row, column]
            positions[1] = absolute_positions[row - 1, column]
            positions[2] = absolute_positions[row, column - 1]
            positions[3] = absolute_positions[row - 1, column - 1]

    offset = np.min(positions, axis=0)
    for a in range(len(positions)):
        positions[a] -= offset
    offsetsize = np.max(positions, axis=0)
    canv = np.zeros(imshape + offsetsize)
    canvmask = np.zeros(imshape + offsetsize)
    for a in range(len(positions)):
        canv[
            positions[a, 0] : positions[a, 0] + imshape[0],
            positions[a, 1] : positions[a, 1] + imshape[1],
        ] += images[ims[a]]
        canvmask[
            positions[a, 0] : positions[a, 0] + imshape[0],
            positions[a, 1] : positions[a, 1] + imshape[1],
        ] += mask

    canvmask[canvmask == 0] = 1
    res = canv / canvmask

    # shaperes=res.shape
    if i < tile_dimensions[1]:  # first row
        x = res.shape[1]
        xmin = int(x * (zoom / 2))
        xmax = int(x - xmin)
        res = res[:, xmin:xmax]
        for a in range(len(positions)):
            positions[a, 1] -= xmin
    else:
        if i % tile_dimensions[1] == 0:  # first element of a row
            y = res.shape[0]
            ymin = int(y * (zoom / 2))
            res = res[ymin:, :]
            for a in range(len(positions)):
                positions[a, 0] -= ymin
        else:
            y = res.shape[0]
            x = res.shape[1]
            xmin = int(x * (zoom / 2))
            xmax = int(x - xmin)
            ymin = int(y * (zoom / 2))
            res = res[ymin:, xmin:xmax]
            for a in range(len(positions)):
                positions[a, 0] -= ymin
                positions[a, 1] -= xmin
    return res, positions[0]


image_counter = 0

#%% manual_correction
def manual_correction(images, absolute_positions, tile_dimensions, mask, zoom=0.5):
    abspos = deepcopy(absolute_positions)

    apshape = np.shape(abspos)

    # functions
    def press(event):
        global image_counter
        row = image_counter // tile_dimensions[1]
        column = image_counter % tile_dimensions[1]
        deltax = -abspos[row, column, 0] + absolute_positions[row, column, 0]
        deltay = -abspos[row, column, 1] + absolute_positions[row, column, 1]

        if event.key == "enter":
            image_counter += 1

            for j in range(column + 1, apshape[1]):
                absolute_positions[row, j, 0] += deltax
                absolute_positions[row, j, 1] += deltay
                abspos[row, j, 0] += deltax
                abspos[row, j, 1] += deltay

            for i in range(row + 1, apshape[0]):
                for j in range(column, apshape[1]):
                    absolute_positions[i, j, 0] += deltax
                    absolute_positions[i, j, 1] += deltay
                    abspos[i, j, 0] += deltax
                    abspos[i, j, 1] += deltay

            deltax = 0
            deltay = 0

            if image_counter == len(images):
                plt.close()
                absolute_positions[:, :, 0] -= np.min(absolute_positions[:, :, 0])
                absolute_positions[:, :, 1] -= np.min(absolute_positions[:, :, 1])
                image_counter = 0
                return 0
        if event.key == "left":
            absolute_positions[row, column, 1] += -1
            deltay += -1
        if event.key == "right":
            absolute_positions[row, column, 1] += 1
            deltay += 1
        if event.key == "up":
            absolute_positions[row, column, 0] += -1
            deltax += -1
        if event.key == "down":
            absolute_positions[row, column, 0] += 1
            deltax += 1
        if event.key == "backspace":
            if image_counter == 1:
                pass
            else:
                image_counter -= 1

        # print(deltax,deltay)

        # deltax=-abspos[row,column,0]+absolute_positions[row,column,0]
        # deltay=-abspos[row,column,1]+absolute_positions[row,column,1]

        pic, pos = two_imshow(
            images, absolute_positions, tile_dimensions, image_counter, mask, zoom
        )
        imshape = images[0].shape
        ax.cla()
        ax.imshow(pic, cmap="gray")
        ax.set_title(
            "image "
            + str(image_counter)
            + "   in row "
            + str(row)
            + "  and column "
            + str(column)
            + "\n $\Delta x=$"
            + str(deltay)
            + "        $\Delta y=$"
            + str(deltax)
        )
        xsmin = np.min([pos[0] + imshape[0], pic.shape[0]])
        xs = [pos[0], pos[0], xsmin, xsmin]
        ysmin = np.min([pos[1] + imshape[1], pic.shape[1]])
        ys = [pos[1], ysmin, pos[1], ysmin]
        ax.plot(ys, xs, "+", c="r", markersize=15)
        plt.gcf().canvas.draw()

        if event.key == "escape":
            plt.close()
            image_counter = 0

    # start program
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect("key_press_event", press)

    ax.plot([0, 1], [0, 1], c="w")
    ax.text(0.3, 0.8, "Manual Image Stitching")
    ax.text(
        0.1,
        0.1,
        "Use enter to start the program,\n"
        + "for moving to the next image, when satisfied,\n"
        + "and to end the program \n(when pressing enter after the last image).\n"
        + "\nPress backspace to go back to a previous image.\n"
        + "\nUse left, right, up and down arrow keys to position \nthe image marked by red + showing the corners.\n"
        + "\nFor closing the program at any point press esc \n(to reset internal counters).",
    )
    plt.show()




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