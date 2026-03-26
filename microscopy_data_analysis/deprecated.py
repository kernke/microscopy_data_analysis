"""
@author: kernke
"""
from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.integrate import quad
from scipy.spatial.distance import cdist

from .general_util import make_mask
from .image_aligning import phase_correlation
from .image_processing import img_rotate_bound, img_to_half_int16


#%%
#%% fine_tuning_shifts (real space align)
def fine_tuning_shifts(aligned_stack,delta=4):
    stack=np.zeros([aligned_stack.shape[0],aligned_stack.shape[1]+1,aligned_stack.shape[2]+1],dtype=np.int16)
    for i in range(len(stack)):

        stack[i,:-1,:-1]=img_to_half_int16(aligned_stack[i])
        
    changes=np.zeros([2*delta+1,2*delta+1])
    shifts=np.zeros([len(stack),2],dtype=int)

    deltarange=np.arange(-delta,delta+1,1)
    Nd=len(deltarange)
    for i in range(len(stack)-1):
        img0=stack[i][delta:-delta-1,delta:-delta-1]
        for j in range(Nd):
            deltaj=deltarange[j]
            for k in range(Nd):
                deltak=deltarange[k]
                #least squares
                #changes[j,k]=np.sum((img0-stack[i+1][delta+deltaj:-(delta-deltaj)-1,delta+deltak:-(delta-deltak)-1])**2)
                #correlation
                changes[j,k]=-np.sum((img0*stack[i+1][delta+deltaj:-(delta-deltaj)-1,delta+deltak:-(delta-deltak)-1])**2)
        idx0,idx1=np.where(changes==np.min(changes))
        shifts[i]=idx0[0],idx1[0]
    shifts+= -delta
    return -np.cumsum(shifts,axis=0)







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
big_steps=False

#%% manual_correction
def manual_correction(images, absolute_positions, tile_dimensions, mask, zoom=0.5):
    abspos = deepcopy(absolute_positions)

    apshape = np.shape(abspos)

    # functions
    def press(event):
        global image_counter
        global big_steps
        row = image_counter // tile_dimensions[1]
        column = image_counter % tile_dimensions[1]
        deltax = -abspos[row, column, 0] + absolute_positions[row, column, 0]
        deltay = -abspos[row, column, 1] + absolute_positions[row, column, 1]

        if event.key == "b":
            if big_steps:
                big_steps=False
                print("big steps deactivated")
            else:
                big_steps=True
                print("big steps activated")
                
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
            if big_steps:
                absolute_positions[row, column, 1] += -10
                deltay += -10                
            else:
                absolute_positions[row, column, 1] += -1
                deltay += -1
        if event.key == "right":
            if big_steps:
                absolute_positions[row, column, 1] += 10
                deltay += 10                
            else:
                absolute_positions[row, column, 1] += 1
                deltay += 1
        if event.key == "up":
            if big_steps:
                absolute_positions[row, column, 0] += -10
                deltax += -10                
            else:
                absolute_positions[row, column, 0] += -1
                deltax += -1
        if event.key == "down":
            if big_steps:
                absolute_positions[row, column, 0] += 10
                deltax += 10                
            else:
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
        + "\nUse left, right, up and down arrow keys to position \nthe image marked "
        + "by red + showing the corners.\n"
        + "\nPress 'b' to switch between big steps (10 pixels) "
        + "and normal steps (1 pixel)"
        + "\nFor closing the program at any point press esc \n"
        +"(to reset internal counters).",
    )
    plt.show()

#%% pos_from_pcm
def _pos_from_pcm(pcm, overlap_limits, mode, tolerance, imdim, rdrift, cdrift):
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
    # images: list of images as a series of rows from top to bottom 
    # and within the row from left to right
    # tile_dimensions: tuple consisting of first number of rows 
    # and second number of columns
    # overlap: tuple of values between 0.0 and 1.0 indicating 
    # the expected relative overlap of pictures
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

    # mask areas far from the overlap, to ensure that even if the side 
    # opposite to the stitching edge
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
                    dist0, dist1, pcms = _pos_from_pcm(
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
                dist0, dist1, pcms = _pos_from_pcm(
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

                    dist0, dist1, pcms = _pos_from_pcm(
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

#%% relative_stitching_positions
def relative_stitching_sift(
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
                    dist0, dist1, pcms = _pos_from_pcm(
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
                dist0, dist1, pcms = _pos_from_pcm(
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

                    dist0, dist1, pcms = _pos_from_pcm(
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
def contrast_correction(images):
    """
    normalize the brightness of all pictures in the series, by multiplying each image
    with a factor, so that the maximum of the pixel-value-histogram of every image is
    at the same position

    Args:
        images (list of images): 
            input.

    Returns:
        images_corrected (list of images): 
            output.

    """
    # 
    imdim = images[0].shape
    hists = []
    for i in range(len(images)):
        flatim = np.reshape(images[i], imdim[0] * imdim[1])
        vals, bins = np.histogram(flatim, 100)
        maxpos = np.argmax(vals[5:]) + 5
        brightness = (bins[maxpos] + bins[maxpos + 1]) / 2
        hists.append(brightness)

    ref = np.mean(hists)
    images_corrected = []
    for i in range(len(images)):
        images_corrected.append(images[i] * ref / hists[i])
    return images_corrected


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

                    dist0, dist1, pcms = _pos_from_pcm(
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
                dist0, dist1, pcms = _pos_from_pcm(
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

                    dist0, dist1, pcms = _pos_from_pcm(
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



#%%

#            if self.modifiable_file is None:
#                print("warning: deprecated")
#                h5file="temp.h5"
#                #self.modifiable_file=h5file
#                with h5py.File(h5file, "a") as f:
#                    if "z_transform_name_list" in f:
#                        del f["z_transform_name_list"]    
#                    f["z_transform_name_list"]=self.img_list
#                    maxnum=len(self.img_list)
#                    zfillnum=int(np.ceil(np.log10(maxnum)))
#                    for i in range(len(self.img_list)):
#                        num=str(i).zfill(zfillnum)
#                        img=self.get_img(i)
#                        minimum=np.min(img)
#                        mean=np.mean(img)
#                        std=np.std(img)
#                        if "z_transform/"+num in f:
#                            del f["z_transform/"+num]
#                        f["z_transform/"+num]=(img-mean)/std
#                        most_negative=min((minimum-mean)/std,most_negative)
#                        new_img_list.append("z_transform/"+num)
#                    
#                    if offset_positive:
#                        for i in range(len(self.img_list)):
#                            num=str(i).zfill(zfillnum)
#                            arr=f["z_transform/"+num][:]
#                            arr-=most_negative
#                            f["z_transform/"+num][:]=arr-most_negative
#
#                return new_img_list


#%%
def split_function_into_equal_area_parts(func,number_of_parts,prec=10**-6,limits=[-np.inf,np.inf],
                                            printing=False,max_iterations=1000):
    """calculate the split positions to divide a function into equal area parts, best suited for smooth functions
    """
    total_area_first,error=quad(func,limits[0],limits[1]) #get total area
    quad_abs_prec=0.5*prec /number_of_parts#*total_area_first/ #determine absolute precision for integration
    adj_func=lambda x: func(x)/total_area_first
    total_area,error=quad(adj_func,limits[0],limits[1],epsabs=quad_abs_prec)

    desired_area=total_area/number_of_parts
    if printing:
        print(quad_abs_prec)
        print(error)
        print(total_area)
        print(desired_area)        
        
    tolerance_area=prec*desired_area

    areas=[]
    cuts=[]
    lower_cut,lower_area=find_next_cut(adj_func,limits[0],desired_area,tolerance_area,quad_abs_prec,max_iterations)    

    cuts.append(lower_cut)
    areas.append(lower_area)
    start_step_size=1
    for i in range(number_of_parts-2):
        cut,area=find_next_cut(adj_func,cuts[-1],desired_area,tolerance_area,quad_abs_prec,max_iterations,step_size=start_step_size)
        
        areas.append(area)
        cuts.append(cut)
        start_step_size=cuts[-1]-cuts[-2]    
        
    remaining=quad(adj_func,cuts[-1],limits[1],epsabs=quad_abs_prec)[0]
    areas.append(remaining)
     
    return cuts,areas

def find_next_cut(func,firstcut,desired_area,tolerance_area,quad_abs_prec,
                         max_iterations,step_size=1):
    area_delta=2*tolerance_area

    if firstcut==-np.inf:
        estimated_cut_position=0
    else:
        estimated_cut_position=firstcut+step_size
        
    step_right_before=None
    counter=0
    while True:
        estimated_area=quad(func,firstcut,estimated_cut_position,epsabs=quad_abs_prec)[0]        
        
        area_delta=desired_area-estimated_area
        if np.abs(area_delta)<tolerance_area:
            break
                
        if area_delta<0: # estimated > desired --> move left 
            step_right=False
            estimated_cut_position -= step_size            
        else: # estimated < desired --> move right
            step_right=True
            estimated_cut_position += step_size

        
        if step_right==step_right_before: #no direction change
            step_size *=2
        else:
            step_size *=0.5
        step_right_before=step_right
                
        counter+=1
        if counter>max_iterations:
            print("maximum number of iterations reached")
            break
            
    return estimated_cut_position,estimated_area

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
            rot, log = img_rotate_bound(image_roi, angle + pmangle[i])
            drot, log = img_rotate_bound(dummy, angle + pmangle[i], bm=0)
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
        rot, log = img_rotate_bound(image_roi, res)
        drot, log = img_rotate_bound(dummy, res, bm=0)
        mask = drot > 0.1
        # obtain_snr(rot, mask, line,True,minlength)
        plt.imshow(rot * mask)
        plt.show()

    return res


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



#%% enhance_lines_prototype
def enhance_lines_prototype(
    image, angle, ksize=None, dist=1, iterations=2, line="dark"
):

    if ksize is None:
        ksize=3
    
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

    # res=np.copy(tres)

    for i in range(iterations):

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

#%%
def normalized_spectra(data,se_image,wavelengths,boundary_offset=0,minimum_pix=25,
                       ksize1=20,ksize2=40,ksize3=10,upper_threshold=1.1,lower_threshold=0.95):
    """
    ksize1:proportional to the boundary between island and ring
    ksize2:proportional to the width of the ring
    ksize3:closing gaps in the mask to obtain the spectrum of SiO
    """
    # obtain binary image with areas holding the value 1 represent the crystal islands
    # and the area with value 0 is the SiO-matrix
    binary_image=se_image>np.mean(se_image)*upper_threshold
    binary_image=binary_image*1
    if boundary_offset>0:
        mask=np.zeros(binary_image.shape)
        mask[boundary_offset:-boundary_offset,boundary_offset:-boundary_offset]=1
        binary_image=(binary_image*mask).astype(np.uint8)

    # label all the islands, that contain atleast the minimum amount of pixels
    # also get a center-position of island by a mean of all pixel-positions of that island
    separated_image,number_of_islands=ndimage.label(binary_image)
    spots=[]
    centers=[]
    for i in range(number_of_islands):
        index1,index2=np.where(separated_image==i)
        if len(index1)<minimum_pix:
            pass
        else:
            if i ==0:
                #pass
                si_matrix=(index1,index2)
            else:
                x=np.mean(index1)
                y=np.mean(index2)
                centers.append((x,y))
                spots.append((index1,index2))

    # get the actual CL-spectrum of each island, 
    # by taking the mean over the pixels of each island at the respective wavelength   
    spectra=np.zeros([len(spots),len(wavelengths)])
    for k in range(len(wavelengths)):
        for i in range(len(spots)):
            index1,index2=spots[i]
            spot_pixels=np.zeros(len(index1))
            for j in range(len(index1)):
                spot_pixels[j]=data[k,index1[j],index2[j]]
            
            spectra[i,k]=np.mean(spot_pixels)


    
    # To obtain a spectrum of the SiO-matrix, we want to make sure to not get any effect from the nano-islands
    # or the spots, where the SiO-matrix is interrupted with no island
    # so now we create a mask not only using a threshold for the bright islands (upper_threshold)
    # but also a threshold for the dark spots, where no islands sit anymore (lower_threshold)
    center=(ksize3-1)//2
    kernel=cv2.circle(np.zeros([ksize3,ksize3],dtype=np.uint8),(center,center),center,1,-1)
    SiO_image=(se_image<np.mean(se_image)*upper_threshold) * (se_image>np.median(se_image)*lower_threshold )*1.
    if boundary_offset>0:
        SiO_image=SiO_image*mask
    SiO_image=cv2.erode(SiO_image,kernel)

    # Show the result next to the original SEM-image
    fig,ax=plt.subplots(1,3)
    fig.set_figwidth(12)
    ax[1].imshow(se_image,cmap="gray")
    ax[1].set_title("original SEM-image")
    ax[0].imshow(binary_image)
    ax[0].set_title("obtained mask/binary image")
    ax[2].imshow(SiO_image)
    ax[2].set_title("yellow area for SiO-spectrum")
    plt.show()
    
    # get the mean background
    mean_background=np.zeros(len(wavelengths))
    for k in range(len(wavelengths)):
        mean_background[k]=np.mean(data[k][SiO_image==1])

    # For that reason we increase the masked area of the island 
    center=(ksize1-1)//2
    kernel=cv2.circle(np.zeros([ksize1,ksize1],dtype=np.uint8),(center,center),center,1,-1)
    si_matrix_mask=np.zeros(binary_image.shape,dtype=np.uint8)
    index1,index2=si_matrix
    si_matrix_mask[index1,index2]=1
    si_matrix_mask2=cv2.erode(si_matrix_mask,kernel)



    # by increasing the masked area further and subtracting the original mask,
    # we obtain rings surrounding each spot, 
    # which we can use to subtract the local background at each spot.
    center=(ksize2-1)//2
    kernel=cv2.circle(np.zeros([ksize2,ksize2],dtype=np.uint8),(center,center),center,1,-1)
    si_matrix_mask3=cv2.erode(si_matrix_mask2,kernel)
    ring_image=si_matrix_mask3-si_matrix_mask2

    separated_image,number_of_islands=ndimage.label(ring_image)
    rings=[]
    rcenters=[]
    for i in range(number_of_islands):
        index1,index2=np.where(separated_image==i)
        if len(index1)<minimum_pix:
            pass
        else:
            if i ==0:
                pass
            else:
                x=np.mean(index1)
                y=np.mean(index2)
                rcenters.append((x,y))
                rings.append((index1,index2))

    fig,ax=plt.subplots(1,3)
    fig.set_figwidth(12)
    ax[0].imshow(binary_image)
    ax[0].set_title("labeled islands")
    counter=0
    for i in centers:
        ax[0].text(i[1],i[0],str(counter),c="r")
        counter+=1
    ax[1].imshow(si_matrix_mask2)
    ax[1].set_title("increased island areas (inside of rings)")
    ax[2].imshow(ring_image)
    ax[2].set_title("rings for local background-subtraction")
    plt.show()

    # Because some islands are very close to each other,
    # they are surrounded by a single ring.
    # Thus the number of islands is not the same as the number of rings.
    # So we calculate the distance of the center of each island to the center of each ring,
    # the the island and the ring with minimal distance are paired 
    distance_matrix=cdist(centers,rcenters)
    spot_ring_relation=np.argmin(distance_matrix,axis=1)

    # obtain the spectra of the all the rings    
    rspectra=np.zeros([len(rings),len(wavelengths)])
    for k in range(len(wavelengths)):
        for i in range(len(rings)):
            index1,index2=rings[i]
            ring_pixels=np.zeros(len(index1))
            for j in range(len(index1)):
                ring_pixels[j]=data[k,index1[j],index2[j]]
            
            rspectra[i,k]=np.mean(ring_pixels)

    # subtract the local background by the corresponding ring from each island
    # also correct brightness variations in the image, due to the mirror-geometry
    # by normalizing the local SiO(ring)-spectra by the mean-SiO-spectrum
    nspectra=np.zeros(spectra.shape)
    for k in range(len(wavelengths)):
        for i in range(len(spots)):
            nspectra[i,k]=spectra[i,k]-rspectra[spot_ring_relation[i],k]
            #fac=mean_background[k]/rspectra[i,k]
            nspectra[i,k] *= mean_background[k]/rspectra[spot_ring_relation[i],k]
            
    return nspectra,mean_background,centers

#%% eliminate_side_maxima_image
"""
    def eliminate_side_maxima_image(
        self, image, shiftrange=2, tol=1, valfactor=2.5, line="dark", test=False
    ):
        tms = self.slope_groups
        conpois = self.conpois
        # image=self.image
            #imcheck= _check_image(conpois[i], conlens[i], checkmaps[i])
            #if line == 'dark':
            #    cond=imcheck>medbrightness*med_ratio_threshold
            #else:
            #    cond=imcheck*med_ratio_threshold<medbrightness


        sortoutids = []
        sortout = []

        if line == "bright":

            for i in range(len(conpois)):
                sortout.append([])
                sortoutids.append([])

                if tms[i] > 1:
                    for j in range(len(conpois[i])):
                        check = _getcheck1(shiftrange, conpois[i][j], image)

                        vcheck = np.max(check[shiftrange - tol : shiftrange + tol])
                        check = check[check.astype(bool)]
                        checkmed = np.median(check)

                        if vcheck < valfactor * checkmed:

                            sortout[-1].append(conpois[i][j])
                            sortoutids[-1].append(j)

                else:
                    for j in range(len(conpois[i])):
                        check = _getcheck0(shiftrange, conpois[i][j], image)

                        vcheck = np.max(check[shiftrange - tol : shiftrange + tol])
                        check = check[check.astype(bool)]
                        checkmed = np.median(check)
                        if vcheck < valfactor * checkmed:
                            sortout[-1].append(conpois[i][j])
                            sortoutids[-1].append(j)

                print(len(sortout[-1]))

        else:
            for i in range(len(conpois)):
                sortout.append([])
                sortoutids.append([])

                if tms[i] > 1:
                    for j in range(len(conpois[i])):
                        check = _getcheck1(shiftrange, conpois[i][j], image)

                        vcheck = np.max(check[shiftrange - tol : shiftrange + tol])
                        check = check[check.astype(bool)]
                        checkmed = np.median(check)

                        if vcheck > valfactor * checkmed:

                            sortout[-1].append(conpois[i][j])
                            sortoutids[-1].append(j)

                else:
                    for j in range(len(conpois[i])):
                        check = _getcheck0(shiftrange, conpois[i][j], image)

                        vcheck = np.max(check[shiftrange - tol : shiftrange + tol])
                        check = check[check.astype(bool)]
                        checkmed = np.median(check)
                        if vcheck > valfactor * checkmed:
                            sortout[-1].append(conpois[i][j])
                            sortoutids[-1].append(j)

                print(len(sortout[-1]))

        if not test:
            self.sort_ids_out(sortoutids)

        return sortout
"""
#%% Enhance Lines2
"""
def line_enhance2(
    image, angle, ksize=None, dist=1, iterations=2, line="dark"
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

        middle = np.median(msrot)

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
"""
#%% Enhance Lines
"""
def line_enhance(
    image, angle, ksize=None, dist=1, iterations=2, line="dark"
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

        middle = np.median(msrot)

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

"""
#%% process2
"""
# smoothsize=35
def line_process2(
    images,
    rotangles,
    lowhigh,
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
    qkeys = []
    qkeys = list(images.keys())

    qcheck_images = {}

    for m in range(len(qkeys)):
        line_images = []
        check_images = []
        image = images[qkeys[m]]

        print(str(m + 1) + " / " + str(len(qkeys)))
        print(qkeys[m])
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

            #new = nclean[newmask > 0]

            check_images.append(img_rotate_back(nclean, log))
            #line_images.append(np.zeros(check_images[-1].shape))

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
        


    return check_images
"""

#%% eliminate_side_maxima_image
"""
    def eliminate_side_maxima_image(self,shiftrange=2,tol=1,line='dark',test=False):
        tms=self.tms
        conpois=self.conpois
        image=self.image

        sortoutids=[]
        sortout=[]

        for i in range(len(conpois)):
            sortout.append([])
            sortoutids.append([])

            #image=checkmaps[i]
            if tms[i]>1:
                for j in range(len(conpois[i])):

                    #a=np.max(conpois[i][j][:,1])
                    #b=np.min(conpois[i][j][:,1])
                    #if a+shiftrange >= image.shape[1] or b-shiftrange<0:
                    #    pass
                    #else:
                    check=getcheck1(shiftrange,conpois[i][j],image)

                    icheck=np.argmax(check)
                    checkval=np.max(check)
                    #print(icheck)
                    if icheck < shiftrange-tol or icheck > shiftrange+tol:

                        #checkmedian=np.median(check)
                        #mad=np.median(np.abs(check-checkmedian))
                        #if check[shiftrange] < checkmedian+mad_threshold*mad:
                        sortout[-1].append(conpois[i][j])
                        sortoutids[-1].append(j)

            else:
                for j in range(len(conpois[i])):

                    #a=np.max(conpois[i][j][:,0])
                    #b=np.min(conpois[i][j][:,0])
                    #if a+shiftrange >= image.shape[0] or b-shiftrange<0:
                    #    pass
                    #else:
                    check=getcheck0(shiftrange,conpois[i][j],image)
                    icheck=np.argmax(check)
                    #print(icheck)

                    if icheck < shiftrange-tol or icheck > shiftrange+tol:

                        #checkmedian=np.median(check)
                        #mad=np.median(np.abs(check-checkmedian))
                        #if check[shiftrange] < checkmedian+mad_threshold*mad:
                        sortout[-1].append(conpois[i][j])
                        sortoutids[-1].append(j)

            print(len(sortout[-1]))

        if not test:
            self.sort_ids_out(sortoutids)

        return sortout
    
"""   
    
    #%% make_interaction_dictionnary
"""
    def make_interaction_dictionnary(self):
        
        ext_slists = self.extended_slists
        shr_slists = self.shrinked_slists
        ext_elists = self.extended_elists
        shr_elists = self.shrinked_elists
        
        check_ext_shr_intersection(ext_slists,shr_slists,ext_elists,shr_elists)
        
        
        crosslines = self.crosslines
        crosslineset = set(map(tuple, crosslines))
        s1 = self.all
        s2 = self.cross_and_block
        s3 = self.crossings

        slists = self.extended_slists
        elists = self.extended_elists

        blockings = s2.difference(s3)
        corners = s1.difference(s2)
        gothroughs = s3


        corners_dic = {}
        gothroughs_dic = {}
        blockings_dic = {}

        for i in corners:
            (i1, i2), (i3, i4) = i
            a = slists[i1][i2]
            b = elists[i1][i2]
            c = slists[i3][i4]
            d = elists[i3][i4]
            corners_dic[i] = lineIntersection(a, b, c, d)

        for i in blockings:
            (i1, i2), (i3, i4) = i
            a = slists[i1][i2]
            b = elists[i1][i2]
            c = slists[i3][i4]
            d = elists[i3][i4]
            blockings_dic[i] = lineIntersection(a, b, c, d)

        for i in gothroughs:
            (i1, i2), (i3, i4) = i
            a = slists[i1][i2]
            b = elists[i1][i2]
            c = slists[i3][i4]
            d = elists[i3][i4]
            gothroughs_dic[i] = lineIntersection(a, b, c, d)

        return corners_dic, blockings_dic, gothroughs_dic
"""
    #%% get_line_lengths
"""    
    def _get_line_lengths(self):
        slists = self.slists
        elists = self.elists

        lengths = []

        for i in range(len(slists)):
            lengths.append([])
            for j in range(len(slists[i])):
                d = slists[i][j] - elists[i][j]
                lengths[-1].append(math.sqrt(np.sum(d * d)))

        self.lengths = lengths
        _add_attr("lengths", self.line_vars)

"""


    #%% eliminate_side_maxima
"""    
    def eliminate_side_maxima(self, mad_threshold=2, shiftrange=10, test=False):
        tms = self.slope_groups
        conpois = self.conpois
        checkmaps = self.checkmaps

        sortoutids = []
        sortout = []

        for i in range(len(conpois)):
            sortout.append([])
            sortoutids.append([])

            image = checkmaps[i]
            if tms[i] > 1:
                for j in range(len(conpois[i])):

                    a = np.max(conpois[i][j][:, 1])
                    b = np.min(conpois[i][j][:, 1])
                    if a + shiftrange >= image.shape[1] or b - shiftrange < 0:
                        pass
                    else:
                        check = _getcheck1(shiftrange, conpois[i][j], image)
                        if np.argmax(check) != shiftrange:

                            checkmedian = np.median(check)
                            mad = np.median(np.abs(check - checkmedian))
                            if check[shiftrange] < checkmedian + mad_threshold * mad:
                                sortout[-1].append(conpois[i][j])
                                sortoutids[-1].append(j)

            else:
                for j in range(len(conpois[i])):

                    a = np.max(conpois[i][j][:, 0])
                    b = np.min(conpois[i][j][:, 0])
                    if a + shiftrange >= image.shape[0] or b - shiftrange < 0:
                        pass
                    else:
                        check = _getcheck0(shiftrange, conpois[i][j], image)
                        if np.argmax(check) != shiftrange:

                            checkmedian = np.median(check)
                            mad = np.median(np.abs(check - checkmedian))
                            if check[shiftrange] < checkmedian + mad_threshold * mad:
                                sortout[-1].append(conpois[i][j])
                                sortoutids[-1].append(j)

            print(len(sortout[-1]))

        if not test:
            self.sort_ids_out(sortoutids)

        return sortout

"""
