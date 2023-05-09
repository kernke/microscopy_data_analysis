# -*- coding: utf-8 -*-
"""
Created on Mon May  8 15:31:33 2023

@author: kernke
"""
import h5py
from datetime import datetime
from .general_util import folder_file,assure_multiple,make_mask,take_map
from .image_processing import img_rotate_bound,img_morphLaplace,img_to_uint8
from .image_aligning import align_images,align_image_fast1
from .line_detection import line_process_partial,line_check_angle_s
import numpy as np
import cv2
import time

#%% merge_h5pys


def h5_merge(newh5, *h5files):
    with h5py.File(newh5, "w") as res:
        for i in h5files:
            with h5py.File(i, "r") as hf:
                for j in hf.keys():
                    hf.copy(hf[j], res, j)


def h5_merge_files(newh5, *h5files):
    with h5py.File(newh5, "w") as res:
        for i in h5files:
            pathname, groupname = folder_file(i)
            with h5py.File(i, "r") as hf:
                for j in hf.keys():
                    hf.copy(hf[j], res, groupname[:-3] + "/" + j)
                    
                    


def h5_get_keys(path_to_data, printing=True):
    with h5py.File(path_to_data, "r") as hf:
        keylist = list(hf.keys())
        keylist = np.array(keylist)
        nums = np.zeros(len(keylist), dtype=int)
        for i in range(len(keylist)):
            end = keylist[i].find(".")
            start = keylist[i].find("_0001_") + len("_0001_")
            nums[i] = int(keylist[i][start:end])
        sortindex = np.argsort(nums)
        skeylist = keylist[sortindex]

        titles = []
        for i in skeylist:
            titles.append(hf[i + "/title"].asstr()[()])

    if printing:
        for i in range(len(titles)):
            print(skeylist[i] + "    " + titles[i])

    return skeylist, titles


def h5_widths_and_relative_times(
    phi0, path_to_data, startnums, skeylist, titles, loopscans=True, orig_width=2560
):
    alltimes = []
    numbers = []
    newwidths = []

    with h5py.File(path_to_data, "r") as hf:
        s = hf[skeylist[startnums[0]] + "/start_time"].asstr()[()]
        end = s.find("+")
        starttime = datetime.strptime(s[:end], "%Y-%m-%dT%H:%M:%S.%f")

        for i in range(startnums[0], len(skeylist)):

            if loopscans and ("dscan" in titles[i] or "ct " in titles[i]):
                pass
            else:
                s = hf[skeylist[i] + "/start_time"].asstr()[()]
                end = s.find("+")
                d = datetime.strptime(s[:end], "%Y-%m-%dT%H:%M:%S.%f")
                if skeylist[i] + "/instrument/elapsed_time/value" in hf:
                    h5path_to_time = skeylist[i] + "/instrument/elapsed_time/value"
                else:
                    h5path_to_time = skeylist[i] + "/instrument/elapsed_time/data"
                times = hf[h5path_to_time][()]
                alltimes.append(times + (d - starttime).total_seconds())
                nu = hf[skeylist[i] + "/instrument/positioners/nu"][()]
                phi = hf[skeylist[i] + "/instrument/positioners/phi"][()]
                alphaf = nu - (phi - phi0)
                ratio = 1 / np.sin(alphaf / 180 * np.pi)
                numbers.append(i)
                newwidths.append((np.round((ratio * orig_width) / 2) * 2).astype(int))

    return newwidths, alltimes, numbers


def h5_make_temp_rois(numbers, alltimes, startnums):

    counter = 1
    tpicnums = [0]
    i = 0
    temporder = [[]]
    while i < len(numbers):
        if numbers[i] < startnums[counter]:
            if hasattr(alltimes[i], "__len__"):
                tpicnums[-1] += len(alltimes[i])
            else:
                tpicnums[-1] += 1
            temporder[-1].append(numbers[i])
            i += 1
        else:
            counter += 1
            tpicnums.append(0)
            temporder.append([])

    return tpicnums, temporder


#%% go_over_data
a = 3

def h5_go_over_data(
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
            drot, log = img_rotate_bound(dummy, rotangles[i], bm=0)
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

                    lapl = img_morphLaplace(img, kernel)

                    summed = np.zeros(lapl.shape, dtype=np.double)
                    summed += 255 - lapl
                    summed += img
                    copt = img_to_uint8(summed)

                    images, lowhigh = take_map(copt, size, overlap)
                    checkmaps = line_process_partial(
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
def h5_align_data(
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


def h5_enhance_and_align(
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
            drot, log = img_rotate_bound(dummy, rotangles[i], bm=0)
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

                    lapl = img_morphLaplace(img, kernel)

                    summed = np.zeros(lapl.shape, dtype=np.double)
                    summed += 255 - lapl
                    summed += img
                    copt = img_to_uint8(summed)

                    images, lowhigh = take_map(copt, size, overlap)
                    checkmaps = line_process_partial(
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
def h5_thresholding(newh5, oldh5, rotangles, tnames, inames, params, roi=None):
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
                        img_to_uint8(workimg), 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
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

                    newlines = line_check_angle_s(lines, rotangles[i], deg_tol)
                    # newlines=lines
                    newimg = np.zeros(srb.shape)
                    for points in newlines:
                        x1, y1, x2, y2 = points  # [0]
                        cv2.line(newimg, (x1, y1), (x2, y2), 255, 1)

                    prefix = "check" + str(i) + "_"
                    res[prefix + inames[j]][k] = newimg

                res["srb_" + inames[j]][k] = (np.sum(srbs, axis=0) > 0).astype(np.uint8)

    return np.sum(srbs, axis=0) > 0, newlines

                    