# -*- coding: utf-8 -*-
"""
@author: kernke
"""
import h5py
from datetime import datetime
from .general_util import folder_file,assure_multiple,make_mask,take_map
from .image_processing import img_rotate_bound,img_morphLaplace,img_to_uint8
from .image_aligning import align_images,align_image_fast1
from .line_detection import line_process_partial,line_check_angle_s,line_process_vis
import numpy as np
import cv2
import time

#%% h5_sortout_0frames_in_raw
def h5_sortout_0frames_in_raw(rawh5):
    with h5py.File(rawh5,'r+') as h5:
        keylist=list(h5.keys())
        for i in keylist:
            data= h5[i+"/imgs"][()]
            time= h5[i+"/time"][()]
            maxvals=[np.max(img) for img in data]
            keepvals=np.argwhere(maxvals)[:,0]
            if len(maxvals) > len(keepvals):
                print('frame is sorted out')
                del h5[i+"/imgs"] 
                del h5[i+"/time"]
                h5_images_dataset(h5, i+"/imgs", [len(keepvals),*data[0].shape])
                h5[i+"/imgs"][:]=data[keepvals]
                h5[i+"/time"]=time[keepvals]
                

    
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
                    
                    
#%% get_keys

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

#%% widths_and_relative_times
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

#%% temp rois
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

#%% create dataset
def h5_images_dataset(h5,name,shape,
                      dtype=np.uint8,
                      compression="gzip",compression_opts=2):

    h5.create_dataset(
        name,
        shape=shape,
        dtype=dtype,
        chunks=(1, shape[1], shape[2]),
        compression=compression,
        compression_opts=compression_opts,
    )

#%% go_over_data vis


def h5_go_over_data_vis(
    newh5, oldh5, rotangles, tempnames, params, roi=None, no_output=False, timed=False
):
    rotangles, tempnames = assure_multiple(rotangles, tempnames)
    
    (
        anms_threshold,
        ksize_anms,
        ksize_erodil,
        line,
        kernel,
        iterations,
    ) = params


    with h5py.File(newh5, "w") as res, h5py.File(oldh5, "r") as hf:

        imshape = hf[tempnames[0] + "/imgs"][0].shape

        dummy = np.ones(imshape)
        masks = []
        for i in range(len(rotangles)):
            drot, log = img_rotate_bound(dummy, rotangles[i], bm=0)
            newmask = make_mask(drot, 2)
            masks.append(newmask)

        for j in range(len(tempnames)):

            print(tempnames[j])
            print("________________________________")
            if timed:
                time_now = time.time()

            times = hf[tempnames[j] + "/time"][()]

            if roi is None:
                iroi = np.arange(len(times))
            else:
                iroi = roi
                times=times[iroi]

            time0 = times[0]

            for ccounter in range(len(rotangles)):
                name = "/check" + str(ccounter)
                h5_images_dataset(res, tempnames[j] + name, [len(iroi), *imshape])

            res.create_dataset(tempnames[j] + "/time", shape=(len(iroi)), dtype="f")

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

                    checkmaps = line_process_vis(
                        copt,
                        rotangles,
                        masks,
                        line=line,
                        anms_threshold=anms_threshold,
                        ksize_anms=ksize_anms,
                        ksize_erodil=ksize_erodil,
                        iterations=iterations
                    )

                res[tempnames[j] + "/time"][i] = times[i] - time0
                for ccounter in range(len(rotangles)):
                    name = "/check" + str(ccounter)
                    cm = checkmaps[ccounter] / np.max(checkmaps[ccounter]) * 255
                    res[tempnames[j] + name][i] = cm.astype(np.uint8)
    return checkmaps

#%% go_over_data

def h5_go_over_data(
    newh5, oldh5, rotangles, tempnames, params, roi=None, no_output=False, timed=False
):
    rotangles, tempnames = assure_multiple(rotangles, tempnames)
    
    (
        anms_threshold,
        ksize_anms,
        ksize_erodil,
        line,
        db_dist,
        kernel,
        iterations,
        ksize_smooth,
    ) = params


    with h5py.File(newh5, "w") as res, h5py.File(oldh5, "r") as hf:

        imshape = hf[tempnames[0] + "/imgs"][0].shape
        dummy = np.ones(imshape)
        masks = []
        for i in range(len(rotangles)):
            drot, log = img_rotate_bound(dummy, rotangles[i], bm=0)
            newmask = make_mask(drot, 2)
            masks.append(newmask)

        for j in range(len(tempnames)):

            print(tempnames[j])
            print("________________________________")
            if timed:
                time_now = time.time()

            times = hf[tempnames[j] + "/time"][()]

            if roi is None:
                iroi = np.arange(len(times))
            else:
                iroi = roi
                times=times[iroi]

            time0 = times[0]

            for ccounter in range(len(rotangles)):
                name = "/check" + str(ccounter)
                h5_images_dataset(res, tempnames[j] + name, [len(iroi), *imshape])
                
            res.create_dataset(tempnames[j] + "/time", shape=(len(iroi)), dtype="f")

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


                    checkmaps = line_process_partial(
                        copt,
                        rotangles,
                        masks,
                        line=line,
                        anms_threshold=anms_threshold,
                        ksize_anms=ksize_anms,
                        ksize_erodil=ksize_erodil,
                        db_dist=db_dist,
                        iterations=iterations,
                        ksize_smooth=ksize_smooth,
                    )

                res[tempnames[j] + "/time"][i] = times[i] - time0
                for ccounter in range(len(rotangles)):
                    name = "/check" + str(ccounter)
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

            times = hf[tempnames[j] + "/time"][()]
            res[tempnames[j] + "/time"] = times



            if roi is None:
                iroi = np.arange(len(times))
            else:
                iroi = roi
                times = times[iroi]

            cdi = 0

            for ccounter in range(len(rotangles)):
                name = "/check" + str(ccounter)
                res.create_dataset(
                    tempnames[j] + name,
                    shape=[len(iroi), imshape[0], imshape[1]],
                    dtype=np.uint8,
                    chunks=(1, imshape[0], imshape[1]),
                    compression="gzip",
                    compression_opts=2,
                )

            res.create_dataset(
                tempnames[j]+"/ref",
                shape=[len(iroi), imshape[0], imshape[1]],
                dtype=np.uint8,
                chunks=(1, imshape[0], imshape[1]),
                compression="gzip",
                compression_opts=2,
            )


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
                        name = "/check" + str(ccounter) 
                        cm = hf[tempnames[j]+name][i]
                        alignedmaps[ccounter] = align_image_fast1(
                            cm, matrix, reswidth, resheight
                        )

                res[tempnames[j]+"/ref"][i] = refim
                for ccounter in range(len(rotangles)):
                    name = "/check" + str(ccounter) 
                    res[tempnames[j]+name][i] = alignedmaps[ccounter].astype(
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

            #check ####
            ccounter=0

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


#%% thresholding
def h5_thresholding(newh5, oldh5, rotangles, tempnames, params, roi=None):
    houghdist, Hthreshold, Hminlength, Hmaxgap, deg_tol, im2 = params
    with h5py.File(newh5, "w") as res, h5py.File(oldh5, "r") as hf:
        
        if roi is None:
            testindex = 0
        else:
            testindex = roi[0]
            
        imshape = hf[tempnames[0] + "/imgs"][testindex].shape

        for j in range(len(tempnames)):

            times = hf[tempnames[j] + "/time"][()]
            res[tempnames[j] + "/time"] = times
            ### check
            inames=[""]
            iroi=[""]
            
            name = "check0_" + inames[j]
            img = hf[name][0]
            imshape = img.shape

            for ccounter in range(len(rotangles)):
                name = "/check" + str(ccounter)
                res.create_dataset(
                    tempnames[j] + name,
                    shape=[len(iroi), imshape[0], imshape[1]],
                    dtype=np.uint8,
                    chunks=(1, imshape[0], imshape[1]),
                    compression="gzip",
                    compression_opts=2,
                )


            if roi is None:
                iroi = np.arange(len(times))
            else:
                iroi = roi

            for i in iroi:
                pass
                #print(k)
                # if k==1:
                #    break

                srbs = []
                for ccounter in range(len(rotangles)):
                    name = "/check" + str(ccounter) 

                    workimg = np.zeros(imshape, dtype=np.double)
                    workimg += hf[name][i]
                    workimg *= im2

                    (thresh, srb) = cv2.threshold(
                        hf[name][i], 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
                    )
                    
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

                    res[name][i] = newimg


    return srb, newlines

                    