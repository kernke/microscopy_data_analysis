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
import os

import ncempy.io as nio
from contextlib import redirect_stdout
import io
#%% stitch any two curves with overlap and equidistant sampling

def stitch_overlap(wavelengths1,spectrum1,wavelengths2,spectrum2,wavelength_bin_direction="center",
                   scale_adjustment=True,newbins=False):
    """
    stitch two spectra with overlapping regions including scale adjustment between the two curves
    the two spectra need to have uniformly spaced wavelengths in increasing order 
    in the overlap region a mean of the two spectra (after scale adjustment) is used
    within the overlap region the finer wavelength resolution is kept
    (scale adjustment adjusts towards the finer resolution data)
    (any interpolation is done linearly)
    """
    
    if len(wavelengths1) != len(spectrum1) or len(wavelengths2) != len(spectrum2):
        print("Corresponding wavelengths and spectrum need to have identical shape.")
        print("In case of bins choose either the lower or upper bound and use 'up' or 'down', respectively,")
        print("for the argument 'wavelength_bin_direction'.")
    
    # prepare data
    wavelengths1=np.array(wavelengths1)
    wavelengths2=np.array(wavelengths2)
    spectrum1=np.array(spectrum1)
    spectrum2=np.array(spectrum2)

    delta_wavelengths1=(wavelengths1[1]-wavelengths1[0])/2
    delta_wavelengths2=(wavelengths2[1]-wavelengths2[0])/2
    if wavelength_bin_direction=="up":
        wavelengthcenters1=wavelengths1+delta_wavelengths1
        wavelengthcenters2=wavelengths2+delta_wavelengths2
    elif wavelength_bin_direction=="down":
        wavelengthcenters1=wavelengths1-delta_wavelengths1
        wavelengthcenters2=wavelengths2-delta_wavelengths2
    elif wavelength_bin_direction=="center":
        wavelengthcenters1=wavelengths1
        wavelengthcenters2=wavelengths2

    # make sure the stepsize of spectrum1 is greater or equal than the stepsize of spectrum2
    if delta_wavelengths1<delta_wavelengths2:
        delta_wavelengths1,delta_wavelengths2=delta_wavelengths2,delta_wavelengths1
        spectrum1,spectrum2=spectrum2,spectrum1
        wavelengthcenters1,wavelengthcenters2=wavelengthcenters2,wavelengthcenters1

    max_overlap_distance=delta_wavelengths1+delta_wavelengths2
    full_overlap_distance=delta_wavelengths1-delta_wavelengths2
    ratio=delta_wavelengths2/delta_wavelengths1
    
    overlap_indicator1=np.zeros(len(wavelengthcenters1))
    
    overlap_neighbours1=[[] for i in wavelengthcenters1]
    overlap_neighbours2=[[] for i in wavelengthcenters2]
    weights1=[[] for i in wavelengthcenters1]
    weights2=[[] for i in wavelengthcenters2]
    
    for index1,value1 in enumerate(wavelengthcenters1):
        for index2,value2 in enumerate(wavelengthcenters2):
            
            distance=np.abs(value2-value1)
            if distance<max_overlap_distance:
                overlap_neighbours1[index1].append(index2)
                overlap_neighbours2[index2].append(index1)
                
                if distance <= full_overlap_distance:
                    weights1[index1].append(ratio)
                    weights2[index2].append(1)
                    overlap_indicator1[index1]+=ratio

                else:
                    weight=(delta_wavelengths1-distance+delta_wavelengths2)/(2*delta_wavelengths2)
                    weights1[index1].append(ratio*weight)
                    weights2[index2].append(weight)
                    overlap_indicator1[index1]+=ratio*weight
    
    new_wavelengths=wavelengthcenters2.tolist()
    from_wavelengths1=np.zeros(len(wavelengthcenters2),dtype=bool).tolist()
    wavelengths_index=np.arange(len(wavelengthcenters2),dtype=int).tolist()
    
    for i in range(len(wavelengthcenters1)):
        if overlap_indicator1[i]<0.5:
            new_wavelengths.append(wavelengthcenters1[i])
            from_wavelengths1.append(True)
            wavelengths_index.append(i)
            
    new_wavelengths=np.array(new_wavelengths)
    from_wavelengths1=np.array(from_wavelengths1)
    wavelengths_index=np.array(wavelengths_index,dtype=int)
    sortindex=np.argsort(new_wavelengths)
    new_wavelengths=new_wavelengths[sortindex]
    from_wavelengths1=from_wavelengths1[sortindex]
    wavelengths_index=wavelengths_index[sortindex]

    
    newspectrum1=np.zeros(len(new_wavelengths))
    newspectrum2=np.zeros(len(new_wavelengths))
    newspectrum_weights1=np.zeros(len(new_wavelengths))
    newspectrum_weights2=np.zeros(len(new_wavelengths))
    
    for i in range(len(new_wavelengths)):
        if from_wavelengths1[i]:
            newspectrum1[i]=spectrum1[wavelengths_index[i]]
            newspectrum_weights1[i]=1
            if len(overlap_neighbours1[wavelengths_index[i]])>0:     
                weight=0
                for index,value in enumerate(overlap_neighbours1[wavelengths_index[i]]):
                    newspectrum2[i]+=spectrum2[value]*weights1[wavelengths_index[i]][index]
                    weight +=weights1[wavelengths_index[i]][index]
                newspectrum2[i]/=weight
                newspectrum_weights2[i]=weight
        else:       
            newspectrum2[i]=spectrum2[wavelengths_index[i]]
            newspectrum_weights2[i]=1
            if len(overlap_neighbours2[wavelengths_index[i]])>0:
                weight=0
                for index,value in enumerate(overlap_neighbours2[wavelengths_index[i]]):
                    newspectrum1[i]+=spectrum1[value]*weights2[wavelengths_index[i]][index]
                    weight +=weights2[wavelengths_index[i]][index]
                newspectrum1[i]/=weight
                newspectrum_weights1[i]=weight

    overlap_region=(newspectrum_weights1*newspectrum_weights2)>0
    overlap1=newspectrum1[overlap_region]
    overlap2=newspectrum2[overlap_region]

    factor1=np.mean(overlap2)/np.mean(overlap1)
    if scale_adjustment:
        newspectrum1*= factor1

    newspectrum=newspectrum1*newspectrum_weights1 + newspectrum2*newspectrum_weights2
    newspectrum /= (newspectrum_weights1+newspectrum_weights2)
    
    if newbins:
        nbins=[]
        for i in range(len(new_wavelengths)-1):
            nbins.append(0.5*(new_wavelengths[i]+new_wavelengths[i+1]))
        startdiff=nbins[1]-nbins[0]
        enddiff=nbins[-1]-nbins[-2]
        nbins=[nbins[0]-startdiff]+nbins
        nbins.append(nbins[-1]+enddiff)
        return new_wavelengths,newspectrum,nbins
        
    return new_wavelengths,newspectrum
    

#%% get_dm4_with_metadata
def get_dm4_with_metadata(filepath):
    """read a dm4-file and return the variables data and metadata as dictionaries
    data includes everything immediately important
    metadata contains all other information
    
    Important missing parameters are  current and for spectra:
    acquisition time and acquisition date are missing
    """


    relevant_metadata_list=["Grating","Objective focus (um)","Stage X","Stage Y","Stage Z","Stage Beta","Stage Alpha","Indicated Magnification",
    "Bandpass","Detector","Filter","PMT HV","Sensitivity","Slit Width","Sample Time","Dwell time (s)","Image Height","Image Width",
     "Number Summing Frames","Voltage","Lightpath","Signal Name","Acquisition Time","Acquisition Date"]    
    
    f = io.StringIO()
    with redirect_stdout(f):
        data=nio.dm.dmReader(filepath,verbose=True)
    all_metadata = f.getvalue()
    
    metadata=dict()

    for i in range(len(relevant_metadata_list)):
        start=all_metadata.find("curTagValue = "+relevant_metadata_list[i])
        end=all_metadata[start:].find("\n")
        if start != -1:
            end+=start
            start+=len("curTagValue = "+relevant_metadata_list[i])+2
        metadata[relevant_metadata_list[i]]=all_metadata[start:end]

    return data,metadata   


#%% get_files_of_format
def get_files_of_format(path,ending):
    files = os.listdir(path)
    desired_format=ending
    pathlist=[]
    
    if path[-1]=="/" or path[-1]=="\\":
        pass
    else:
        path+="/"
    
    for i in range(len(files)):
        if files[i][-len(ending):] == desired_format:
            pathlist.append(path+files[i])
    return pathlist
#%%

def get_all_files(folder,ending=None,start=None):
    filenames=[]#os.listdir(folder)
    for path, subdirs, files in os.walk(folder):
        for name in files:
            condition=False
            if ending is None and start is None: condition=True 
            elif name[-len(ending):]==ending and start is None: condition=True
            elif ending is None and start==name[:len(start)]: condition=True 
            elif name[-len(ending):]==ending and start==name[:len(start)]: condition=True

            if condition: filenames.append(os.path.join(path, name))
    
    return filenames

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
        if hasattr(i, '__iter__'):#len(np.shape(i)) == 0:
            res.append(i)
        else:
            res.append([i])

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

def fft_circ_mask(imshape, mask_radius=680, mask_sigma=50):
    kernel_size = 7 * mask_sigma
    if kernel_size % 2 == 0:
        kernel_size += 1
    mask = make_circular_mask(
        (imshape[0] - 1) / 2, (imshape[1] - 1) / 2, mask_radius, np.zeros(imshape)
    )
    maskb = cv2.GaussianBlur(
        mask.astype(np.double), [kernel_size, kernel_size], mask_sigma
    )

    return maskb



#%% peak_com2d


def peak_com2d(data, delta=None, roi=None):
        
    if delta is None:
        delt=np.zeros(2)+max(data.shape)
    else:
        if isinstance(delta,int):
            delt = np.array([delta, delta]).astype(int)
        else:
            delt = np.array(delta).astype(int)

    if roi is None:
        dat = data
        roi = [[0, data.shape[0]], [0, data.shape[1]]]
    else:
        dat = data[roi[0][0] : roi[0][1], roi[1][0] : roi[1][1]]

    dist = np.argmax(dat)
    disty = dist % dat.shape[1]
    distx = dist // dat.shape[1]

    delt[0] = min(distx, dat.shape[0] - distx,delt[0])
    delt[1] = min(disty, dat.shape[1] - disty,delt[1])
    

    xstart = distx - delt[0]
    xend = distx + delt[0]+1
    ystart = disty - delt[1]
    yend = disty + delt[1]+1

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

    return np.array([xpos, ypos]), np.array([mvposx, mvposy]),delt


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

def point_in_convex_ccw_roi(xroi, yroi, xpoint, ypoint):
    signs = np.zeros(len(xroi))
    for i in range(len(xroi)):
        signs[i] = isleft(xroi[i - 1], yroi[i - 1], xroi[i], yroi[i], xpoint, ypoint)
    return np.sum(np.sign(signs)) == len(xroi)

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





#%% get_angular_dist
def get_angular_dist(image, borderdist=100, centerdist=20, plotcheck=False):
    """
    angles measured in degrees starting from horizontal line counterclockwise 
    (like phi in polar coordinate)    

    Parameters
    ----------
    image : MxN array
    borderdist : int, optional
        DESCRIPTION. The default is 100.
    centerdist : int, optional
        DESCRIPTION. The default is 20.
    plotcheck : bool, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    angles:
    intensity:
    """
    if image.shape[0] != image.shape[1]:
        print("Warning: image is cropped to square")
        img = img_make_square(image)
    else:
        img=image

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
    """Returns the n-th row of Pascal's triangle"""
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


