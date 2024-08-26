# -*- coding: utf-8 -*-
"""
submodule covering some 1D, 2D, mixed and other functions
"""

import numpy as np
import cv2
#import copy
from numba import njit
import scipy.special

from skimage.draw import circle_perimeter
from skimage.draw import line_aa
import matplotlib.pyplot as plt

from .image_processing import img_make_square
import os

#%%
def bin_centering(x_bins,additional_boundary_bin_threshold=None):
    """
    transform bin thresholds to x values

    Args:
        x_bins (list or array_like): with length N, in strictly increasing order.
        
        additional_boundary_bin_threshold (float, optional): option to give an additional bin,
        in case the given values, represent only lower or upper bin boundaries. Defaults to None.

    Returns:
        centers (array_like): with length N-1 or length N when an additional bin is given.

    """
    if not isinstance(x_bins,list):
        x_bins=x_bins.tolist()

    if additional_boundary_bin_threshold is not None:    
        if additional_boundary_bin_threshold>np.max(x_bins):
            x_bins.append(additional_boundary_bin_threshold)
        elif additional_boundary_bin_threshold<np.min(x_bins): 
            x_bins.insert(0,additional_boundary_bin_threshold)
        else:
            raise ValueError("additional_boundary_bin_threshold within the range of x_bins")
        
    bin_diff=np.diff(x_bins)
    if np.min(bin_diff)<=0:
        raise ValueError("input x_bins is not strictly increasing")
    
    centers=x_bins[:-1]+bin_diff/2

    return centers

def create_bins(x):
    """
    create bin thresholds between points of x,
    always in the middle between two points, 
    with mirroring boundary conditions

    Args:
        x (list or array_like): with length N, in strictly increasing order.

    Returns:
        bins (list): with length N+1.

    """
    if np.min(np.diff(x))<=0:
        raise ValueError("input x is not strictly increasing")
    
    bins=[]
    for i in range(len(x)-1):
        bins.append(0.5*(x[i]+x[i+1]))
    startdiff=bins[1]-bins[0]
    enddiff=bins[-1]-bins[-2]
    bins=[bins[0]-startdiff]+bins
    bins.append(bins[-1]+enddiff)
    return bins


#%% stitch any two curves with overlap and equidistant sampling

def stitch_1d_overlap(x1,y1,x2,y2,scale_adjustment=True,newbins=False,verbose=False):
    """
    stitch two 1d-signals (x1,y1 and x2,y2) containing some overlap in x,
    x1 and x2 must be uniformly spaced and in increasing order
    within the overlap region the finer resolution in x is kept
    
    adjusting the scale for a smooth transition from y1 to y2 
    happens towards data with finer resolution, by using the mean within the overlap region
    (any interpolation in this function is done linearly)
    
    (typical use case: two spectroscopic measurements with different settings,
    yielding two spectra of different wavelength-regions with some overlap)

    Args:
        x1 (list or array_like): same length as y1.
        
        y1 (list or array_like): same length as x1.
        
        x2 (list or array_like): same length as y2.
        
        y2 (list or array_like): same length as x2.
                
        scale_adjustment (bool, optional): turn multiplicative adjustement on (True) or off (False). 
        Defaults to True.
        
        verbose (bool, optional): prints out the scale_adjustment factor, if set to True. 
        Defaults to False.
        
    Returns:
        new_x (array_like): stitched signal x.
        
        new_y (array_like): stitched signal y.

    """
    
    if len(x1) != len(y1) or len(x2) != len(y2):
        print("Corresponding wavelengths and spectrum need to have identical shape.")
        print("In case of bins choose either the lower or upper bound and use 'up' or 'down', respectively,")
        print("for the argument 'wavelength_bin_direction'.")
    
    # prepare data
    if isinstance(x1,list):
        x1=np.array(x1)
    if isinstance(x2,list):
        x2=np.array(x2)
    if isinstance(y1,list):
        y1=np.array(y1)    
    if isinstance(y2,list):
        y2=np.array(y2)

    delta_x1=(x1[1]-x1[0])/2
    delta_x2=(x2[1]-x2[0])/2

    # make sure the stepsize of y1 is greater or equal than the stepsize of y2
    switched=False
    if delta_x1<delta_x2:
        delta_x1,delta_x2=delta_x2,delta_x1
        y1,y2=y2,y1
        x1,x2=x2,x1
        switched=True

    max_overlap_distance=delta_x1+delta_x2
    full_overlap_distance=delta_x1-delta_x2
    
    
    overlap_indicator1=np.zeros(len(x1))
    
    #determine points in the overlapping region
    overlap_neighbours1=[[] for i in x1]
    overlap_neighbours2=[[] for i in x2]
    #give weights for interpolation
    weights1=[[] for i in x1]
    weights2=[[] for i in x2]
    #weights are given corresponding to the width of a datapoint in x
    ratio=delta_x2/delta_x1
    
    
    for index1,value1 in enumerate(x1):
        for index2,value2 in enumerate(x2):
            
            distance=np.abs(value2-value1)
            if distance<max_overlap_distance:
                overlap_neighbours1[index1].append(index2)
                overlap_neighbours2[index2].append(index1)
                
                if distance <= full_overlap_distance:
                    weights1[index1].append(ratio)
                    weights2[index2].append(1)
                    overlap_indicator1[index1]+=ratio

                else:
                    weight=(delta_x1-distance+delta_x2)/(2*delta_x2)
                    weights1[index1].append(ratio*weight)
                    weights2[index2].append(weight)
                    overlap_indicator1[index1]+=ratio*weight
    
    #take all points of the signal with finer resolution in x as new points
    new_x=x2.tolist()
    from_x1=np.zeros(len(x2),dtype=bool).tolist()
    x_index=np.arange(len(x2),dtype=int).tolist()
    
    # add all points of the signal with coarse resolution, 
    # if less than half their width in x is covered by the points of finer resolution
    for i in range(len(x1)):
        if overlap_indicator1[i]<0.5:
            new_x.append(x1[i])
            from_x1.append(True)
            x_index.append(i)
  
    # prepare the new points in x (transforming to arrays and sorting)
    new_x=np.array(new_x)
    from_x1=np.array(from_x1)
    x_index=np.array(x_index,dtype=int)
    sortindex=np.argsort(new_x)
    new_x=new_x[sortindex]
    from_x1=from_x1[sortindex]
    x_index=x_index[sortindex]

    
    # initialize variables for calculating the new points in y
    newy1=np.zeros(len(new_x))
    newy2=np.zeros(len(new_x))
    new_y_weights1=np.zeros(len(new_x))
    new_y_weights2=np.zeros(len(new_x))
    
    for i in range(len(new_x)):
        # if a point comes from x1, newy1 is the corresponding value in y1
        # and newy2 is the weighted sum from neighbouring points in y2 (linear interpolation)
        # else it goes the other way around
        if from_x1[i]:
            newy1[i]=y1[x_index[i]]
            new_y_weights1[i]=1
            if len(overlap_neighbours1[x_index[i]])>0:     
                weight=0
                for index,value in enumerate(overlap_neighbours1[x_index[i]]):
                    newy2[i]+=y2[value]*weights1[x_index[i]][index]
                    weight +=weights1[x_index[i]][index]
                newy2[i]/=weight
                new_y_weights2[i]=weight
        else:       
            newy2[i]=y2[x_index[i]]
            new_y_weights2[i]=1
            if len(overlap_neighbours2[x_index[i]])>0:
                weight=0
                for index,value in enumerate(overlap_neighbours2[x_index[i]]):
                    newy1[i]+=y1[value]*weights2[x_index[i]][index]
                    weight +=weights2[x_index[i]][index]
                newy1[i]/=weight
                new_y_weights1[i]=weight

    # produce a mask for the new points that is True within the overlap region and otherwise False
    overlap_region=(new_y_weights1*new_y_weights2)>0
    overlap1=newy1[overlap_region]
    overlap2=newy2[overlap_region]

    # adjust the scale by comparing the mean within the overlap region 
    factor1=np.mean(overlap2)/np.mean(overlap1)
    if not scale_adjustment:
        factor1=1
        
    newy1*= factor1
    if verbose:
        if switched:
            print("scale factor adjusting y2 to y1 is "+str(factor1))
        else:
            print("scale factor adjusting y1 to y2 is "+str(factor1))
            
    # actually execute the weighted sum 
    new_y=newy1*new_y_weights1 + newy2*new_y_weights2
    new_y /= (new_y_weights1+new_y_weights2)
        
    return new_x,new_y
    
#%%


#%% get_files_of_format
def get_files_of_format(path,ending):
    """
    searches files with the given ending within the directory

    Args:
        path (string): relative or absolute path to a directory.
        
        ending (string): typical usecase: ".png" to get all png-images.

    Returns:
        pathlist (list): list of the paths of files with the specific ending

    """
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

def get_all_files(folder='.',ending=None,start=None):
    """
    searches recursively (including all subdirectories) 
    for all files with the given start and ending 
    within the given folder

    Args:
        folder (string, optional): directory name. Defaults to '.'.
        
        ending (string, optional): typical usecase: ".png" to get all png-images. 
        Defaults to None.
        
        start (string, optional): pattern in the beginning of each filename. 
        for example use "img" here: "img1.png,img2.tif,img3.jpeg" 
        Defaults to None.
        
    Returns:
        filepaths (list): list of paths.

    """
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
    """
    split a string presenting a path with the filename into the filename and the 
    directory-path. (works with slash, backslash or double backslash as seperator)

    Args:
        path_string (string): absolute or relative path.

    Returns:
        directory_path (string): folder.
        
        filename (string): file.

    """
    
    name = path_string.replace("\\", "/")
    pos = name[::-1].find("/")
    return name[:-pos], name[-pos:]

#%% assure_multiple
def assure_multiple(*x):
    """
    pass through one or more variables and check, if they are effectively iterable,
    meaning the format supports iteration and the variable contains more than 
    one element. If just one element is present, this element is extracted from
    its iterable container.

    Args:
        \*x (TYPE): DESCRIPTION.
        
    Returns:
        TYPE: DESCRIPTION.

    """
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

    Args:
        image (TYPE): DESCRIPTION.
        
        borderdist (TYPE, optional): DESCRIPTION. 
        Defaults to 100.
        
        centerdist (TYPE, optional): DESCRIPTION. 
        Defaults to 20.
        
        plotcheck (TYPE, optional): DESCRIPTION. 
        Defaults to False.

    Returns:
        angledeg (TYPE): DESCRIPTION.
        
        values (TYPE): DESCRIPTION.

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
    r2=r**2
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if (i - x0) ** 2 + (j - y0) ** 2 < r2:
                mask[i, j] = 1
    return mask


def make_circular_mask(x0, y0, r, image):
    if x0 % 2 != 0 or y0 % 2 != 0:
        return circ_mask(x0, y0, r, image)
    else:
        mask = np.zeros(np.shape(image))
        return cv2.circle(mask, [x0, y0], r, 1, -1)


#%% get_max deprecated
#def get_max(z):
#    "returns the indices of the maximum value of an array MxN"
#    maxpos = np.argmax(z)
#    x0 = maxpos // z.shape[1]
#    y0 = maxpos % z.shape[1]
#    return np.array([x0, y0]).astype(int)


#%% pascal triangle
def _pascal_numbers(n):
    """Returns the n-th row of Pascal's triangle"""
    return scipy.special.comb(n, np.arange(n + 1))


#%% smoothbox_kernel
def smoothbox_kernel(kernel_size):
    """Gaussian Smoothing kernel approximated by integer-values obtained via binomial distribution"""
    r = kernel_size[0]
    c = kernel_size[1]
    sb = np.zeros([r, c])

    row = _pascal_numbers(r - 1)
    col = _pascal_numbers(c - 1)

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


