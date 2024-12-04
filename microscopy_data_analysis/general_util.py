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
from skimage.draw import disk

import matplotlib.pyplot as plt

from .image_processing import img_make_square
import os



from ipywidgets import widgets
from matplotlib.ticker import MultipleLocator
import matplotlib as mpl
import ipywidgets
import json
#from IPython import display


def _cmap_from_setpoints(cmapname,setpoints):
    cm = mpl.colormaps[cmapname]
    n=len(setpoints)
    cvals=np.linspace(0,1,n)
    mean_diff=cvals[1]-cvals[0]
    colors_per_set=int(np.round(512/(n-1)))
    
    number_of_colors=int(np.round((setpoints[1]-setpoints[0])/mean_diff * colors_per_set))
    newcolors=cm(np.linspace(cvals[0],cvals[1],number_of_colors))
    for i in range(1,n-1):
        number_of_colors=int(np.round((setpoints[i+1]-setpoints[i])/mean_diff * colors_per_set))
        appendcolors=cm(np.linspace(cvals[i],cvals[i+1],number_of_colors))
        newcolors=np.vstack((newcolors,appendcolors))
    return mpl.colors.ListedColormap(newcolors)

def custom_colormap(img,colormap_name,number_of_colorsliders=3):
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    
    setpoints=np.linspace(0,1,number_of_colorsliders+2)
    nc=len(setpoints)
    
    val,bins=np.histogram(np.ravel(img),256)
    contrastmin=np.min(img)
    contrastmax=np.max(img)
    contrastrange=contrastmax-contrastmin
    
    #readoutbool=True
    wmin=0
    wmax=1
    wstep=0.01
    wreadout=True
    wupdate=True#False    
    sliders=[]
    arg_slider_dict={}
    for i in range(nc):
        sliders.append(widgets.FloatSlider(value=setpoints[i],min=wmin,max=wmax,step=wstep,readout=wreadout,continuous_update=wupdate))
        arg_slider_dict["s"+str(i)]=sliders[-1]
    links=[]
    for i in range(nc-1):
        links.append(ipywidgets.link((sliders[i], 'value'), (sliders[i+1], 'min')))            
    ui = widgets.HBox(sliders)

    def plot(*args):
        svals=list(args)
        fig,axs=plt.subplots(nrows=2,ncols=1,figsize=[8,10])
        axs[0].plot(-.6-2*val/np.max(val),c='k')
        axs[0].set_yticks([])
        xtics=np.linspace(0,256,11)
        xlabs=np.round(np.linspace(0,1,11),2)
        axs[0].set_xticks(xtics,xlabs)
        axs[0].xaxis.set_minor_locator(MultipleLocator(256/100))
        axs[0].set_xlim([-1,256.5])
        secax = axs[0].secondary_xaxis('top')
        thirdax=axs[0].secondary_xaxis('bottom')
        thirdax.set_xticks(np.linspace(0,256,11)+25.6/2,[])
        secax.set_xticks([0,256],[np.round(contrastmin,3),np.round(contrastmax,3)])
        newcmp=_cmap_from_setpoints(colormap_name,svals)
        vmin=contrastmin + svals[0] * contrastrange
        vmax=contrastmin + svals[-1] * contrastrange
        
        axs[1].imshow(img,cmap=newcmp,vmin=vmin,vmax=vmax)        
        axs[0].imshow(gradient,aspect=10,cmap=newcmp,extent=(svals[0]*256,svals[-1]*256,1.5,-.5))
  
        for i in svals:
            axs[0].plot([i*256,i*256],(-0.5,2/3-0.5),c='w')
            axs[0].plot([i*256,i*256],(1.5-2/3,1.5),c='k')
        
        cmap_dict["colormap"]=newcmp
        cmap_dict["vmin"]=vmin
        cmap_dict["vmax"]=vmax
        cmap_dict["setpoints"]=svals
        
        plt.tight_layout()
        plt.show()
    
    cmap_dict={}
    cmap_dict["name"]=colormap_name
    
    if number_of_colorsliders==1:
        def meta_plot(s0,s1,s2):
            plot(s0,s1,s2)
    elif number_of_colorsliders==2:
        def meta_plot(s0,s1,s2,s3):
            plot(s0,s1,s2,s3)
    elif number_of_colorsliders==3:
        def meta_plot(s0,s1,s2,s3,s4):
            plot(s0,s1,s2,s3,s4)
    elif number_of_colorsliders==4:
        def meta_plot(s0,s1,s2,s3,s4,s5):
            plot(s0,s1,s2,s3,s4,s5)
    elif number_of_colorsliders==5:
        def meta_plot(s0,s1,s2,s3,s4,s5,s6):
            plot(s0,s1,s2,s3,s4,s5,s6)
    else:
        print("number of colorsliders must be smaller or equal 5")

    out = widgets.interactive_output(meta_plot, arg_slider_dict)
    display(ui, out)
    out.layout.height = '800px'

    return cmap_dict#return_dict["colormap"],return_dict["vmin"],return_dict["vmax"]

def save_cmap(filename,cmap_dict):
    save_dict={}
    save_dict["name"]=cmap_dict["name"]
    save_dict["setpoints"]=cmap_dict["setpoints"]
    save_dict["vmin"]=cmap_dict["vmin"]
    save_dict["vmax"]=cmap_dict["vmax"]

    with open(filename, 'w') as fp:
        json.dump(save_dict,fp)
    print("saved")

def load_cmap(filename):
    with open(filename, 'r') as fp:
        cmap_dict= json.load(fp)
    cmap_dict["colormap"]=_cmap_from_setpoints(cmap_dict["name"],cmap_dict["setpoints"])
    return cmap_dict


#%% dashed line in image

def draw_dashed_line(img,start_point,end_point,color,thickness=1,segment_length=15,gap_length=10):
    """
    draw a dashed line on a 2D rasterized image

    Args:
        img (array_like): DESCRIPTION.
        start_point (tuple or array_like): DESCRIPTION.
        end_point (tuple or array_like): DESCRIPTION.
        color (3 tupel): ( r , g, b ).
        thickness (int, optional): DESCRIPTION. Defaults to 1.
        segment_length (int, optional): DESCRIPTION. Defaults to 15.
        gap_length (int, optional): DESCRIPTION. Defaults to 10.

    Returns:
        None.

    """
    startv=np.array(start_point)
    endv=np.array(end_point)
    line_length=np.sqrt(np.sum((endv-startv)**2))
    unit_step=(endv-startv) / line_length
    
    current = np.copy(startv)
    length_counter=0
    while length_counter <line_length:
        if line_length-length_counter<segment_length:
            end=end_point
        else:
            end = current + segment_length*unit_step
        
        cv2.line(img, np.round(current).astype(int), np.round(end).astype(int), color, thickness)
        current = end + gap_length*unit_step
        length_counter += segment_length + gap_length


#%% bins for histograms
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
    stitch two 1d-signals (x1,y1 and x2,y2) containing some overlap in x.
    x1 and x2 must be uniformly spaced and in increasing order.
    within the overlap region the finer resolution in x is kept
    (any interpolation in this function is done linearly).
    
    adjust the scale of y1 and y2 such that they yield the same mean value
    in the overlap region, by multiplying both signals with a fixed factor
    
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
    overlapmean1=np.mean(overlap1)
    overlapmean2=np.mean(overlap2)
    omean=(overlapmean1+overlapmean2)/2
    
    factor1=omean/overlapmean1
    factor2=omean/overlapmean2
    #factor1=np.mean(overlap2)/np.mean(overlap1)
    if not scale_adjustment:
        factor1=1
        factor2=1
        
    newy1*= factor1
    newy2*= factor2
    if verbose:
        if switched:
            print("scale factor adjusting y1 is "+str(factor2))
            print("scale factor adjusting y2 is "+str(factor1))

        else:
            print("scale factor adjusting y1 is "+str(factor1))
            print("scale factor adjusting y2 is "+str(factor2))
            
    # actually execute the weighted sum 
    new_y=newy1*new_y_weights1 + newy2*new_y_weights2
    new_y /= (new_y_weights1+new_y_weights2)
        
    return new_x,new_y
    

#%% path (get_files_) functions
def get_files_of_format(path,ending):
    """
    searches files with the given ending within the directory

    Args:
        path (string): 
            relative or absolute path to a directory.
        
        ending (string): 
            typical usecase: ".png" to get all png-images.

    Returns:
        pathlist (list): 
            list of the paths of files with the specific ending

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


def get_all_files(folder='.',ending=None,start=None):
    """
    searches recursively (including all subdirectories) 
    for all files with the given start and ending 
    within the given folder

    Args:
        folder (string, optional): 
            directory name. Defaults to '.'.
        
        ending (string, optional): 
            typical usecase: ".png" to get all png-images. 
            Defaults to None.
        
        start (string, optional): 
            pattern in the beginning of each filename. 
            for example use "img" here: "img1.png,img2.tif,img3.jpeg" 
            Defaults to None.
        
    Returns:
        filepaths (list): 
            list of paths.

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


def folder_file(path_string):
    """
    split a string presenting a path with the filename into the filename and the 
    directory-path. (works with slash, backslash or double backslash as seperator)

    Args:
        path_string (string): 
            absolute or relative path.

    Returns:
        directory_path (string): 
            folder.
        
        filename (string): 
            file.

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
        \*x (TYPE): 
            DESCRIPTION.
        
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


#%% circular masks 


def circle_perimeter_points(row, col, r, image,accurate=False):
    if isinstance(row,float) or isinstance(col,float) or isinstance(r,float):
        accurate=True
    if accurate:
        mask=make_circular_mask(row, col, r, image)
        kernel=np.zeros([3,3],dtype=np.uint8)
        kernel[:,1]=1
        kernel[1,:]=1
        res=cv2.dilate(mask,kernel)-mask
        rr,cc=np.where(res==1)
    else:
        rr, cc=circle_perimeter(row,col, r,shape=image.shape)
    return np.array((rr,cc))



def make_circular_mask(row, col, r, image=None,imshape=None):
    """
    create a circular mask, with row , col and r either being integers or float


    Args:
        row (TYPE): DESCRIPTION.
        col (TYPE): DESCRIPTION.
        r (TYPE): DESCRIPTION.
        image (TYPE, optional): DESCRIPTION. Defaults to None.
        imshape (TYPE, optional): DESCRIPTION. Defaults to None.

    Raises:
        ValueError: DESCRIPTION.

    Returns:
        mask (TYPE): DESCRIPTION.

    """
    if imshape is None:
        if image is None:
            raise ValueError("either image and imshape must be given")
        else:
            imshape=image.shape
        
    mask = np.zeros(imshape)
    ind=disk((row, col), r,shape=imshape)
    mask[ind]=1
    return mask

#%% fftmasks

def rfft_circ_mask(imshape, mask_radius=680, mask_sigma=0):
    rfftimshape=(imshape[0],int((imshape[1]+2)/2))

    mask = make_circular_mask(
        int(imshape[0] / 2), 0, mask_radius, imshape=rfftimshape
    )
    if mask_sigma==0:
        pass
    else:
        kernel_size = 6 * mask_sigma +1

        mask = cv2.GaussianBlur(
            mask, [kernel_size, kernel_size], mask_sigma)
        mask /= np.max(mask)

    mask = np.roll(mask, -int(mask.shape[0] / 2), axis=0)
    return mask

def fft_circ_mask(imshape, mask_radius=680, mask_sigma=0):
    mask = make_circular_mask(
        int(imshape[0] / 2), int(imshape[1] / 2), mask_radius, imshape)
    if mask_sigma==0:
        pass
    else:
        kernel_size = 6 * mask_sigma +1
        mask = cv2.GaussianBlur(
            mask, [kernel_size, kernel_size], mask_sigma )
        mask /= np.max(mask)

    return mask


def rfft_to_fft(rfft_img,fullshape):
    fullfft=np.zeros(fullshape)
    fullfft[:,:rfft_img.shape[1]]=rfft_img[::-1,::-1]
    fullfft[:,-rfft_img.shape[1]:]=rfft_img
    return fullfft        


def rfft_starmask(angles,imshape,mask_sigma=0):
    

    # transform real space angles to reciprocal space radians
    recanglerads=(angles-90)/180*np.pi
    
    rfftimshape=(imshape[0],int((imshape[1]+2)/2))
    maskshift=np.zeros(rfftimshape)
    center=np.array([int(imshape[0]/2),0])
    
    bounding_points_0=[0 for i in range(rfftimshape[1])]
    bounding_points_0+=[i for i in range(rfftimshape[0])]
    bounding_points_0+=[rfftimshape[0]-1 for i in range(rfftimshape[1])]
    
    bounding_points_1=[i for i in range(rfftimshape[1])]
    bounding_points_1+=[rfftimshape[1]-1 for i in range(rfftimshape[0])]
    bounding_points_1+=[i for i in range(rfftimshape[1]-1,-1,-1)]
    
    bps=np.array((bounding_points_0,bounding_points_1))

    anglerads=-np.arctan2((bps[0]-center[0]).astype(np.double),bps[1].astype(np.double))

    for i in range(len(recanglerads)):
        pointind=np.argmin(np.abs(anglerads-recanglerads[i]))
                
        rp,cp=bps.T[pointind]
        rr,cc,vv=line_aa(center[0],center[1],rp,cp)
        maskshift[rr,cc]=1   

    if mask_sigma==0:
        pass
    else:
        kernel_size = 6 * mask_sigma +1
        maskshift = cv2.GaussianBlur(
        maskshift, [kernel_size, kernel_size], mask_sigma )
        maskshift /= np.max(maskshift)

    mask=np.roll(maskshift, -int(imshape[0] / 2), axis=0)
    return mask
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


#%% geometrical functioins
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
    angles measured in degrees starting from horizontal line counterclockwise (real space)
    (like phi in polar coordinate)  

    Args:
        image (TYPE): 
            DESCRIPTION.
        
        borderdist (TYPE, optional): 
            DESCRIPTION. 
            Defaults to 100.
        
        centerdist (TYPE, optional): 
            DESCRIPTION. 
            Defaults to 20.
        
        plotcheck (TYPE, optional): 
            DESCRIPTION. 
            Defaults to False.

    Returns:
        angledeg (TYPE): 
            DESCRIPTION.
        
        values (TYPE): 
            DESCRIPTION.

    """
    if image.shape[0] != image.shape[1]:
        print("Warning: image is cropped to square")
        img = img_make_square(image)
    else:
        img=image

    fftimage = np.fft.rfft2(img)
    halfheight = int(img.shape[0] / 2)
    rffti = np.roll(fftimage, halfheight, axis=0)
    fim = np.log(np.abs(rffti))
    radius = halfheight - borderdist
    center = np.array([halfheight, 0], dtype=int)

    rcirc, ccirc = circle_perimeter(*center, radius, shape=fim.shape)

    checkcirc = np.array([rcirc,ccirc]).T

    dy,dx =rcirc-halfheight,ccirc
    angles= - np.arctan2(dy, dx)
    checkcenter= center + centerdist * np.array([dy, dx]).T / radius
    checkcenter=np.round(checkcenter).astype(int)

    values = np.zeros(len(rcirc))    
    for i in range(len(rcirc)):
        rr, cc, vv = line_aa(*checkcenter[i], *checkcirc[i])
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



#%% smoothbox_kernel

def _pascal_numbers(n):
    """Returns the n-th row of Pascal's triangle"""
    return scipy.special.comb(n, np.arange(n + 1))


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



#%% make border Mask


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




#%% homogeneous image threshold

def determine_thresh(image):
    Nrows, Ncols = image.shape
    nr = Nrows // 3
    nc = Ncols // 3
    roi = image[nr : 2 * nr, nc : 2 * nc]
    i_mean, i_std = np.mean(roi), np.std(roi)
    thresh = i_mean - 2.575829 * i_std
    return thresh


