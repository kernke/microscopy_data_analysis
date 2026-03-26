"""
@author: kernke
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .general_util import max_from_2d, peak_com2d
from .image_processing import (
    img_add_weighted_rgba,
    img_gray_to_rgba,
    img_padding_attenuation,
    img_periodic_tiling,
    img_to_uint16,
)


def close_translation_by_phase_correlation(im1, im2, 
                                            sigma=1, max_transl=None):
    """
    calculate translation vector between images by phase correlation,
    assumes 'closeness' as translations less than half of image dimensions
    
    Args:
        im1 (array_like):
            image 1
        
        im2 (array_like):
            image 2

        sigma (float):
            width of Gaussian smoothing of correlation matrix

        max_transl (tuple):
            maximal translation limit enforced by masking the correlation matrix

    Returns:
        translation_vector (array_like, tuple):
        
        certainty (float):
            normed signal noise ration (max-mean)/std
    """
    # dims=im1.shape

    mat = img_correlation(im1-np.mean(im1), im2-np.mean(im2))
    # matb=cv2.blur(mat,[blur,blur])
    if sigma is None:
        mat0=mat-np.min(mat)
    else:
        ksize = int(sigma * 4)
        if ksize % 2 == 0:
            ksize += 1
        matb = cv2.GaussianBlur(mat, [ksize, ksize], sigma)  # ,cv2.BORDER_WRAP)
        mat0 = matb - np.min(matb)

    dims = mat0.shape

    mean = np.mean(mat0)
    std = np.std(mat0)
    # print(std)
    if max_transl:
        img_mask = np.zeros(dims)
        img_mask[:max_transl[0] + 1, :max_transl[1] + 1] = 1
        img_mask[:max_transl[0] + 1, -max_transl[1]:] = 1
        img_mask[-max_transl[0]:, :max_transl[1] + 1] = 1
        img_mask[-max_transl[0]:, -max_transl[1]:] = 1
        # plt.imshow(img_mask)
    else:
        img_mask = np.ones(dims)

    matm = mat0 * img_mask

    transl, maxval = max_from_2d(matm)
    if transl[0] > dims[0] / 2:
        transl[0] -= dims[0]
    if transl[1] > dims[1] / 2:
        transl[1] -= dims[1]

    certainty = (maxval - mean) / std
    return transl, certainty


#%% phase_correlation
def img_correlation(a, b, padding_ratio=4, pad_mode="mean", pad_attenuation="linear",
                      pad_parameters=None,cross_correlation=False):
    """
    calculate the pase correlation between two images a,b (best with even dimensions)
    with same shape MxN

    Args:
        a (MxN array_like): 
            first image.
        
        b (MxN array_like): 
            second image.

        padding_ratio (float):
            ratio of biggest dimension of image
            Defaults to 4    
        
        pad_mode (string):
            same modes as in numpy.pad
            Defaults to 'mean'
        
        pad_attenuation (string):
            For details see img_padding_attenuation
            Defaults to 'linear'

        pad_parameters (dictionary):
            For details see img_padding_attenuation
            Defaults to None

        cross_correlation (bool):
            set to 'True' for additionally returning cross correlation matrix
            Defaults to 'False'
    Returns:
        phase_r (MxN array_like): 
            phase correlation matrix.

        cross_r (MxN array_like):
            cross correlation matrix
    """
    M,N=np.shape(a)
    padval=int(max(M,N)//padding_ratio)
    statlength=int(np.ceil(padval/2))
    pa=np.pad(a,padval,mode=pad_mode,stat_length=statlength)
    pb=np.pad(b,padval,mode=pad_mode,stat_length=statlength)
    pa=img_padding_attenuation(pa,padval,mode=pad_attenuation,parameters=pad_parameters)
    pb=img_padding_attenuation(pb,padval,mode=pad_attenuation,parameters=pad_parameters)
    if M%2==0 and N%2==0:
        G_a = np.fft.rfft2(pa)
        G_b = np.fft.rfft2(pb)
        conj_b = np.conjugate(G_b)
        R = G_a * conj_b
        if cross_correlation:
            phaseR = R/np.absolute(R)
            cross_r = np.fft.irfft2(R)
            phase_r = np.fft.irfft2(phaseR)
            return phase_r,cross_r  
        else:
            R /= np.absolute(R)
            phase_r = np.fft.irfft2(R)
            return phase_r
    else:
        G_a = np.fft.fft2(pa)
        G_b = np.fft.fft2(pb)
        conj_b = np.conjugate(G_b)
        R = G_a * conj_b
        if cross_correlation:
            phaseR = R/np.absolute(R)
            cross_r = np.fft.ifft2(R).real
            phase_r = np.fft.ifft2(phaseR).real
            return phase_r,cross_r  
        else:
            R /= np.absolute(R)
            phase_r = np.fft.ifft2(R).real
            return phase_r


#%% plain phase correlation
def phase_correlation(a, b):
    """
    calculate the pase correlation between two images a,b
    with same shape MxN

    Args:
        a (MxN array_like): 
            first image.
        
        b (MxN array_like): 
            second image.

    Returns:
        r (MxN array_like): 
            phase correlation matrix.

    """
    G_a = np.fft.rfft2(a)
    G_b = np.fft.rfft2(b)
    conj_b = np.conjugate(G_b)
    R = G_a * conj_b
    R /= np.absolute(R)
    r = np.fft.irfft2(R)
    return r


#%% stitching

def stitch(im1, im2):
    """
    stitch two images together to one, by correcting a translational offset
    The images must have the the same shape (MxN) and some overlap 
    
    Args:
        im1 (MxN array_like): 
            first image.
        
        im2 (MxN array_like): 
            second image.

    Returns:
        stitched (KxL array_like): 
            montage of the two images.

    """

    pc = align(im1, im2)
    
    pcs=im1.shape
    
    sheet = np.zeros(pcs + np.abs(pc))
    sheetdiv = np.zeros(pcs + np.abs(pc))

    im1pos=np.zeros(2,dtype=int)
    im2pos=np.copy(pc)

    for i in range(2):
        if pc[i]<0:
            im1pos[i]=-pc[i]
            im2pos[i]=0

    sheet[im1pos[0] : im1pos[0] + pcs[0], im1pos[1] : im1pos[1] + pcs[1]] += im1
    sheetdiv[im1pos[0] : im1pos[0] + pcs[0], im1pos[1] : im1pos[1] + pcs[1]] += \
                                                                        np.ones(pcs)

    sheet[im2pos[0] : im2pos[0] + pcs[0], im2pos[1] : im2pos[1] + pcs[1]] += im2
    sheetdiv[im2pos[0] : im2pos[0] + pcs[0], im2pos[1] : im2pos[1] + pcs[1]] += \
                                                                        np.ones(pcs)

    sheetdiv[sheetdiv == 0] = -1
    stitched=sheet / sheetdiv
    return stitched




def stitch_given_shift(im1, im2,pc):
    """
    stitch two images together to one, by correcting a translational offset
    The images must have the the same shape (MxN) and some overlap 
    
    Args:
        im1 (MxN array_like): 
            first image.
        
        im2 (MxN array_like): 
            second image.

    Returns:
        stitched (KxL array_like): 
            montage of the two images.

    """
    
    
    sim1=im1.shape
    sim2=im2.shape
    
    sres=np.zeros(2,dtype=int)
    im1pos=np.zeros(2,dtype=int)
    im2pos=np.zeros(2,dtype=int)

    
    for i in range(2):
        if pc[i]>0:
            sres[i]=max(sim1[i],sim2[i]+pc[i])
            im2pos[i]=pc[i]
            #im1pos[i]=0
        else:
            im1pos[i]=-pc[i]
            #im2pos[i]=0
            if sim2[i]+pc[i] > sim1[i]:
                sres[i]=sim2[i]
            else:    
                sres[i]=sim1[i]-pc[i]

        
    sheet = np.zeros(sres)
    sheetdiv = np.zeros(sres)

    sheet[im1pos[0] : im1pos[0] + sim1[0], im1pos[1] : im1pos[1] + sim1[1]] += im1
    sheetdiv[im1pos[0] : im1pos[0] + sim1[0], im1pos[1] : im1pos[1] + sim1[1]] += \
                                                                            np.ones(sim1)

    sheet[im2pos[0] : im2pos[0] + sim2[0], im2pos[1] : im2pos[1] + sim2[1]] += im2
    sheetdiv[im2pos[0] : im2pos[0] + sim2[0], im2pos[1] : im2pos[1] + sim2[1]] += \
                                                                            np.ones(sim2)

    sheetdiv[sheetdiv == 0] = -1
    stitched=sheet / sheetdiv
    return stitched


#%% align

def align(im1, im2,printing=False,_verbose=False):
    """
    calculate the translational offset of image im2 relative to image im1
    using phase correlation between the two image
    The images must have the the same shape (MxN) and some overlap 

    Args:
        im1 (MxN array_like): 
            first image.
        
        im2 (MxN array_like): 
            second image.
        
        printing (bool, optional): 
            set to "True" for printing more information about the function execution. 
            Defaults to False.
        
        verbose (bool,optional): 
            set to "True" for additionally returning indices about the relative
            positioning. Defaults to False.

    Returns:
        offset (tuple): 
            containing two integers.

    """
    pcm = phase_correlation(im1, im2)
    pc,pcval = max_from_2d(pcm)
    pcs = np.array(np.shape(pcm))
    
    optimal_solution=True

    hs = int(np.ceil(pcs[0] / 2))
    ws = int(np.ceil(pcs[1] / 2))
    hs2= int(hs/2)
    ws2= int(ws/2)
    delta_d0=pcs[0]-hs
    delta_d1=pcs[1]-ws

    pcm_a = phase_correlation(im1[hs2:hs2+hs, :], im2[hs2:hs2+hs, :])
    pcm_b = phase_correlation(im1[:hs, :], im2[:hs, :])
    pcm_c = phase_correlation(im1[-hs:, :], im2[-hs:, :])
    pcm_d = phase_correlation(im1[:hs, :], im2[-hs:, :])
    pcm_e = phase_correlation(im1[-hs:, :], im2[:hs, :])
 
    pc0vals=np.zeros(5)   
    pc0s=np.zeros([5,2],dtype=int)
    pc0s[0],pc0vals[0] = max_from_2d(pcm_a)
    pc0s[1],pc0vals[1] = max_from_2d(pcm_b)
    pc0s[2],pc0vals[2] = max_from_2d(pcm_c)
    pc0s[3],pc0vals[3] = max_from_2d(pcm_d)
    pc0s[4],pc0vals[4] = max_from_2d(pcm_e)

    pc0s[4,0]+=delta_d0
    optimal_solution *= pc[0] in pc0s[:,0]
    cond0= pc[0] == pc0s[:,0]   

    if optimal_solution:
        if np.sum(cond0)>1:
            if cond0[0]:
                index0=0
            else:
                index0=np.argmax(pc0vals*cond0)
        else:
            index0=np.argwhere(cond0)[0][0]
    else:
        index0=np.argmax(pc0vals)
    
    pcm_a = phase_correlation(im1[:,ws2:ws2+ws], im2[:,ws2:ws2+ws])
    pcm_b = phase_correlation(im1[:,:ws], im2[:,:ws])
    pcm_c = phase_correlation(im1[:,-ws:], im2[:,-ws:])
    pcm_d = phase_correlation(im1[:,:ws], im2[:,-ws:])
    pcm_e = phase_correlation(im1[:,-ws:], im2[:,:ws])
    
    pc1vals=np.zeros(5)
    pc1s=np.zeros([5,2],dtype=int)
    pc1s[0],pc1vals[0] = max_from_2d(pcm_a)
    pc1s[1],pc1vals[1] = max_from_2d(pcm_b)
    pc1s[2],pc1vals[2] = max_from_2d(pcm_c)
    pc1s[3],pc1vals[3] = max_from_2d(pcm_d)
    pc1s[4],pc1vals[4] = max_from_2d(pcm_e)

    pc1s[4,1]+=delta_d1
    optimal_solution *= pc[1] in pc1s[:,1]
    cond1= pc[1] == pc1s[:,1]
    
    if optimal_solution:
        if np.sum(cond1)>1:
            if cond1[0]:
                index1=0
            else:
                index1=np.argmax(pc1vals*cond1)
        else:
            index1=np.argwhere(cond1)[0][0]
    else:
        index1=np.argmax(pc1vals)
    
    if not optimal_solution:
        print("Warning: optimal solution not found")
        
    if printing:
        print("Maximum position of whole phase correlation matrix:")
        print(pc)
        print("Order of partial correlations (1d):")
        print("(central,central)")
        print("(left,left)")
        print("(right,right)")
        print("(right,left)")
        print("(left,right)")
        print('Axis 0 --------------')
        print("partial maximum positions:")
        print(pc0s)
        print("maximum values:")
        print(pc0vals)
        print("resulting index")
        print(index0)
        print("Axis 1--------------")
        print("partial maximum positions")
        print(pc1s)
        print("maximum values:")
        print(pc1vals)
        print("resulting index")
        print(index1) 

    if index0<3:
        if pc[0] > pcs[0] / 2:
            pc[0] = pc[0] - pcs[0]
    elif index0==3:
        pc[0] = pc[0]-pcs[0]
        
    if index1<3:
        if pc[1] > pcs[1] / 2:
            pc[1] = pc[1] - pcs[1]
    elif index1==3:
        pc[1] = pc[1]-pcs[1]

    if _verbose:
        return pcm,(index0,index1)
    else:
        return pc


def align_com_precise(im1, im2,delta=None,show=False,artifacts=None):
    """
    

    Args:
        im1 (TYPE): 
            DESCRIPTION.
        im2 (TYPE): 
            DESCRIPTION.
        delta (TYPE, optional): 
            DESCRIPTION. 
            Defaults to None.
        show (TYPE, optional): 
            DESCRIPTION. 
            Defaults to False.
        artifacts (TYPE, optional): 
            DESCRIPTION. Defaults to None.

    Returns:
        pc (TYPE): 
            DESCRIPTION.

    """
    
    pcm,(index0,index1) = align(im1,im2,_verbose=True)

    pcm -= np.min(pcm)
    pcb,orig=img_periodic_tiling(pcm)
    rows=int(orig[0][0]//2)
    cols=int(orig[1][0]//2)
    pcr=pcb[rows:3*rows,cols:3*cols]
    if delta is None:
        delta=min(int(rows//2),int(cols//2))

    if artifacts is None:
        pass
    else:
        center=pcr.shape[0]//2,pcr.shape[1]//2
        pcr[center[0],:]*=artifacts
        pcr[:,center[1]]*=artifacts
    
    delt=2*delta
    compos,maxpos,delta_used=peak_com2d(pcr,delta=delta)

    if show:
        plt.imshow(pcr[maxpos[0]-delt:maxpos[0]+delt,maxpos[1]-delt:maxpos[1]+delt])
        plt.plot(compos[1]-(maxpos[1]-delt),compos[0]-(maxpos[0]-delt),'rx',
                 label='center of mass')
        plt.plot(delt,delt,'wx',label='global max')
        plt.plot(-maxpos[1]+pcr.shape[1]//2+delt,-maxpos[0]+pcr.shape[0]//2+delt,'x',c='fuchsia',label='origin')
        plt.legend()
        plt.colorbar()
        plt.show()
        
    pc=np.zeros(2)
    pc[0]=(compos[0]+rows)%orig[0][0]
    pc[1]=(compos[1]+cols)%orig[1][0]
    
    pcs = np.array(np.shape(pcm))

    if index0<3:
        if pc[0] > pcs[0] / 2:
            pc[0] = pc[0] - pcs[0]
    elif index0==3:
        pc[0] = pc[0]-pcs[0]
        
    if index1<3:
        if pc[1] > pcs[1] / 2:
            pc[1] = pc[1] - pcs[1]
    elif index1==3:
        pc[1] = pc[1]-pcs[1]
        
    return pc

#%% stacks
def stack_crop_shifts(stack,shifts):
    delta=np.max(shifts,axis=0)-np.min(shifts,axis=0)
    delta=delta.astype(int)
    if delta[0]==0:
        res=stack[:,:,delta[1]:-delta[1]]
    elif delta[1]==0:
        res=stack[:,delta[0]:-delta[0],:]
    else:        
        res=stack[:,delta[0]:-delta[0],delta[1]:-delta[1]]
    return res    

def stack_shift_precise(imgs,delta=None,show=False,artifacts=None):
    #img=imgs[0]
    shifts=np.zeros([len(imgs),2])
    for i in range(len(imgs)-1):
        shifts[i+1]=align_com_precise(imgs[i],imgs[i+1],delta=delta,show=show,artifacts=artifacts)
        
    return np.cumsum(shifts,axis=0)#shifts

def stack_align_com_precise(imgs,shifts):

    ma_sh=np.max(shifts,axis=0)
    mi_sh=np.min(shifts,axis=0)
    to_sh=ma_sh-mi_sh
    to_sh_int=np.round(to_sh).astype(int)
    
    size=np.array(imgs[0].shape,dtype=int)
    newsize=size+to_sh_int
 
    x, y = np.meshgrid(np.arange(newsize[1]), np.arange(newsize[0]))
    
    res=[]

    nshifts = shifts- mi_sh
    for i in range(len(imgs)):
        newx=x-nshifts[i,1]
        newy=y-nshifts[i,0]
        

        tres=cv2.remap(img_to_uint16(imgs[i]), newx.astype(np.float32), 
                       newy.astype(np.float32), 
                          interpolation=cv2.INTER_CUBIC,
                          borderValue= 0, borderMode=cv2.BORDER_CONSTANT)
        
        res.append(tres)
    
    return res

def stack_shifting(imgs):
    #img=imgs[0]
    shifts=np.zeros([len(imgs),2],dtype=int)
    for i in range(len(imgs)-1):
        shifts[i+1]=align(imgs[i],imgs[i+1])
    return np.cumsum(shifts,axis=0)#shifts

def stack_align(imgs,shifts):
    shifts=shifts.astype(int)
    ma_sh=np.max(shifts,axis=0)
    mi_sh=np.min(shifts,axis=0)
    to_sh=ma_sh-mi_sh
    
    size=imgs[0].shape
    new=np.zeros([len(imgs),size[0]+to_sh[0],size[1]+to_sh[1]])
    nshifts = shifts-mi_sh
    for i in range(len(imgs)):        
        new[i,nshifts[i,0]:nshifts[i,0]+size[0],nshifts[i,1]:nshifts[i,1]+size[1]]=imgs[i]
    
    return new




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
    #global list_of_points

    if event.button == 3:  # right clicking

        x = event.xdata
        y = event.ydata

        list_of_points.append([y, x])
        ax.plot(x, y, "o")
        print(y, x)
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
    #im1s,p1s=assure_multiple(im1s,p1s)
    single_image=False
    if not len(im1s) == len(p1s):
        single_image=True        
        im1s=[im1s]
        p1s=[p1s]

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

    if len(im2.shape)>2: 
        img2Reg = np.zeros([resheight, reswidth,im2.shape[2]],dtype=np.uint8)
        img2Reg[
            height_shift : height_shift + im2.shape[0],
            width_shift : width_shift + im2.shape[1],
        ] = im2

    else:
        
        img2Reg = np.zeros([resheight, reswidth],dtype=np.uint8)
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
    
    if single_image:
        if verbose:
            return im1res[0], img2Reg, matrices, reswidth, \
                    resheight, width_shift, height_shift
        else:
            return im1res[0], img2Reg
    else:
        if verbose:
            return im1res, img2Reg, matrices, reswidth, \
                    resheight, width_shift, height_shift
        else:
            return im1res, img2Reg



def align_pair(im1, im2, p1, p2, verbose=False):
    
    allwidths = []
    allheights = []

    matrix = cv2.estimateAffinePartial2D(p2, p1)[0]

    scale=np.sqrt(matrix[0,0]**2+matrix[0,1]**2)
    rotation = np.degrees(np.arctan2(-matrix[0,1], matrix[0,0]))
    translation = matrix[:,2]#[::-1]
    
    
    xf=[0,im2.shape[1],im2.shape[1],0]
    yf=[0,0,im2.shape[0],im2.shape[0]]
    
    img_matrix = np.stack([xf, yf, np.ones(len(xf))])
    
    res = np.tensordot(matrix, img_matrix, axes=1)
    
    # consider the scaling of the border pixels
    extra=np.abs(scale-1)

    im1shift=np.zeros(2,dtype=int)
    resmin=np.min(res,axis=-1)-extra
    resmax=np.max(res,axis=-1)
    if True in (res[0]<0):
        matrix[0,2]-=resmin[0]
        im1shift[1]=-np.round(resmin[0])
        
    if True in (res[1]<0):
        matrix[1,2]-=(resmin[1])
        im1shift[0]=-np.round(resmin[1])

    allwidths.append( min(resmin[0], 0))
    allheights.append(min(resmin[1], 0))

    allwidths.append(max(resmax[0], im1.shape[1]) - allwidths[-1])
    allheights.append(max(resmax[1], im1.shape[0]) - allheights[-1])

    reswidth = np.round(np.max(allwidths)).astype(int)
    resheight = np.round(np.max(allheights)).astype(int)
    
    if len(im1.shape)>2: 
        im1orig = np.zeros([resheight, reswidth,im1.shape[2]],dtype=im1.dtype)
        im1orig[
            im1shift[0] : im1shift[0] + im1.shape[0],
            im1shift[1] : im1shift[1] + im1.shape[1],
        ] = im1

    else:
        
        im1orig = np.zeros([resheight, reswidth],dtype=im1.dtype)
        im1orig[
            im1shift[0] : im1shift[0] + im1.shape[0],
            im1shift[1] : im1shift[1] + im1.shape[1],
        ] = im1
    
    if scale>0.95:
        interpolation=cv2.INTER_CUBIC
    else:
        interpolation=cv2.INTER_AREA
        
    im2transformed = cv2.warpAffine(im2,matrix,(reswidth,resheight), 
                                    flags=interpolation)
    
    if verbose:
        params=dict()
        params["translation"]=translation
        params["rotation"]=rotation
        params["scale"]=scale
        params["im1shift"]=im1shift
        return im1orig,im2transformed,params
    else:
        return im1orig,im2transformed


def align_pair_special(im1, im2, p1, p2,relative_scale_limit=None,
                       translation_only=False):
    
    allwidths = []
    allheights = []

    matrix = cv2.estimateAffinePartial2D(p2, p1)[0]
    if matrix is None:
        print("could not estimate")
        return None,None
    if translation_only:
        matrix[0,0]=1
        matrix[1,1]=1
        matrix[0,1]=0
        matrix[1,0]=0
    
    
    scale=np.sqrt(matrix[0,0]**2+matrix[0,1]**2)
    if relative_scale_limit is not None:
        if np.abs(1-scale)>relative_scale_limit:
            return None,None
    rotation = np.degrees(np.arctan2(-matrix[0,1], matrix[0,0]))
    translation = matrix[:,2]#[::-1]
    
    
    xf=[0,im2.shape[1],im2.shape[1],0]
    yf=[0,0,im2.shape[0],im2.shape[0]]
    
    img_matrix = np.stack([xf, yf, np.ones(len(xf))])
    
    res = np.tensordot(matrix, img_matrix, axes=1)
    
    # consider the scaling of the border pixels
    extra=np.abs(scale-1)

    im1shift=np.zeros(2,dtype=int)
    resmin=np.min(res,axis=-1)-extra
    resmax=np.max(res,axis=-1)
    if True in (res[0]<0):
        matrix[0,2]-=resmin[0]
        im1shift[1]=-np.round(resmin[0])
        
    if True in (res[1]<0):
        matrix[1,2]-=(resmin[1])
        im1shift[0]=-np.round(resmin[1])

    allwidths.append( min(resmin[0], 0))
    allheights.append(min(resmin[1], 0))

    allwidths.append(max(resmax[0], im1.shape[1]) - allwidths[-1])
    allheights.append(max(resmax[1], im1.shape[0]) - allheights[-1])

    reswidth = np.round(np.max(allwidths)).astype(int)
    resheight = np.round(np.max(allheights)).astype(int)
    
    
    if scale>0.95:
        interpolation=cv2.INTER_CUBIC
    else:
        interpolation=cv2.INTER_AREA
        
    im2transformed = cv2.warpAffine(im2,matrix,(reswidth,resheight), 
                                    flags=interpolation)
    

    params=dict() 
    params["translation"]=translation
    params["rotation"]=rotation
    params["scale"]=scale
    params["im1shift"]=im1shift
    return im2transformed,params






def stitch_pair(im1,im2,verbose=False):
    """
    only works with uint8

    Args:
        im1 (TYPE): DESCRIPTION.
        im2 (TYPE): DESCRIPTION.
        verbose (TYPE, optional): DESCRIPTION. Defaults to False.

    Returns:
        TYPE: DESCRIPTION.

    """
    if len(im1.shape)==2:
        im1=img_gray_to_rgba(im1)
        
    if len(im2.shape)==2:
        im2=img_gray_to_rgba(im2)
            
    pts1,pts2 = sift_align_matches(im1, im2)
    if verbose:
        im1orig,im2transformed,params = align_pair(im1, im2, pts1, pts2,verbose=True)
        stitched=img_add_weighted_rgba(im1orig, im2transformed)
        return stitched,params
    else:
        im1orig,im2transformed = align_pair(im1, im2, pts1, pts2)
        stitched=img_add_weighted_rgba(im1orig, im2transformed)
        return stitched
    
#%% sift align matches
def sift_align_matches(img1,img2,ratio_threshold=0.5,verbose=False):
    """
    only works with uint8 -dtype

    Args:
        img1 (TYPE): DESCRIPTION.
        img2 (TYPE): DESCRIPTION.
        ratio_threshold (TYPE, optional): DESCRIPTION. Defaults to 0.5.

    Returns:
        Matched (TYPE): DESCRIPTION.
        ptsA (TYPE): DESCRIPTION.
        ptsB (TYPE): DESCRIPTION.

    """

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    #img=cv2.drawKeypoints(img1,kp,img1)
    """
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # Need to draw only good matches, so create a mask
    #good_matches = [[0,0] for i in range(len(matches))]
    good=[]
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            #good_matches[i]=[1,0]
            good.append(m)
    
    """
    #Initialize the BFMatcher for matching 
    BFMatch = cv2.BFMatcher() 
    Matches = BFMatch.knnMatch(des1,des2,k=2) 
      
    # Need to draw only good matches, so create a mask 
    good_matches = [[0,0] for i in range(len(Matches))] 
    
    good=[] 
    
    # ratio test as per Lowe's paper 
    for i,(m,n) in enumerate(Matches): 
        if m.distance < ratio_threshold*n.distance: 
            good_matches[i]=[1,0] 
            good.append(m)
    
    ptsA = np.zeros((len(good), 2), dtype="float")
    ptsB = np.zeros((len(good), 2), dtype="float")
    
    # loop over the top matches
    for i,m in enumerate(good):
        ptsA[i]= (kp1[m.queryIdx].pt[0],kp1[m.queryIdx].pt[1])
        ptsB[i]= (kp2[m.trainIdx].pt[0],kp2[m.trainIdx].pt[1])

    if verbose:
        # Draw the matches using drawMatchesKnn() 
        Matched = cv2.drawMatchesKnn(img1,       
                                     kp1,    
                                     img2, 
                                     kp2, 
                                     Matches, 
                                     outImg = None, 
                                     matchColor = (0,0,255),   
                                     singlePointColor = (0,255,255), 
                                     matchesMask = good_matches, 
                                     flags = 0
                                    )
        return Matched,ptsA,ptsB
    else:
        return ptsA,ptsB
    

#%% stack_sift_align
def stack_sift_align_to_first(stack,ratio=0.5,verbose=False):
    
    #get keypoints and good macthes
    ptsBs=[]
    ptsAs=[]
    for i in range(len(stack)-1):    
        matched,ptsA,ptsB=sift_align_matches(stack[0], 
                                                stack[i+1],ratio_threshold=ratio)
        ptsAs.append(ptsA)
        ptsBs.append(ptsB)

    # make a dictionary of matching points over the whole stack 
    kpdict=dict()
    for i in range(len(ptsAs)):
        for j in range(len( ptsAs[i])):
            ptA=(ptsAs[i][j][0],ptsAs[i][j][1])
            ptB=(ptsBs[i][j][0],ptsBs[i][j][1])
            
            if ptA in kpdict:
                kpdict[ptA].append(ptB)
            else:
                kpdict[ptA]=[ptB]
    # choose only keypoints that persist in all images
    persistent_ptsA=[]
    persistent_ptsBs=[]
    for i in kpdict:
        if len(kpdict[i])==len(ptsAs):
            persistent_ptsA.append(i)
            persistent_ptsBs.append(kpdict[i])
    resulting_ptsA=np.array(persistent_ptsA)
    number_of_images_to_align=len(ptsAs)
    number_of_persistent_matches=len(resulting_ptsA)
    
    print(number_of_persistent_matches)
    #restructure points B
    resulting_ptsBs=np.zeros([number_of_images_to_align,
                     number_of_persistent_matches,
                     2])    
    for i in range(number_of_persistent_matches):
        resulting_ptsBs[:,i,:]=persistent_ptsBs[i]
    
    
    #calculate the homography matrices and apply them    
    
    if not verbose:
        im1s,img0=align_images(stack[1:],stack[0], resulting_ptsBs[:], resulting_ptsA)
        imlist=[img0]+im1s
        return imlist
    else:
        (im1s, img0, matrices, reswidth, resheight, 
         width_shift, height_shift)=align_images(
                                                 stack[1:],stack[0], 
                                                 resulting_ptsBs[:], 
                                                 resulting_ptsA,verbose=True)
        imlist=[img0]+im1s
        metadata=dict()
        metadata["matrices"]=matrices
        metadata["reswidth"]=reswidth
        metadata["resheight"]=resheight
        metadata["width_shift"]=width_shift
        metadata["height_shift"]=height_shift
        
        return imlist,metadata
    
#%% stack_align_from_matrices   
def stack_align_from_matrices(stack,metadata):
    """
    Aligns a stack of images using the homographic transformation calculated
    by the function stack_sift_align_to_first() with argument verbose=True

    Args:
        stack (KxMxN array_like or list of MxN array_likes): 
            stack of images 
            (different image dimensionss, e.g. one with 1024x512 
            and another with 234x653, are possible).
        
        metadata (dictionary): 
            transformational information as given by stack_sift_align_to_first.

    Returns:
        imlist (list of MxN): 
            list of aligned images.

    """
    
    reswidth=metadata["reswidth"]
    resheight=metadata["resheight"]
    width_shift=metadata["width_shift"]    
    height_shift=metadata["height_shift"]
    matrices=metadata["matrices"]
    
    im1s=[]
    for i in range(len(stack)-1):
        img=align_image_fast1(stack[i+1], matrices[i], reswidth, resheight)
        im1s.append(img)

    img0=align_image_fast2(stack[0], reswidth, resheight, width_shift, height_shift)
    
    imlist=[img0]+im1s
    return imlist  
    