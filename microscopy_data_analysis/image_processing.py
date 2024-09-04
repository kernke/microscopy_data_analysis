# -*- coding: utf-8 -*-
"""
submodule focussed completely on images, all functions with prefix "img" 
take an image as main input and return a processed image
"""
import cv2
import numpy as np
from skimage import  exposure
import copy
from numba import njit

#%% autoclipping
def img_autoclip(img,ratio=0.001):
    """
    auto clipping removing small and big outliers 

    Args:
        img (TYPE): DESCRIPTION.
        ratio (TYPE, optional): DESCRIPTION. Defaults to 0.001.

    Returns:
        TYPE: DESCRIPTION.

    """
    imgdata=np.sort(np.ravel(img))
    threshold=int(len(imgdata)*ratio)
    return np.clip(img,imgdata[threshold],imgdata[-threshold])


#%% morphLaplace
def img_morphLaplace(image, kernel):
    """
    morphological Laplace Filter

    Args:
        image (MxN array_like): 
            np.uint8.
        
        kernel (TYPE): 
            DESCRIPTION.

    Returns:
        filtered_image (MxN array_like): 
            np.uint8.

    """
    return cv2.erode(image, kernel) + cv2.dilate(image, kernel) - 2 * image - 128


#%% gammaCorrection
def img_gammaCorrection(img, gamma):
    """
    transformation adjusting the gamma level
    (gamma=1 means no change)

    Args:
        img (MxN array_like): 
            np.uint8.
        
        gamma (float): 
            gamma value between 0 and ~10.

    Returns:
        gamma_transformed_image (MxN array_like): 
            np.uint8.
    """
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(img, table)

#%% format image dtypes
def img_to_uint8(img,imgmax=255):
    """
    transform contrast range to unsigned integer 8bit

    Args:
        img (MxN array_like): 
            input image.
        
        imgmax (int, optional): 
            optionally reduce contrast range, 
            by setting a lower, than datatype given, threshold to the maximum value. 
            Defaults to 255.

    Returns:
        datatype_conform_image (MxN array_like): 
            np.uint8.

    """
    img -= np.min(img)
    return (img / np.max(img) * (imgmax+0.5)).astype(np.uint8)

def img_to_uint16(img,imgmax=65535):
    """
    transform contrast range to unsigned integer 16bit

    Args:
        img (MxN array_like): 
            input image.
        
        imgmax (int, optional): 
            optionally reduce contrast range, 
            by setting a lower, than datatype given, threshold to the maximum value. 
            Defaults to 65535.

    Returns:
        datatype_conform_image (MxN array_like): 
            np.uint16.

    """
    img -= np.min(img)
    return (img / np.max(img) * (imgmax+0.5)).astype(np.uint16)

def img_to_int8(img,imgmax=127):
    """
    transform contrast range to signed integer 8bit

    Args:
        img (MxN array_like): 
            input image.
        
        imgmax (int, optional): 
            optionally reduce contrast range, 
            by setting a lower, than datatype given, threshold to the maximum value. 
            Defaults to 127.

    Returns:
        datatype_conform_image (MxN array_like): 
            np.int8.

    """
    imgmax+=128
    img -= np.min(img)
    return (img / np.max(img) * (imgmax+0.5) -128).astype(np.int8)

def img_to_int16(img,imgmax=32767):
    """
    transform contrast range to signed integer 16bit

    Args:
        img (MxN array_like): 
            input image.
        
        imgmax (int, optional): 
            optionally reduce contrast range, 
            by setting a lower, than datatype given, threshold to the maximum value. 
            Defaults to 32767.

    Returns:
        datatype_conform_image (MxN array_like): 
            np.int16.

    """
    imgmax+=32768
    img -= np.min(img)
    return (img / np.max(img) * (imgmax+0.5) -32768).astype(np.int16)


def img_to_half_int8(img):
    """
    transform contrast to half the range of signed integer 8bit,
    so values are between -64 and 63 for possible range -128 to 127

    Args:
        img (MxN array_like): 
            input image.

    Returns:
        datatype_conform_image (MxN array_like): 
            np.int8.

    """
    img -= np.min(img)
    return (img / np.max(img) * 63.5 -64).astype(np.int8)


def img_to_half_int16(img):
    """
    transform contrast to half the range of signed integer 16bit,
    so values are between -16384 and 16383 for possible range -32768 to 32767

    Args:
        img (MxN array_like): 
            input image.

    Returns:
        datatype_conform_image (MxN array_like): 
            np.int16.

    """
    img -= np.min(img)
    return (img / np.max(img) * 16383.5-16384).astype(np.int16)


#%% noise_line_suppression


def img_noise_line_suppression(image, ksize):
    """
    morphological opening with a horizontal line as structuring element
    (first erode, than dilate ; only horizotal lines with length ksize remain)

    Args:
        image (array_like): 
            input.
        
        ksize (int): 
            uneven integer.

    Returns:
        processed_image (array_like): 
            output.

    """
    erod_img = cv2.erode(image, np.ones([1, ksize]))
    return cv2.dilate(erod_img, np.ones([1, ksize]))

#%% rebin
def img_rebin_by_mean(image, new_shape):
    """
    reduce the resolution of an image MxN to mxn by taking an average,
    whereby M and N must be multiples of m and n
    
    Args:
        image (MxN array_like): 
            image.
        new_shape (tuple): 
            containing two integers with the new shape.

    Returns:
        rebinned_image (mxn array_like): 
            smaller image.

    """
    if image.shape[0]%new_shape[0] !=0 or image.shape[1]%new_shape[1]:
        raise ValueError("image shape is not a multiple of new_shape")

    shape = (
        new_shape[0],
        image.shape[0] // new_shape[0],
        new_shape[1],
        image.shape[1] // new_shape[1],
    )

    return image.reshape(shape).mean(-1).mean(1)

#%% make_square


def img_make_square(image, startindex=None):
    """
    crops the largest square image from the original, by default from the center.
    The position of the cropped square can be specified via startindex,
    moving the frame from the upper left corner at startindex=0
    to the lower right corner at startindex=abs(M-N)   

    Args:
        image (MxN array_like): 
            DESCRIPTION.
        
        startindex (int, optional): 
            must be within 0 <= startindex <= abs(M-N). 
            Defaults to None.

    Returns:
        square_image (array_like): 
            either MxM or NxN array.

    """

    ishape = image.shape
    index_small, index_big = np.argsort(ishape)

    roi = np.zeros([2, 2], dtype=int)

    roi[index_small] = [0, ishape[index_small]]

    delta = np.abs(ishape[1] - ishape[0])

    if startindex is None:
        startindex = np.floor(delta / 2)
    else:
        if startindex > delta or startindex < 0:
            print("Error: Invalid startindex")
            print("0 <= startindex <= " + str(delta))

    roi[index_big] = startindex, startindex + ishape[index_small]

    square_image = image[roi[0, 0] : roi[0, 1], roi[1, 0] : roi[1, 1]]

    return square_image


#%% image rotation

# this function is a modified version of the original from
# https://github.com/PyImageSearch/imutils/blob/master/imutils/convenience.py#L41
def img_rotate_bound(image, angle, flag="cubic", bm=1):
    """
    rotates an image by the given angle clockwise;
    The rotated image is given in a rectangular bounding box
    without cutting off parts of the original image.
    
    Args:
        image (MxN array_like): 
            np.uint8.
        
        angle (float): 
            angle given in degrees.
        
        flag (string, optional): 
            possibilities:"cubic","linear";
            sets the method of interplation. 
            Defaults to "cubic".
        
        bm (int, optional): 
            sets the border mode, 
            extrapolating from the borders of the image.
            0: continues the image by padding zeros
            1: continues the image by repeating the border-pixel values. 
            Defaults to 1.

    Returns:
        rotated_image (KxL array_like): 
            np.uint8.
        
        log (list): 
            looking like [M,N,inverse_rotation_matrix],
            contains the original shape M,N and the matrix needed 
            to invert the rotation for the function img_rotate_back.

    """


    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int(np.round((h * sin) + (w * cos)))
    nH = int(np.round((h * cos) + (w * sin)))

    if bm == 0:
        bm = cv2.BORDER_CONSTANT
    elif bm == 1:
        bm = cv2.BORDER_REPLICATE

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    invRotateMatrix = cv2.invertAffineTransform(M)
    log = [(h, w), invRotateMatrix]
    # perform the actual rotation and return the image
    if flag == "cubic":
        return (
            cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=bm),
            log,
        )
    else:
        return (
            cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=bm),
            log,
        )


def img_rotate_back(image, log, flag="cubic", bm=1):
    """
    invert the rotation done by img_rotate_bound returning the image
    to its original shape MxN, cutting away padded values for the
    bounding box generated by img_rotate_bound

    Args:
        image (KxL array_like):
            np.uint8.
        
        log (list): 
            [M,N,inverse_rotation_matrix],
            contains the original shape M,N and the matrix needed 
            to invert the rotation. log is given by the function img_rotate_bound.
        
        flag (string, optional): 
            possibilities:"cubic","linear";
            sets the method of interplation. 
            Defaults to "cubic".
        
        bm (int, optional): 
            possibilities: 0,1;
            sets the border mode, extrapolating from the borders of the image.
            0: continues the image by padding zeros
            1: continues the image by repeating the border-pixel values. 
            (bm=1 allows more exact back transformation, avoiding the 
            decrease of border-pixel values due to interpolation with zeros.)
            Defaults to 1.

    Returns:
        inverse_rotated_image (MxN array_like): 
            np.uint8.
            
    """

    (h, w), invM = log

    if bm == 0:
        bm = cv2.BORDER_CONSTANT
    elif bm == 1:
        bm = cv2.BORDER_REPLICATE

    if flag == "cubic":
        return cv2.warpAffine(image, invM, (w, h), flags=cv2.INTER_CUBIC, borderMode=bm)
    else:
        return cv2.warpAffine(
            image, invM, (w, h), flags=cv2.INTER_LINEAR, borderMode=bm
        )



#%% tiling


def img_periodic_tiling(img, tiles=3):
    """
    takes an image as a tile and creates a tiling of
    tiles x tiles by duplicating it. 
    
    Args:
        img (MxN array_like): 
            2d-dataset / image.
        
        tiles (int, optional): 
            number of tiles in vertical and horizontal direction. 
            the number of tiles must be uneven.
            Defaults to 3.

    Returns:
        tiled (array_like): 
            with shape tiles*M x tiles*N.
        
        orig (tuples): 
            containing the bounding coordinates of the center image
            (lower_row_limit,upper_row_limit),(lower_column_limit,upper_column_limit).

    """
    s = np.array(img.shape)
    tiled = np.zeros(s * tiles, dtype=img.dtype)
    for i in range(tiles):
        for j in range(tiles):
            tiled[s[0] * i : s[0] * (i + 1), s[1] * j : s[1] * (j + 1)] = img

    oij = tiles // 2
    orig = (s[0] * oij, s[0] * (oij + 1)), (s[1] * oij, s[1] * (oij + 1))
    return tiled, orig


#%% special image transformations

def img_transform(image, imshape, rfftmask, rebin=True):
    """
    special function that resizes an image to imshape,
    afterwards applies a Fourier-space mask given by rfftmask,
    and finally rebins squares of 4 pixels to 1 pixel, if rebin=True
    
    Args:
        image (MxN array_like): 
            DESCRIPTION.
        imshape (tuple): 
            if rebin=True both integers of imshape must be even.
        rfftmask (Mx(N/2+1) array_like): 
            mask in Fourier space.
        rebin (bool, optional): 
            DESCRIPTION. Defaults to True.

    Returns:
        transformed_image (KxL array_like): 
            result.

    """
    # imshape must be even for rebin
    image[image <= 0] = 1
    image = np.log(image)
    resized = cv2.resize(image, imshape[::-1])
    fftimage = np.fft.rfft2(resized)
    inv = np.fft.irfft2(rfftmask * fftimage).real
    equ = exposure.equalize_adapthist(
        inv / np.max(inv), kernel_size=[128, 128], nbins=256
    )
    if rebin:
        rebin_shape = np.array(imshape) // 2
        equ = img_rebin_by_mean(equ, rebin_shape)
    equ -= np.min(equ)
        
    return (equ / np.max(equ) * 254 + 1).astype(np.uint8)


def img_transform_minimal(image, imshape,kernel):
    """
    special function that resizes an image to imshape,
    
    Args:
        image (MxN array_like): 
            DESCRIPTION.
        
        imshape ([int,int]): 
            if rebin=True both integers of imshape must be even 
        
        kernel (TYPE): 
            DESCRIPTION.

    Returns:
        transformed_image (KxL array_like): 
            result

    """
    image[image <= 0] = 1
    image = np.log(image)
    equ = cv2.resize(image, imshape[::-1])
    equ=(equ / np.max(equ) * 254 + 1).astype(np.uint8)
    lapl = img_morphLaplace(equ, kernel)
    summed = np.zeros(lapl.shape, dtype=np.double)
    summed += 255 - lapl
    summed += equ
    copt = img_to_uint8(summed)
    
    new = exposure.equalize_adapthist(
        copt / np.max(copt), kernel_size=[32, 32], nbins=256
    )
    return img_to_uint8(new)#(equ / np.max(equ) * 254 + 1).astype(np.uint8)


#%% asymmetric non maximum supppression

@njit
def _aysmmetric_non_maximum_suppression(
    newimg, img, cimg, rimg, mask, thresh_ratio, ksize, asympix, damping
):
    ioffs = ksize // 2
    joffs = ksize // 2 + asympix // 2

    for i in range(ioffs, img.shape[0] - ioffs):
        for j in range(joffs, img.shape[1] - joffs):
            if not mask[i, j]:
                pass
            elif (
                not mask[i - ioffs, j - joffs]
                or not mask[i + ioffs, j + joffs]
                or not mask[i - ioffs, j + joffs]
                or not mask[i + ioffs, j - joffs]
            ):
                pass
            else:
                v = max(cimg[i, j - joffs : j + joffs + 1])
                h = max(rimg[i - ioffs : i + ioffs + 1, j]) * ksize / (ksize + asympix)

                if h > v * thresh_ratio:
                    newimg[i, j] = img[i, j]
                else:
                    newimg[i, j] = (
                        img[i, j] / damping
                    )  # np.min(img[i-ioffs:i+ioffs+1,j-joffs:j+joffs+1])
    return newimg



def img_anms(img, mask, thresh_ratio=1.5, ksize=5, asympix=0, damping=5):
    """
    asymmetric non maximum supppression

    Args:
        img (array_like): 
            DESCRIPTION.
        
        mask (TYPE): 
            DESCRIPTION.
        
        thresh_ratio (float, optional): 
            DESCRIPTION. 
            Defaults to 1.5.
        
        ksize (int, optional): 
            uneven integer. 
            Defaults to 5.
        
        asympix (TYPE, optional): 
            DESCRIPTION. 
            Defaults to 0.
        
        damping (TYPE, optional): 
            DESCRIPTION. 
            Defaults to 5.

    Returns:
        processed_image (array_like): 
            DESCRIPTION.

    """
    newimg = copy.deepcopy(img)
    cimg = cv2.sepFilter2D(
        img, cv2.CV_64F, np.ones(1), np.ones(ksize), borderType=cv2.BORDER_ISOLATED
    )
    rimg = cv2.sepFilter2D(
        img,
        cv2.CV_64F,
        np.ones(ksize + asympix),
        np.ones(1),
        borderType=cv2.BORDER_ISOLATED,
    )
    return _aysmmetric_non_maximum_suppression(
        newimg, img, cimg, rimg, mask, thresh_ratio, ksize, asympix, damping
    )


