# -*- coding: utf-8 -*-
"""
Created on Mon May  8 22:21:07 2023

@author: kernke
"""

#%% asymmetric non maximum supppression median


def anms_median(img, mask, thresh_ratio=1.5, ksize=5, asympix=0):
    newimg = copy.deepcopy(img)
    return aysmmetric_non_maximum_suppression_median(
        newimg, img, mask, thresh_ratio, ksize, asympix
    )


@njit
def aysmmetric_non_maximum_suppression_median(
    newimg, img, mask, thresh_ratio, ksize, asympix
):
    ioffs = ksize // 2  # +asympix//2
    joffs = ksize // 2 + asympix // 2
    # newimg=img#np.zeros(img.shape)
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
                newimg[i, j] = img[i, j]
            else:
                g = img[i - ioffs : i + ioffs + 1, j - joffs : j + joffs + 1]
                v = max(np.sum(g, axis=0))  # * ksize/(ksize+asympix)
                h = max(np.sum(g, axis=1)) * ksize / (ksize + asympix)
                if h > v * thresh_ratio:
                    newimg[i, j] = img[i, j]
                else:
                    newimg[i, j] = np.median(g)
    return newimg


#%% noise level determination from aysmmetric_non_maximum_suppression


@njit
def anms_noise(img, mask, thresh_ratio, ksize, asympix):
    ioffs = ksize // 2  # +asympix//2
    joffs = ksize // 2 + asympix // 2
    npix = ksize * (ksize + asympix)
    noisemean = []
    noisemax = []
    noisestd = []
    # newimg=img#np.zeros(img.shape)
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
                #    newimg[i,j]=img[i,j]
            else:
                g = img[i - ioffs : i + ioffs + 1, j - joffs : j + joffs + 1]
                v = max(np.sum(g, axis=0))  # * ksize/(ksize+asympix)
                h = max(np.sum(g, axis=1))
                if h * ksize / (ksize + asympix) > v * thresh_ratio:
                    pass
                    # newimg[i,j]=img[i,j]
                else:
                    ave = (v + h) / npix
                    g = (g - ave) ** 2
                    noisemean.append(ave)
                    noisemax.append(np.max(g))
                    std = np.sum(g)
                    noisestd.append(std)
                    # newimg[i,j]=np.min(g)
    return noisemax, noisemean, noisestd



def determine_noise_threshold(img, mask, thresh_ratio, ksize, asympix):
    npix = ksize * (ksize + asympix)

    noisemax, noisemean, noisestd = anms_noise(img, mask, thresh_ratio, ksize, asympix)
    nma = np.array(noisemax)
    nme = np.array(noisemean)
    nms = np.array(noisestd)
    return np.sqrt(nma), nme, np.sqrt(nms / npix)



