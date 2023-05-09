# -*- coding: utf-8 -*-
"""
Created on Mon May  8 20:56:03 2023

@author: kernke
"""

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
            rot, log = rotate_bound(image_roi, angle + pmangle[i])
            drot, log = rotate_bound(dummy, angle + pmangle[i], bm=0)
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
        rot, log = rotate_bound(image_roi, res)
        drot, log = rotate_bound(dummy, res, bm=0)
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


#%% processing
def processing(
    image, angles, rois, filterheight, asympix=20, thresh_ratio=1.5, smoothing=11
):

    res = []

    for i in range(len(angles)):
        dark, nmask, log = enhance_lines(image, angles[i], line="dark")
        bright, nmask1, log = enhance_lines(image, angles[i], line="bright")
        smoothkernel = smoothbox_kernel([1, smoothing])[0]
        dres = cv2.sepFilter2D(dark, cv2.CV_64F, smoothkernel, np.ones(1))
        bres = cv2.sepFilter2D(bright, cv2.CV_64F, smoothkernel, np.ones(1))

        dres = anms(
            dres, nmask, thresh_ratio=thresh_ratio, ksize=filterheight, asympix=asympix
        )
        bres = anms(
            bres, nmask, thresh_ratio=thresh_ratio, ksize=filterheight, asympix=asympix
        )

        drows = np.sum(dres, axis=1)
        brows = np.sum(bres, axis=1)

        dpos = peak_com(drows, filterheight, rois[i])
        bpos = peak_com(brows, filterheight, rois[i])

        delta = int(np.round((dpos - bpos) / 2) * 2)
        nres = np.zeros(dres.shape)
        if delta < 0:
            delta = -delta
            nres[: -delta // 2] += bres[delta // 2 :]
            nres[delta // 2 :] += dres[: -delta // 2]
            nres[delta // 2 : -delta // 2] /= 2.0

        elif delta == 0:
            nres += bres
            nres += dres
            nres /= 2.0

        else:
            nres[: -delta // 2] += dres[delta // 2 :]
            nres[delta // 2 :] += bres[: -delta // 2]
            nres[delta // 2 : -delta // 2] /= 2.0

        res1 = rotate_back(nres, log)
        # rotate_bound(nres,-angles[i],flag2='back')
        offset = np.zeros(2, dtype=int)
        offset = (np.array(res1.shape) - np.array(image.shape)) / 2
        offset = offset.astype(int)
        gres = res1[offset[0] : -offset[0], offset[1] : -offset[1]]

        res.append(gres)
    return res[0] + res[1]

#%% enhance_lines_prototype
def enhance_lines_prototype(
    image, angle, number_of_bins=61, ksize=None, dist=1, iterations=2, line="dark"
):

    dummy = np.ones(image.shape)
    rot, log = rotate_bound(image, angle)
    drot, log = rotate_bound(dummy, angle, bm=0)
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
        if ksize is None:
            srot = cv2.Sobel(tres, cv2.CV_64F, 0, 1)
        else:
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
    # def eliminate_side_maxima_image(self,shiftrange=2,tol=1,line='dark',test=False):
    #     tms=self.tms
    #     conpois=self.conpois
    #     image=self.image

    #     sortoutids=[]
    #     sortout=[]

    #     for i in range(len(conpois)):
    #         sortout.append([])
    #         sortoutids.append([])

    #         #image=checkmaps[i]
    #         if tms[i]>1:
    #             for j in range(len(conpois[i])):

    #                 #a=np.max(conpois[i][j][:,1])
    #                 #b=np.min(conpois[i][j][:,1])
    #                 #if a+shiftrange >= image.shape[1] or b-shiftrange<0:
    #                 #    pass
    #                 #else:
    #                 check=getcheck1(shiftrange,conpois[i][j],image)

    #                 icheck=np.argmax(check)
    #                 checkval=np.max(check)
    #                 #print(icheck)
    #                 if icheck < shiftrange-tol or icheck > shiftrange+tol:

    #                     #checkmedian=np.median(check)
    #                     #mad=np.median(np.abs(check-checkmedian))
    #                     #if check[shiftrange] < checkmedian+mad_threshold*mad:
    #                     sortout[-1].append(conpois[i][j])
    #                     sortoutids[-1].append(j)

    #         else:
    #             for j in range(len(conpois[i])):

    #                 #a=np.max(conpois[i][j][:,0])
    #                 #b=np.min(conpois[i][j][:,0])
    #                 #if a+shiftrange >= image.shape[0] or b-shiftrange<0:
    #                 #    pass
    #                 #else:
    #                 check=getcheck0(shiftrange,conpois[i][j],image)
    #                 icheck=np.argmax(check)
    #                 #print(icheck)

    #                 if icheck < shiftrange-tol or icheck > shiftrange+tol:

    #                     #checkmedian=np.median(check)
    #                     #mad=np.median(np.abs(check-checkmedian))
    #                     #if check[shiftrange] < checkmedian+mad_threshold*mad:
    #                     sortout[-1].append(conpois[i][j])
    #                     sortoutids[-1].append(j)

    #         print(len(sortout[-1]))

    #     if not test:
    #         self.sort_ids_out(sortoutids)

    #     return sortout