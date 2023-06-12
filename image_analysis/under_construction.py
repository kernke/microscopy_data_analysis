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


#%%
"""
            print(str(event.xdata)+" , "+str(event.ydata))
            
            sortindex=np.argsort(obj.x)
            newx=np.array(obj.x)[sortindex]
            newy=np.array(obj.y)[sortindex]
            
            obj.length=dl
            angle=np.arctan2(dy,dx)/np.pi *180
            if angle<0:
                angle=180-angle
            obj.angle=angle
            if arg_dic["snap_to_angle"]:
                angles_index=np.argmin(np.abs(np.array(arg_dic["angles"])-angle))
            
                meanx=newx[0]+dx/2
                meany=newy[0]+dy/2
    
                m=arg_dic["ms"][angles_index]
                n=meany-m*meanx
                angle_for_cos=arg_dic["angles"][angles_index]/180 * np.pi
                if angle_for_cos > np.pi/2:
                    angle_for_cos=np.pi-angle_for_cos
                ratio_length_to_x=np.cos(angle_for_cos)
                dl2=dl/2 * ratio_length_to_x
                obj.x[0]=meanx-dl2
                obj.x[1]=meanx+dl2
                obj.y[0]=m*(meanx-dl2)+n
                obj.y[1]=m*(meanx+dl2)+n

            
            dim=images[arg_dic["image_counter"]].shape[::-1]

            bsd=arg_dic["border_snap_distance"]
            xroi=[bsd,dim[0]-bsd,dim[0]-bsd,bsd]
            yroi=[bsd,bsd,dim[1]-bsd,dim[1]-bsd]
            
            cond0=point_in_convex_ccw_roi(xroi, yroi, obj.x[0]+shift[0], obj.y[0]+shift[1])
            cond1=point_in_convex_ccw_roi(xroi, yroi, obj.x[1]+shift[0], obj.y[1]+shift[1])

            if cond0 and cond1:
                pass
            else:
                a=np.array((obj.x[0]+shift[0],obj.y[0]+shift[1]))
                b=np.array((obj.x[1]+shift[0],obj.y[1]+shift[1]))

                points=[]
                points.append(np.array(lineIntersection(a, b,(0,0),(0,dim[1])) ))
                points.append(np.array(lineIntersection(a, b,(0,0),(dim[0],0)) ))
                points.append(np.array(lineIntersection(a, b,(0,dim[1]),(dim[0],dim[1]))))
                points.append(np.array(lineIntersection(a, b,(dim[0],0),(dim[0],dim[1]))))

                if not cond0:
                    dists=np.zeros(4)
                    for i in range(4):
                        dists[i]=np.sum((a-points[i])*(a-points[i]))
                    res_index=np.argmin(dists)
                    
                    obj.x[0]=points[res_index][0]-shift[0]
                    obj.y[0]=points[res_index][1]-shift[1]

                if not cond1:
                    dists=np.zeros(4)
                    for i in range(4):
                        dists[i]=np.sum((b-points[i])*(b-points[i]))
                    res_index=np.argmin(dists)
                    
                    obj.x[1]=points[res_index][0]-shift[1]
                    obj.y[1]=points[res_index][1]-shift[1]
                

"""




