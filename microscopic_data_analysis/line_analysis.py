# -*- coding: utf-8 -*-
"""
@author: kernke
"""

import numpy as np
from skimage.draw import line_aa
from numba import njit
import matplotlib.pyplot as plt
import math
from .general_util import intersect, lineIntersection
from .image_processing import img_to_uint8
from scipy.sparse import csr_matrix
import cv2


#https://stackoverflow.com/questions/33281957/faster-alternative-to-numpy-where
def compute_M(data):
    cols = np.arange(data.size)
    return csr_matrix((cols, (data.ravel(), cols)),
                      shape=(data.max() + 1, data.size))

def get_indices_sparse(data):
    M = compute_M(data)
    return [np.unravel_index(row.data, data.shape) for row in M]


#%% get_connected_points
def get_connected_points(srb,minimal_points=5):
    srb=img_to_uint8(1*(srb))
    num_labels, labels_im = cv2.connectedComponents(srb,connectivity=4)
    indices = get_indices_sparse(labels_im)

    conpoi=[]
    conlen = []
    for i in range(1,len(indices)):
        points=np.array(indices[i]).T
        if len(points)>minimal_points:
            conpoi.append(points)
            conlen.append(len(points))
    
    return conpoi,conlen


    #%%
def make_line_overview(conpois,img):
    """
    create an image where 
    """
    if len(np.shape(conpois[0][0][0]))==0:
        conpois=[conpois]
    newimg=np.zeros(img.shape)
    shape=[]
    for i in conpois:
        shape.append(len(i))
    separator=np.cumsum(shape)
    for i in range(len(conpois)):
        for k in range(len(conpois[i])):
            newimg[conpois[i][k][:,0],conpois[i][k][:,1]]=k+separator[i]
    return newimg,separator


#%% check_checkmap
def _check_image(conpoi, conlen, image):
    concheck = np.zeros(len(conpoi))
    for i in range(len(conpoi)):
        for j in conpoi[i]:
            concheck[i] += image[j[0], j[1]]
    return concheck / np.array(conlen)

#%% calc_m_n_t
@njit
def _calc_m_n_t_l(points, singlemap):
    """
    slope: m
    intercept: n
    thickness: t
    length: l
    """
    xstartindex = np.argmin(points[:, 1])
    xendindex = np.argmax(points[:, 1])
    ystart, xstart = points[xstartindex]
    yend, xend = points[xendindex]
    dy = yend - ystart
    dx = xend - xstart


    t = int(np.ceil(len(points) / np.sqrt(dy * dy + dx * dx)))
    startmean = np.zeros(2)
    endmean = np.zeros(2)
    cstart = np.zeros(2)
    cend = np.zeros(2)
    for i in range(t):
        for j in range(-t, t):
            ys = ystart + j
            xs = xstart + i
            if singlemap[ys, xs] != 0:
                startmean[0] += ys
                startmean[1] += xs
                cstart[0] += 1
                cstart[1] += 1
            ye = yend + j
            xe = xend - i
            if singlemap[yend + j, xend - i] != 0:
                endmean[0] += ye
                endmean[1] += xe
                cend[0] += 1
                cend[1] += 1

    s = startmean / cstart
    e = endmean / cend

    dy = e[0] - s[0]
    dx = e[1] - s[1]
    if  dx == 0:
        dx=10**(-20)
        print("Warning")
        print("number of points "+str(len(points)))
        print("division by zero")

    m = dy / dx
    l = math.sqrt(dy * dy + dx * dx)

    y = (e[0] + s[0]) / 2
    x = (e[1] + s[1]) / 2
    n = y - m * x

    return [m, n, t, l], s, e


#%% getcheck
@njit
def _getcheck0(shiftrange, points, image):
    check = np.zeros(2 * shiftrange + 1)
    checklen = np.zeros(2 * shiftrange + 1) + len(points)
    cshape0, cshape1 = image.shape
    for i in range(-shiftrange, shiftrange + 1):
        for j in points:
            if j[0] + i >= cshape0 or j[1] >= cshape1 or j[0] + i < 0 or j[1] < 0:
                checklen[i + shiftrange] -= 1
            else:
                check[i + shiftrange] += image[j[0] + i, j[1]]
    checklen[checklen == 0] = 1
    return check / checklen


@njit
def _getcheck1(shiftrange, points, image):
    check = np.zeros(2 * shiftrange + 1)
    checklen = np.zeros(2 * shiftrange + 1) + len(points)
    cshape0, cshape1 = image.shape
    for i in range(-shiftrange, shiftrange + 1):
        for j in points:
            if j[0] >= cshape0 or j[1] + i >= cshape1 or j[0] < 0 or j[1] + i < 0:
                checklen[i + shiftrange] -= 1
            else:
                check[i + shiftrange] += image[j[0], j[1] + i]
    checklen[checklen == 0] = 1
    return check / checklen


#%% getcheck
@njit
def _getcheck2(shiftrange, points, image,mask):
    check = np.zeros(2 * shiftrange + 1)
    checklen = np.zeros(2 * shiftrange + 1) + len(points)
    cshape0, cshape1 = image.shape
    for i in range(-shiftrange, shiftrange + 1):
        for j in points:
            if j[0] + i >= cshape0 or j[1] >= cshape1 or j[0] + i < 0 or j[1] < 0:
                checklen[i + shiftrange] -= 1
            else:
                if mask[j[0]+i,j[1]]:
                    checklen[i + shiftrange] -= 1
                else:
                    check[i + shiftrange] += image[j[0] + i, j[1]]
    checklen[checklen == 0] = 1
    return check / checklen


@njit
def _getcheck3(shiftrange, points, image,mask):
    check = np.zeros(2 * shiftrange + 1)
    checklen = np.zeros(2 * shiftrange + 1) + len(points)
    cshape0, cshape1 = image.shape
    for i in range(-shiftrange, shiftrange + 1):
        for j in points:
            if j[0] >= cshape0 or j[1] + i >= cshape1 or j[0] < 0 or j[1] + i < 0:
                checklen[i + shiftrange] -= 1
            else:
                if mask[j[0],j[1]+i]:
                    checklen[i + shiftrange] -= 1
                else:
                    check[i + shiftrange] += image[j[0], j[1] + i]
    checklen[checklen == 0] = 1
    return check / checklen



#%% _order_points
def _order_points(s1, e1, s2, e2):
    """
    Returns outerstart, outerend, start, end
    """
    if sum((s1 - e2) * (s1 - e2)) > sum((s2 - e1) * (s2 - e1)):
        return s1, e2, e1, s2
    else:
        return s2, e1, s1, e2


#%% line_analysis_object (class)


class line_analysis_object:
    """
    object containing images, lines and points
    """

    def __init__(self, image, singlemaps, checkmaps):

        self.image = image
        self.singlemaps = singlemaps
        self.checkmaps = checkmaps
        self.binmap = np.sum(singlemaps,axis=0)>0

        self.properties = []
        self.image_vars = ["image", "singlemaps", "checkmaps","binmap"]
        self.line_vars = []
        self.point_vars = []

    def _add_attr(self, attr_name, attrlist, attr):
        setattr(self, attr_name, attr)
        if attr_name not in attrlist:
            attrlist.append(attr_name)

    #%%
    def _update_confidence(self, confidence):
        for i in range(len(confidence)):
            for j in range(len(confidence[i])):
                self.confidence[i][j] += confidence[i][j]

    #%% get methods
    def get_methods(self):
        """
        Returns a list containing all methods of this class
        """
        object_methods = [
            method_name for method_name in dir(self) if callable(getattr(self, method_name))
        ]

        methods = [method for method in object_methods if method[0] != "_"]

        return methods

    #%% sort_ids_out
    def sort_ids_out(self, sortoutids):
        """
        Delete list of indices refering to lines
        """
        for attr in self.line_vars:
            var = getattr(self, attr)
            for s in range(len(sortoutids)):
                tosortout = sortoutids[s]
                for i in sorted(tosortout, reverse=True):
                    del var[s][i]

    #%%
    def check_line_vars(self, printing=True):
        """
        Check that all variables scaling with the number of lines,
        have the same dimensions.
        """
        checkshape = None
        returnvalue = True
        for attr in self.line_vars:
            var = getattr(self, attr)
            if checkshape is None:
                checkshape = np.zeros(len(var))
                for i in range(len(var)):
                    checkshape[i] = len(var[i])
                if printing:
                    print(attr)
                    print(checkshape)
            else:
                newshape = np.zeros(len(checkshape))
                for i in range(len(var)):
                    newshape[i] = len(var[i])
                if printing:
                    print(attr)
                    print(newshape)
                if not np.allclose(newshape, checkshape):
                    returnvalue = False
        return returnvalue

    #%% all_connected_points
    def all_connected_points(self, printing=True):
        singlemaps = self.singlemaps
        conpois = []
        conlens = []
        shape = []
        confidence = []
        for i in range(len(singlemaps)):
            conpoi, conlen = get_connected_points(singlemaps[i])
            conpois.append(conpoi)
            conlens.append(conlen)
            shape.append(len(conlen))
            confidence.append(np.zeros(len(conlen)).tolist())

        self._add_attr("conpois", self.line_vars, conpois)
        self._add_attr("confidence", self.line_vars, confidence)
        self._add_attr("conlens", self.line_vars, conlens)

        if printing:
            print("Total number of lines: {:.0f}".format(np.sum(shape)))
            print(shape)
        return conpois, conlens

    #%% print_number_of_lines
    def print_number_of_lines(self):
        shape = []
        for i in self.conpois:
            shape.append(len(i))
        print("Total number of lines: {:.0f}".format(np.sum(shape)))
        print(shape)

    #%%
    def make_line_overview(self):
        """
        create an image where 
        """        
        newimg,separator=make_line_overview(self.conpois, self.image)
        self.line_image=newimg
        self.separator=separator
        self.properties.append("separator")
        self.image_vars.append("line_image")
        return newimg
    
    #%% sortout_by_value

    def sortout_by_value(self, mad_threshold=4, plot=False, test=False):
        conpois = self.conpois
        conlens = self.conlens
        checkmaps = self.checkmaps

        val_med_mad = []
        confidence = []
        sortout = []
        sortoutids = []
        mad_t = np.zeros(len(conpois))
        if len(np.shape(mad_threshold)) == 0:
            mad_t += mad_threshold
        else:
            mad_t = mad_threshold
        for i in range(len(conpois)):
            sortout.append([])
            sortoutids.append([])
            conpoi = conpois[i]
            conlen = conlens[i]

            concheck = _check_image(conpoi, conlen, checkmaps[i])

            med_mad = np.zeros(2)
            med_mad[0] = np.median(concheck)
            md_concheck = concheck - med_mad[0]
            med_mad[1] = np.median(np.abs(md_concheck))
            threshold = med_mad[0] - mad_t[i] * med_mad[1]
            wrong = np.where(concheck < threshold)[0]
            confidence.append((concheck - threshold).tolist())

            print(len(wrong))
            if plot:
                plt.plot(concheck, "o")
                plt.title(str(i))
                plt.hlines(med_mad[0], 0, len(concheck),colors='b')
                plt.hlines(threshold, 0, len(concheck),colors='r')
                plt.show()

            for j in wrong:
                sortout[-1].append(conpoi[j])
                sortoutids[-1].append(j)

            val_med_mad.append(med_mad)

        locmax = []
        for i in range(len(confidence)):
            locmax.append(np.max(confidence[i]))
        confmax = np.max(locmax)

        if confmax <= 0:
            confmax = 1.0

        for i in range(len(confidence)):
            for j in range(len(confidence[i])):
                confidence[i][j] /= confmax

        if not test:

            self._update_confidence(confidence)

            self.sort_ids_out(sortoutids)

            self._add_attr("val_med_mad", self.properties, val_med_mad)

        return sortout

    #%% sortout_by_angle

    def sortout_by_angle(self, mad_threshold=8, test=False):

        conpois = self.conpois
        conlens = self.conlens
        singlemaps = self.singlemaps

        confidence = []
        sortoutids = []
        sortout = []
        ns = []
        ls = []
        slists = []
        elists = []
        tms = []
        for j in range(len(conpois)):
            sortout.append([])
            sortoutids.append([])

            conpoi = conpois[j]
            conlen = conlens[j]
            singlemap = singlemaps[j]

            testindex = np.argmax(conlen)
            [m, n, t, l], s, e = _calc_m_n_t_l(conpoi[testindex], singlemap)
            testm = np.abs(m)
            # tms.append(testm)

            mntl = np.zeros([len(conpoi), 4])
            yx = np.zeros([len(conpoi), 2])
            slist = []
            elist = []
            for i in range(len(conpoi)):
                if testm > 1:
                    mntl[i], s, e = _calc_m_n_t_l(conpoi[i][:, ::-1], singlemap.T)
                    yx[i] = (e[0] + s[0]) / 2, (e[1] + s[1]) / 2
                    slist.append(s[::-1])
                    elist.append(e[::-1])
                else:
                    mntl[i], s, e = _calc_m_n_t_l(conpoi[i], singlemap)
                    yx[i] = (e[0] + s[0]) / 2, (e[1] + s[1]) / 2
                    slist.append(s)
                    elist.append(e)

            m = np.median(mntl[:, 0])
            tms.append(m)
            deg = np.arctan(mntl[:, 0])
            mdeg = np.median(deg)
            ad = np.abs(deg - mdeg)
            mad = np.median(ad)

            wrong = np.where(ad > mad_threshold * mad)[0]
            print(len(wrong))

            confidence.append((mad_threshold * mad - ad).tolist())

            ratio = 1 / math.sqrt(m * m + 1)
            nvals = (yx[:, 0] - m * yx[:, 1]) * ratio

            nvals = []
            cslist = []
            celist = []
            lvals = []
            for i in range(len(conpoi)):
                if i in wrong:
                    sortout[-1].append(conpoi[i])
                    sortoutids[-1].append(i)
                else:
                    nvals.append((yx[i, 0] - m * yx[i, 1]) * ratio)
                    lvals.append(mntl[i, 3])
                    cslist.append(slist[i])
                    celist.append(elist[i])

            ls.append(lvals)
            ns.append(nvals)
            slists.append(cslist)
            elists.append(celist)

        locmax = []
        for i in range(len(confidence)):
            locmax.append(np.max(confidence[i]))
        confmax = np.max(locmax)

        if confmax <= 0:
            confmax = 1.0

        for i in range(len(confidence)):
            for j in range(len(confidence[i])):
                confidence[i][j] /= confmax

        if not test:
            self._update_confidence(confidence)

            self.sort_ids_out(sortoutids)

            self._add_attr("lengths", self.line_vars, ls)
            self._add_attr("ns", self.line_vars, ns)
            self._add_attr("slists", self.line_vars, slists)
            self._add_attr("elists", self.line_vars, elists)
            self._add_attr("slope_groups", self.properties, tms)
            self._add_attr("meanthickness", self.properties, np.mean(mntl[:, 2]))

        return sortout

    #%% eliminate_side_maxima_checkmaps
    def eliminate_side_maxima_checkmaps(
        self, shiftrange=20, ratio_threshold=2., test=False
    ):
        tms = self.slope_groups
        conpois = self.conpois
        checkmaps = self.checkmaps
        conlens=self.conlens
        

        sortoutids = []
        sortout = []

        for i in range(len(conpois)):
            sortout.append([])
            sortoutids.append([])

                
            binmap=np.zeros(checkmaps[i].shape,dtype=bool)
            for j in conpois[i]:
                binmap[j[:,0],j[:,1]]=True

            if tms[i] > 1:
                for j in range(len(conpois[i])):
                    check = _getcheck3(shiftrange, conpois[i][j], checkmaps[i],binmap)
                    gcheck=max(check)
                    
                    vcheck=0
                    for k in conpois[i][j]:
                        vcheck += checkmaps[i][k[0], k[1]]
                    vcheck /= conlens[i][j]
                    
                    if vcheck * ratio_threshold < gcheck:

                        sortout[-1].append(conpois[i][j])
                        sortoutids[-1].append(j)

            else:
                for j in range(len(conpois[i])):
                    check = _getcheck2(shiftrange, conpois[i][j], checkmaps[i],binmap)
                    gcheck=max(check)
                    
                    vcheck=0
                    for k in conpois[i][j]:
                        vcheck += checkmaps[i][k[0], k[1]]
                    vcheck /= conlens[i][j]
                    
                    if vcheck * ratio_threshold < gcheck:

                        sortout[-1].append(conpois[i][j])
                        sortoutids[-1].append(j)

            print(len(sortout[-1]))

        if not test:
            self.sort_ids_out(sortoutids)

        return sortout,check 

    #%% eliminate_side_maxima_checkmaps
    def eliminate_side_maxima_image(
        self, shiftrange=20, ratio_threshold=0.75,image=None,line="dark", test=False
    ):
        tms = self.slope_groups
        conpois = self.conpois
        conlens=self.conlens

        

        if line=="dark":
            if image is None:
                image=np.max(self.image)-self.image
            else:
                image=np.max(image)-image
        else:
            if image is None:
                image=self.image
            

        binmap=np.zeros(image.shape,dtype=bool)
        for i in range(len(conpois)):
            for j in conpois[i]:
                binmap[j[:,0],j[:,1]]=True

            
        sortoutids = []
        sortout = []

        for i in range(len(conpois)):
            sortout.append([])
            sortoutids.append([])

                
            if tms[i] > 1:
                for j in range(len(conpois[i])):
                    check = _getcheck3(shiftrange, conpois[i][j], image,binmap)#checkmaps[i])
                    gcheck=max(check)
                    vcheck=0
                    for k in conpois[i][j]:
                        vcheck += image[k[0], k[1]]
                    vcheck /= conlens[i][j]
                    

                    if vcheck * ratio_threshold < gcheck:# and cond[i]:

                        sortout[-1].append(conpois[i][j])
                        sortoutids[-1].append(j)

            else:
                for j in range(len(conpois[i])):
                    check = _getcheck2(shiftrange, conpois[i][j],image,binmap)#checkmaps[i])
                    gcheck=max(check)
                    vcheck=0
                    for k in conpois[i][j]:
                        vcheck += image[k[0], k[1]]
                    vcheck /= conlens[i][j]
                    
                    if vcheck * ratio_threshold < gcheck:# and cond[i]:

                        sortout[-1].append(conpois[i][j])
                        sortoutids[-1].append(j)

            print(len(sortout[-1]))

        if not test:
            self.sort_ids_out(sortoutids)

        return sortout,check 


    #%% merge_conpoi

    def merge_conpoi(
        self,
        ratio_stitched_to_orig_below=2,
        merge_below_distance=5,
        val_threshold=6,
        closeness=2,
        test=False,
    ):

        cns = self.ns
        ls = self.lengths
        slists = self.slists
        elists = self.elists
        checkmaps = self.checkmaps
        val_med_mad = self.val_med_mad
        conpois = self.conpois
        confidences = self.confidence

        sortoutids = []
        mergeparts = []

        newconpois = []
        newslists = []
        newelists = []
        newlengths = []
        newconfs = []
        newns = []
        newconlens = []

        for g in range(len(conpois)):
            sortoutids.append([])
            mergeparts.append([])

            newconpois.append([])
            newslists.append([])
            newelists.append([])
            newconfs.append([])
            newns.append([])
            newlengths.append([])
            newconlens.append([])

            ns = np.array(cns[g])

            si = np.argsort(ns)
            nsorted = ns[si]
            ndiff = np.zeros(len(ns))
            ndiff[-1] = 2 * closeness
            ndiff[:-1] = np.diff(nsorted)

            merges = np.where(ndiff < closeness)[0]

            for i in merges:
                k = si[i]
                if k not in sortoutids[-1]:
                    s1, e1 = slists[g][k], elists[g][k]
                    length1 = ls[g][k]
                    newconf = confidences[g][k]
                    newn = ns[k]
                    newconpoi = conpois[g][k]

                    already_merged = set()
                    diff = ndiff[i]
                    counter = 1
                    while diff < closeness:
                        j = si[i + counter]
                        diff += ndiff[i + counter]
                        counter += 1
                        if j not in already_merged:
                            s2, e2 = slists[g][j], elists[g][j]
                            length2 = ls[g][j]

                            outerstart, outerend, start, end = _order_points(s1, e1, s2, e2)
                            rr, cc, val = line_aa(*start.astype(int), *end.astype(int))
                            dl = np.stack((rr, cc)).T

                            tester = 0
                            count = 0
                            for p in dl:
                                tester += checkmaps[g][p[0], p[1]]
                                count += 1
                            tester /= count

                            stitchlength = math.sqrt(sum((start - end) * (start - end)))

                            cond1 = tester > val_med_mad[g][0] - val_threshold * val_med_mad[g][1]
                            cond2 = (
                                stitchlength < (length1 + length2) * ratio_stitched_to_orig_below
                            )
                            cond3 = stitchlength < merge_below_distance

                            if (cond1 and cond2) or cond3:
                                already_merged.add(j)
                                counter = 1
                                diff /= 2

                                s1, e1 = outerstart, outerend
                                newconf += confidences[g][j]
                                newconf /= 2
                                newn += ns[j]
                                newn /= 2
                                newconpoi = np.concatenate((newconpoi, dl, conpois[g][j]), axis=0)
                                newlength = math.sqrt(
                                    sum((outerstart - outerend) * (outerstart - outerend))
                                )
                                newconlen = len(newconpoi)

                                mergeparts[-1].append(dl)

                    print(len(mergeparts[-1]))
                    if len(already_merged) > 0:
                        sortoutids[-1].append(k)
                        sortoutids[-1] += list(already_merged)

                        newconpois[-1].append(newconpoi)
                        newslists[-1].append(s1)
                        newelists[-1].append(e1)
                        newconfs[-1].append(newconf)
                        newns[-1].append(newn)
                        newlengths[-1].append(newlength)
                        newconlens[-1].append(newconlen)

        if not test:
            self.sort_ids_out(sortoutids)

            for g in range(len(conpois)):
                self.conlens[g] += newconlens[g]
                self.conpois[g] += newconpois[g]
                self.slists[g] += newslists[g]
                self.elists[g] += newelists[g]
                self.confidence[g] += newconfs[g]
                self.ns[g] += newns[g]
                self.lengths[g] += newlengths[g]

        return mergeparts, newconpois

    #%% make_sets

    def _make_sets(self):
        conpois = self.conpois
        linesets = []
        for i in conpois:
            linesets.append([])
            for j in i:
                linesets[-1].append(set(map(tuple, j)))

        self._add_attr("linesets", self.line_vars, linesets)
        return linesets

    #%% get_connections

    def get_connections(self):
        if "linesets" not in self.line_vars:
            self._make_sets()

        linesets = self.linesets

        crosspoints = []
        crosslines = []
        crosslens = []

        connections = []
        for i in linesets:
            connections.append([])
            for j in i:
                connections[-1].append(set())

        for i in range(len(linesets)):
            for k in range(len(linesets[i])):
                line1 = linesets[i][k]
                for j in range(i + 1, len(linesets)):
                    for l in range(len(linesets[j])):
                        line2 = linesets[j][l]
                        cp = line1.intersection(line2)
                        if cp:
                            crosspoints.append(cp)
                            crosslens.append(len(cp))
                            crosslines.append([(i, k), (j, l)])
                            # crossdic[(i,k),(j,l)]=cp
                            connections[i][k].add((j, l))
                            connections[j][l].add((i, k))

        self._add_attr("crosspoints", self.point_vars, crosspoints)
        self._add_attr("crosslens", self.point_vars, crosslens)
        self._add_attr("crosslines", self.point_vars, crosslines)
        self._add_attr("connections", self.line_vars, connections)

        return crosspoints, crosslines, crosslens

    #%% sortout_by_confidence
    def sortout_by_confidence(self, confidence_threshold=0.3, test=False):
        sortoutids = []
        sortout = []
        confidence = self.confidence

        flatconf = []
        for i in confidence:
            flatconf += i
        threshold = np.median(flatconf) * confidence_threshold

        for i in range(len(confidence)):
            sortoutids.append([])
            sortout.append([])
            for j in range(len(confidence[i])):
                if threshold > confidence[i][j]:
                    sortoutids[-1].append(j)
                    sortout[-1].append(self.conpois[i][j])
            print(len(sortoutids[-1]))

        if not test:
            self.sort_ids_out(sortoutids)
        return sortout

    #%% shrink_extend_line
    def shrink_extend_line(self, deltapix):
        lengths = self.lengths
        slists = self.slists
        elists = self.elists

        extended_slists = []
        extended_elists = []
        shrinked_slists = []
        shrinked_elists = []

        for i in range(len(slists)):
            extended_slists.append([])
            shrinked_slists.append([])
            extended_elists.append([])
            shrinked_elists.append([])
            for j in range(len(slists[i])):
                p0 = slists[i][j]
                p1 = elists[i][j]
                dp = p1 - p0
                l = lengths[i][j]
                s_ext = (l + deltapix) / l
                s_shr = (l - deltapix) / l

                extended_elists[-1].append(p0 + dp * s_ext)
                shrinked_elists[-1].append(p0 + dp * s_shr)

                s_ext = 1 - s_ext
                s_shr = 1 - s_shr

                extended_slists[-1].append(p0 + dp * s_ext)
                shrinked_slists[-1].append(p0 + dp * s_shr)

        self._add_attr("extended_elists", self.line_vars, extended_elists)
        self._add_attr("extended_slists", self.line_vars, extended_slists)
        self._add_attr("shrinked_elists", self.line_vars, shrinked_elists)
        self._add_attr("shrinked_slists", self.line_vars, shrinked_slists)

    #%% check_ext_shr_intersection
    def check_intersection_type(self, deltapix=None):

        if deltapix is not None:
            self.shrink_extend_line(deltapix)

        ext_slists = self.extended_slists
        shr_slists = self.shrinked_slists

        ext_elists = self.extended_elists
        shr_elists = self.shrinked_elists

        s1 = set()  # test extended and all extended
        s2 = set()  # test extended and all shrinked
        s3 = set()  # test shrinked and all shrinked

        for i in range(len(ext_slists)):
            for k in range(len(ext_slists[i])):
                a1 = ext_slists[i][k]
                b1 = ext_elists[i][k]

                a2 = shr_slists[i][k]
                b2 = shr_elists[i][k]
                if i + 1 == len(ext_slists):
                    pass
                else:
                    for j in range(i + 1, len(ext_slists)):
                        for l in range(len(ext_slists[j])):
                            c1 = ext_slists[j][l]
                            d1 = ext_elists[j][l]

                            c2 = shr_slists[j][l]
                            d2 = shr_elists[j][l]

                            if intersect(a1, b1, c1, d1):
                                s1.add(((i, k), (j, l)))

                            if intersect(a1, b1, c2, d2) or intersect(a2, b2, c1, d1):
                                s2.add(((i, k), (j, l)))

                            if intersect(a2, b2, c2, d2):
                                s3.add(((i, k), (j, l)))

        blockings = s2.difference(s3)
        corners = s1.difference(s2)
        gothroughs = s3

        # crosslines = self.crosslines
        # crosslineset = set(map(tuple, crosslines))

        corners_dic = {}
        gothroughs_dic = {}
        blockings_dic = {}

        for i in corners:
            (i1, i2), (i3, i4) = i
            a = ext_slists[i1][i2]
            b = ext_elists[i1][i2]
            c = ext_slists[i3][i4]
            d = ext_elists[i3][i4]
            corners_dic[i] = lineIntersection(a, b, c, d)

        for i in blockings:
            (i1, i2), (i3, i4) = i
            a = ext_slists[i1][i2]
            b = ext_elists[i1][i2]
            c = ext_slists[i3][i4]
            d = ext_elists[i3][i4]
            blockings_dic[i] = lineIntersection(a, b, c, d)

        for i in gothroughs:
            (i1, i2), (i3, i4) = i
            a = ext_slists[i1][i2]
            b = ext_elists[i1][i2]
            c = ext_slists[i3][i4]
            d = ext_elists[i3][i4]
            gothroughs_dic[i] = lineIntersection(a, b, c, d)

        return corners_dic, blockings_dic, gothroughs_dic

    #%% sortout_zero_connections
    def sortout_zero_connections(self, test=False):
        connections = self.ext_connections
        sortoutids = []
        sortout = []

        for i in range(len(connections)):
            sortoutids.append([])
            sortout.append([])
            for j in range(len(connections[i])):
                if len(connections[i][j]) == 0:
                    sortoutids[-1].append(j)
                    sortout[-1].append(self.conpois[i][j])
            print(len(sortout[-1]))

        if not test:
            self.sort_ids_out(sortoutids)

        return sortout

    #%% sortout_short
    def sortout_short(self, threshold=20, test=False):
        if "lengths" not in self.line_vars:
            self._get_line_lengths()

        lengths = self.lengths
        sortoutids = []
        sortout = []

        for i in range(len(lengths)):
            sortoutids.append([])
            sortout.append([])
            for j in range(len(lengths[i])):
                if lengths[i][j] < threshold:
                    sortoutids[-1].append(j)
                    sortout[-1].append(self.conpois[i][j])
        
        for i in sortoutids:
            print(len(i))

        if not test:
            self.sort_ids_out(sortoutids)

        return sortout

    #%% check_misclassification
    def check_misclassification(self, shortratio=3, longratio=4, test=False):
        if "crosslines" not in self.point_vars:
            self.get_connections()

        linesets = self.linesets
        crosslines = self.crosslines
        crosslens = self.crosslens
        meanthickness = self.meanthickness

        threshold = int(np.ceil(2 * meanthickness**2))
        tocheck = np.where(np.array(crosslens) > threshold)[0]

        sortoutids = []
        sortout = []
        for i in range(len(linesets)):
            sortoutids.append([])
            sortout.append([])

        for i in tocheck:
            line1, line2 = crosslines[i]

            if len(linesets[line1[0]][line1[1]]) < shortratio * crosslens[i]:
                if len(linesets[line2[0]][line2[1]]) > longratio * crosslens[i]:
                    sortoutids[line1[0]].append(line1[1])
                    sortout[line1[0]].append(self.conpois[line1[0]][line1[1]])

            elif len(linesets[line2[0]][line2[1]]) < shortratio * crosslens[i]:
                if len(linesets[line1[0]][line1[1]]) > longratio * crosslens[i]:
                    sortoutids[line2[0]].append(line2[1])
                    sortout[line2[0]].append(self.conpois[line2[0]][line2[1]])

        for i in sortoutids:
            print(len(i))

        if not test:

            self.sort_ids_out(sortoutids)

            self.get_connections()
        return sortout

    #%% get_number_of_connections

    def get_number_of_connections(self):
        crosslines = self.crosslines
        connections = []
        for i in range(len(self.linesets)):
            connections.append()
        for i in crosslines:
            pass

    #%% pinpoint_crossings
    def pinpoint_crossings(self):
        slists = self.slists
        elists = self.elists
        crosslines = self.crosslines

        crossings = np.zeros([len(crosslines), 2])

        for i in range(len(crosslines)):
            line1, line2 = crosslines[i]
            a = slists[line1[0]][line1[1]]
            b = elists[line1[0]][line1[1]]
            c = slists[line2[0]][line2[1]]
            d = elists[line2[0]][line2[1]]
            crossings[i] = lineIntersection(a, b, c, d)

        self.crossings = crossings
        return crossings

    #%% check_ends
    def check_ends(self):
        slists = self.slists
        elists = self.slists

        crosslines = self.crosslines

        crosslineset = set(map(tuple, crosslines))

        singleends = []
        corners = []
        blockings = []

        for i in range(len(slists)):
            for j in range(len(slists[i])):
                start = slists[i][j]
                end = elists[i][j]

                meets = []
                for k in range(i + 1, len(slists)):
                    for l in range(len(slists[k])):
                        cstart = slists[k][l]
                        cend = elists[k][l]

                # check start
                # check end
