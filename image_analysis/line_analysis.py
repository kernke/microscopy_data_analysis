import numpy as np
import cv2
from skimage.draw import line_aa
from numba import njit
import matplotlib.pyplot as plt
import math


#%% get_connected_points
def get_connected_points(srb):
    indices = np.argwhere(srb)
    pointset = set()
    for i in range(len(indices)):
        pointset.add((indices[i, 0], indices[i, 1]))

    conpoi = []
    conlen = []
    while len(pointset) > 0:
        conpoi.append([])
        conlen.append(1)
        point0, point1 = pointset.pop()
        conpoi[-1].append((point0, point1))
        points_to_check = [(point0, point1)]
        while len(points_to_check) > 0:
            newpoints_to_check = []

            for i in range(len(points_to_check)):
                point0, point1 = points_to_check[i]

                if point0 + 1 < srb.shape[0] and srb[point0 + 1, point1]:
                    pointlabel = (point0 + 1, point1)
                    if pointlabel in pointset:
                        conlen[-1] += 1
                        conpoi[-1].append(pointlabel)
                        pointset.discard(pointlabel)
                        newpoints_to_check.append(pointlabel)
                if point1 + 1 < srb.shape[1] and srb[point0, point1 + 1]:
                    pointlabel = (point0, point1 + 1)
                    if pointlabel in pointset:
                        conlen[-1] += 1
                        conpoi[-1].append(pointlabel)
                        pointset.discard(pointlabel)
                        newpoints_to_check.append(pointlabel)
                if point0 > 0 and srb[point0 - 1, point1]:
                    pointlabel = (point0 - 1, point1)
                    if pointlabel in pointset:
                        conlen[-1] += 1
                        conpoi[-1].append(pointlabel)
                        pointset.discard(pointlabel)
                        newpoints_to_check.append(pointlabel)
                if point1 > 0 and srb[point0, point1 - 1]:
                    pointlabel = (point0, point1 - 1)
                    if pointlabel in pointset:
                        conlen[-1] += 1
                        conpoi[-1].append(pointlabel)
                        pointset.discard(pointlabel)
                        newpoints_to_check.append(pointlabel)

            points_to_check = newpoints_to_check

    for i in range(len(conpoi)):
        conpoi[i] = np.array(conpoi[i])

    return conpoi, conlen


#%% check_checkmap
def check_checkmap(conpoi, conlen, checkmap):
    concheck = np.zeros(len(conpoi))
    for i in range(len(conpoi)):
        for j in conpoi[i]:
            concheck[i] += checkmap[j[0], j[1]]
    return concheck / np.array(conlen)


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


#%% calc_m_n_t


@njit
def calc_m_n_t(points, singlemap):
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
    m = dy / dx

    y = (e[0] + s[0]) / 2
    x = (e[1] + s[1]) / 2
    n = y - m * x

    return [m, n, t], s, e


#%% getcheck
@njit
def getcheck0(shiftrange, points, image):
    check = np.zeros(2 * shiftrange + 1)
    checklen = np.zeros(2 * shiftrange + 1) + len(points)
    cshape0, cshape1 = image.shape
    # cshape0 -= 1
    # cshape1 -= 1
    for i in range(-shiftrange, shiftrange + 1):
        for j in points:
            if j[0] + i >= cshape0 or j[1] >= cshape1 or j[0] + i < 0 or j[1] < 0:
                checklen[i + shiftrange] -= 1
            else:
                check[i + shiftrange] += image[j[0] + i, j[1]]
    checklen[checklen == 0] = 1
    return check / checklen


@njit
def getcheck1(shiftrange, points, image):
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


# @njit
# def getcheck1(shiftrange,points,image):
#    check=np.zeros(2*shiftrange+1)
#    for i in range(-shiftrange,shiftrange+1):
#        for j in points:
#            check[i+shiftrange]+=image[j[0],j[1]+i]
#    return check

#%% line_analysis_object (class)


class line_analysis_object:
    def __init__(self, image, singlemaps, checkmaps):

        self.image = image
        self.singlemaps = singlemaps
        self.checkmaps = checkmaps

        self.reducibles = []
        # self.reduc_crossings=[]

    #%% all_connected_points
    def all_connected_points(self):
        singlemaps = self.singlemaps
        conpois = []
        conlens = []
        for i in range(len(singlemaps)):
            conpoi, conlen = get_connected_points(singlemaps[i])
            conpois.append(conpoi)
            conlens.append(conlen)

        self.conpois = conpois
        self.conlens = conlens

        self.reducibles.append("conpois")
        self.reducibles.append("conlens")
        return conpois, conlens

    #%% sort_ids_out
    def sort_ids_out(self, sortoutids):
        for attr in self.reducibles:
            var = getattr(self, attr)
            for s in range(len(sortoutids)):
                tosortout = sortoutids[s]
                for i in sorted(tosortout, reverse=True):
                    del var[s][i]

    #%% sortout_by_value

    def sortout_by_value(self, mad_threshold=4, plot=False, test=False):
        conpois = self.conpois
        conlens = self.conlens
        checkmaps = self.checkmaps

        concheckparams = []
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
            # confidence.append([])
            sortoutids.append([])
            conpoi = conpois[i]
            conlen = conlens[i]

            concheck = check_checkmap(conpoi, conlen, checkmaps[i])

            concheckparam = np.zeros(2)
            concheckparam[0] = np.median(concheck)
            md_concheck = concheck - concheckparam[0]
            concheckparam[1] = np.median(np.abs(md_concheck))

            threshold = concheckparam[0] - mad_t[i] * concheckparam[1]
            wrong = np.where(concheck < threshold)[0]

            confidence.append((concheck - threshold).tolist())

            print(len(wrong))
            if plot:
                plt.plot(concheck, "o")
                plt.title(str(i))
                plt.hlines(-mad_t[i] * concheckparam[1], 0, len(concheck))
                plt.show()

            for j in wrong:
                sortout[-1].append(conpoi[j])
                sortoutids[-1].append(j)

            concheckparams.append(concheckparam)

        locmax = []
        for i in range(len(confidence)):
            locmax.append(np.max(confidence[i]))
        confmax = np.max(locmax)

        if confmax <= 0:
            confmax = 1.0

        for i in range(len(confidence)):
            for j in range(len(confidence[i])):
                confidence[i][j] /= confmax

        self.confidence = confidence
        self.reducibles.append("confidence")

        if not test:
            self.sort_ids_out(sortoutids)

            self.concheckparams = concheckparams

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
            [m, n, t], s, e = calc_m_n_t(conpoi[testindex], singlemap)
            testm = np.abs(m)
            tms.append(testm)

            mnt = np.zeros([len(conpoi), 3])
            yx = np.zeros([len(conpoi), 2])
            slist = []
            elist = []
            for i in range(len(conpoi)):
                if testm > 1:
                    mnt[i], s, e = calc_m_n_t(conpoi[i][:, ::-1], singlemap.T)
                    yx[i] = (e[0] + s[0]) / 2, (e[1] + s[1]) / 2
                    slist.append(s[::-1])
                    elist.append(e[::-1])
                else:
                    mnt[i], s, e = calc_m_n_t(conpoi[i], singlemap)
                    yx[i] = (e[0] + s[0]) / 2, (e[1] + s[1]) / 2
                    slist.append(s)
                    elist.append(e)

            m = np.median(mnt[:, 0])
            deg = np.arctan(mnt[:, 0])
            mdeg = np.median(deg)
            ad = np.abs(deg - mdeg)
            mad = np.median(ad)

            wrong = np.where(ad > mad_threshold * mad)[0]
            print(len(wrong))

            confidence.append((mad_threshold * mad - ad).tolist())

            ratio = 1 / np.sqrt(m * m + 1)
            nvals = (yx[:, 0] - m * yx[:, 1]) * ratio

            nvals = []
            cslist = []
            celist = []
            for i in range(len(conpoi)):
                if i in wrong:
                    sortout[-1].append(conpoi[i])
                    sortoutids[-1].append(i)
                else:
                    nvals.append((yx[i, 0] - m * yx[i, 1]) * ratio)
                    cslist.append(slist[i])
                    celist.append(elist[i])

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
            for i in range(len(confidence)):
                for j in range(len(confidence[i])):
                    self.confidence[i][j] += confidence[i][j]

            self.sort_ids_out(sortoutids)

            self.ns = ns
            self.slists = slists
            self.elists = elists
            self.tms = tms
            self.meanthickness = np.mean(mnt[:, 2])

            self.reducibles.append("ns")
            self.reducibles.append("slists")
            self.reducibles.append("elists")

        return sortout

    #%% eliminate_side_maxima_checkmaps
    def eliminate_side_maxima_checkmaps(
        self, shiftrange=2, tol=1, valfactor=2.5, test=False
    ):
        tms = self.tms
        conpois = self.conpois
        checkmaps = self.checkmaps

        sortoutids = []
        sortout = []

        for i in range(len(conpois)):
            sortout.append([])
            sortoutids.append([])

            image = checkmaps[i]
            if tms[i] > 1:
                for j in range(len(conpois[i])):
                    check = getcheck1(shiftrange, conpois[i][j], image)

                    vcheck = np.max(check[shiftrange - tol : shiftrange + tol])
                    check = check[check.astype(bool)]
                    checkmed = np.median(check)

                    if vcheck < valfactor * checkmed:

                        sortout[-1].append(conpois[i][j])
                        sortoutids[-1].append(j)

            else:
                for j in range(len(conpois[i])):
                    check = getcheck0(shiftrange, conpois[i][j], image)

                    vcheck = np.max(check[shiftrange - tol : shiftrange + tol])
                    check = check[check.astype(bool)]
                    checkmed = np.median(check)
                    if vcheck < valfactor * checkmed:
                        sortout[-1].append(conpois[i][j])
                        sortoutids[-1].append(j)

            print(len(sortout[-1]))

        if not test:
            self.sort_ids_out(sortoutids)

        return sortout

    #%% eliminate_side_maxima_image
    def eliminate_side_maxima_image(
        self, image, shiftrange=2, tol=1, valfactor=2.5, line="dark", test=False
    ):
        tms = self.tms
        conpois = self.conpois
        # image=self.image

        sortoutids = []
        sortout = []

        if line == "bright":

            for i in range(len(conpois)):
                sortout.append([])
                sortoutids.append([])

                if tms[i] > 1:
                    for j in range(len(conpois[i])):
                        check = getcheck1(shiftrange, conpois[i][j], image)

                        vcheck = np.max(check[shiftrange - tol : shiftrange + tol])
                        check = check[check.astype(bool)]
                        checkmed = np.median(check)

                        if vcheck < valfactor * checkmed:

                            sortout[-1].append(conpois[i][j])
                            sortoutids[-1].append(j)

                else:
                    for j in range(len(conpois[i])):
                        check = getcheck0(shiftrange, conpois[i][j], image)

                        vcheck = np.max(check[shiftrange - tol : shiftrange + tol])
                        check = check[check.astype(bool)]
                        checkmed = np.median(check)
                        if vcheck < valfactor * checkmed:
                            sortout[-1].append(conpois[i][j])
                            sortoutids[-1].append(j)

                print(len(sortout[-1]))

        else:
            for i in range(len(conpois)):
                sortout.append([])
                sortoutids.append([])

                if tms[i] > 1:
                    for j in range(len(conpois[i])):
                        check = getcheck1(shiftrange, conpois[i][j], image)

                        vcheck = np.max(check[shiftrange - tol : shiftrange + tol])
                        check = check[check.astype(bool)]
                        checkmed = np.median(check)

                        if vcheck > valfactor * checkmed:

                            sortout[-1].append(conpois[i][j])
                            sortoutids[-1].append(j)

                else:
                    for j in range(len(conpois[i])):
                        check = getcheck0(shiftrange, conpois[i][j], image)

                        vcheck = np.max(check[shiftrange - tol : shiftrange + tol])
                        check = check[check.astype(bool)]
                        checkmed = np.median(check)
                        if vcheck > valfactor * checkmed:
                            sortout[-1].append(conpois[i][j])
                            sortoutids[-1].append(j)

                print(len(sortout[-1]))

        if not test:
            self.sort_ids_out(sortoutids)

        return sortout

    # #%%
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

    #%% eliminate_side_maxima
    def eliminate_side_maxima(self, mad_threshold=2, shiftrange=10, test=False):
        tms = self.tms
        conpois = self.conpois
        checkmaps = self.checkmaps

        sortoutids = []
        sortout = []

        for i in range(len(conpois)):
            sortout.append([])
            sortoutids.append([])

            image = checkmaps[i]
            if tms[i] > 1:
                for j in range(len(conpois[i])):

                    a = np.max(conpois[i][j][:, 1])
                    b = np.min(conpois[i][j][:, 1])
                    if a + shiftrange >= image.shape[1] or b - shiftrange < 0:
                        pass
                    else:
                        check = getcheck1(shiftrange, conpois[i][j], image)
                        if np.argmax(check) != shiftrange:

                            checkmedian = np.median(check)
                            mad = np.median(np.abs(check - checkmedian))
                            if check[shiftrange] < checkmedian + mad_threshold * mad:
                                sortout[-1].append(conpois[i][j])
                                sortoutids[-1].append(j)

            else:
                for j in range(len(conpois[i])):

                    a = np.max(conpois[i][j][:, 0])
                    b = np.min(conpois[i][j][:, 0])
                    if a + shiftrange >= image.shape[0] or b - shiftrange < 0:
                        pass
                    else:
                        check = getcheck0(shiftrange, conpois[i][j], image)
                        if np.argmax(check) != shiftrange:

                            checkmedian = np.median(check)
                            mad = np.median(np.abs(check - checkmedian))
                            if check[shiftrange] < checkmedian + mad_threshold * mad:
                                sortout[-1].append(conpois[i][j])
                                sortoutids[-1].append(j)

            print(len(sortout[-1]))

        if not test:
            self.sort_ids_out(sortoutids)

        return sortout

    #%% merge_conpoi

    def merge_conpoi(
        self,
        relative_length=2,
        absolute_distance=5,
        val_threshold=6,
        closeness=2,
        test=False,
    ):

        cns = self.ns
        slists = self.slists
        elists = self.elists
        checkmaps = self.checkmaps
        concheckparams = self.concheckparams
        conpois = self.conpois
        confidences = self.confidence

        newconpois = []
        newslists = []
        newelists = []
        mergedones = []
        newconf = []

        for g in range(len(conpois)):
            confidence = confidences[g]
            slist = slists[g]
            elist = elists[g]
            checkmap = checkmaps[g]
            newconpoi = []
            ns = np.array(cns[g])
            newslists.append([])
            newelists.append([])
            newconf.append([])
            concheckparam = concheckparams[g]
            conpoi = conpois[g]

            si = np.argsort(ns)

            nsorted = ns[si]
            ndiff = np.diff(nsorted)

            merges = np.where(ndiff < closeness)[0]
            # print(merges)
            merge = False
            mergedones.append([])

            for i in range(len(si)):
                k = si[i]

                if merge:
                    s2, e2 = slist[k], elist[k]

                    if np.sum((s1 - e2) * (s1 - e2)) > np.sum((s2 - e1) * (s2 - e1)):
                        outerstart = s1
                        outerend = e2
                        start = e1
                        end = s2
                    else:
                        outerstart = s2
                        outerend = e2
                        start = s1
                        end = e2
                    rr, cc, val = line_aa(*start.astype(int), *end.astype(int))
                    dl = np.stack((rr, cc)).T

                    tester = 0
                    count = 0
                    for j in dl:
                        tester += checkmap[j[0], j[1]]
                        count += 1
                    tester /= count

                    stitchlength = np.sqrt(np.sum((start - end) * (start - end)))
                    length1 = np.sqrt(np.sum((s1 - e1) * (s1 - e1)))
                    length2 = np.sqrt(np.sum((s1 - e1) * (s1 - e1)))

                    cond1 = tester > concheckparam[0] - val_threshold * concheckparam[1]
                    cond2 = relative_length * stitchlength < length1 + length2
                    cond3 = stitchlength < absolute_distance
                    if (cond1 and cond2) or cond3:

                        if i in merges:
                            newconpoi[-1] = np.concatenate(
                                (newconpoi[-1], dl, conpoi[k]), axis=0
                            )
                            merge = True
                            s1, e1 = outerstart, outerend
                            newslists[-1][-1] = outerstart
                            newelists[-1][-1] = outerend
                            newconf[-1][-1] += confidence[k]
                            newconf[-1][-1] /= 2
                            mergedones[-1].append(dl)
                        else:

                            newconpoi[-1] = np.concatenate(
                                (newconpoi[-1], dl, conpoi[k]), axis=0
                            )
                            merge = False
                            newslists[-1][-1] = outerstart
                            newelists[-1][-1] = outerend
                            newconf[-1][-1] += confidence[k]
                            newconf[-1][-1] /= 2
                            mergedones[-1].append(dl)

                    else:
                        newconpoi.append(conpoi[k])
                        newslists[-1].append(slist[k])
                        newelists[-1].append(elist[k])
                        newconf[-1].append(confidence[k])
                        merge = False
                        if i in merges:
                            merge = True
                            s1, e1 = slist[k], elist[k]

                else:
                    newconpoi.append(conpoi[k])
                    newslists[-1].append(slist[k])
                    newelists[-1].append(elist[k])
                    newconf[-1].append(confidence[k])
                    if i in merges:
                        merge = True
                        s1, e1 = slist[k], elist[k]

            print(len(mergedones[-1]))

            newconpois.append(newconpoi)

        if not test:
            newconlens = []
            for i in newconpois:
                newconlens.append([])
                for j in i:
                    newconlens[-1].append(len(i))
            self.conlens = newconlens
            self.conpois = newconpois
            self.slists = newslists
            self.elists = newelists
            self.confidence = newconf

        return mergedones, newconpois

    #%% make_sets

    def make_sets(self):
        conpois = self.conpois
        linesets = []
        for i in conpois:
            linesets.append([])
            for j in i:
                linesets[-1].append(set(map(tuple, j)))
        self.linesets = linesets
        self.reducibles.append("linesets")
        return linesets

    #%% get_connections

    def get_connections(self):
        linesets = self.linesets

        crosspoints = []
        crosslines = []
        crosslens = []
        # crossdic={}

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

        self.crosspoints = crosspoints
        self.crosslines = crosslines
        self.crosslens = crosslens
        self.connections = connections
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

    #%% get_line_lengths
    def get_line_lengths(self):
        slists = self.slists
        elists = self.elists

        lengths = []

        for i in range(len(slists)):
            lengths.append([])
            for j in range(len(slists[i])):
                d = slists[i][j] - elists[i][j]
                lengths[-1].append(math.sqrt(np.sum(d * d)))

        self.lengths = lengths

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

        self.extended_elists = extended_elists
        self.extended_slists = extended_slists
        self.shrinked_elists = shrinked_elists
        self.shrinked_slists = shrinked_slists

    #%% check_ext_shr_intersection
    def check_ext_shr_intersection(self):
        ext_slists = self.extended_slists
        shr_slists = self.shrinked_slists

        ext_elists = self.extended_elists
        shr_elists = self.shrinked_elists

        # ext_connections=[]
        # shr_connections=[]
        # ext_crosslineset=set()
        # shr_crosslineset=set()
        s1 = set()  # test extended and all extended
        s2 = set()  # test extended and all shrinked
        s3 = set()  # test shrinked and all shrinked
        # for i in ext_slists:
        #    ext_connections.append([])
        #    shr_connections.append([])
        #    for j in i:
        #        ext_connections[-1].append(set())
        #        shr_connections[-1].append(set())

        for i in range(len(ext_slists)):
            for k in range(len(ext_slists[i])):
                a1 = ext_slists[i][k]
                b1 = ext_elists[i][k]

                a2 = shr_slists[i][k]
                b2 = shr_elists[i][k]
                for j in range(len(ext_slists)):
                    if i == j:
                        pass
                    else:
                        for l in range(len(ext_slists[j])):
                            c1 = ext_slists[j][l]
                            d1 = ext_elists[j][l]

                            c2 = shr_slists[j][l]
                            d2 = shr_elists[j][l]

                            if intersect(a1, b1, c1, d1):
                                s1.add(((i, k), (j, l)))
                                # ext_connections[i][k].add((j,l))
                                # ext_connections[j][l].add((i,k))
                                # ext_crosslineset.add(((i,k),(j,l)))
                            if intersect(a1, b1, c2, d2):
                                s2.add(((i, k), (j, l)))

                            if intersect(a2, b2, c2, d2):
                                s3.add(((i, k), (j, l)))

                            # shr_connections[i][k].add((j,l))
                            # shr_connections[j][l].add((i,k))
                            # shr_crosslineset.add(((i,k),(j,l)))

        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
        # self.ext_connections=ext_connections
        # self.shr_connections=shr_connections
        # self.ext_crosslineset=ext_crosslineset
        # self.shr_crosslineset=shr_crosslineset

    #%% make_interaction_dictionnary
    def make_interaction_dictionnary(self):
        crosslines = self.crosslines
        crosslineset = set(map(tuple, crosslines))
        s1 = self.s1
        s2 = self.s2
        s3 = self.s3

        slists = self.extended_slists
        elists = self.extended_elists

        corners = s1.difference(s2)
        gothroughs = s3
        blockings = s2.difference(s3)

        corners_dic = {}
        gothroughs_dic = {}
        blockings_dic = {}

        for i in corners:
            (i1, i2), (i3, i4) = i
            a = slists[i1][i2]
            b = elists[i1][i2]
            c = slists[i3][i4]
            d = elists[i3][i4]
            corners_dic[i] = lineIntersection(a, b, c, d)

        for i in blockings:
            (i1, i2), (i3, i4) = i
            a = slists[i1][i2]
            b = elists[i1][i2]
            c = slists[i3][i4]
            d = elists[i3][i4]
            blockings_dic[i] = lineIntersection(a, b, c, d)

        for i in gothroughs:
            (i1, i2), (i3, i4) = i
            a = slists[i1][i2]
            b = elists[i1][i2]
            c = slists[i3][i4]
            d = elists[i3][i4]
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

        if not test:
            self.sort_ids_out(sortoutids)

        return sortout

    #%% check_misclassification
    def check_misclassification(self, shortratio=3, longratio=4, test=False):
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
