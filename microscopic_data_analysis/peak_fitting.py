# -*- coding: utf-8 -*-
"""
@author: kernke
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import copy

#%% peak_com
def peak_com(y,x=None,roi=None,bins=None):
    if x is None:
        xx=np.arange(len(y))
    else:
        xx=x
        
    if roi is None:
        roi0=0
        roi1=len(y)-1
        pos = xx[np.argmax(y)]
    else:
        roi0=np.where(xx>=roi[0])[0]
        if len(roi0)==0:
            roi0=0
        else:
            roi0=roi0[0]

        roi1=np.where(xx>=roi[1])[0]
        if len(roi1)==0:
            roi1=len(y)-1
        else:
            roi1=roi1[0]
            
        pos = xx[roi0+np.argmax(y[roi0 : roi1+1])]

    start=roi0
    end=roi1+1

    if bins is None:
        nbins=[]
        for i in range(len(xx)-1):
            nbins.append(0.5*(xx[i]+xx[i+1]))
        startdiff=nbins[1]-nbins[0]
        enddiff=nbins[-1]-nbins[-2]
        nbins=[nbins[0]-startdiff]+nbins
        nbins.append(nbins[-1]+enddiff)
    else:
        nbins=bins
        
    xwidths=np.diff(nbins)
    
    res=np.sum(xx[start: end]*xwidths[start:end] * y[start:end]) / np.sum(y[start:end]*xwidths[start:end])

    return res, pos



#def peak_com(y, delta=None, roi=None):
#    if roi is None:
#        pos = np.argmax(y)
#    else:
#        pos = roi[0] + np.argmax(y[roi[0] : roi[1]])
#    if delta is None:
#        delta = min(pos, len(y) - pos)
#        print(delta)
#        start = pos - delta
#        end = pos + delta
#    else:
#        start = max(0, pos - delta)
#        end = min(len(y), pos + delta)
#    return np.sum(np.arange(start, end) * y[start:end]) / np.sum(y[start:end]), pos


#%%
def get_n_peaks_1d(y, x=None, delta=0, n=5, roi=None):
    """
    Obtain n maxima from y in descending order
    Calculated by consecutively finding the maximum of y
    and setting values near the maximum with distance +-delta
    to a median value of y, then repeating the process n times

    Args:
        y (array_like): input data.
        
        x (array_like, optional): same length as y, in increasing order, 
        if x is None, it is constructed as indices enumerating y 
        Defaults to None.
        
        delta (float, optional): radius masking nearby points around a  maximum
        for the iterative search of further maxima. 
        Defaults to 0.
        
        n (int, optional): number of peaks. 
        Defaults to 5.
        
        roi (tuple, optional): upper and lower threshold of region of interest. 
        Defaults to None.

    Returns:
        n_peaks (array_like): with length n containing the positions of the n peaks.

    """

    if x is None:
        x = np.arange(len(y))
    else:
        dist = 1
        while x[dist] - x[0] == 0:
            dist += 1
        if x[dist] - x[0] < 0:
            sortindex = np.argsort(x)
            x = x[sortindex]
            y = y[sortindex]

    if roi is None:
        newy = copy.deepcopy(y)
        xadd = 0
    else:
        start = np.where(x >= roi[0])[0][0]
        if roi[1] >= np.max(x):
            end = len(x)
        else:
            end = np.where(x > roi[1])[0][0]
        xadd = start
        newy = copy.deepcopy(y[start:end])

    ymaxpos = np.zeros(n, dtype=int)
    med = np.min(y)
    for i in range(n):
        pos = np.argmax(newy)
        ymaxpos[i] = pos + xadd

        delstart = np.where(x >= x[ymaxpos[i]] - delta)[0][0] - xadd
        delend = np.where(x <= x[ymaxpos[i]] + delta)[0][-1] + 1 - xadd
        if delstart < 0:
            delstart = 0
        newy[delstart:delend] = med

    return x[ymaxpos]

#%%

#https://www.webpages.uidaho.edu/brauns/vibspect1.pdf
def Asym_Pseudo_Voigt(x,x0,A,eta,gamma_0,a):
    """
    normalized asymmetric pseudo-Voigt function (area under curve equals one 
    for amplitude A=1, meaning A can be interpreted as sample size with given 
    probability density, also implying A can differ from peakheight)
    
    
    definitions following the paper: doi:10.1016/j.vibspec.2008.02.009

    Args:
        x (float or array_like): input.
        x0 (float): maximum position.
        A (float): amplitude.
        eta (float): ratio of Lorentzian to Gaussian (between 0 and 1).
        gamma_0 (float): width.
        a (float): asymmetry.

    Returns:
        y (like x): output.

    """
    xcentered=x-x0
    gamma=2*gamma_0/(1+np.exp(a*xcentered))
    gauss= 1/gamma* np.sqrt(4*np.log(2)/np.pi) *np.exp(-4*np.log(2) *(xcentered/gamma)**2)    
    lorentz=(2/(np.pi*gamma ))/(1+4*(xcentered/gamma)**2)
    voigt= ((1-eta)*gauss+eta*lorentz)*A
    return voigt


#https://docs.mantidproject.org/v6.1.0/fitting/fitfunctions/PseudoVoigt.html
def Pseudo_Voigt(x,x0,A,eta,gamma,sigma):
    """
    calculates a normalized pseudo-Voigt profile (area under curve equals one 
    for amplitude A=1, meaning A can be interpreted as sample size with given 
    probability density, also implying A can differ from peakheight)
    
    pseudo-Voigt is a linear combination of Lorentzian and Gaussian,
    instead of a convolution in case of the real Voigt-profile
    
    definitions here following:    
    http://dx.doi.org/10.5286/SOFTWARE/MANTID#.#    
    
    Args:
        x (float or array_like): input.
        
        x0 (float): maximum position.
        
        A (float): amplitude.
        
        eta (float): ratio of Lorentzian to Gaussian (between 0 and 1).
        
        gamma (float): Lorentzian width (full width half maximum) .
        
        sigma (float): Gaussian width (full width half maximum)/2.355 .

    Returns:
        y (like x): output.

    """
    xcentered=x-x0
    gauss=np.exp(-xcentered**2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    lorentz=1/np.pi*0.5*gamma /(0.25*gamma**2+xcentered**2)
    voigt= ((1-eta)*gauss+eta*lorentz)*A
    return voigt

def Asym_Pseudo_Voigt_peakheight(A,eta,gamma0,a):
    """
    transform amplitude A to peakheight using the further necessary function parameters 

    Args:
        A (float): amplitude.
        
        eta (float): ratio of Lorentzian to Gaussian (between 0 and 1).
        
        gamma0 (float): width.
        
        a (float): asymmetry.

    Returns:
        peakheight (float).

    """
    return A*( (1-eta)/gamma0* np.sqrt(4*np.log(2)/np.pi) +eta*2/(np.pi*gamma0)) 

def Pseudo_Voigt_peakheight(A,eta,gamma,sigma):
    """
    transform amplitude A to peakheight using the further necessary function parameters 

    Args:
        A (float): amplitude.
        
        eta (float): ratio of Lorentzian to Gaussian (between 0 and 1).
        
        gamma (float): Lorentzian width.

        sigma (TYPE): Gaussian width.

    Returns:
        peakheight (float).

    """
    return A*( (1-eta)/(sigma * np.sqrt(2 * np.pi)) +eta*2/(np.pi*gamma)) 

def Gaussian(x,x0,A,sigma):
    """
    normalized Gaussian curve (area under curve equals one for amplitude A=1, 
    meaning A can be interpreted as sample size with given probability density,
    also implying A can differ from peakheight)

    Args:
        x (float or array_like): input value.
        
        x0 (float): maximum position.
        
        A (float): amplitude.
        
        sigma (float): width.

    Returns:
        y (like x): output value.

    """
    xcentered=x-x0
    return A/(sigma * np.sqrt(2 * np.pi))*np.exp(- 0.5*(xcentered/sigma)**2)

def Gaussian_A(peakheight,sigma):
    """
    calculate amplitude A from peakheight and width

    Args:
        peakheight (float): positive value.
        
        sigma (float): width.

    Returns:
        amplitude (float) .

    """
    return peakheight*(sigma * np.sqrt(2 * np.pi)) 
    
def Lorentzian_A(peakheight,gamma):
    """
    calculate amplitude A from peakheight and width

    Args:
        peakheight (float): positive value.
        
        gamma (float): width.

    Returns:
        amplitude (float) .

    """
    return peakheight*np.pi*gamma/2

def Gaussian_peakheight(A,sigma):
    """
    calculate peakheight from amplitude and width

    Args:
        A (float): amplitude.
        
        sigma (float): width.

    Returns:
        peakheight (float).

    """
    return A/(sigma * np.sqrt(2 * np.pi)) 
    
def Lorentzian_peakheight(A,gamma):
    """
    calculate peakheight from amplitude and width

    Args:
        A (float): amplitude.
        
        gamma (float): width.

    Returns:
        peakheight (float).

    """
    return A/(np.pi*gamma/2)

def Lorentzian(x,x0,A,gamma):
    """
    normalized Lorentzian curve (area under curve equals one for amplitude A=1, 
    meaning A can be interpreted as sample size with given probability density,
    also implying A can differ from peakheight)

    Args:
        x (float or array_like): input value.
        
        x0 (float): maximum position.
        
        A (float): amplitude.
        
        gamma (float): width.

    Returns:
        y (like x): output value.

    """
    xcentered=x-x0
    return A*gamma /(2*np.pi*(0.25*gamma**2+xcentered**2))

#%%
def calculate_FWHM(x_data,y_data,superres=2):
    """
    Returns fwhm, fwhm_positions, peak_height, peak_positon
    """
    #cubic_interp = interp1d(x_data,y_data, kind='cubic') #interpolation for better fit

    x=np.linspace(x_data[0],x_data[-1],len(x_data)*superres)
    
    I=np.interp(x,x_data,y_data)#cubic_interp(x) 
    
    H=np.max(I)/2
    maxpos=np.argmax(I)

    previous_point_was_below=True
    lefts=[]
    for i in range(maxpos):
        if I[i]>=H and previous_point_was_below:   #check over to get posi of FWHM
            lefts.append(x[i])
            previous_point_was_below=False
        elif I[i] < H and not previous_point_was_below:
            previous_point_was_below=True
            
    previous_point_was_below=False
    rights=[]
    for i in range(maxpos,len(I)):
        if I[i]<=H and not previous_point_was_below:
            rights.append(x[i])
            previous_point_was_below=True
        elif I[i] > H and previous_point_was_below:
            previous_point_was_below=False
    
    if len(lefts)==0 or len(rights)==0:
        print("full width half maximum condition not fulfilled with peak height: "+str(2*H))

    left=np.mean(lefts)
    right=np.mean(rights)

    fwhm=right-left
    fwhm_positions=np.array([left,right])
    peak_height=2*H
    peak_positon=x[maxpos]
    return fwhm,fwhm_positions,peak_height,peak_positon    

#%%
def peak_fit(y_data,x_data=None,roi=None,plot=False,orders_of_deviation=2,verbose=False):
    orders=orders_of_deviation
    if roi is None:
        roi0=0
        roi1=len(y_data)
        if x_data is None:
            x=np.arange(len(y_data))
        else:
            x=x_data
    else:
        if x_data is None:
            x=np.arange(len(y_data))
            roi0=roi[0]
            roi1=roi[1]
        else:
            x=x_data
            roi0=np.where(x>=roi[0])[0]
            if len(roi0)==0:
                roi0=0
            else:
                roi0=roi0[0]
            roi1=np.where(x>=roi[1])[0]
            if len(roi1)==0:
                roi1=len(y_data)
            else:
                roi1=roi1[0]

    x=x[roi0:roi1]
    y=y_data[roi0:roi1]
    
    fwhm,fwhm_positions,A,A_position=calculate_FWHM(x,y)
    
    estimated_x0=(A_position+np.mean(fwhm_positions))/2
    estimated_gamma=fwhm
    estimated_sigma=fwhm/(2*np.sqrt(2*np.log(2)))
    estimated_gaussian_A=Gaussian_A(A,estimated_sigma)
    estimated_lorentzian_A=Lorentzian_A(A,estimated_gamma)

    if verbose:
        print("Estimates")
        print("x0: "+str(estimated_x0))
        print("sigma: "+str(estimated_sigma))
        print("gamma: "+str(estimated_gamma))

    bounds=([x[0],     0,     0  ],
            [x[-1] ,np.inf, 2*(x[-1]-x[0]) ])
    gp0=(estimated_x0,estimated_gaussian_A,estimated_sigma)
    gparams, gcv = curve_fit(Gaussian, x,y,p0=gp0,bounds=bounds)

    lp0=(estimated_x0,estimated_lorentzian_A,estimated_gamma)
    lparams, lcv = curve_fit(Lorentzian,x,y,p0=lp0,bounds=bounds)

    gA=Gaussian_peakheight(gparams[1],gparams[2])
    lA=Lorentzian_peakheight(lparams[1],lparams[2])
    dA=max(gA,lA,A)-min(lA,gA,A)

    l_ratio=dA/np.abs(lA-A)
    g_ratio=dA/np.abs(gA-A)
    estimated_eta=l_ratio/(l_ratio+g_ratio)
    
    estimated_pV_A=estimated_eta*estimated_lorentzian_A+(1-estimated_eta)*estimated_gaussian_A

    pVp0 = (estimated_x0,estimated_pV_A,estimated_eta,estimated_gamma,estimated_sigma)
    bounds=([x[0],     0,0,      estimated_gamma*10**-orders,estimated_sigma*10**-orders],#-np.inf,-np.inf],
            [x[-1],np.inf,1, estimated_gamma*10**orders, estimated_sigma*10**orders])
    
    if verbose:
        print(pVp0)
    pVparams, pVcv = curve_fit(Pseudo_Voigt, x,y,p0=pVp0,bounds=bounds)#,method="dogbox")

    #dx=np.abs(fwhm_positions-A_position)
    #if dx[0]>dx[1]:
    #    estimated_a=1-dx[1]/dx[0]
    #else:
    #    estimated_a=dx[0]/dx[1]-1

    estimated_a=0
    estimated_eta=0.5
    estimated_gamma0=(estimated_gamma+estimated_sigma)/2

    
    apVp0 = (estimated_x0,estimated_pV_A,estimated_eta,estimated_gamma0,estimated_a)
    bounds=([x[0],      0,0,estimated_gamma*10**-orders,-np.inf],
            [x[-1],np.inf,1, estimated_gamma*10**orders, np.inf])
    apVparams, apVcv = curve_fit(Asym_Pseudo_Voigt, x,y, p0=apVp0,bounds=bounds)

    if plot:
        xx=np.linspace(x[0],x[-1],int(len(y)*2))
        plt.plot(x,y,'k.',label='data')
        plt.plot(xx,Gaussian(xx,*gparams),label='Gaussian')
        plt.plot(xx,Lorentzian(xx,*lparams),label='Lorentzian')
        plt.plot(xx,Pseudo_Voigt(xx,*pVparams),label='Pseudo Voigt')
        plt.plot(xx,Asym_Pseudo_Voigt(xx,*apVparams),label='Asym. Pseudo Voigt')
        plt.legend()

    return gparams,lparams,pVparams,apVparams
    
#%%
def sequential_peak_fit(y_data,x_data=None,regions_of_interest=[],plot=False,verbose=False):
    if x_data is None:
        x=np.arange(len(y_data))
        rois=regions_of_interest
    else:
        x=x_data
        rois=[]
        for roi in regions_of_interest:
            roi0=np.where(x>=roi[0])[0]
            if len(roi0)==0:
                roi0=0
            else:
                roi0=roi0[0]
            roi1=np.where(x>=roi[1])[0]
            if len(roi1)==0:
                roi1=len(y_data)
            else:
                roi1=roi1[0]
            rois.append([roi0,roi1])


    n=len(rois)
    params=[[] for i in range(n)]
    maxval=np.zeros(n)
    for i in range(n):
        maxval[i]=np.max(y_data[rois[i][0]:rois[i][1]])
    order=np.argsort(maxval)[::-1]

    newy=np.zeros(len(y_data))
    for i in range(n):
        roi=regions_of_interest[order[i]]
        y=y_data-newy
        if np.min(y[rois[order[i]][0]:rois[order[i]][1]])<0:
            y-=np.min(y[rois[order[i]][0]:rois[order[i]][1]])
        gparams,lparams,pVparams,apVparams=peak_fit(y,x,roi=roi,plot=plot,verbose=verbose)
        newy+=Asym_Pseudo_Voigt(x,*apVparams)
        params[order[i]]=apVparams
    return params,newy


#%%
#following
#doi:10.1016/j.nima.2008.11.132

def snip_pure(y_data,m):
    """
    statistics-sensitive non-linear iterative peak-clipping
    (without smoothing, vulnerable to noise)
    following the paper: doi:10.1016/j.nima.2008.11.132

    Args:
        y_data (array_like): uniformly spaced points recommended.
        
        m (int): optimal m should be half the width w of features (w=2m+1).

    Returns:
        background_curve (array_like): signal without peaks.

    """
    y=np.log(np.log(np.sqrt(y_data+1)+1)+1)
    n=len(y)
    z=np.zeros(n)
    
    for p in range(m,0,-1):
        for i in range(p, n - p):
            b = (y[i - p] + y[i + p]) / 2
            z[i] = min(y[i], b)
        y[p:n-p]=z[p:n-p]
    return (np.exp(np.exp(y)-1)-1)**2-1

# with smoothing
# TODO: allow sw>1 after taking care of boundaries
def snip(y_data,m):
    """
    statistics-sensitive non-linear iterative peak-clipping
    (with smoothing width = 1, more robust to noise)
    following the paper: doi:10.1016/j.nima.2008.11.132

    Args:
        y_data (array_like): uniformly spaced points recommended.
        
        m (int): optimal m should be half the width w of features (w=2m+1).

    Returns:
        background_curve (array_like): signal without peaks.

    """
    y=np.log(np.log(np.sqrt(y_data+1)+1)+1)
    n=len(y)
    sw=1
    z=np.zeros(n)
    for p in range(m,0,-1):
        for i in range(p, n - p):
            b = (y[i - p] + y[i + p]) / 2
            if b<y[i]:
                z[i]=b
            else:
                z[i]=1/(2*sw+1) *np.sum(y[i-sw:i+sw+1])
        y[p:n-p]=z[p:n-p]
    return (np.exp(np.exp(y)-1)-1)**2-1
#%%
def multi_ident_func_fit(func,p0_lists,x,y,single_upper_bounds=None,single_lower_bounds=None):

    n=len(p0_lists)
    subn=len(p0_lists[0])
    
    if single_upper_bounds is None:
        single_upper_bounds=[np.inf for i in range(subn)]
        single_lower_bounds=[-np.inf for i in range(subn)]
    
    ub=[]
    for i in single_upper_bounds:
        ub.append(i)
    lb=[]
    for i in single_lower_bounds:
        lb.append(i)
    ubs=[]
    lbs=[]
    for i in range(n):
        ubs+=ub
        lbs+=lb
    bounds=(lbs,ubs)

    flat_p0=[]
    for i in p0_lists:
        for j in i:
            flat_p0.append(j)

    peak_func=lambda x,*params: np.sum([func(x,*params[i*subn:(i+1)*subn]) for i in range(n)],axis=0) 

    pars,cov=curve_fit(peak_func,x,y,p0=flat_p0,bounds=bounds)

    res=[]
    for i in range(n):
        res.append(pars[i*subn:(i+1)*subn])
    
    return res,peak_func(x,*pars)