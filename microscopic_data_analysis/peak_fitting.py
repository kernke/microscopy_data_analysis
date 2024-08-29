# -*- coding: utf-8 -*-
"""
submodule focussed completely on 1D-data
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numba import njit
import pandas as pd

#%% peak_com
def center_of_mass(y,x=None,bins=None,roi=None):
    """
    calculate the center of mass of the given data y, within the region-of-interest.
    Either x or bins can be given as x-axis data, otherwise
    x is constructed as the indices enumerating y     

    Args:
        y (array_like): data value array with length N.

        x (array_like, optional): x-positions of the y-points with length N. 
        Defaults to None.

        bins (array_like, optional): thresholds surrounding x-positions of the y-points,
        with length N+1.
        Defaults to None.

        roi (tuple, optional): two thresholds in increasing order. 
        (thresholds of roi, cut only at the datapoints (no interpolation))
        Defaults to None.

    Returns:
        com (float): center of mass.

        maxpos (float or int): position of biggest value.

    """
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


#%% get_n_peaks_1d
def get_n_peaks_1d(y, x=None, n=5, roi=None,noise_integration_width=7):
    """
    Obtain n peaks from y in descending order with respect to peakheight
    Calculated by consecutively searching the maximum of y and masking it, 
    aswell as masking the decreasing flanks of the peak, 
    in order to find the next maximum in the following iteration
    to a median value of y, then repeating the process n times

    Args:
        y (array_like):
            input data.
        
        x (array_like, optional): 
            same length as y, in increasing order, 
            if x is None, it is constructed as indices enumerating y 
            Defaults to None.
        
        n (int, optional): 
            number of peaks. 
            Defaults to 5.
        
        roi (tuple, optional): 
            lower and upper threshold of region of interest.
            If None the whole dataset is evaluated
            Defaults to None.

        noise_integration_width (int, optional):
            uneven integer giving the window size for determining
            the decreasing flanks
            Defaults to 7.
            

    Returns:
        n_peaks (array_like): 
            with length n containing the positions of the n peaks.
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
        newy = np.copy(y)
        xadd = 0
    else:
        start = np.where(x >= roi[0])[0][0]
        if roi[1] >= np.max(x):
            end = len(x)
        else:
            end = np.where(x > roi[1])[0][0]
        xadd = start
        newy = np.copy(y[start:end])

    ymaxpos = np.zeros(n, dtype=int)
    maskval = np.min(y)

    startparams=np.zeros(4)
    nparams=np.zeros(4)

    pos = np.argmax(newy)
    ymaxpos[0] = pos + xadd

    for i in range(1,n):
        low=int(noise_integration_width/2)
        up=int(noise_integration_width/2)+1
        poslow=pos-low
        posup=pos+up
        
        startsection=newy[poslow:posup]
        startparams[0]=max(startsection)
        startparams[1]=min(startsection)
        startparams[2]=sum(startsection)
        startparams[3]=0

        sparams=np.copy(startparams)
        maskend=len(newy)
        for j in range(noise_integration_width,len(newy)-posup,noise_integration_width):
            nextsection=newy[poslow+j:posup+j]
            nparams[0]=max(nextsection)
            nparams[1]=min(nextsection)
            nparams[2]=sum(nextsection)
            nparams[3]=nextsection[-1]-nextsection[0]
            cond_count=np.sum(nparams>sparams)

            if cond_count > 2:
                maskend=pos+j
                break
            sparams=np.copy(nparams)

        
        startsection=newy[poslow:posup]
        sparams=np.copy(startparams)
        maskstart=0
        for j in range(noise_integration_width,poslow,noise_integration_width):
            nextsection=newy[poslow-j:posup-j]
            nparams[0]=max(nextsection)
            nparams[1]=min(nextsection)
            nparams[2]=sum(nextsection)
            nparams[3]=nextsection[0]-nextsection[-1]
            cond_count=np.sum(nparams>sparams)
            if cond_count > 2:
                maskstart=pos-j
                break
            sparams=np.copy(nparams)
                
        newy[maskstart:maskend] = maskval
        pos = np.argmax(newy)
        ymaxpos[i] = pos + xadd

    return x[ymaxpos]

#%% create rois
def create_rois(y_data,peak_positions,x_data=None):
    """
    create roi tuples, by using the minimas between peaks and mirror-boundary conditions

    Args:
        y_data (array_like): DESCRIPTION.
        
        peak_positions (list, array_like): DESCRIPTION.
        
        x_data (array_like, optional): DESCRIPTION. Defaults to None.

    Returns:
        rois (list of tuples): DESCRIPTION.

    """
    
    peak_positions=np.sort(peak_positions)
    if x_data is None:
        x_data=np.arange(len(y_data))

    ymax=np.max(y_data)
    min_indices_between_peaks=[]
    peak_indices=[]
    for i in range(len(peak_positions)-1):
        mask=np.zeros(len(y_data))
        mask[x_data<peak_positions[i]]+=ymax
        mask[x_data>peak_positions[i+1]]+=ymax
        peak_indices.append(np.where(mask==0)[0][0])                        

        min_index=np.argmin(y_data+mask)
        min_indices_between_peaks.append(min_index)

    peak_indices.append(np.where((x_data<peak_positions[-1])==0)[0][0])                        

    start=peak_indices[0]-(min_indices_between_peaks[0]-peak_indices[0])
    if start<0:
        start==0

    end=peak_indices[-1]+(peak_indices[-1]-min_indices_between_peaks[-1])
    if end >= len(x_data):
        end=len(x_data)-1
    
    thresholds=[start]+min_indices_between_peaks+[end]
    
    
    rois=[]
    for i in range(len(peak_positions)):
        rois.append((x_data[thresholds[i]],x_data[thresholds[i+1]]))

    return rois
        
    



#%% asym_pseudo_voigt ...

#https://www.webpages.uidaho.edu/brauns/vibspect1.pdf
def asym_pseudo_voigt(x,x0,A,eta,gamma_0,a):
    """
    normalized asymmetric pseudo-Voigt function (area under curve equals one 
    for amplitude A=1, meaning A can be interpreted as sample size with given 
    probability density, also implying A can differ from peakheight)
    
    
    definitions following the paper: doi:10.1016/j.vibspec.2008.02.009

    Args:
        x (float or array_like): 
            input.
        
        x0 (float): 
            maximum position.
        
        A (float): 
            amplitude.
        
        eta (float): 
            ratio of Lorentzian to Gaussian (between 0 and 1).
        
        gamma_0 (float): 
            width.
        
        a (float): 
            asymmetry.

    Returns:
        y (like x): 
            output.

    """
    xcentered=x-x0
    gamma=2*gamma_0/(1+np.exp(a*xcentered))
    gauss= 1/gamma* np.sqrt(4*np.log(2)/np.pi) *np.exp(-4*np.log(2) *(xcentered/gamma)**2)    
    lorentz=(2/(np.pi*gamma ))/(1+4*(xcentered/gamma)**2)
    voigt= ((1-eta)*gauss+eta*lorentz)*A
    return voigt

def asym_pseudo_voigt_parameter_bounds():
    """
    retrieve global bounds for the curve parameters

    Returns:
        bounds (list of lists): 
            containing lower bounds in the first and upper bounds in the second list.

    """
    lower_bounds=[-np.inf,     0,0,     0,-np.inf]
    upper_bounds=[ np.inf,np.inf,1,np.inf, np.inf]
    bounds=[lower_bounds,upper_bounds]
    return bounds

def asym_pseudo_voigt_normalized_asym(a,gamma_0):
    """
    transform the asymmetry parameter a into a range between -1 and 1
    (erases the influence of gamma_0 on a, to compare the value for different peaks)

    Args:
        a (float): 
            asymmetry parameter.
            
        gamma_0 (float): 
            width parameter.

    Returns:
        asym (float): 
            asymmetry between 0 and 1.

    """    
    return 1-2/(1+np.exp(a*gamma_0))



def asym_pseudo_voigt_center(x0,A,eta,gamma_0,a):
    """
    center position of the peak

    Args:
        x0 (float): 
            maximum position.
        A (float): 
            amplitude.
        eta (float): 
            ratio of Lorentzian to Gaussian mixture.
        gamma_0 (float): 
            width.
        a (float): 
            asymmetry.

    Returns:
        center (float): 
            center of mass.

    """
    start=x0-5*gamma_0
    end=x0+5*gamma_0
    x=np.linspace(start,end,10000)
    y=asym_pseudo_voigt(x, x0, A, eta, gamma_0, a)
    center=center_of_mass(y,x)[0]
    return center    


def asym_pseudo_voigt_peakheight(A,eta,gamma0,a):
    """
    transform amplitude A to peakheight using the further necessary function parameters 

    Args:
        A (float): 
            amplitude.
        
        eta (float): 
            ratio of Lorentzian to Gaussian (between 0 and 1).
        
        gamma0 (float): 
            width.
        
        a (float): 
            asymmetry.

    Returns:
        peakheight (float).

    """
    return A*( (1-eta)/gamma0* np.sqrt(4*np.log(2)/np.pi) +eta*2/(np.pi*gamma0)) 

    
def asym_pseudo_voigt_table(params,show=True,verbose=True):
    """
    show the fitting results in a table (pandas-dataframe, for use within jupyter)
    

    Args:
        params (list or array_like): 
            fit parameter.
        show (bool, optional): 
            set to False, to return the table only as a variable. 
            Defaults to True.
        verbose (bool, optional): 
            set to False, to return just the raw fit parameter. 
            Defaults to False.

    Returns:
        df (dataframe): 
            pandas table.

    """

    indices=[]
    for i in range(len(params)):
        indices.append("peak "+str(i+1))
    pd.set_option('colheader_justify', 'center')
    if verbose:
        res=np.zeros([len(params),11])
        res[:,0]=[elem[0] for elem in params]
        res[:,2]=[elem[1] for elem in params]    
        res[:,4]=[elem[2] for elem in params]
        res[:,7]=[elem[3] for elem in params]
        res[:,8]=[elem[4] for elem in params]
        res[:,1]=[asym_pseudo_voigt_center(*elem) for elem in params]
        res[:,3]=[asym_pseudo_voigt_peakheight(*elem[1:]) for elem in params]
        res[:,6]=(1-res[:,4])*100
        res[:,5]=res[:,4]*100
        res[:,9]=[asym_pseudo_voigt_normalized_asym(*elem[-2:][::-1]) for elem in params]
        res[:,10]=res[:,0]-res[:,1]
        
        cols=[r"$x_0$",r"$x_{center}$",r"amplitude $A$","peakheight",r"$\eta$","Gaussian","Lorentzian" 
              ,r'width $\gamma_0$',r"asym. $a$",r"norm. $a$",r"$x_0-x_{center}$"]
        df = pd.DataFrame(res, index=indices, columns=cols)

        mapper =  {r"$x_0$": '{:.3e}',
                   r"$x_{center}$": '{:.3e}',
                   r"amplitude $A$": '{:.3e}',
                   "peakheight": '{:.3e}',
                   r"$\eta$": '{0:.3f}',
                   "Gaussian": '{0:.1f}%',
                   "Lorentzian" : '{0:.1f}%',
                   r'width $\gamma_0$': '{:.3e}',
                   r"asym. $a$": '{:.3e}',
                   r"norm. $a$": '{0:.3f}',
                   r"$x_0-x_{center}$": '{:.3e}'
                    }
        
    else:

        cols=[r"$x_0$",r"$A$", r"$\eta$",r'$\gamma_0$',r"$a$"]  
        df = pd.DataFrame(params, index=indices, columns=cols)
        
        mapper =  {r'$x_0$': '{:.3e}',
                   r'$A$': '{:.3e}',
                   r'$\eta$': '{0:.3f}',
                   r'$\gamma_0$': '{:.3e}',
                  r'$a$': '{:.3e}'}
    
    if show:
        display(df.style.format(mapper))
    return df

#%% pseudo_voigt ...
#https://docs.mantidproject.org/v6.1.0/fitting/fitfunctions/PseudoVoigt.html
def pseudo_voigt(x,x0,A,eta,gamma,sigma):
    """
    calculates a normalized pseudo-Voigt profile (area under curve equals one 
    for amplitude A=1, meaning A can be interpreted as sample size with given 
    probability density, also implying A can differ from peakheight)
    
    pseudo-Voigt is a linear combination of Lorentzian and Gaussian,
    instead of a convolution in case of the real Voigt-profile
    
    definitions here following:    
    http://dx.doi.org/10.5286/SOFTWARE/MANTID#.#    
    
    Args:
        x (float or array_like): 
            input.
        
        x0 (float): 
            maximum position.
        
        A (float): 
            amplitude.
        
        eta (float): 
            ratio of Lorentzian to Gaussian (between 0 and 1).
        
        gamma (float): 
            Lorentzian width (full width half maximum) .
        
        sigma (float): 
            Gaussian width (full width half maximum)/2.355 .

    Returns:
        y (like x): 
            output.

    """
    xcentered=x-x0
    gauss=np.exp(-xcentered**2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    lorentz=1/np.pi*0.5*gamma /(0.25*gamma**2+xcentered**2)
    voigt= ((1-eta)*gauss+eta*lorentz)*A
    return voigt

def pseudo_voigt_parameter_bounds():
    """
    retrieve global bounds for the curve parameters

    Returns:
        bounds (list of lists): 
            containing lower bounds in the first and upper bounds in the second list.

    """
    lower_bounds=[-np.inf,     0,0,     0,     0]
    upper_bounds=[ np.inf,np.inf,1,np.inf,np.inf]
    bounds=[lower_bounds,upper_bounds]
    return bounds


def pseudo_voigt_peakheight(A,eta,gamma,sigma):
    """
    transform amplitude A to peakheight using the further necessary function parameters 

    Args:
        A (float): 
            amplitude.
        
        eta (float): 
            ratio of Lorentzian to Gaussian (between 0 and 1).
        
        gamma (float): 
            Lorentzian width.

        sigma (TYPE): 
            Gaussian width.

    Returns:
        peakheight (float).

    """
    return A*( (1-eta)/(sigma * np.sqrt(2 * np.pi)) +eta*2/(np.pi*gamma)) 

#%% gaussian ...

def gaussian(x,x0,A,sigma):
    """
    normalized Gaussian curve (area under curve equals one for amplitude A=1, 
    meaning A can be interpreted as sample size with given probability density,
    also implying A can differ from peakheight)

    Args:
        x (float or array_like): 
            input value.
        
        x0 (float): 
            maximum position.
        
        A (float): 
            amplitude.
        
        sigma (float): 
            width.

    Returns:
        y (like x): 
            output value.

    """
    xcentered=x-x0
    return A/(sigma * np.sqrt(2 * np.pi))*np.exp(- 0.5*(xcentered/sigma)**2)

def gaussian_A(peakheight,sigma):
    """
    calculate amplitude A from peakheight and width

    Args:
        peakheight (float): 
            positive value.
        
        sigma (float): 
            width.

    Returns:
        amplitude (float) .

    """
    return peakheight*(sigma * np.sqrt(2 * np.pi)) 

def gaussian_peakheight(A,sigma):
    """
    calculate peakheight from amplitude and width

    Args:
        A (float): 
            amplitude.
        
        sigma (float): 
            width.

    Returns:
        peakheight (float).

    """
    return A/(sigma * np.sqrt(2 * np.pi)) 


def gaussian_parameter_bounds():
    """
    retrieve global bounds for the curve parameters

    Returns:
        bounds (list of lists): 
            containing lower bounds in the first and upper bounds in the second list.

    """
    lower_bounds=[-np.inf,0,0]
    upper_bounds=[np.inf,np.inf,np.inf]
    bounds=[lower_bounds,upper_bounds]
    return bounds



#%% lorentzian ...    
def lorentzian_A(peakheight,gamma):
    """
    calculate amplitude A from peakheight and width

    Args:
        peakheight (float): 
            positive value.
        
        gamma (float): 
            width.

    Returns:
        amplitude (float) .

    """
    return peakheight*np.pi*gamma/2


    
def lorentzian_peakheight(A,gamma):
    """
    calculate peakheight from amplitude and width

    Args:
        A (float): 
            amplitude.
        
        gamma (float): 
            width.

    Returns:
        peakheight (float).

    """
    return A/(np.pi*gamma/2)

def lorentzian(x,x0,A,gamma):
    """
    normalized Lorentzian curve (area under curve equals one for amplitude A=1, 
    meaning A can be interpreted as sample size with given probability density,
    also implying A can differ from peakheight)

    Args:
        x (float or array_like): 
            input value.
        
        x0 (float): 
            maximum position.
        
        A (float): 
            amplitude.
        
        gamma (float): 
            width.

    Returns:
        y (like x): 
            output value.

    """
    xcentered=x-x0
    return A*gamma /(2*np.pi*(0.25*gamma**2+xcentered**2))

def lorentzian_parameter_bounds():
    """
    retrieve global bounds for the curve parameters

    Returns:
        bounds (list of lists): 
            containing lower bounds in the first and upper bounds in the second list.

    """
    lower_bounds=[-np.inf,0,0]
    upper_bounds=[np.inf,np.inf,np.inf]
    bounds=[lower_bounds,upper_bounds]
    return bounds


#%% calculate FWHM
def calculate_FWHM(x_data,y_data,superres=2):
    """
    calculate full width half maximum
    (robust to non uniform spacing and noise)

    Args:
        x_data (list or array_like): 
            input data.
        
        y_data (list or array_like): 
            input data.
        
        superres (float, optional): 
            DESCRIPTION. 
            Defaults to 2.

    Returns:
        fwhm (float): 
            full width half maximum value.
        
        fwhm_positions (tuple): 
            left and right peak flank positions.
        
        peak_height (float).
        
        peak_positon (float).

    """
    x=np.linspace(x_data[0],x_data[-1],int(len(x_data)*superres))
    
    I=np.interp(x,x_data,y_data)
    
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

#%% peak fit
def peak_fit(y_data,x_data=None,roi=None,plot=False,orders_of_deviation=2,verbose=False):
    """
    peak fitting routine, that contains 4 peak functions:
    Gaussian, Lorentzian, Pseudo-Voigt, Asymmetric-Pseudo-Voigt
    

    Args:
        y_data (array_like): 
            DESCRIPTION.
        
        x_data (array_like, optional): 
            if x_data is None, it is constructed as the 
            indices enumerating y_data. 
            Defaults to None.
        
        roi (tuple, optional): 
            lower and upper threshold limiting the maximum position of the peak. 
            Defaults to None.
        
        plot (bool, optional): 
            show the input data and the 4 fit-routines compare to each other  
            Defaults to False.
        
        orders_of_deviation (int, optional): 
            abbrevated as ood.
            For robustness, the optimization range Delta is limited by thresholds 
            relative to an estimate p like this: p*10^(-ood) < Delta < p*10^(ood)
            Defaults to 2.
        
        verbose (bool, optional): 
            print estimated values for parameters before optmization. 
            Defaults to False.

    Returns:
        gparams (list): 
            Gaussian fit-parameters.
        
        lparams (list): 
            Lorentzian fit-parameters.
        
        pVparams (list): 
            pseudo-Voigt fit-parameters.
        
        apVparams (list): 
            asymmetric-pseudo-Voigt fit-parameters.

    """
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
    estimated_gaussian_A=gaussian_A(A,estimated_sigma)
    estimated_lorentzian_A=lorentzian_A(A,estimated_gamma)

    if verbose:
        print("Estimates")
        print("x0: "+str(estimated_x0))
        print("sigma: "+str(estimated_sigma))
        print("gamma: "+str(estimated_gamma))

    bounds=([x[0],     0,     0  ],
            [x[-1] ,np.inf, np.inf ]) #sigma/gamma_bound=2*(x[-1]-x[0])
    gp0=(estimated_x0,estimated_gaussian_A,estimated_sigma)
    gparams, gcv = curve_fit(gaussian, x,y,p0=gp0,bounds=bounds)

    lp0=(estimated_x0,estimated_lorentzian_A,estimated_gamma)
    lparams, lcv = curve_fit(lorentzian,x,y,p0=lp0,bounds=bounds)

    gA=gaussian_peakheight(gparams[1],gparams[2])
    lA=lorentzian_peakheight(lparams[1],lparams[2])
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

    pVparams, pVcv = curve_fit(pseudo_voigt, x,y,p0=pVp0,bounds=bounds)#,method="dogbox")

    estimated_a=0
    estimated_eta=0.5
    estimated_gamma0=(estimated_gamma+estimated_sigma)/2

    
    apVp0 = (estimated_x0,estimated_pV_A,estimated_eta,estimated_gamma0,estimated_a)
    bounds=([x[0],      0,0,estimated_gamma*10**-orders,-np.inf],
            [x[-1],np.inf,1, estimated_gamma*10**orders, np.inf])
    apVparams, apVcv = curve_fit(asym_pseudo_voigt, x,y, p0=apVp0,bounds=bounds)

    if plot:
        colors=['tab:blue','tab:orange','tab:green','tab:red']
        xx=np.linspace(x[0],x[-1],int(len(y)*2))
        plt.plot(x,y,'k.',label='data')
        plt.plot(xx,gaussian(xx,*gparams),c=colors[0],label='Gaussian')
        plt.plot(xx,lorentzian(xx,*lparams),c=colors[1],label='Lorentzian')
        plt.plot(xx,pseudo_voigt(xx,*pVparams),c=colors[2],label='Pseudo Voigt')
        plt.plot(xx,asym_pseudo_voigt(xx,*apVparams),c=colors[3],label='Asym. Pseudo Voigt')
        plt.plot(xx,pseudo_voigt(xx,*pVparams),c=colors[2])
        plt.legend()
        

    return gparams,lparams,pVparams,apVparams
    
#%% sequential_peak_fit
#TODO output for other functions, not only asym pseudo Voigt
def sequential_peak_fit(y_data,x_data=None,regions_of_interest=[],plot=False,verbose=False):
    """
    fit multiple peaks succesively in descending order with respect to their peakheight. 
    the number of peaks is given by the number of tuples for regions_of_interest

    Args:
        y_data (array_like): 
            input data.
        
        x_data (array_like, optional): 
            if x_data is None, it is constructed as the 
            indices enumerating y_data. 
            Defaults to None.
        
        regions_of_interest (list of tuples, optional): 
            with each tuple containing 
            the lower and upper threshold limiting the maximum position of one peak.
            if no region is specified, the global maximum is fitted
            Defaults to [].
        
        plot (bool, optional): 
            create one plot per peak, showing each single fit. 
            Defaults to False.
        
        verbose (bool, optional): 
            print out information about each single fit. 
            Defaults to False.

    Returns:
        params (list of lists): 
            with one list with the fit-parameters for each peak.
        
        newy (array_like): 
            fitting curve, using asym_pseudo_voigt for all peaks.

    """
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
        try: 
            gparams,lparams,pVparams,apVparams=peak_fit(y,x,roi=roi,plot=plot,verbose=verbose)
        except RuntimeError:
            try:
                gparams,lparams,pVparams,apVparams=peak_fit(y,x,roi=[roi[0]*0.9,roi[1]*1.1],
                                                           plot=plot,verbose=verbose)
            except RuntimeError:
                gparams,lparams,pVparams,apVparams=peak_fit(y,x,roi=[roi[0]*1.05,roi[1]*0.95],
                                                            plot=plot,verbose=verbose)
            
        if plot:
            plt.show()
        newy+=asym_pseudo_voigt(x,*apVparams)
        params[order[i]]=apVparams
    return params,newy



#%% multiple identical functions fit
def multi_ident_func_fit(func,p0_lists,x,y,single_upper_bounds=None,single_lower_bounds=None):
    """
    simultaneous fit of multiple peaks/features with identical fitfunctions

    Args:
        func (function): 
            single fit function.
        
        p0_lists (list of lists): 
            every list containing the single fit estimates.
            As given for example  by the function sequential_peak_fit
        
        x (array_like): 
            input data.
        
        y (array_like): 
            input data.
        
        single_upper_bounds (list or array_like, optional): 
            boundaries for fit-parameters 
            applied for all peaks/features. If not given +inf is used for all parameters.
            Defaults to None.
        
        single_lower_bounds (list or array_like, optional): 
            boundaries for fit-parameters 
            applied for all peaks/features. If not given -inf is used for all parameters.
            Defaults to None.

    Returns:
        fit_parameter_lists (list of lists): 
            with one list with the fit-parameters for each peak/feature.
        
        fit_curve_y (array_like): 
            fitting curve, to compare to input y.

    """

    n=len(p0_lists)
    subn=len(p0_lists[0])
    
    if single_upper_bounds is None:
        single_upper_bounds=[np.inf for i in range(subn)]
        single_lower_bounds=[-np.inf for i in range(subn)]
    else:
        if not isinstance(single_upper_bounds,list):
            single_upper_bounds=single_upper_bounds.tolist()
        if not isinstance(single_lower_bounds,list):
            single_lower_bounds=single_lower_bounds.tolist()
            
    ubs=[]
    lbs=[]
    for i in range(n):
        ubs+=single_upper_bounds
        lbs+=single_lower_bounds
    bounds=(lbs,ubs)

    flat_p0=[]
    for i in p0_lists:
        for j in i:
            flat_p0.append(j)

    peak_func=lambda x,*params: np.sum([func(x,*params[i*subn:(i+1)*subn]) for i in range(n)],axis=0) 

    pars,cov=curve_fit(peak_func,x,y,p0=flat_p0,bounds=bounds)

    fit_parameter_lists=[]
    for i in range(n):
        fit_parameter_lists.append(pars[i*subn:(i+1)*subn])
    fit_curve_y=peak_func(x,*pars)
    
    return fit_parameter_lists,fit_curve_y



#%% SNIP

@njit
def _snip_pure_fast(y_data,m):
    y=np.log(np.log(np.sqrt(y_data+1)+1)+1)
    n=len(y)
    z=np.zeros(n)
    
    for p in range(m,0,-1):
        for i in range(p, n - p):
            b = (y[i - p] + y[i + p]) / 2
            z[i] = min(y[i], b)
        y[p:n-p]=z[p:n-p]
    return (np.exp(np.exp(y)-1)-1)**2-1    

def snip_pure(y_data,m,x_data=None):
    """
    statistics-sensitive non-linear iterative peak-clipping
    (without smoothing, vulnerable to noise)
    following the paper: doi:10.1016/j.nima.2008.11.132
    (data points should be uniformly spaced)

    Args:
        y_data (array_like): 
            uniformly spaced points recommended.
                
        m (int, float if x_data is given): 
            optimal m should be half the width w of features (w=2m+1).

        x_data (array_like,optional): 
            if x_data is None, it is constructed as the 
            indices enumerating y_data.
            Defaults to  None


    Returns:
        background_curve (array_like): 
            signal without peaks.

    """
    if x_data is None:
        x_data=np.arange(len(y_data))
    else:
        deltax=x_data[1]-x_data[0]
        m=int(np.round(m/deltax))
    
    return _snip_pure_fast(y_data,m)


# with smoothing
# TODO: allow sw>1 after taking care of boundaries
@njit
def _snip_fast(y_data,m):
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

def snip(y_data,m,x_data=None):
    """
    statistics-sensitive non-linear iterative peak-clipping
    (with smoothing width = 1, more robust to noise)
    following the paper: doi:10.1016/j.nima.2008.11.132
    (data points should be uniformly spaced)

    Args:
        y_data (array_like): 
            uniformly spaced points recommended.
        
        m (int, float if x_data is given): 
            optimal m should be half the width w of features (w=2m+1).
        
        x_data (array_like,optional): 
            if x_data is None, it is constructed as the 
            indices enumerating y_data.
            Defaults to  None

    Returns:
        background_curve (array_like): signal without peaks.

    """
    if x_data is None:
        x_data=np.arange(len(y_data))
    else:
        deltax=x_data[1]-x_data[0]
        m=int(np.round(m/deltax))

    return _snip_fast(y_data,m)

