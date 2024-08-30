# -*- coding: utf-8 -*-
"""
create dummy data 1D
"""

import numpy as np
import microscopy_data_analysis as mda


def run_script():
    # create x-data 
    xbins=np.linspace(0,5,1001)
    x=mda.bin_centering(xbins)
    
    # start with some curve
    y=mda.gaussian(x,x0=2,A=5000,sigma=2)
    
    # add some noise to the curve
    random_sample=np.random.uniform(0,5,size=10**6)
    hist,bin_edges=np.histogram(random_sample,bins=xbins)
    y+=hist
    
    
    # add gaussian peaks
    peak_positions=np.array([1,1.8,3,4.2])
    peak_widths=np.array([0.08,0.18,0.13,0.1])
    peak_amplitudes=np.array([10**5,4*10**4,1*10**5,2*10**5])
    
    for i in range(4):
        random_sample=np.random.normal(peak_positions[i],peak_widths[i],peak_amplitudes[i])
        hist,bin_edges=np.histogram(random_sample,bins=xbins)
        y+=hist

    # manipulate the third peak to create some asymmetry
    asym_sample=np.random.normal(peak_positions[2],peak_widths[2]*3,int(peak_amplitudes[2]*2))
    hist,bin_edges=np.histogram(asym_sample,bins=xbins)
    y+=hist*(1.6+np.arctan(x**3-3**3))
    
    # add Lorentzian contributions to the peaks
    peak_amplitude_factors=np.array([2,1,0.5,1])
    cauchy_peak_amplitudes=(peak_amplitude_factors*peak_amplitudes).astype(int)
    cauchy_peak_widths=peak_widths*np.array([2.4,1.2,3.6,1])
    cauchy_pos=peak_positions+np.array([0.05,0.1,0.,0.])
    
    for i in range(4):
        random_sample=np.random.standard_cauchy(cauchy_peak_amplitudes[i])*cauchy_peak_widths[i]+cauchy_pos[i]
        hist,bin_edges=np.histogram(random_sample,bins=xbins)
        y+=hist
    
    # finally, some manipulation to make it more realistic and not get perfect peak-shapes    
    y*=np.log(x+2)

    # as an extra a peak with more overlap to another is added
    random_sample=np.random.normal(0.75,0.07,6*10**4)
    hist,bin_edges=np.histogram(random_sample,bins=xbins)
    y+=hist
    
    return x,y

