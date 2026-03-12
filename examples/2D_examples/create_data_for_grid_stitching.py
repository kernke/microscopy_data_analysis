# -*- coding: utf-8 -*-
"""
@author: kernke
"""
import numpy as np
import cv2
import scipy.datasets

def run_script():
    
    #%% take example image and transform to gray-levels and desired size   

    test = scipy.datasets.face()
    test=np.mean(test,axis=-1)
    test=cv2.resize(test[:,200:968], dsize=[2500,2500])
       
    #%% cut the big image into small pictures, that form with minor deviations a grid
    # produce also an inhomogeneous illumination for each picture and add noise
    
    b=np.linspace(-1,0.2,500)
    b=1/(b**2+0.6)
    inhom_illumination=np.outer(b,b)
  
    tests=[]
    rstart=0
    cstart=0
    randomshift=8

    for i in range(5):
        for j in range(5):
           
            noise=np.random.normal(0,np.max(test)/4,[500,500])
            tests.append(test[rstart:rstart+500,cstart:cstart+500]*inhom_illumination+noise)
    
            if j==4:
                cstart += -4*375 +8 +np.random.randint(-randomshift,randomshift)
                rstart += 375 +4 +np.random.randint(-randomshift,randomshift)
            else:
                cstart += 375 +8 +np.random.randint(-randomshift,randomshift)
                rstart +=4 +np.random.randint(-randomshift,randomshift)
            
    #%% save the series of images forming a grid
    for i in range(len(tests)):
        tests[i]=tests[i]-np.min(tests[i])
        tests[i]=tests[i]/(np.max(tests[i]))+np.random.uniform(0.2,4) #offset and change the contrast a little randomly
        tests[i]=tests[i]/(np.max(tests[i]))*255
        cv2.imwrite('image_'+str(i).zfill(2)+'.tif', tests[i].astype(np.uint8))

#%% Execute the script  
if __name__ == '__main__':
    run_script()
    