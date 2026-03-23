"""
@author: kernke
"""
import os

import cv2
import numpy as np
import scipy.datasets


def run_script():
    # create folder for example images
    os.makedirs("example_images", exist_ok=True)

    # take example image and transform to gray-levels and desired size   
    test = scipy.datasets.face()
    test=np.mean(test,axis=-1)
    test=cv2.resize(test[:,200:968], dsize=[2500,2500])
       
    # cut the big image into small pictures, that form with minor deviations a grid
    # produce also an inhomogeneous illumination for each picture and add noise 
    square_tile_dimension=500

    b=np.linspace(-1,0.2,square_tile_dimension)
    b=1/(b**2+0.6)
    inhom_illumination=np.outer(b,b)
  
    tests=[]
    rstart=0
    cstart=0
    shift=30
    offset=shift
    hdrift=0
    vdrift=0

    for i in range(5): 
        for j in range(5):
           
            noise=np.random.normal(0,np.max(test)/4,[square_tile_dimension,square_tile_dimension])

            cstart = offset + j * 375 + np.random.randint(-shift,shift) + hdrift
            rstart = offset + i * 375 + np.random.randint(-shift,shift) + vdrift
            hdrift += np.random.randint(30)
            vdrift += np.random.randint(20)
            tests.append(test[rstart:rstart+square_tile_dimension,
                              cstart:cstart+square_tile_dimension]*inhom_illumination+noise)


    # save the series of images forming a grid
    for i in range(len(tests)):
        tests[i]=tests[i]-np.min(tests[i])

        #offset and change the contrast a little randomly
        tests[i]=tests[i]/(np.max(tests[i]))+np.random.uniform(0.2,4) 
        
        tests[i]=tests[i]/(np.max(tests[i]))*255
        cv2.imwrite('example_images/image_'+str(i).zfill(2)+'.tif', 
                    tests[i].astype(np.uint8))

# Execute the script  
if __name__ == '__main__':
    run_script()
    