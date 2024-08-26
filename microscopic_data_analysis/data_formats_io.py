# -*- coding: utf-8 -*-
"""
submodule focussed on dataformats
"""
import numpy as np
import h5py
import ast
import datetime
import os
import cv2

import ncempy.io as nio
from contextlib import redirect_stdout
import io

from .image_processing import img_to_uint8



def get_emd_with_metadata(filepath):
    """
    Read TEM-imgages as .emd-files 

    Args:
        filepath (string): relative or absolute path to the file.

    Returns:
        image (MxN array_like): 2D image with only one intensity-channel (gray-scale).
        
        metadata (dict): Dictionary containing the most important metadata.
    """
    image_with_metadata=h5py.File(filepath)
    metadata=dict()
    kw=list(image_with_metadata["Data/Image"].keys())[0]
    image=image_with_metadata["Data/Image/"+kw+"/Data"][:]

    ascii_char=""
    for num in image_with_metadata["Data/Image/"+kw+"/Metadata"][:,0]:
        ascii_char += chr(num)
        
    allmetadata=ast.literal_eval(ascii_char[:ascii_char.find("\n")])
    #print(allmetadata)
    
    unixtimestamp=allmetadata["Acquisition"]["AcquisitionStartDatetime"]["DateTime"]
    metadata["unix_timestamp"]=int(unixtimestamp)
    uts=datetime.datetime.fromtimestamp(int(unixtimestamp))
    metadata["datetime"]=uts.strftime('%Y.%m.%d, %H:%M:%S')

    metadata["beam_mode"]=allmetadata["Optics"]["IlluminationMode"]
    metadata["detector_name"]=allmetadata["BinaryResult"]["Detector"]


    pix_to_nm=np.double(allmetadata["BinaryResult"]["PixelSize"]["width"])*10**9
    metadata["pixelsize_nm"]=pix_to_nm
    
    metadata["camera_length_mm"]=np.double(allmetadata["Optics"]["CameraLength"])*1000
    metadata["frame_time_sec"]=np.double(allmetadata["Scan"]["FrameTime"])
    if metadata["detector_name"]=="HAADF":    
        metadata["dwell_time_microsec"]=np.double(allmetadata["Scan"]["DwellTime"])*10**6
    else:
        metadata["exposure_time_sec"]=np.double(allmetadata["Detectors"]["Detector-0"]["ExposureTime"])
    
    metadata["x_mm"]=np.double(allmetadata["Stage"]["Position"]["x"])*1000
    metadata["y_mm"]=np.double(allmetadata["Stage"]["Position"]["y"])*1000
    metadata["z_mm"]=np.double(allmetadata["Stage"]["Position"]["z"])*1000
    
    # it seems weird, that scan rotation is only saved with this detector
    if metadata["detector_name"]=="HAADF":
        metadata["scan_rotation_rad"]=np.double(allmetadata["Scan"]["ScanRotation"])
    
    #assuming quadratic camera chip
    size=pix_to_nm*image.shape[0]
    metadata["field_of_view_microns"]=size/1000
    metadata["image_shape"]=image.shape[:2]

    metadata["acceleration_volt"]=np.double(allmetadata["Optics"]["AccelerationVoltage"])

    if metadata["beam_mode"] != "Parallel":
        metadata["beam_convergence_mrad"]=np.double(allmetadata["Optics"]["BeamConvergence"])*1000
        
        metadata["collection_angle_start_mrad"]=np.double(
            allmetadata["Detectors"]["Detector-1"]["CollectionAngleRange"]["begin"])*1000
        metadata["collection_angle_end_mrad"]=np.double(
            allmetadata["Detectors"]["Detector-1"]["CollectionAngleRange"]["end"])*1000
    
    image=image[:,:,0]
    return image,metadata


#%% get_dm4_with_metadata
def get_dm4_with_metadata(filepath):
    """
    read a dm4-file and return the variables data and metadata as dictionaries
    data includes everything immediately important
    metadata contains all other information
    
    Important missing parameters are electric current and  only for spectra:
    acquisition time and acquisition date are missing

    Args:
        filepath (string): relative or absolute path to the file.

    Returns:
        data (dict): data["data"] yields the data other keywords contain the most important metadata.
        
        metadata (dict): Dictionary containing further metadata.
    """

    relevant_metadata_list=["Grating","Objective focus (um)","Stage X","Stage Y","Stage Z","Stage Beta","Stage Alpha","Indicated Magnification",
    "Bandpass","Detector","Filter","PMT HV","Sensitivity","Slit Width","Sample Time","Dwell time (s)","Image Height","Image Width",
     "Number Summing Frames","Voltage","Lightpath","Signal Name","Acquisition Time","Acquisition Date"]    
    
    f = io.StringIO()
    with redirect_stdout(f):
        data=nio.dm.dmReader(filepath,verbose=True)
    all_metadata = f.getvalue()
    
    metadata=dict()

    for i in range(len(relevant_metadata_list)):
        start=all_metadata.find("curTagValue = "+relevant_metadata_list[i])
        end=all_metadata[start:].find("\n")
        if start != -1:
            end+=start
            start+=len("curTagValue = "+relevant_metadata_list[i])+2
        metadata[relevant_metadata_list[i]]=all_metadata[start:end]

    return data,metadata   




def save_dm4_spectra_as_csv(spectrum_paths):
    for i in spectrum_paths:
        data,metadata=get_dm4_with_metadata(i)
        x=data["coords"][1]
        y=data["data"][0]
        xy=np.zeros([2,len(x)])
        xy[0]=x
        xy[1]=y
        
        np.savetxt(i[:-3]+"csv",xy.T,delimiter=",",fmt="%i")

def convert_dm4_to_png(dm4_filepaths,logscale=False):
    #stringoffset=2
    for i in dm4_filepaths:
        dmd,meta=get_dm4_with_metadata(i)
        new=i[:-4]
        if len(dmd["data"].shape)==3:
            if not os.path.exists(new):
                os.mkdir(new)
            halfbandpass=(dmd["coords"][0][1]-dmd["coords"][0][0])/2
            wavelengths=np.round(dmd["coords"][0]+halfbandpass,1)
            if dmd["pixelUnit"][0] !='nm': 
                print("other pixelsize than 'nm' occured")
            pixelSizeString=str(np.round(dmd["pixelSize"][1],4)*1000)[:6]+"nm"
            if dmd["pixelUnit"][1] !='µm':
                print("other pixelsize than 'µm' occured")
            if logscale:
                newdat=dmd["data"]-np.min(dmd["data"])
                newdat= newdat +np.min(newdat[newdat>0]) * 0.01

            for j in range(len(wavelengths)):
                newname=new+"/"+"img"+str(j)+"_at_"+str(wavelengths[j])+"nm_with_pixelSize_"+pixelSizeString
                if logscale:
                    image=img_to_uint8(np.log(newdat[j]))
                    newname=newname+"_logscale.png"#[:newname.find(".png")]+
                    cv2.imwrite(newname,image)        
                else:
                    image=img_to_uint8(dmd["data"][j])
                    newname=newname+".png"
                    cv2.imwrite(newname,image)        
        else:    
            revnew=new[::-1]
            magnification_start=revnew.find("_X")+1
            magnification_end=magnification_start+revnew[magnification_start:].find("_")
            pixelSizeString=str(np.round(dmd["pixelSize"][0],4)*1000)[:6]+"nm"
            if dmd["pixelUnit"][0] !='µm':
                print("other pixelsize than 'µm' occured")
            
            new[-magnification_end:-magnification_start]
            newname=new[:-magnification_end]+"pixelSize"+pixelSizeString +new[-magnification_start:]
    
            if logscale:
                newdat=dmd["data"]-np.min(dmd["data"])
                newdat= newdat +np.min(newdat[newdat>0]) * 0.01
                image=img_to_uint8(np.log(newdat))
                newname=newname+"_logscale.png"#[:newname.find(".png")]+
            else:
                image=img_to_uint8(dmd["data"]) 
                newname =newname+".png"
            
            cv2.imwrite(newname,image)
