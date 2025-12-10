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

from .image_processing import img_to_uint8,img_normalize



def imsave(filename,img,floatnormalize=True):
    """
    writes single (grayscale), triple (RGB), quadrupel (RGBA) channel images, 
    for uint8 and uint16 formats ".png" is used as default, 
    for float32 ".tif" is used as default
    double channel images (gray,alpha) will be transformed to RGBA
    filename extensions (".png", ".jpg", ...) are optional 
    and can be given to obtain a format differing from default 

    Args:
        filename (string): 
            filename with or without extension determining the format.
        img (array_like): 
            image matrix, either MxN, MxNx1, MxNx3 or MxNx4.
        floatnormalize (bool): 
            transform the color range between 0 and 1 
            (for better compatibility with most image viewers)
            Defaults to True.

    Returns:
        success_info (bool): 
            True, when succesfully written.

    """
    invfilename=filename[::-1]
    # filename should not include other "." not related to file format
    dotpos=invfilename.find('.')
    if dotpos==-1:
        file_extension=".png"
    else:
        file_extension=invfilename[:dotpos+1][::-1]
        filename=filename[:-dotpos-1]
    
    if np.issubdtype(img.dtype,np.floating):
        file_extension=".tif"
        if floatnormalize:
            img=img_normalize(img)
        img=img.astype(np.float32)
    
    fn=filename+file_extension
    
    if len(img.shape)==3:
        if img.shape[2]==2:
            #converting double channel to RGBA
            img=np.dstack((img[:,:,0],img[:,:,0],img[:,:,0],img[:,:,1]))
        else:
            # changing from rgb to bgr, as cv2 saves as bgr, but rgb is wanted here
            img[:,:,[0,2]]=img[:,:,[2,0]]    
    print(img.shape)
    return cv2.imwrite(fn,img)
    

def imsave_multi(filename,stack,floatnormalize=True):
    """
    save multiple (single channel / greyscale) images as tiff-stack
    (supported formats uint8, uint16, float32)

    Args:
        filename (string): 
            filename with or without .tiff extension.
        stack (list of images): 
            grayscale images only containing one channel.
        floatnormalize (bool): 
            transform the color range between 0 and 1 
            (for better compatibility with most image viewers)
            Defaults to True.

    Returns:
        success_info (bool): 
            True, when succesfully written.

    """
    invfilename=filename[::-1]
    dotpos=invfilename.find('.')
    if dotpos==-1:
        file_extension=".tiff"
    else:
        file_extension=invfilename[:dotpos+1][::-1]
        filename=filename[:-dotpos-1]
    
    if np.issubdtype(stack[0].dtype,np.floating):
        file_extension=".tiff"
        for i in range(len(stack)):
            if floatnormalize:
                stack[i]=img_normalize(stack[i]).astype(np.float32)
            else:
                stack[i]=stack[i].astype(np.float32)
        
    fn=filename+file_extension        
    return cv2.imwrite_multi(fn,stack)
    
    
    

def get_emd_with_metadata(filepath):
    """
    Read TEM-imgages as .emd-files 

    Args:
        filepath (string): 
            relative or absolute path to the file.

    Returns:
        image (MxN array_like): 
            2D image with only one intensity-channel (gray-scale).
        
        metadata (dict): 
            Dictionary containing the most important metadata.
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
        filepath (string): 
            relative or absolute path to the file.

    Returns:
        data (dict): 
            data["data"] yields the data other keywords contain the most important metadata.
        
        metadata (dict): 
            Dictionary containing further metadata.
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




def get_SEMtif_with_metadata(filepath):
    with open(filepath, 'r', encoding="utf8",errors="ignore") as fp:
        contents=fp.read()    
    text=contents[contents.find("Date"):]
    # qtu <-- quantitiy_type_unit
    #Beam
    qtu=[["\nSystemType=","str",None],
         ["\nTimeOfCreation=","str",None],
         
         ["\nPixelWidth=","np.float64","m"],
         ["\nPixelHeight=","np.float64","m"],         
    #Microscope     
         ["\nHV=","np.float64","V"],
         ["\nBeamCurrent=","np.float64","A"],
         ["\nWorkingDistance=","np.float64","m"],
         ["\nDwelltime=","np.float64","s"],
         
         ["\nSpot=","np.float64","nm"],
         ["\nStigmatorX=","np.float64",None],
         ["\nStigmatorY=","np.float64",None],
         ["\nBeamShiftX=","np.float64",None],
         ["\nBeamShiftY=","np.float64",None],
         ["\nSourceTiltX=","np.float64",None],
         ["\nSourceTiltY=","np.float64",None],
         ["\nEmissionCurrent=","np.float64","A"],
         ["\nSpecimenCurrent=","np.float64","A"],
         #["\nApertureDiameter=","np.float64","m"],
         #["\nATubeVoltage=","np.float64","V"],
    #Scan     
         ["\nUseCase=","str",None],
         ["\nTiltCorrectionIsOn=","str",None],#yes,no 
         
         ["\nScanRotation=","np.float64","rad"],


    #CompoundLens
         ["\nIsOn=","str",None], #On,Off
         ["\nThresholdEnergy=","np.float64","eV"],
    #Stage          
         ["\nStageX=","np.float64","m"],
         ["\nStageY=","np.float64","m"],
         ["\nStageZ=","np.float64","m"],
         ["\nStageR=","np.float64","rad"],
         ["\nStageTa=","np.float64","rad"],
         ["\nStageTb=","np.float64","rad"],
         ["\nStageBias=","np.float64","V"],
         ["\nChPressure=","np.float64","Pa"],

    #Detecor
         ["\nName=","str",None],
         ["\nMode=","str",None],
         
         ["\nSignal=","str",None],
         ["\nContrast=","np.float64",None],
         ["\nBrightness=","np.float64",None],
         ["\nContrastDB=","np.float64","dB"],
         ["\nBrightnessDB=","np.float64","dB"],
         ["\nAverage=","int",None],
         ["\nIntegrate=","int",None],
         ["\nResolutionX=","int",None],
         ["\nResolutionY=","int",None],
         ["\nHorFieldsize=","np.float64","m"],
         ["\nVerFieldsize=","np.float64","m"],
         ["\nFrameTime=","np.float64","s"],
    #Digital
         ["\nDigitalContrast=","np.float64",None],
         ["\nDigitalBrightness=","np.float64",None],
         ["\nDigitalGamma=","np.float64",None]]
    
    res=[]
    
    typechanges=[]
    counter=0
    for i in qtu:
        kw=i[0]
        start=text.find(kw)+len(kw)
        end=text[start:].find("\n")
        if end == 0 or start==-1 or kw not in text:
            res.append(None)
        else:
            if i[1]=="int":    
                res.append(int(text[start:start+end]))
            elif i[1]=="np.float64":
                res.append(np.double(text[start:start+end]))
            elif i[1]=="str" or i[1]=="string":
                if text[start:start+end] in ["no","No","off","Off","false","False"]:
                    res.append(False)
                    typechanges.append(counter)
                elif text[start:start+end] in ["yes","Yes","on","On","true","True"]:
                    res.append(True)
                    typechanges.append(counter)
                elif ":" in text[start:start+end]:
                    datestring=text[start:start+end]
                    
                    datestring=datestring.replace("."," ")
                    day,month,year,hour=datestring.split(" ")
                    newdate=year+"-"+month+"-"+day+" "+hour
                    newdate +=".000Z"
                    res.append(newdate)
                else:
                    res.append(text[start:start+end])
        counter +=1

    for j in typechanges:
        qtu[j][1]="bool"
    



    res.append(os.path.abspath(filepath).replace("\\","/"))
    qtu.append(["PathToImage","str",None])
               
    qtu_dict=readable_names(res,qtu)
    
    #additional detector information
    if qtu_dict["Detector"]["value"] == "ETD":
        kw="\nGrid="
        start=text.find(kw)+len(kw)
        end=text[start:].find("\n")
        Grid_Voltage=np.double(text[start:start+end])
        qtu_dict["Grid_Voltage"]=dict()
        qtu_dict["Grid_Voltage"]["dtype"]="np.float64"
        qtu_dict["Grid_Voltage"]["unit"]="V",
        qtu_dict["Grid_Voltage"]["value"]=Grid_Voltage
    
    return cv2.imread(filepath,-1),qtu_dict


def readable_names(res,qtu):
    
    nice_names=dict()

    nice_names["\nSystemType="]="Microscope"
    nice_names["\nTimeOfCreation="]="Time_of_Creation" 
    nice_names["\nPixelWidth="]="Pixel_Width"
    nice_names["\nPixelHeight="]="Pixel_Height"
    nice_names["\nHV="]="Acceleration_Voltage"
    nice_names["\nBeamCurrent="]="Beam Current"
    nice_names["\nWorkingDistance="]="Working_Distance"
    nice_names["\nDwelltime="]="Dwell_Time"
    nice_names["\nSpot="]="Spot_Diameter_(estimated)"    
    nice_names["\nStigmatorX="]="Stigmator_X"
    nice_names["\nStigmatorY="]="Stigmator_Y"
    nice_names["\nBeamShiftX="]="Beam_Shift_X"
    nice_names["\nBeamShiftY="]="Beam_Shift_Y"
    nice_names["\nSourceTiltX="]="Source_Tilt_X"
    nice_names["\nSourceTiltY="]="Source_Tilt_Y"
    nice_names["\nEmissionCurrent="]="Emission_Current"
    nice_names["\nSpecimenCurrent="]="Specimen_Current"
    nice_names["\nUseCase="]="SEM_Mode"
    nice_names["\nTiltCorrectionIsOn="]="Tilt_Correction"
    nice_names["\nScanRotation="]="Scan_Rotation"
    nice_names["\nIsOn="] ="Compound_Lens"           
    nice_names["\nThresholdEnergy="]="Compound_Lens_Threshold_Energy"            
    nice_names["\nStageX="]="Stage_X"            
    nice_names["\nStageY="]="Stage_Y"            
    nice_names["\nStageZ="]="Stage_Z"
    nice_names["\nStageR="]="Stage_Rotation"
    nice_names["\nStageTa="]="Stage_Tilt_alpha"
    nice_names["\nStageTb="]="Stage_Tilt_beta"
    nice_names["\nStageBias="]="Stage_Bias"
    nice_names["\nChPressure="]="Chamber_Pressure"
    nice_names["\nName="]="Detector"
    nice_names["\nMode="]="Detector_Mode"
    nice_names["\nSignal="]="Signal_Type"
    nice_names["\nContrast="]="Contrast"
    nice_names["\nContrastDB="]="Contrast_DB"
    nice_names["\nBrightness="]="Brightness"
    nice_names["\nBrightnessDB="]="Brightness_DB"
    nice_names["\nAverage="]="Average"
    nice_names["\nIntegrate="]="Integrate"
    nice_names["\nResolutionX="]="Resolution_X"
    nice_names["\nResolutionY="]="Resolution_Y"
    nice_names["\nHorFieldsize="]="Horizontal_Fieldsize"
    nice_names["\nVerFieldsize="]="Vertical_Fieldsize"
    nice_names["\nFrameTime="]="Frame_Time"
    nice_names["\nDigitalContrast="]="Digital_Contrast"
    nice_names["\nDigitalBrightness="]="Digital_Brightness"
    nice_names["\nDigitalGamma="]="Digital_Gamma"
    nice_names["PathToImage"]="Path_to_Image"
    

    qtu_dict=dict()       
    for i in range(len(qtu)):
        qtu_dict[nice_names[qtu[i][0]]]=dict()
        qtu_dict[nice_names[qtu[i][0]]]["dtype"]=qtu[i][1]
        qtu_dict[nice_names[qtu[i][0]]]["unit"]=qtu[i][2]
        qtu_dict[nice_names[qtu[i][0]]]["value"]=res[i]#qtu[i][1]
        #qtu_dict[nice_names[qtu[i][0]]].append(res[i])
        #qtu[i][0]=nice_names[qtu[i][0]]
    return qtu_dict