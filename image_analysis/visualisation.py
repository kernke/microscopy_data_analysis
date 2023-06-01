# -*- coding: utf-8 -*-
"""
Created on Mon May  8 15:32:54 2023

@author: kernke
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio
import copy
from matplotlib.widgets import TextBox
import math
from .general_util import point_in_convex_ccw_roi,lineIntersection
from .image_aligning import align_images

#%% make_scale_bar
def vis_make_scale_bar(
    images, pixratios, lengthperpix, barlength, org, thickness=4, color=(255, 0, 0)
):

    org = np.array(org)
    for i in range(len(images)):
        pixlength = (barlength / lengthperpix) / pixratios[i]
        pixlength = np.round(pixlength).astype(int)
        pt2 = org + np.array([0, pixlength])
        cv2.line(images[i], org[::-1], pt2[::-1], color, thickness=thickness)



#%% make_mp4
def vis_make_mp4(filename, images, fps):

    with imageio.get_writer(filename, mode="I", fps=fps) as writer:
        for i in range(len(images)):
            writer.append_data(images[i])

    return True


#%% zoom


def vis_zoom(img, zoom_center, final_height, steps, gif_resolution_to_final=1):

    iratio = img.shape[0] / img.shape[1]

    final_size = np.array([final_height, final_height / iratio])

    startpoints = np.zeros([4, 2], dtype=int)
    endpoints = np.zeros([4, 2], dtype=int)

    startpoints[2, 1] = img.shape[1] - 1
    startpoints[1, 0] = img.shape[0] - 1
    startpoints[3] = img.shape
    startpoints[3] -= 1

    endpoints[0] = np.round(zoom_center - final_size / 2).astype(int)
    endpoints[3] = np.round(zoom_center + final_size / 2).astype(int)

    tocorner = np.array([-final_size[0], final_size[1]]) / 2
    endpoints[1] = np.round(zoom_center - tocorner).astype(int)
    tocorner = np.array([final_size[0], -final_size[1]]) / 2
    endpoints[2] = np.round(zoom_center - tocorner).astype(int)

    steps += 1
    cornerpoints = np.zeros([steps, 4, 2], dtype=int)
    pixratios = np.zeros(steps)
    for i in range(4):
        for j in range(2):
            cornerpoints[:, i, j] = np.round(
                np.linspace(startpoints[i, j], endpoints[i, j], steps)
            ).astype(int)

    final_resolution = np.round(final_size * gif_resolution_to_final).astype(int)
    images = np.zeros([steps, final_resolution[0], final_resolution[1]])
    for i in range(steps):
        pixratios[i] = (
            cornerpoints[i, 1, 0] - cornerpoints[i, 0, 0]
        ) / final_resolution[0]

        roi_img = img[
            cornerpoints[i, 0, 0] : cornerpoints[i, 1, 0],
            cornerpoints[i, 0, 1] : cornerpoints[i, 2, 1],
        ]

        ratio = roi_img.shape[0] / final_resolution[0]  # size
        sigma = ratio / 4

        ksize = np.round(5 * sigma).astype(int)
        if ksize % 2 == 0:
            ksize += 1
        roi_img = cv2.GaussianBlur(roi_img, [ksize, ksize], sigma)
        images[i] = cv2.resize(roi_img, final_resolution[::-1], cv2.INTER_AREA)

    return images, pixratios

#%%  plot_sortout
def vis_plot_line_ids(image, sortout, legend=True, alpha=0.5, markersize=0.5):
    plt.figure(figsize=(12, 10))
    plt.imshow(image, cmap="gray")
    colors = ["b", "r", "g", "c", "m", "y"]
    for j in range(len(sortout)):
        count = 0
        for i in sortout[j]:
            if count == 0:
                plt.plot(
                    i[:, 1],
                    i[:, 0],
                    "o",
                    c=colors[j],
                    alpha=alpha,
                    label=str(j),
                    markersize=markersize,
                )
            else:
                plt.plot(
                    i[:, 1],
                    i[:, 0],
                    "o",
                    c=colors[j],
                    alpha=alpha,
                    markersize=markersize,
                )
            count += 1
    if legend == True:
        plt.legend()

#%% interactive line plotting

def _get_shift(arg_dic):
    keylist=np.sort(list(arg_dic["shifts"].keys()))
    for i in range(1,len(keylist)):
        if keylist[i] > arg_dic["image_counter"]:
            return arg_dic["shifts"][keylist[i-1]]
        else:
            return arg_dic["shifts"][keylist[-1]]


def _plot_overlay(arg_dic,ax,artists,line_objs):
    shift=_get_shift(arg_dic)
    for key in line_objs[arg_dic["image_counter"]]:
        obj=line_objs[arg_dic["image_counter"]][key]
        artist = ax.plot(obj.x-shift[1], obj.y-shift[0], 'x-', picker=5,alpha=0.6,c='c')[0]
        artist.obj = obj  
        artists[key]=artist
    
    if arg_dic["active"]:
        artists[arg_dic["lineindex"]].remove()
        del artists[arg_dic["lineindex"]]
        obj=line_objs[arg_dic["image_counter"]][arg_dic["lineindex"]]
        artist = ax.plot(obj.x-shift[1], obj.y-shift[0], 'x-', picker=5,alpha=0.6,c='r')[0]
        artist.obj = obj  
        artists[key]=artist
    
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())

def _progress_to_next_image(next_image_counter,line_objs):
    for i in line_objs[next_image_counter-1]:
        if i in line_objs[next_image_counter]:
            change_index0=line_objs[next_image_counter-1][i].changed
            change_index1=line_objs[next_image_counter][i].changed
            
            if change_index0 > change_index1:
                line_objs[next_image_counter][i]=copy.deepcopy(line_objs[next_image_counter-1][i])

        else:
            line_objs[next_image_counter][i]=copy.deepcopy(line_objs[next_image_counter-1][i])


def _press(event,ax,img_plot,arguments):
    if event.inaxes in [ax]:
        images,artists,line_objs,set_of_lines,arg_dic=arguments
        
        if event.key == "b":
            arg_dic["active"]=False
            if arg_dic["image_counter"]==0:
                print('no previous image')
            else:
                arg_dic["image_counter"] -=1
                img_plot.set_data(images[arg_dic["image_counter"]])
                ax.set_title("image "+str(arg_dic["image_counter"]))
                
                if arg_dic["overlayed"]:
                    for i in list(artists.keys()):
                        artists[i].remove()
                        del artists[i]
                    
                    _plot_overlay(arg_dic, ax, artists, line_objs) 


                    
                plt.gcf().canvas.draw() 
        if event.key == "n":
            arg_dic["active"]=False
            if arg_dic["image_counter"]==len(images)-1:
                print('end of image stack')
            else:
                arg_dic["image_counter"] +=1    
                _progress_to_next_image(arg_dic["image_counter"],line_objs)

                img_plot.set_data(images[arg_dic["image_counter"]])
                ax.set_title("image "+str(arg_dic["image_counter"]))
                
                if arg_dic["overlayed"]:
                    for i in list(artists.keys()):
                        artists[i].remove()
                        del artists[i]
                    
                    _plot_overlay(arg_dic, ax, artists, line_objs)   

                plt.gcf().canvas.draw() 

        if event.key == "h":
            if not arg_dic["shift_active"]:
                print("shift activated")
                arg_dic["shift_active"]=True
                if arg_dic["overlayed"]:
                    for i in list(artists.keys()):
                        artists[i].remove()
                        del artists[i]
                arg_dic["overlayed"]=True
                _plot_overlay(arg_dic,ax,artists,line_objs)
                plt.gcf().canvas.draw() 
            else:
                arg_dic["shift_active"]=False
                print("shift deactivated resulting with:")
                print(arg_dic["shifts"][arg_dic["image_counter"]])

        if event.key == "up":
            if arg_dic["shift_active"]:
                if arg_dic["overlayed"]:
                    for i in list(artists.keys()):
                        artists[i].remove()
                        del artists[i]
                arg_dic["shifts"][arg_dic["image_counter"]][0] +=1
                _plot_overlay(arg_dic,ax,artists,line_objs)
                plt.gcf().canvas.draw() 
                
        if event.key == "down":
            if arg_dic["shift_active"]:
                if arg_dic["overlayed"]:
                    for i in list(artists.keys()):
                        artists[i].remove()
                        del artists[i]
                arg_dic["shifts"][arg_dic["image_counter"]][0] -=1
                _plot_overlay(arg_dic,ax,artists,line_objs)
                plt.gcf().canvas.draw() 

        if event.key == "left":
            if arg_dic["shift_active"]:
                if arg_dic["overlayed"]:
                    for i in list(artists.keys()):
                        artists[i].remove()
                        del artists[i]
                arg_dic["shifts"][arg_dic["image_counter"]][1] +=1
                _plot_overlay(arg_dic,ax,artists,line_objs)
                plt.gcf().canvas.draw() 

        if event.key == "right":
            if arg_dic["shift_active"]:
                if arg_dic["overlayed"]:
                    for i in list(artists.keys()):
                        artists[i].remove()
                        del artists[i]
                arg_dic["shifts"][arg_dic["image_counter"]][1] -=1
                _plot_overlay(arg_dic,ax,artists,line_objs)
                plt.gcf().canvas.draw() 


                

        if event.key == "u":
            if arg_dic["active"]:
                if not arg_dic["undo"]:
                    print("change to line "+str(arg_dic["lineindex"])+" will be undone?")
                    print("Confirm with 'u', cancel with 'i'")
                    arg_dic["undo"]=True
                else:
                    if len(arg_dic["backup"][0])==0:
                        set_of_lines.remove(arg_dic["lineindex"])
                        del line_objs[arg_dic["image_counter"]][arg_dic["lineindex"]]
                        artists[arg_dic["lineindex"]].remove()
                        del artists[arg_dic["lineindex"]]
                        
                    else:
                        artists[arg_dic["lineindex"]].remove()
                        del artists[arg_dic["lineindex"]]
                        obj=line_objs[arg_dic["image_counter"]][arg_dic["lineindex"]]
                        obj.x=arg_dic["backup"][0] 
                        obj.y=arg_dic["backup"][1] 
                        obj.length=arg_dic["backup"][2]
                        obj.angle=arg_dic["backup"][3]
                        obj.changed=arg_dic["backup"][4]
                        artist = ax.plot(obj.x, obj.y, 'x-', picker=5,alpha=0.6,c='c')[0]
                        artist.obj = obj  
                        artists[arg_dic["lineindex"]]=artist
                        
                        ax.set_xlim(ax.get_xlim())
                        ax.set_ylim(ax.get_ylim())
                        plt.gcf().canvas.draw()
                        arg_dic["overlayed"]=True
                        
                    arg_dic["undo"]=False                       
                    arg_dic["active"]=False
                    ax.set_xlim(ax.get_xlim())
                    ax.set_ylim(ax.get_ylim())
                    plt.gcf().canvas.draw()
            else:
                print("no line selected")
            
        if event.key == "c":

            if arg_dic["cl_overlay"]<2:
                shift=_get_shift(arg_dic)
                aligns=arg_dic["aligns"]
                cl=arg_dic["cl"]
                cl_cold=arg_dic["cl_cold"]
                dim=images[arg_dic["image_counter"]].shape
                p2=copy.deepcopy(aligns[:,:2])
                for i in range(len(p2)):
                    p2[i]-=shift[::-1]
                
                [im1a,im1b],im2, matrices, reswidth, resheight, width_shift, height_shift=align_images(
                    [cl,cl_cold], images[arg_dic["image_counter"]], [aligns[:,2:4],aligns[:,4:6]], p2,verbose=True)
                
                
            if arg_dic["cl_overlay"]==0:                
                
                newim=im1a[height_shift:height_shift+dim[0],width_shift:width_shift+dim[1]]
                img_plot.set_data(newim)
                ax.set_title("image "+str(arg_dic["image_counter"]))
                
                arg_dic["cl_overlay"]+=1
                if arg_dic["overlayed"]:
                    for i in list(artists.keys()):
                        artists[i].remove()
                        del artists[i]
                    
                    _plot_overlay(arg_dic, ax, artists, line_objs)
                    
            elif arg_dic["cl_overlay"]==1:
                newim=im1b[height_shift:height_shift+dim[0],width_shift:width_shift+dim[1]]
                img_plot.set_data(newim)                
                
                arg_dic["cl_overlay"]+=1                
                if arg_dic["overlayed"]:
                    for i in list(artists.keys()):
                        artists[i].remove()
                        del artists[i]
                    
                    _plot_overlay(arg_dic, ax, artists, line_objs) 
                
            else:
                arg_dic["cl_overlay"]=0
                img_plot.set_data(images[arg_dic["image_counter"]])
                ax.set_title("image "+str(arg_dic["image_counter"]))
                
                if arg_dic["overlayed"]:
                    for i in list(artists.keys()):
                        artists[i].remove()
                        del artists[i]
                    
                    _plot_overlay(arg_dic, ax, artists, line_objs) 

            #im2[height_shift:height_shift+dim[0],width_shift:width_shift+dim[1]]

            
                img_plot.set_data(images[arg_dic["image_counter"]])
                ax.set_title("image "+str(arg_dic["image_counter"]))
            
            plt.gcf().canvas.draw()

        if event.key == 'o':
            if not arg_dic["overlayed"]:
                arg_dic["overlayed"]=True
                _plot_overlay(arg_dic, ax, artists, line_objs) 

            else:
                for i in list(artists.keys()):
                    artists[i].remove()
                    del artists[i]
                arg_dic["overlayed"]=False
            plt.gcf().canvas.draw() 

        if event.key == 'd':
            if arg_dic["active"]:
                if not arg_dic["delete"]:
                    print("line "+str(arg_dic["lineindex"])+" will be deleted?")
                    print("Confirm with 'd', cancel with 'i'")
                    arg_dic["delete"]=True
                else:
                    set_of_lines.remove(arg_dic["lineindex"])
                    for i in range(len(images)):
                        if arg_dic["lineindex"] in line_objs[i]: 
                            del line_objs[i][arg_dic["lineindex"]]
            else:
                print("no line selected to delete")


        if event.key == 'i':
            if arg_dic["active"]:
                arg_dic["active"]=False
                arg_dic["delete"]=False
                arg_dic["undo"]=False

                artists[arg_dic["lineindex"]].remove()
                del artists[arg_dic["lineindex"]]
                
                obj=line_objs[arg_dic["image_counter"]][arg_dic["lineindex"]]
                artist = ax.plot(obj.x, obj.y, 'x-', picker=5,alpha=0.6,c='c')[0]
                artist.obj = obj  
                artists[arg_dic["lineindex"]]=artist
                
                ax.set_xlim(ax.get_xlim())
                ax.set_ylim(ax.get_ylim())
                plt.gcf().canvas.draw()

                print("deactivated")
            else:
                print("no line selected")                
            #print("next line will be "+str(lineindex))

def _click(event,ax,img_plot,arguments):

    
    if event.button == 3:
        images,artists,line_objs,set_of_lines,arg_dic=arguments
        shift=_get_shift(arg_dic)
        
        if arg_dic["active"]:
            obj=line_objs[arg_dic["image_counter"]][arg_dic["lineindex"]]

            if len(obj.x)>1:
                arg_dic["backup"][0]=copy.deepcopy(obj.x)
                arg_dic["backup"][1]=copy.deepcopy(obj.y)
                arg_dic["backup"][2]=copy.deepcopy(obj.length)
                arg_dic["backup"][3]=copy.deepcopy(obj.angle)
                arg_dic["backup"][4]=copy.deepcopy(obj.changed)
                
                dist0=(event.xdata-shift[0]-obj.x[0])**2+(event.ydata-shift[1]-obj.y[0])**2
                dist1=(event.xdata-shift[0]-obj.x[1])**2+(event.ydata-shift[1]-obj.y[1])**2
                if dist0 > dist1:                   
                    obj.x[1]=event.xdata-shift[0]
                    obj.y[1]=event.ydata-shift[1]
                else:
                    obj.x[0]=event.xdata-shift[0]
                    obj.y[0]=event.ydata-shift[1]
                    
            else:
                obj.x.append(event.xdata-shift[0])
                obj.y.append(event.ydata-shift[1])
            
            print(str(event.xdata)+" , "+str(event.ydata))
            
            sortindex=np.argsort(obj.x)
            newx=np.array(obj.x)[sortindex]
            newy=np.array(obj.y)[sortindex]
            
            dx=newx[1]-newx[0]
            dy=newy[1]-newy[0]
            dl=math.sqrt(dx*dx+dy*dy)
            if dl < arg_dic["backup"][2]:
                print("Change not accepted, lenght must increase")
                obj.x=arg_dic["backup"][0] 
                obj.y=arg_dic["backup"][1] 
                #obj.length=arg_dic["backup"][2]
                #obj.angle=arg_dic["backup"][3]
            else:
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
                    
                    
            artists[arg_dic["lineindex"]].remove()
            del artists[arg_dic["lineindex"]]
            obj.changed.append(arg_dic["image_counter"])#line_objs[arg_dic["image_counter"]][arg_dic["lineindex"]]
            artist = ax.plot(obj.x, obj.y, 'x-', picker=5,alpha=0.6,c='c')[0]
            artist.obj = obj  
            artists[arg_dic["lineindex"]]=artist
            
            ax.set_xlim(ax.get_xlim())
            ax.set_ylim(ax.get_ylim())
            plt.gcf().canvas.draw()
            arg_dic["overlayed"]=True
            print("line "+str(arg_dic["lineindex"])+ " written")
                  
            arg_dic["active"]=False
        else:
            if event.inaxes in [ax]:

                if len(set_of_lines)==0:
                    max_index=-1
                else:
                    max_index=max(set_of_lines)

                if max_index >len(set_of_lines)-1:
                    for i in range(max_index):
                        if i not in set_of_lines:
                            arg_dic["lineindex"]=i
                            break
                else:
                    arg_dic["lineindex"]=len(set_of_lines)       

                print("add line "+str(arg_dic["lineindex"]))
                line_objs[arg_dic["image_counter"]][arg_dic["lineindex"]]=line_object(
                    event.xdata-shift[0],event.ydata-shift[1],arg_dic["lineindex"],arg_dic["image_counter"]) 

                obj=line_objs[arg_dic["image_counter"]][arg_dic["lineindex"]]
                artist = ax.plot(obj.x, obj.y, 'x-', picker=5,alpha=0.6,c='r')[0]
                artist.obj = obj  
                artists[arg_dic["lineindex"]]=artist

                ax.set_xlim(ax.get_xlim())
                ax.set_ylim(ax.get_ylim())
                plt.gcf().canvas.draw()
                arg_dic["overlayed"]=True
                                                                                
                set_of_lines.add(arg_dic["lineindex"])
                print(str(event.xdata)+" , "+str(event.ydata))
                arg_dic["active"]=True
            
            
def _on_pick(event,ax,img_plot,arguments):
    images,artists,line_objs,set_of_lines,arg_dic=arguments
    arg_dic["lineindex"]=event.artist.obj.index
    arg_dic["active"]=True
    
    shift=_get_shift(arg_dic)
    artists[arg_dic["lineindex"]].remove()
    obj=line_objs[arg_dic["image_counter"]][arg_dic["lineindex"]]
    artist = ax.plot(obj.x-shift[1], obj.y-shift[0], 'x-', picker=5,alpha=0.6,c='r')[0]
    artist.obj = obj  
    artists[arg_dic["lineindex"]]=artist

    print("line "+str(event.artist.obj.index)+" activated")


class line_object:
    def __init__(self, x, y, index,image_counter):
        self.x = [x]
        self.y = [y]
        self.length=0
        self.angle=None
        self.changed=[image_counter]
        self.index = index

def initialize_plot_arguments(images,angles,cl,cl_cold,aligns,line_objs=None,set_of_lines=None,shifts=None):
    arg_dic={}
    arg_dic["image_counter"]=0
    arg_dic["active"]=False
    arg_dic["overlayed"]=False
    arg_dic["delete"]=False
    arg_dic["undo"]=False
    arg_dic["lineindex"]=0
    arg_dic["border_snap_distance"]=10
    arg_dic["snap_to_angle"]=False
    arg_dic["angles"]=angles#[58.5,64.,118.5,124.3]
    anglerad=np.array(arg_dic["angles"])/180 *np.pi
    arg_dic["ms"]=np.tan(anglerad)
    arg_dic["shift_active"]=False
    arg_dic["shifts"]=shifts
    arg_dic["cl"]=cl
    arg_dic["cl_cold"]=cl_cold
    arg_dic["cl_overlay"]=0
    arg_dic["aligns"]=aligns    
    arg_dic["backup"]=[[],[],0,None,None]
    artists={}
    if set_of_lines is None:
        line_objs=[{} for i in range(len(images))]
        set_of_lines=set()
    
    return images,artists,line_objs,set_of_lines,arg_dic

def _submit(expression,ax,img_plot,text_box,arguments):
    images,artists,line_objs,set_of_lines,arg_dic=arguments
    new_image_counter=int(expression)
    if new_image_counter < arg_dic["image_counter"]:
        pass
    else:
        for j in range(arg_dic["image_counter"]+1,new_image_counter+1):
            _progress_to_next_image(j,line_objs)
            
    arg_dic["image_counter"]=new_image_counter
    img_plot.set_data(images[arg_dic["image_counter"]])
    ax.set_title("image "+str(arg_dic["image_counter"]))
    if arg_dic["overlayed"]:
        for i in list(artists.keys()):
            artists[i].remove()
            del artists[i]
        
        _plot_overlay(arg_dic, ax, artists, line_objs)   
    
    plt.gcf().canvas.draw() 
    
    
def interactive_plot(images,arguments):
    #global text_box
    fig,ax  = plt.subplots()
    ax.cla()
    plt.subplots_adjust(bottom=0.2)
    img_plot=ax.imshow(images[0],cmap='gray')
    
    ax.set_title("image "+str(0))
    axbox = fig.add_axes([0.05, 0.05, 0.05, 0.075])
    text_box = TextBox(axbox, "Goto", textalignment="center")
    text_box.set_val("0")
    fig.tight_layout()
    text_box.on_submit(lambda expression: _submit(expression,ax,img_plot,text_box,arguments))
    fig.canvas.mpl_connect('key_press_event', lambda event: _press(event,ax,img_plot,arguments))
    fig.canvas.mpl_connect('button_press_event', lambda event: _click(event,ax,img_plot,arguments))
    fig.canvas.callbacks.connect('pick_event',  lambda event: _on_pick(event,ax,img_plot,arguments))