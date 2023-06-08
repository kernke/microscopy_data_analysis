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
from .general_util import point_in_convex_ccw_roi,lineIntersection,assure_multiple
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

#%%
"""            
            print(str(event.xdata)+" , "+str(event.ydata))
            
            sortindex=np.argsort(obj.x)
            newx=np.array(obj.x)[sortindex]
            newy=np.array(obj.y)[sortindex]
            
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
                
"""               

    
    
#%% good stuff

class line_object:
    __slots__="x","y","length","changed","index"
    def __init__(self, x, y, index,image_counter):
        self.x = [x]
        self.y = [y]
        self.length=0
        self.changed=[image_counter]
        self.index = index


    
def _progress_to_next_image(next_image_counter,line_objs):
    for i in line_objs[next_image_counter-1]:
        if i in line_objs[next_image_counter]:
            change_index0=line_objs[next_image_counter-1][i].changed
            change_index1=line_objs[next_image_counter][i].changed
            
            if change_index0 > change_index1:
                line_objs[next_image_counter][i]=copy.deepcopy(line_objs[next_image_counter-1][i])

        else:
            line_objs[next_image_counter][i]=copy.deepcopy(line_objs[next_image_counter-1][i])



#%% image plotting
class image_plotting:
    def __init__(self,images,image_counter=0):
        
        self.images=assure_multiple(images)
        self.image_counter=image_counter
        self.line_overlay=False
        self._main_args=[]
        self._main_funcs=[]

        plt.ioff()
        self.fig,self.ax  = plt.subplots()
        self.ax.cla()

    def show(self):
        self.img_plot=self.ax.imshow(self.images[self.image_counter],cmap='gray')
        self.ax.set_title("image "+str(self.image_counter))
        self.fig.tight_layout()
        plt.ion()
        for i in range(len(self._main_funcs)):
            self._main_funcs[i](*self._main_args[i])
        plt.show()


    def add_keyboard(self):
        self._main_funcs.append(self.fig.canvas.mpl_connect)
        self._main_args.append(['key_press_event',self._keyboard_input])
        self.keyboard_funcs={}
        

    def _keyboard_input(self,event):
        if event.key in self.keyboard_funcs:
            self.keyboard_funcs[event.key]()
            
            if self.line_overlay:
                self._plot_overlay()
    
            plt.gcf().canvas.draw() 
        
           
    #%%% image series navigation b/n
    def addfunc_image_series(self,times=None):
        if times is None:
            times=np.arange(len(self.images))
        else:
            self.times=times
            
        if not hasattr(self,"keyboard_funcs"):
            self.add_keyboard()
        
        self.keyboard_funcs["n"]=self._next_image
        self.keyboard_funcs["b"]=self._before_image

    def _next_image(self):
        if self.image_counter==len(self.images)-1:
            print('end of image stack')
        else:
            self.image_counter +=1    
            self.img_plot.set_data(self.images[self.image_counter])
            self.ax.set_title("image "+str(self.image_counter))
            
            if hasattr(self, "line_objs"):
                _progress_to_next_image(self.image_counter, self.line_objs) 

    def _before_image(self):
        if self.image_counter==0:
            print('no previous image')
        else:
            self.image_counter -=1
            self.img_plot.set_data(self.images[self.image_counter])
            self.ax.set_title("image "+str(self.image_counter))

    #%%% shifts
    def addfunc_shifts(self,shifts=None):
        if not hasattr(self,"keyboard_funcs"):
            self.add_keyboard()
            
        if shifts is None:
            self.shifts={}
            self.shifts[0]=np.array([0,0])
            self.shifts[len(self.images)]=np.array([0,0])
        else:
            self.shifts=copy.deepcopy(shifts)

        self.shift_active=False

        self.keyboard_funcs["h"]=self._manual_shift
        self.keyboard_funcs["up"]=self._move_up
        self.keyboard_funcs["down"]=self._move_down
        self.keyboard_funcs["left"]=self._move_left
        self.keyboard_funcs["right"]=self._move_right
            
    def _manual_shift(self):        
        shift=self.get_shift(self.image_counter, self.shifts)

        if not self.shift_active:
            print("shift activated")
            self.shift_active=True
            print("shift is x="+str(shift[0])+" , y="+str(shift[1]))
            if self.image_counter not in self.shifts:
                self.shifts[self.image_counter]=shift
            
        else:
            self.shift_active=False
            print("shift deactivated resulting with:")
            print("x="+str(shift[0])+" , y="+str(shift[1]))

            if self.image_counter>0:
                oldshift=self.get_shift(self.image_counter-1, self.shifts)
                if oldshift[0]==shift[0] and oldshift[1]==shift[1]:
                    print("no change happened")
                    del self.shifts[self.image_counter]
                    
    def _move_up(self):
        if self.shift_active:
            self.shifts[self.image_counter][1] +=1
                 
    def _move_down(self):
        if self.shift_active:
            self.shifts[self.image_counter][1] -=1
               
    def _move_left(self):
        if self.shift_active:
            self.shifts[self.image_counter][0] +=1
                
    def _move_right(self):
        if self.shift_active:
            self.shifts[self.image_counter][0] -=1
                
    @staticmethod
    def get_shift(image_counter,shifts):
        keylist=np.sort(list(shifts.keys()))
        for i in range(1,len(keylist)):
            if keylist[i] > image_counter:
                return shifts[keylist[i-1]]
            else:
                return shifts[keylist[-1]]

    #%%% image overlays
    def addfunc_img_overlays(self,orig_points,overlay_imgs,overlay_points):
        if not hasattr(self,"keyboard_funcs"):
            self.add_keyboard()
            
        if not hasattr(self,"shifts"):
            self.shifts={}
            self.shifts[0]=[0,0]
            self.shifts[len(self.images)]=[0,0]
            
        self.img_overlay=0
        self.img_max=len(overlay_imgs)
        self.orig_points=orig_points
        self.overlay_imgs=overlay_imgs
        self.overlay_points=overlay_points
        
        self.keyboard_funcs["c"]=self._image_overlay


    def _image_overlay(self):           
        if self.img_overlay<self.img_max:
            shift=self.get_shift(self.image_counter,self.shifts)
            dim=self.images[self.image_counter].shape
            p2=copy.deepcopy(self.orig_points)
            for i in range(len(p2)):
                p2[i]-=shift#[::-1]
            
            im1s,im2, matrices, reswidth, resheight, width_shift, height_shift=align_images(
                self.overlay_imgs, self.images[self.image_counter], self.overlay_points, p2,verbose=True)
            
            newim=im1s[self.img_overlay][height_shift:height_shift+dim[0],width_shift:width_shift+dim[1]]
            
            self.img_plot.set_data(newim)
            self.ax.set_title("overlay")
            self.img_overlay+=1

        else:
            self.img_overlay=0
            self.img_plot.set_data(self.images[self.image_counter])
            self.ax.set_title("image "+str(self.image_counter))
                

    #%%% line features
    def addfunc_line_features(self,line_objs=None):
        if not hasattr(self,"keyboard_funcs"):
            self.add_keyboard()

        if not hasattr(self,"shifts"):
            self.shifts={}
            self.shifts[0]=[0,0]
            self.shifts[len(self.images)]=[0,0]
        
        if line_objs is None:
            self.line_objs=[{} for i in range(len(self.images))]
            self.line_set=set()
            self.line_index=0
        else:
            self.line_objs=line_objs
            self.line_set=set()
            for i in self.line_objs:
                for j in i:
                    self.line_set.add(j)
            self.line_index=0
            
        self.artists={}
        self.line_active=False
        self.line_activated_at=0
        self.line_delete=False
        self.line_undo=False

        self.keyboard_funcs["o"]=self._line_overlay        
        self.keyboard_funcs["i"]=self._inactivate_lines
        self.keyboard_funcs["u"]=self._undo_last_line_change
        self.keyboard_funcs["d"]=self._delete_lines
        
        self._main_funcs.append(self.fig.canvas.callbacks.connect)
        self._main_args.append(['pick_event',self._pick_lines])

        self._main_funcs.append(self.fig.canvas.mpl_connect)
        self._main_args.append(['button_press_event',self._generate_lines])  
        
        #self.addfunc_border_snapping()
        #self.addfunc_snap_to_angle()


    #%%%% line overlay
    def _line_overlay(self):
        if not self.line_overlay:
            self.line_overlay=True                
        else:
            for i in list(self.artists.keys()):
                self.artists[i].remove()
                del self.artists[i]
            self.line_overlay=False

    def _plot_overlay(self):
        shift=self.get_shift(self.image_counter, self.shifts) 
                
        for i in self.artists:
            self.artists[i].remove()
        self.artists={}
        
        for key in self.line_objs[self.image_counter]:
            obj=self.line_objs[self.image_counter][key]
            artist = self.ax.plot(np.array(obj.x)-shift[0], np.array(obj.y)-shift[1], 'x-', picker=5,alpha=0.6,c='c')[0]
            artist.index = obj.index  
            self.artists[key]=artist
    
        if self.line_active and self.line_index in self.line_objs[self.image_counter]:
            self.artists[self.line_index].remove()
            del self.artists[self.line_index]
            obj=self.line_objs[self.image_counter][self.line_index]
            if self.image_counter == self.line_activated_at:
                artist = self.ax.plot(np.array(obj.x)-shift[0], np.array(obj.y)-shift[1], 'x-', picker=5,alpha=0.6,c='r')[0]
            else:
                artist = self.ax.plot(np.array(obj.x)-shift[0], np.array(obj.y)-shift[1], 'x-', picker=5,alpha=0.6,c='y')[0]
            artist.index = obj.index  
            self.artists[key]=artist
        
        self.ax.set_xlim(self.ax.get_xlim())
        self.ax.set_ylim(self.ax.get_ylim())
   
    #%%%% line picking
    def _pick_lines(self,event):
        #leftclick_id=1
        if event.mouseevent.button == 1:
            self.line_index=event.artist.index
            self.line_active=True
            self.line_activated_at=self.image_counter
            
            print("line "+str(event.artist.index)+" activated")
            
            if self.line_overlay:
                self._plot_overlay()
            plt.gcf().canvas.draw()
    
    #%%%% line inactivating
    def _inactivate_lines(self):
        if not self.line_active:
            print("no line selected")      
        else:                
            self.line_active=False
            self.line_delete=False
            self.line_undo=False

            obj=self.line_objs[self.image_counter][self.line_index]

            if len(obj.x)==1:
                self.line_set.remove(self.line_index)
                for i in range(len(self.images)):
                    if self.line_index in self.line_objs[i]: 
                        del self.line_objs[i][self.line_index]
                        
                self.artists[self.line_index].remove()
                del self.artists[self.line_index]
                
                print("line "+str(self.line_index)+" deleted")

            else:
                print("line "+str(self.line_index)+" inactivated")
                    
        
    
    #%%%% line generating
    def _generate_lines(self,event):
        #rightclick_id=3
        if event.button == 3:
            shift=self.get_shift(self.image_counter, self.shifts)
            if not self.line_active:    
                self.line_active=True
                self.line_activated_at=self.image_counter
                self._first_point_of_line(event,shift)
                
            else:
                
                obj=self.line_objs[self.image_counter][self.line_index]
    
                if len(obj.x)==1:
                    if self.image_counter==self.line_activated_at:
                        obj=self._add_second_point_of_line(event,obj,shift)
                    else:
                        print("line must be completed in image "+str(self.line_activated_at))
                        print("otherwise delete first point of line with 'i'")
                        return
    
                else:
                    self.backup=copy.deepcopy(obj)
                    obj=self._change_second_point_of_line(event,obj,shift)
                    if obj is None:
                        return
                
                    obj.changed.append(self.image_counter)
    
                
                self.line_overlay=True
                print("line "+str(self.line_index)+ " written")
                      
                self.line_active=False
                
            if self.line_overlay:
                self._plot_overlay()
                plt.gcf().canvas.draw()

    @staticmethod
    def get_next_line_index(line_set):
        if len(line_set)==0:
               max_index=-1
        else:
            max_index=max(line_set)

        if max_index >len(line_set)-1:
            for i in range(max_index):
                if i not in line_set:
                    return i
        else:
            return len(line_set)                
        
    def _first_point_of_line(self,event,shift):
        self.line_index=self.get_next_line_index(self.line_set)    
        print("add line "+str(self.line_index))
        obj=line_object(event.xdata+shift[0],event.ydata+shift[1],self.line_index,self.image_counter) 

        self.line_objs[self.image_counter][self.line_index]=obj

        self.line_overlay=True
        self.line_set.add(self.line_index)
        print(str(event.xdata)+" , "+str(event.ydata))    

    @staticmethod
    def _add_second_point_of_line(event,obj,shift):
        obj.x.append(event.xdata+shift[0])
        obj.y.append(event.ydata+shift[1])
        dx=obj.x[1]-obj.x[0]
        dy=obj.y[1]-obj.y[0]
        obj.length=math.sqrt(dx*dx+dy*dy)
        return obj
    
    @staticmethod
    def _change_second_point_of_line(event,obj,shift): 
        old_x=copy.deepcopy(obj.x)
        old_y=copy.deepcopy(obj.y)
        
        dist0=(event.xdata+shift[0]-obj.x[0])**2+(event.ydata+shift[1]-obj.y[0])**2
        dist1=(event.xdata+shift[0]-obj.x[1])**2+(event.ydata+shift[1]-obj.y[1])**2
        if dist0 > dist1:                   
            obj.x[1]=event.xdata-shift[0]
            obj.y[1]=event.ydata-shift[1]
        else:
            obj.x[0]=event.xdata-shift[0]
            obj.y[0]=event.ydata-shift[1]
            
        dx=obj.x[1]-obj.x[0]
        dy=obj.y[1]-obj.y[0]
        dl=math.sqrt(dx*dx+dy*dy)
        if dl < obj.length:
            print("Change not accepted, length must increase")
            obj.x=old_x 
            obj.y=old_y 
            return None
        else:
            obj.length=dl
            return obj
        

    #%%%% line undo last change  
    def _undo_last_line_change(self):
            
        if not self.line_active:
            print("no line selected")
        else:
            if not self.line_undo:
                print("change to line "+str(self.line_index)+" will be undone?")
                print("Confirm with 'u', cancel with 'i'")
                self.line_undo=True
            else:
                self.line_undo=False                       
                self.line_active=False

                obj=self.line_objs[self.image_counter][self.line_index]
                self.artists[self.line_index].remove()
                del self.artists[self.line_index]
                
                if len(obj.changed)<2:
                    print("to delete press 'd'")
                    
                else:
                    cond0=self.line_index ==self.backup.index
                    cond1=obj.changed[-2] == self.backup.changed[-1]
                    print(self.backup.changed)
                    print(self.image_counter)
                    print(self.line_index)
                    if cond0 and cond1:
                        self.line_objs[self.image_counter][self.line_index]=copy.deepcopy(self.backup)
                    else:
                        print('backup is not aligned with active line')
                            

    #%%%% line deleting
    def _delete_lines(self):
        if self.line_active:
            if not self.line_delete:
                print("line "+str(self.line_index)+" will be deleted?")
                print("Confirm with 'd', cancel with 'i'")
                self.line_delete=True
            else:
                self.line_set.remove(self.line_index)
                for i in range(len(self.images)):
                    if self.line_index in self.line_objs[i]: 
                        del self.line_objs[i][self.line_index]
                        
                
                print("line "+str(self.line_index)+" deleted")
                self.line_delete=False
                self.line_active=False                    
        else:
            print("no line selected to delete")

    
    #%% Text input
    def addfunc_text_input(self):
        self.axbox = self.fig.add_axes([0.05, 0.05, 0.05, 0.075])
        self.text_box = TextBox(self.axbox, "Goto", textalignment="center")
        self.text_box.set_val("0")
        self._main_funcs.append(self.text_box.on_submit)
        self._main_args.append([self._text_input])
    
    def _text_input(self,expression):
        new_image_counter=int(expression)
        if new_image_counter < self.image_counter:
            pass
        else:
            for i in range(self.image_counter+1,new_image_counter+1):
                _progress_to_next_image(i,self.line_objs)
                
        self.image_counter=new_image_counter
        self.img_plot.set_data(self.images[self.image_counter])
        self.ax.set_title("image "+str(self.image_counter))

        

