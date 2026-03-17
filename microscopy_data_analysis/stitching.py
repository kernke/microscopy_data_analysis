# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import cv2
import h5py
import shapely
import os
import networkx as nx
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import imagesize
from tqdm import tqdm

from .image_processing import img_to_uint8
from .image_aligning import phase_correlation,max_from_2d,img_padding_attenuation


class stitching_object:
    def __init__(self,mode="memory",no_print=False):
        """mode can either be 'memory' or 'storage'
        "memory" is the default, assuming the image series being loaded into the RAM
        "storage" is suitable for larger file sizes, when not all images simultaneously fit into the RAM
        (storage supports images as .tif, .png ... or as multiple files in .h5 or as datacube in .h5)"""
        self.mode=mode
        self.no_print=no_print
        self.directory=None
        if not no_print:
            if mode == "storage":
                print("For .h5 provide the path to the h5-file (that serves as h5-directory) via the 'set_directory_path' method")
                print("For .tif, .png, ... image files either provide the folder via 'set_directory_path' or a list of filepaths via 'set_img_list'") 
            if mode == "memory":
                print("please provide the images, as a list of images via 'set_img_list'")
                
    def info(self):
        print("General conditions: ")
        print("All images of the series are assumed, to have no rotation and same pixel size.")
        print("For background subtraction, additionally same dimensions (for example: 512x256,512x256,...) are required.")
            
    def set_directory_path(self,path_string):
        self.directory=path_string
        if path_string[-3:]==".h5":
            self.h5_mode=True
            print("please provide the h5 image file paths as a list via 'set_img_list' or use 'imgs_from_datacube'")
        else:
            print("please provide the image file-paths/names as a list via 'set_img_list'")
            self.h5_mode=False

    def imgs_from_datacube(self,dataset_name):
        """images are assumed to be represented by the first index (for example: 16 x 1024 x 1024)"""
        self.mode="h5_datacube"
        self.dataset_name=dataset_name
        with h5py.File(self.directory,'r') as h5:
            datashape=h5[dataset_name].shape
            self.img_list=np.arange(datashape[0])

    def change_img_list(self,mode="memory",no_print=False):
        """mode can either be 'memory' or 'storage'
        "memory" is the default, assuming the image series being loaded into the RAM
        "storage" is suitable for larger file sizes, when not all images simultaneously fit into the RAM
        (storage supports images as .tif, .png ... or as multiple files in .h5 or as datacube in .h5)"""      
        self.mode=mode
        self.directory=None
        if not no_print:
            if mode == "storage":
                print("For .h5 provide the path to the h5-file (that serves as h5-directory) via the 'set_directory_path' method")
                print("For .tif, .png, ... image files either provide the folder via 'set_directory_path' or a list of filepaths via 'set_img_list'") 
            if mode == "memory":
                print("please provide the images, as a list of images via 'set_img_list'")


    def set_img_list(self,img_list):

        if self.directory is None:
            self.h5_mode=False
            self.img_list=img_list
        else:
            self.img_list=img_list

    
    def get_img(self,index):
        if self.mode=="memory":
            img = self.img_list[index]
        elif self.mode=="storage":
            if self.h5_mode:
                with h5py.File(self.directory,'r') as h5:
                    img=h5[self.img_list[index]][()]
            else:
                img=cv2.imread(self.img_list[index],0)
        elif self.mode=="h5_datacube":
            with h5py.File(self.directory,'r') as h5:
                img=h5[self.dataset_name][index,:,:]
        return img

    def set_positions(self,positions):
        """array or list of tuples, containing x,y""" 
        self.positions=positions

    def set_units_per_pixel(self,units_per_pixel):
        self.units_per_pixel=units_per_pixel

    def sniff_dimensions(self,dimensions=None):
        img=self.get_img(0)
        #self.dtype=img.dtype
        if dimensions is not None:
            self.dimensions=dimensions
        else:
            self.dimensions=np.zeros([len(self.img_list),2],dtype=int)
            
            for i in range(len(self.img_list)):
                if self.h5_mode:
                    img=self.get_img(i)
                    self.dimensions[i]=img.shape
                else:
                    self.dimensions[i]=imagesize.get(self.img_list[i])        
    
    def create_temporary_data(self,filename="temp.h5"):
        self.temp_file=filename
        img=self.get_img(0)
        self.dtype=img.dtype
        if len(np.unique(self.dimensions))>2:
            print("all images must have equal dimensions")
            return 0
        else:
            newshape=[len(self.img_list),self.dimensions[0][0],self.dimensions[0][1]]
        
        with h5py.File(filename, "w") as f:
            f.create_dataset("data", newshape,self.dtype,chunks=(1,self.dimensions[0][0],self.dimensions[0][1]) )#dtype="f4")
            for i in range(len(self.img_list)):
                img=self.get_img(i)
                f["data"][i,:,:]=img

    def subtract_dark_field(self,dark_field):
        with h5py.File(self.temp_file, "r+") as f:
            for i in range(len(self.img_list)):
                f["data"][i,:,:]=np.maximum(f["data"][i,:,:]-dark_field,0)
                
    def flat_field_generation(self,percentile_steps=19,subdiv=4):
        #needs temp data
        #img_series=np.array(img_list)
        #print(percentile_steps)
        step=100/(percentile_steps+1)
        steps=np.linspace(step,100-step,percentile_steps)
        dims=self.dimensions[0]
        flats=np.zeros([percentile_steps,dims[0],dims[1]])
        subdiv_steps=int(dims[0]//4)
        with h5py.File(self.temp_file, "r") as f:
            start_index=0
            for i in range(1,subdiv):
                if not self.no_print:
                    print(str(i)+" out of "+str(subdiv)+" iterations")
                flats[:,start_index:i*subdiv_steps,:]=np.percentile(f["data"][:,start_index:i*subdiv_steps,:],steps,axis=0)                
            if not self.no_print:
                print(str(subdiv)+" out of "+str(subdiv)+" iterations")
            flats[:,(subdiv-1)*subdiv_steps:,:]=np.percentile(f["data"][:,(subdiv-1)*subdiv_steps:,:],steps,axis=0)
        #flats=np.percentile(img_series,steps,axis=0)
        self.flat_field=np.mean(flats,axis=0)
        self.flats=flats
        return self.flat_field

    def dust_from_flat_field(self,flat_field,threshold_block_size=17,morph_closing_size=5,dark_background=True,dilate=0):
        th=  cv2.adaptiveThreshold(img_to_uint8(flat_field),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,threshold_block_size,2) 
        kernel = np.ones((morph_closing_size, morph_closing_size), np.uint8)
        closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    
        if dark_background:
            num,img=cv2.connectedComponents(255-closed)
        else:
            num,img=cv2.connectedComponents(closed)
        if not self.no_print:
            print(str(num)+" dust particles found")
    
        img_d=cv2.dilate(img.astype(np.uint8),np.ones([3,3],dtype=np.uint8))
        last=np.copy(img)
        for i in range(dilate):
            last=np.copy(img_d)
            img_d=cv2.dilate(img_d,np.ones([3,3],dtype=np.uint8))
            
            
        rings=img_d-last
    
        self.dust_dict=dict()
        for i in range(num-1):
            self.dust_dict[i]=list() 
            #dust indices
            self.dust_dict[i].append(np.where(last==i+1))
    
            #ring indices
            self.dust_dict[i].append(np.where(rings==i+1))
        self.dust_dict
        return last

    def dust_removal(self,img,mode="median"):
        #needs temp data
        result=np.copy(img)
        for i in self.dust_dict:
            if len(self.dust_dict[i][1][0])<1:
                print("Warning: empty ring")
            if mode=="median":
                ring_values=np.median(img[self.dust_dict[i][1]])
            elif mode=="mean":        
                ring_values=np.mean(img[self.dust_dict[i][1]])
            elif mode=="normal":
                ring_values=np.mean(img[self.dust_dict[i][1]])
                ring_std=np.std(img[self.dust_dict[i][1]],mean=ring_values)
                ring_values=np.random.normal(ring_values,ring_std,len(img[self.dust_dict[i][1]]))
            
            result[self.dust_dict[i][0]]=ring_values
        return result
    
    def dust_removal_all(self):
        self.flat_field=self.dust_removal(self.flat_field)
        with h5py.File(self.temp_file, "r+") as f:
            for i in range(len(self.img_list)):
                f["data"][i,:,:]=self.dust_removal(f["data"][i,:,:])

    def flat_field_correction(self,flat_field=None):
        if flat_field is None:
            flat_field=self.flat_field
        if np.min(flat_field)<=0:
            flat_field=flat_field.astype(np.double)
            flat_field[flat_field<=0]=np.inf
            flat_field[flat_field==np.inf]=np.min(flat_field)
        
        newshape=[len(self.img_list),self.dimensions[0][0],self.dimensions[0][1]]
        with h5py.File(self.temp_file, "r+") as f:
            f.create_dataset("data_corrected", newshape,dtype="f4",chunks=(1,self.dimensions[0][0],self.dimensions[0][1]))
            for i in range(len(self.img_list)):
                f["data_corrected"][i,:,:]=f["data"][i,:,:]/flat_field  

    def convert_to_uint16(self,real_zero=False):#,newdtype=np.uint16):
        img_maxs=np.zeros(len(self.img_list))
        img_mins=np.zeros(len(self.img_list))        
        with h5py.File(self.temp_file, "r") as f:
            for i in range(len(self.img_list)):
                corr=f["data_corrected"][i,:,:]  
                img_maxs[i]=np.max(corr)
                img_mins[i]=np.min(corr)                     

            img_min=np.min(img_mins)
            img_max=np.max(img_maxs)
            div=img_max-img_min
                
            newname=self.temp_file[:-3]+"_very_temporary.h5"
            newshape=[len(self.img_list),self.dimensions[0][0],self.dimensions[0][1]]
            with h5py.File(newname, "w") as tf:
                tf.create_dataset("data", newshape,np.uint16,chunks=(1,self.dimensions[0][0],self.dimensions[0][1]))
                for i in range(len(self.img_list)):
                    corr=((f["data_corrected"][i,:,:]-img_min)/div) * 65535  
                    tf["data"][i,:,:]=corr.astype(np.uint16)
            if real_zero:
                newname=self.temp_file[:-3]+"_real_zero.h5"
                with h5py.File(newname, "w") as tf:
                    tf.create_dataset("data", newshape,np.uint16,chunks=(1,self.dimensions[0][0],self.dimensions[0][1]))
                    for i in range(len(self.img_list)):
                        corr=(f["data_corrected"][i,:,:]/img_max) * 65535  
                        tf["data"][i,:,:]=corr.astype(np.uint16)

        os.remove(self.temp_file)
        os.rename(self.temp_file[:-3]+"_very_temporary.h5",self.temp_file)


    def make_polygons(self,units_per_pixel=None,positions=None,dimensions=None,orientation=0):
        if units_per_pixel is None:
            units_per_pixel=self.units_per_pixel
        if positions is None:
            positions=self.positions
        if dimensions is None:
            dimensions=self.dimensions

        N=len(positions)
        anchor_points=np.zeros([N,2])
        polygons=[]
        im_size=units_per_pixel*dimensions
        
        side0comp=np.ones(N)
        side1comp=np.zeros(N)
            
        for i in range(N):
            deltax=im_size[i,1]*np.array([side0comp[i],side1comp[i]])
            deltay=im_size[i,0]*np.array([-side1comp[i],side0comp[i]])
            p0=positions[i]+deltax/2-deltay/2
            p1=p0-deltax
            p2=p1+deltay
            p3=p2+deltax
            polygons.append(shapely.geometry.Polygon([p0,p1,p2,p3]))
            if orientation==0:
                anchor_points[i]=p0
            elif orientation==1:
                anchor_points[i]=p1
            elif orientation==2:
                anchor_points[i]=p2
            elif orientation==3:
                anchor_points[i]=p3
        self.polygons=polygons
        self.anchor_points=anchor_points
        return polygons,anchor_points



    def connection_groups(self,polygons=None,units_per_pixel=None,minimal_number_of_pixels=64,inverse=False):
        if polygons is None:
            polygons=self.polygons
        if units_per_pixel is None:
            units_per_pixel=self.units_per_pixel
    
        N=len(polygons)
        square_units=units_per_pixel*units_per_pixel
        G = nx.Graph()
        G.add_nodes_from(np.arange(N))
        
        for i in range(N-1):
            for j in range(i+1,N):
                inter=shapely.intersection(polygons[i],polygons[j])
                area=inter.area/square_units
                cond1=area>minimal_number_of_pixels
                if cond1:
                    if inverse:
                        G.add_edge(i,j,weight=1/area)
                    else:
                        G.add_edge(i,j,weight=area)
    
        con_groups=[]
        for i in nx.connected_components(G):
            con_groups.append(np.array(list(G.subgraph(i).edges)))
        self.G=G
        if len(con_groups)>1:
            print("not all images can be connected via overlap")
            print("sorting out of non-connected images needed")
        self.con_group=con_groups[0]
        return G,con_groups

    def plot_connection_network(self,figsize=[15,15],relative=True):
        plt.figure(figsize=figsize)
        if relative:
            plt.title("percentage of maximum neighbor-overlap")
        else:
            plt.title("neighbor-overlap in pixels")

        nx.draw(self.G, pos=self.positions, with_labels=True,)
        edge_labels = nx.get_edge_attributes(self.G, "weight")
        keylist=list(edge_labels.keys())
        maxval=0
        for i in keylist:
            area=edge_labels[i]
            maxval=np.maximum(area,maxval)
        for i in keylist:
            if relative:
                edge_labels[i]=np.round(edge_labels[i]/maxval * 100,1)
            else:
                edge_labels[i]=int(np.round(edge_labels[i]))
        nx.draw_networkx_edge_labels(self.G, self.positions, edge_labels,label_pos=0.3)
        plt.show()


    def get_outer_polygon_limits(self,polygons):
        minx=np.inf
        miny=np.inf
        maxx=-np.inf
        maxy=-np.inf    
        for polygon in polygons:
            #if not isinstance(i,float):
            x,y=polygon.exterior.xy
            pminx,pmaxx=np.min(x),np.max(x)
            pminy,pmaxy=np.min(y),np.max(y)
    
     
            minx=min(pminx,minx)
            miny=min(pminy,miny)
            maxx=max(pmaxx,maxx)
            maxy=max(pmaxy,maxy)
        
    
        points=np.zeros([4,2])
        points[0]=minx,miny
        points[1]=maxx,miny
        points[2]=maxx,maxy
        points[3]=minx,maxy
        return points
    
    
    def close_translation_by_phase_correlation(self,im1,im2,sigma=1,max_transl=None):
        
        #dims=im1.shape
            
        mat=phase_correlation(im1,im2)
        #matb=cv2.blur(mat,[blur,blur])
        ksize=int(sigma*4)
        if ksize%2==0:
            ksize+=1
        matb=cv2.GaussianBlur(mat,[ksize,ksize],sigma)#,cv2.BORDER_WRAP)
        mat0=matb-np.min(matb)
        
        dims=mat0.shape
        
        mean=np.mean(mat0)
        std=np.std(mat0)
        #print(std)
        if max_transl:
            img_mask=np.zeros(dims)
            img_mask[:max_transl[0]+1,:max_transl[1]+1]=1
            img_mask[:max_transl[0]+1,-max_transl[1]:]=1
            img_mask[-max_transl[0]:,:max_transl[1]+1]=1
            img_mask[-max_transl[0]:,-max_transl[1]:]=1
            #plt.imshow(img_mask)
        else:
            img_mask=np.ones(dims)
            
        matm=mat0*img_mask
        
        transl,maxval=max_from_2d(matm)
        if transl[0]>dims[0]/2:
            transl[0]-=dims[0]
        if transl[1]>dims[1]/2:
            transl[1]-=dims[1]
    
        
        certainty=(maxval-mean)/std
        return transl,certainty
    
    
    def real_to_pixel(self,index,points,orientation=2):
        unit_per_pixel=self.units_per_pixel
        anchor_point=self.anchor_points[index]
        points=np.vstack((points,anchor_point))
        #print("sdfsdf")
        #print(points)
        points=points[:-1]-points[-1]
        
        points /= unit_per_pixel
        points=points[:,::-1] #*-1
        #if orientation==0:
        #    points[:,1] *= -1
        if orientation==2:
            points[:,0] *= -1
        
        return points#np.round(points).astype(int)
    
    def crop_from_points(self,img,points,shape=None):
        vmin,hmin=np.min(points,axis=0)
        vmax,hmax=np.max(points,axis=0)
        
        vmin_int=int(np.round(vmin))
        vmax_int=int(np.round(vmax))
        hmin_int=int(np.round(hmin))
        hmax_int=int(np.round(hmax))
        #print(shape)
        #print("----------")
        #print(vmax-vmin)
        if shape:# is not None:
            if vmax_int-vmin_int > shape[0]: #bigger than intended
                lower_residual=vmin_int-vmin # negative makes smaller
                upper_residual=vmax-vmax_int # positive makes bigger
                if lower_residual<upper_residual:
                    vmin_int+=1
                else:
                    vmax_int-=1
                
            elif vmax_int-vmin_int < shape[0]: #smaller than intended
                lower_residual=vmin_int-vmin # negative makes smaller
                upper_residual=vmax-vmax_int # positive makes bigger
                if lower_residual>upper_residual:
                    vmin_int-=1
                else:
                    vmax_int+=1
                    
            if hmax_int-hmin_int > shape[1]: #bigger than intended
                lower_residual=hmin_int-hmin # negative makes smaller
                upper_residual=hmax-hmax_int # positive makes bigger
                if lower_residual<upper_residual:
                    hmin_int+=1
                else:
                    hmax_int-=1
                
            elif hmax_int-hmin_int < shape[1]: #smaller than intended
                lower_residual=hmin_int-hmin # negative makes smaller
                upper_residual=hmax-hmax_int # positive makes bigger
                if lower_residual>upper_residual:
                    hmin_int-=1
                else:
                    hmax_int+=1
            
        return img[vmin_int:vmax_int,hmin_int:hmax_int]
    
    
    def pixel_to_real(self,index,points):
        anchor_point=self.anchor_points[index]
        unit_per_pixel=self.units_per_pixel
        vec1=np.array([unit_per_pixel,0])
        vec0=np.array([0,unit_per_pixel])
        real_points=np.empty(points.shape)
        for i in range(len(points)):
            real_points[i]=points[i,1]*vec1-points[i,0]*vec0    
            real_points[i]+=anchor_point        
        return real_points

    def check_pairs(self,max_transl_pix=None,sigma=1,check_data=False):
        
        shifts=[]
        if check_data:        
            croplist1=[]
            croplist2=[]
        pairs=np.array(self.G.edges())
        
        for pair in tqdm(pairs):

            index1=pair[0]
            index2=pair[1]
        
            poly1=self.polygons[index1]
            poly2=self.polygons[index2]
        
            overlap=shapely.intersection(poly1,poly2)
            overlap_coord=np.array(overlap.oriented_envelope.exterior.xy).T[:-1]
        
        
            im1_overlap_coord=self.real_to_pixel(index1,
                            overlap_coord)
    
            im2_overlap_coord=self.real_to_pixel(index2,
                            overlap_coord)

            im2=self.get_img(index2)
            im2_crop=self.crop_from_points(im2,im2_overlap_coord)
        

            im1=self.get_img(index1)
            im1_crop=self.crop_from_points(im1,im1_overlap_coord)
            
            if check_data:
                croplist1.append(im1_crop)
                croplist2.append(im2_crop)
            
            pix_transl,certainty=self.close_translation_by_phase_correlation(im1_crop[:,:],im2_crop[:,:],
                                                                             sigma=sigma,max_transl=max_transl_pix)#*im1_crop[:,:,1]
        
            real_transl=self.units_per_pixel*pix_transl[::-1]
            real_transl[1]*=-1
            
            shifts.append(self.positions[index2]-self.positions[index1]+real_transl)

        pcm_distances=dict()
        edge_tuples=list(map(tuple, pairs))
        for i,edge in enumerate(edge_tuples):
            pcm_distances[edge]=shifts[i]

        self.pcm_distances=pcm_distances
        self.edge_tuples=edge_tuples

        if check_data:
            return pcm_distances,croplist1,croplist2
        else:
            return pcm_distances

    @staticmethod
    def _residuals(posflat,edge_tuples,pcm_distances):
        n=int(len(posflat)//2)
        pts = posflat.reshape(n, 2)
        r = []

        for i, j in edge_tuples:
            r.append( np.linalg.norm(pts[j] - pts[i]- pcm_distances[i, j]))

        return np.array(r)

    def optimize_positions(self,verbose=True):

        n_res = len(self.edge_tuples)
        n_var = len(self.positions)*2

        S = lil_matrix((n_res, n_var))

        counter=0
        for i,j in self.edge_tuples:
            S[counter,2*i]=1
            S[counter,2*i+1]=1
            S[counter,2*j]=1
            S[counter,2*j+1]=1
            counter+=1

        x0=np.ravel(self.positions)
        if verbose:
            verbose_level=2
        else:
            verbose_level=0

        result = least_squares(
            self._residuals,
            x0,
            jac_sparsity=S,
            args=(self.edge_tuples,self.pcm_distances),
            method='trf',
            verbose=verbose_level
        )


        final=result.x.reshape(len(self.positions),2)

        moved_polygons = [
            shapely.affinity.translate(poly, xoff=px - poly.centroid.x, yoff=py - poly.centroid.y)
            for poly, (px, py) in zip(self.polygons, final)
        ]
        return moved_polygons


    def map_from_polygons_h5(self,polygons,h5file="temp.h5",blending="average",custom_mask=None):

        start_index=0
        outer_realspace=self.get_outer_polygon_limits(polygons)
        pixelouter=self.real_to_pixel(start_index,outer_realspace)
        pixelouter=np.round(pixelouter).astype(int)
        offset_x_y=-np.min(pixelouter,axis=0)
        image_dims=np.max(pixelouter,axis=0)+offset_x_y

        with h5py.File(h5file, "a") as f:
            division_mask = f.create_dataset(
                        "mask",
                        shape=image_dims,
                        dtype="float32",
                        chunks=(512, 512),
                        fillvalue=0   
                    )
            image = f.create_dataset(
                        "data",
                        shape=image_dims,
                        dtype="float32",
                        chunks=(512, 512),
                        fillvalue=0   
                    )

            chunk_rows, chunk_cols = image.chunks



            for index,poly in enumerate(polygons):
                np_realspace=np.array(poly.oriented_envelope.exterior.xy).T[:-1]
                np_pixelspace=np.round(self.real_to_pixel(start_index,np_realspace)).astype(int)
                image_pixelspace=np_pixelspace+offset_x_y
                img=self.get_img(index)
                img_start=np.min(image_pixelspace,axis=0)
                #img_end=np.max(image_pixelspace,axis=0)
                img_end=img_start+img.shape

                if blending=="average":
                    image[img_start[0]:img_end[0],img_start[1]:img_end[1]]+=img
                    division_mask[img_start[0]:img_end[0],img_start[1]:img_end[1]]+=1



            n_rows, n_cols = image.shape
            for row_start in tqdm(range(0, n_rows, chunk_rows), desc="Rows"):
                row_end = min(row_start + chunk_rows, n_rows)

                # loop over column chunks
                for col_start in range(0, n_cols, chunk_cols):
                    col_end = min(col_start + chunk_cols, n_cols)
                    
                    divider=division_mask[row_start:row_end, col_start:col_end]
                    divider[divider==0]=1
                    # slice the 2D chunk
                    image[row_start:row_end, col_start:col_end] = image[row_start:row_end, col_start:col_end]/divider
          


    def map_from_polygons(self,polygons,blending="average",custom_mask=None):
        start_index=0
        outer_realspace=self.get_outer_polygon_limits(polygons)
        pixelouter=self.real_to_pixel(start_index,outer_realspace)
        pixelouter=np.round(pixelouter).astype(int)
        offset_x_y=-np.min(pixelouter,axis=0)
        image_dims=np.max(pixelouter,axis=0)+offset_x_y


        division_mask=np.zeros(image_dims)#,dtype=np.uint8)
        image=np.zeros(image_dims)
   
        if blending=="minimum":
            image += np.inf

        for index,poly in enumerate(polygons):
            np_realspace=np.array(poly.oriented_envelope.exterior.xy).T[:-1]
            np_pixelspace=np.round(self.real_to_pixel(start_index,np_realspace)).astype(int)
            image_pixelspace=np_pixelspace+offset_x_y
            img=self.get_img(index)
            img_start=np.min(image_pixelspace,axis=0)
            #img_end=np.max(image_pixelspace,axis=0)
            img_end=img_start+img.shape

            if blending=="hard_cut":
                image[img_start[0]:img_end[0],img_start[1]:img_end[1]]=img
            elif blending=="maximum":
                stack=np.empty([img.shape[0],img.shape[1],2])
                stack[:,:,0]=image[img_start[0]:img_end[0],img_start[1]:img_end[1]]
                stack[:,:,1]=img
                image[img_start[0]:img_end[0],img_start[1]:img_end[1]]=np.max(stack,axis=-1)
            elif blending=="minimum":
                stack=np.empty([img.shape[0],img.shape[1],2])
                stack[:,:,0]=image[img_start[0]:img_end[0],img_start[1]:img_end[1]]
                stack[:,:,1]=img
                image[img_start[0]:img_end[0],img_start[1]:img_end[1]]=np.min(stack,axis=-1)
            elif blending=="linear":
                blending_helper=np.ones(img.shape)     
                imshape=np.array(img.shape)
                blending_helper=img_padding_attenuation(blending_helper,imshape//2,mode="linear")   
                image[img_start[0]:img_end[0],img_start[1]:img_end[1]]+=img*blending_helper
                division_mask[img_start[0]:img_end[0],img_start[1]:img_end[1]]+=blending_helper 
            elif blending=="quadratic":
                blending_helper=np.ones(img.shape)     
                imshape=np.array(img.shape)
                blending_helper=img_padding_attenuation(blending_helper,imshape,mode="linear")   
                image[img_start[0]:img_end[0],img_start[1]:img_end[1]]+=img*blending_helper
                division_mask[img_start[0]:img_end[0],img_start[1]:img_end[1]]+=blending_helper 
            elif blending=="average":
                image[img_start[0]:img_end[0],img_start[1]:img_end[1]]+=img
                division_mask[img_start[0]:img_end[0],img_start[1]:img_end[1]]+=1

            elif blending=="custom_single":
                image[img_start[0]:img_end[0],img_start[1]:img_end[1]]+=img*custom_mask
                division_mask[img_start[0]:img_end[0],img_start[1]:img_end[1]]+=custom_mask
            elif blending=="custom_multi":  
                image[img_start[0]:img_end[0],img_start[1]:img_end[1]]+=img*custom_mask[index]
                division_mask[img_start[0]:img_end[0],img_start[1]:img_end[1]]+=custom_mask[index]


        if blending=="minimum":
            image[image==np.inf]=0

        division_mask[division_mask==0]=1
        image[:,:]/=division_mask
        return image

    def z_transform_images(self,h5file="temp.h5",offset_positive=True):
        new_img_list=[]
        most_negative=0
        if self.mode=="memory":
            for img in self.img_list:
                mean=np.mean(img)
                std=np.std(img)
                z_trans=(img-mean)/std
                new_img_list.append(z_trans)
                most_negative=min(most_negative,np.min(z_trans))
 
            if offset_positive:
                for i in range(len(self.img_list)):
                    new_img_list[i] -= most_negative

        else:
            self.temp_file=h5file
            with h5py.File(h5file, "a") as f:
                if "z_transform_name_list" in f:
                    del f["z_transform_name_list"]    
                f["z_transform_name_list"]=self.img_list
                maxnum=len(self.img_list)
                zfillnum=int(np.ceil(np.log10(maxnum)))
                for i in range(len(self.img_list)):
                    num=str(i).zfill(zfillnum)
                    img=self.get_img(i)
                    minimum=np.min(img)
                    mean=np.mean(img)
                    std=np.std(img)
                    if "z_transform/"+num in f:
                        del f["z_transform/"+num]
                    f["z_transform/"+num]=(img-mean)/std
                    most_negative=min((minimum-mean)/std,most_negative)
                    new_img_list.append("z_transform/"+num)
                
                if offset_positive:
                    for i in range(len(self.img_list)):
                        num=str(i).zfill(zfillnum)
                        arr=f["z_transform/"+num][:]
                        arr-=most_negative
                        f["z_transform/"+num][:]=arr-most_negative

        return new_img_list

