# -*- coding: utf-8 -*-
"""
"""
import numpy as np
import cv2
from .image_processing import img_gray_to_rgba,img_add_weighted_rgba,img_autoclip,img_to_uint8_fast
from .image_processing import img_single_to_double_channel,img_add_weighted_gray_alpha,img_rebin_by_mean
from .image_aligning import align_pair, sift_align_matches,phase_correlation,align_pair_special,max_from_2d
import matplotlib.pyplot as plt
import networkx as nx
import shapely
from skimage.registration import phase_cross_correlation
import skimage
#skimage.registration.phase_cross_correlation(btest1,btest2)

def make_polygons(positions,units_per_pixel,dimensions,scan_rotations=None):
    #scan_rotation counterclockwise
    N=len(positions)
    anchor_points=np.zeros([N,2])
    polygons=[]
    if len(units_per_pixel.shape)==1:
        units_per_pixel=np.stack((units_per_pixel,units_per_pixel),axis=1)
    im_size=units_per_pixel*dimensions
    
    if scan_rotations is not None:
        side0comp=np.cos(np.radians(-scan_rotations))
        side1comp=np.sin(np.radians(-scan_rotations))
    else:
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
        anchor_points[i]=p0        
    return polygons,anchor_points


def pixel_to_real_relation(im1,im2,pos1,pos2,check=True):
    #two uint8-images with some overlap, with same magnification(scale), resolution(dimensions) and rotation
    if check:
        if len(im1.shape)==2:
            im1=img_gray_to_rgba(im1)
        if len(im2.shape)==2:
            im2=img_gray_to_rgba(im2)
            
    pts1,pts2=sift_align_matches(im1,im2)
    res1,res2,params=align_pair(im1,im2,pts1,pts2,verbose=True)

    realshift=pos2-pos1
    realdist=np.sqrt(np.sum(realshift*realshift))
    pixshift=params["translation"]
    pixdist=np.sqrt(np.sum(pixshift*pixshift))
    units_per_pixel=realdist/pixdist
    deltarad=np.arctan2(pixshift[1],-pixshift[0])-np.arctan2(realshift[1],realshift[0])
    deltadeg=np.degrees(deltarad)
    if check:
        stitched=img_add_weighted_rgba(res1, res2)
        plt.imshow(stitched)
        
    return units_per_pixel,deltadeg


def real_to_pixel(unit_per_pixel,scan_rotation,anchor_point,points):
    theta=np.radians(scan_rotation)
    a=np.sin(theta)
    b=np.cos(theta)
    rotation_matrix=np.array([[b,a],[-a,b]])
    points=np.vstack((points,anchor_point))
    res=np.tensordot(points,rotation_matrix,axes=1)
    #print(res.shape)
    points=res[:-1]-res[-1]
    points /= unit_per_pixel
    points=points[:,::-1]
    points[:,1] *= -1
    return np.round(points).astype(int)


def pixel_to_real(unit_per_pixel,scan_rotation,anchor_point,points):#,anchor_shift=np.zeros(2)):
    side0comp=np.cos(np.radians(-scan_rotation))*unit_per_pixel
    side1comp=np.sin(np.radians(-scan_rotation))*unit_per_pixel
    vec1=np.array([side0comp,side1comp])
    vec0=np.array([-side1comp,side0comp])
    #anchor_point=anchor_point-anchor_shift[0]*vec0+anchor_shift[1]*vec1
    real_points=np.empty(points.shape)
    for i in range(len(points)):
        real_points[i]=-points[i,1]*vec1+points[i,0]*vec0    
        real_points[i]+=anchor_point
        
    return real_points#,anchor_point


def connection_groups(polygons,units_per_pixel,minimal_number_of_pixels):
    N=len(polygons)
    square_units=units_per_pixel*units_per_pixel
    G = nx.Graph()
    G.add_nodes_from(np.arange(N))
    for i in range(N-1):
        for j in range(i+1,N):
            inter=shapely.intersection(polygons[i],polygons[j])
            area=inter.area
            cond1=area/square_units[i]>minimal_number_of_pixels
            cond2=area/square_units[j]>minimal_number_of_pixels
            #intersection=polygons[i].intersects(polygons[j])
            if cond1 and cond2:#intersection:
                G.add_edge(i,j)

    con_groups=[]
    for i in nx.connected_components(G):
        con_groups.append(np.array(list(G.subgraph(i).edges)))
    return con_groups#,con_graph
#G=nx.from_edgelist(con_groups[0])

def path_through_connected(con_group,start=None):
    if start is None:
        start=np.min(con_group)
    G=nx.from_edgelist(con_group)
    edgepath=[i for i in nx.bfs_edges(G,source=start)]
    #node_dict=dict()
    return np.array(edgepath)

def create_metadata(pathlist,positions,dimensions,units_per_pixel,scan_rotations,polygons,anchor_points):#,connected_groups):
    metadata=dict()
    for i in range(len(pathlist)):
        metadata[i]=dict()
        metadata[i]["filepath"]=pathlist[i]
        metadata[i]["position"]=positions[i]
        metadata[i]["dimensions"]=dimensions[i]
        metadata[i]["units_per_pixel"]=units_per_pixel[i]
        metadata[i]["scan_rotation"]=scan_rotations[i]
        metadata[i]["polygon"]=polygons[i]
        metadata[i]["anchor_point"]=anchor_points[i]
    
    return metadata

def get_outer_polygon_limits(metadata):
    minx=np.inf
    miny=np.inf
    maxx=-np.inf
    maxy=-np.inf    
    for i in metadata.values():
        x,y=i["polygon"].exterior.xy
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

def get_stitchpath(stitchpaths,index):
    if not isinstance(index,list):
        index=[index]
    newindex=stitchpaths[index[-1]]
    if newindex==index[-1]:
        return index
        
    index.append(int(stitchpaths[index[-1]]))
    return get_stitchpath(stitchpaths,index)

def stitchpath_correction(edgepath0,metadata,infolog):

    #startpos=metadata[-1]["position"]
    endpos=metadata[edgepath0]["position"]
    
    return infolog[edgepath0]["im1pos"]-endpos#al_delta-from_pixel_delta#)[::-1]

def process_on_loading(img,subtract_background=75,normalizer=None,clip_ratio=0.01,rebin=None):
    
    if rebin is None:
        img_eff=img
    else:
        img_eff=img_rebin_by_mean(img, rebin)
    
    if subtract_background:
        im1float=img_eff/(1+cv2.blur(img_eff,[subtract_background,subtract_background]))
    else:                                                     
        #if normalizer is None:
        im1float=img_eff
        #else:
        #    im1float=img/normalizer

    if clip_ratio:
        im1float=img_autoclip(im1float,clip_ratio)
    
    resimg=np.empty([im1float.shape[0],im1float.shape[1],2],dtype=np.uint8)
    resimg[:,:,0]=img_to_uint8_fast(im1float)
    if normalizer is None:
        resimg[:,:,1]+=255
    else:
        resimg[:,:,1]=normalizer
    return resimg


def prepare_stitch_object(edgepath,metadata,con_group,padding=0,process_on_loading=None):
    
    index1=edgepath[0,0]    
    
    im1=cv2.imread(metadata[index1]["filepath"],0) 
    if process_on_loading is not None:
        res_img_small=process_on_loading(im1)
    else:
        res_img_small=im1

    
    G=nx.from_edgelist(con_group)
    metadata[-1]=metadata[index1].copy()

    outer_points=get_outer_polygon_limits(metadata)
    
    pixel_points=real_to_pixel(metadata[-1]["units_per_pixel"], metadata[-1]["scan_rotation"], metadata[-1]["anchor_point"],outer_points)
    min0,max0=np.min(pixel_points[:,0]),np.max(pixel_points[:,0])
    min1,max1=np.min(pixel_points[:,1]),np.max(pixel_points[:,1])

    min0-=padding
    min1-=padding
    max0+=padding
    max1+=padding    

    resdim=np.array([max0-min0,max1-min1])        
    res_img=np.zeros([resdim[0],resdim[1],2],dtype=np.uint8)
    res_img[-min0:-min0+res_img_small.shape[0],-min1:-min1+res_img_small.shape[0],:]=img_single_to_double_channel(res_img_small)

    anchor_update=pixel_to_real(metadata[-1]["units_per_pixel"], metadata[-1]["scan_rotation"], metadata[-1]["anchor_point"],np.array([[min0,min1]]))
    metadata[-1]["anchor_point"]=anchor_update[0]
    infolog=metadata[-1].copy()


    infolog[index1]=dict()
    infolog[index1]["step"]=0
    infolog[index1]["translation"]=np.zeros(2)
    infolog[index1]["rotation"]=0
    infolog[index1]["scale"]=1
    infolog[index1]["im1shift"]=np.array([-min0,-min1])
    infolog["stitchpaths"]=dict()
    infolog["stitchpaths"][index1]=index1
    #infolog["stitchindices"]=[index1]
    
    infolog[index1]["im1pos"]=metadata[index1]["position"]

    stitchobject=dict()

    stitchobject["edgepath"]=edgepath
    stitchobject["res_img"]=res_img
    stitchobject["graph"]=G
    stitchobject["counter"]=0
    stitchobject["infolog"]=infolog
    stitchobject["metadata"]=metadata.copy()
    return stitchobject



def stitch_by_overlap_no_copy(im1,im2,index1,index2,metadata,level_lowe=0.5,rotation_limit=None,relative_scale_limit=None,overlap_pad=0,
                              translation_only=False,correct_for_stitchpath=None,infolog=None,plotting=False,
                              minimal_number_of_sift_features=15,minimal_correlation_sigma=5,
                              no_sift=False,force=False):

    #if len(im2.shape)==2:
    #    im2=img_single_to_double_channel(im2)

    if correct_for_stitchpath is None:
        poly2=metadata[index2]["polygon"]
    else:
        #poly3=metadata[index2]["polygon"]
        poly2=shapely.affinity.translate(metadata[index2]["polygon"],correct_for_stitchpath[0],correct_for_stitchpath[1])#1+

    overlap=shapely.intersection(metadata[index1]["polygon"],poly2)
    
    overlap_coord=np.array(overlap.oriented_envelope.exterior.xy).T[:-1]


    if plotting or len(overlap_coord)==0:
        #p3p=-np.array(poly3.exterior.xy)
        p2p=-np.array(poly2.exterior.xy)
        p1p=-np.array(metadata[index1]["polygon"].exterior.xy)
        if len(overlap_coord)==0:
            pass
        else:
            pop=-np.array(overlap.exterior.xy)
            plt.plot(pop[0],pop[1],'r')
        
        #plt.plot(p3p[0],p3p[1],'b')    
        plt.plot(p2p[0],p2p[1],'g')
        plt.plot(p1p[0],p1p[1],'k')
        plt.axis("equal")
        plt.show()

    if len(overlap_coord)==0:
        print("empty overlap")
        #a=5/0
        return None,None,None

    #overlapcenter=np.array(overlap.centroid.coords[0])
    #distances=np.empty(len(p2coord))
    #for i in range(len(distances)):
    #    distances[i]=np.sqrt(np.sum((p2coord[i]-overlapcenter)*(p2coord[i]-overlapcenter)))
    #distance=np.max(distances)
    #im1_points=np.empty([4,2])
    #im1_points[0]=overlapcenter+np.array([distance,distance])
    #im1_points[1]=overlapcenter+np.array([distance,-distance])
    #im1_points[2]=overlapcenter+np.array([-distance,-distance])
    #im1_points[3]=overlapcenter+np.array([-distance,distance])
    #print(overlap_coord)
    #print(overlapcenter)
    #print(distance)
    
    

    im1_coord=real_to_pixel(metadata[index1]["units_per_pixel"],
                            metadata[index1]["scan_rotation"],
                            metadata[index1]["anchor_point"],
              np.array(metadata[index1]["polygon"].exterior.xy).T[:-1])
    
    im1_overlap_coord=real_to_pixel(metadata[index1]["units_per_pixel"],
                            metadata[index1]["scan_rotation"],
                            metadata[index1]["anchor_point"],
                            overlap_coord)

        
    lower1=np.min(im1_overlap_coord,axis=0)-overlap_pad
    upper1=np.max(im1_overlap_coord,axis=0)+overlap_pad

    
    #print(correct_for_stitchpath)
    for i in range(len(overlap_coord)):
        overlap_coord[i]-=correct_for_stitchpath
        
    im2_coord=real_to_pixel(metadata[index2]["units_per_pixel"],
                            metadata[index2]["scan_rotation"],
                            metadata[index2]["anchor_point"],
                            overlap_coord)


    lower2=np.min(im2_coord,axis=0)-overlap_pad
    upper2=np.max(im2_coord,axis=0) +overlap_pad      
    im1_cropdim=upper1-lower1
    im2_cropdim=upper2-lower2
    
    dimdiff=im2_cropdim-im1_cropdim
    if np.sum(np.abs(dimdiff)>0):
        print("dimdiff>0 some rounding")
    if  dimdiff[0]<0:
        lower1[0]+=1
    elif dimdiff[0]>0:
        lower2[0]+=1
        
    if  dimdiff[1]<0:
        lower1[1]+=1
    elif dimdiff[1]>0:
        lower2[1]+=1

    cropdim=upper2-lower2
    
    im2_cropped=np.zeros([cropdim[0],cropdim[1],2],dtype=np.uint8)
    
    offs2=np.zeros(2,dtype=int)
    l2=np.zeros(2,dtype=int)
    if lower2[0] < 0:
        offs2[0]=-lower2[0]
    else:
        l2[0]=lower2[0]
        
    if lower2[1] < 0:
        offs2[1]=-lower2[1]
    else:
        l2[1]=lower2[1]

    ucpd=upper2-l2+offs2
    if upper2[0]>metadata[index2]["dimensions"][0]:
        ucpd[0]+=(metadata[index2]["dimensions"][0]-upper2[0])

    if upper2[1]>metadata[index2]["dimensions"][1]:
        ucpd[1]+=(metadata[index2]["dimensions"][1]-upper2[1])

    im2_cropped[offs2[0]:ucpd[0],offs2[1]:ucpd[1],:]=im2[l2[0]:upper2[0],l2[1]:upper2[1],:]

    im1_cropped=im1[lower1[0]:upper1[0],lower1[1]:upper1[1],:]
    if im1_cropped.shape[0]!=im2_cropped.shape[0] or im1_cropped.shape[1]!=im2_cropped.shape[1]:
        print("--------------------------------------------------------------------")
        print("pad image1 before (phase_correlation)")
        return None,None,None
 
    if no_sift:
        pts1=np.zeros(2)
    else:
        try:
            pts1,pts2 = sift_align_matches(im1_cropped[:,:,0],im2_cropped[:,:,0],level_lowe) #clipped
        except:
            print("no sift matches found")
            pts1=np.zeros(2)
        #return None,None,None
    if len(pts1)<minimal_number_of_sift_features:
        print("warning: "+str(index1)+" , "+str(index2))
        print(len(pts1))
        print("phase correlation used                         phase_correlation")
 
        transl2,certainty=close_translation_by_phase_correlation(im1_cropped[:,:,0],im2_cropped[:,:,0])
        transl=phase_cross_correlation(im1_cropped[:,:,0],im2_cropped[:,:,0],
                                       reference_mask=im1_cropped[:,:,1]>1, 
                                       moving_mask=im2_cropped[:,:,1]>1,
                                       disambiguate=True)[0]
        
        print(transl,certainty)
        if certainty <minimal_correlation_sigma:
            print("no good phase correlation found       sorted out ")
            if not force:
                return None,None,None
            else: 
                transl=np.zeros(2)
        pts1=np.zeros([4,2])#,dtype=int)
        pts1[0]=0,0
        pts1[1]=2,0
        pts1[2]=2,2
        pts1[3]=0,2
        pts2=np.zeros([4,2])#,dtype=int)
        for i in range(4):
            pts2=pts1-transl[::-1]
        coffs=np.min(pts2,axis=0)
        for i in range(4):
            pts1[i]-=coffs
            pts2[i]-=coffs

    
    pts2[:,0]+=lower2[1]
    pts2[:,1]+=lower2[0]
        
    imtransformed,params = align_pair_special(im1_cropped,im2,pts1,pts2,
                                              relative_scale_limit=relative_scale_limit,
                                              translation_only=translation_only)
    if params is None:
        print("Scale Warning")
        return None,None,None
    
    if rotation_limit is not None:
        if np.abs(params["rotation"])>rotation_limit:
            print("Rotation Warning")
            return None,None,None
            
    
    offset=lower1-params["im1shift"]

    if offset[0]<0 or offset[1]<0:#correct_negative0>0 or correct_negative1>0:
        print("--------------------------------------------------------------------")
        print("pad image1 before")
        return None,None,None

    imog=im1[offset[0]:offset[0]+imtransformed.shape[0],offset[1]:offset[1]+imtransformed.shape[1],:]
    
    im1[offset[0]:offset[0]+imtransformed.shape[0],offset[1]:offset[1]+imtransformed.shape[1],:]=img_add_weighted_gray_alpha(imog,imtransformed)

    imtransformed2=img_add_weighted_gray_alpha(imog,imtransformed)
    imgbool=(imog[:,:,1].astype(np.uint16)*imtransformed2[:,:,1].astype(np.uint16))>0
    imgboolnum=imgbool*1
    im1check=img_to_uint8_fast(imgboolnum*imog[:,:,0])
    im2check=img_to_uint8_fast(imgboolnum*imtransformed2[:,:,0])
    #print("PSNR")
    print(skimage.metrics.peak_signal_noise_ratio(im1check,im2check))
    print(skimage.metrics.structural_similarity(im1check, im2check))


    num,newimg=cv2.connectedComponents(
        im1[offset[0]:offset[0]+imtransformed.shape[0],offset[1]:offset[1]+imtransformed.shape[1],1])#nres)
    
    if num>2:
        print("more than two")
        print(num)
        im1[offset[0]:offset[0]+imtransformed.shape[0],offset[1]:offset[1]+imtransformed.shape[1],:]=imog

        return None,None,None



    t0low=min(offset[0],lower1[0],np.min(im1_coord[:,0]))
    t1low=min(offset[1],lower1[1],np.min(im1_coord[:,1]))
    t0up=max(offset[0]+imtransformed.shape[0],upper1[0],np.max(im1_coord[:,0]))
    t1up=max(offset[1]+imtransformed.shape[1],upper1[1],np.max(im1_coord[:,1]))


    thresh,nres=cv2.threshold(imtransformed[:,:,1],1,255,cv2.THRESH_BINARY )

    contours2, hierarchy2 = cv2.findContours(nres, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)#, cv2.CHAIN_APPROX_SIMPLE)
    cont2=contours2[0][:,0,:][:,::-1]    
    cont2[:,0]+=offset[0]
    cont2[:,1]+=offset[1]
    
    #new_poly_points2=pixel_to_real(metadata[index1]["units_per_pixel"],
    #                                     metadata[index1]["scan_rotation"],
    #                                     metadata[index1]["anchor_point"],
    #                                     cont2)
    #polyg2=shapely.geometry.Polygon(new_poly_points2)
    #pc2=polyg2.centroid.coords[0]

    thresh,nres=cv2.threshold(im1[t0low:t0up,t1low:t1up,1],1,255,cv2.THRESH_BINARY )

    contours, hierarchy = cv2.findContours(nres, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)#, cv2.CHAIN_APPROX_SIMPLE)
    cont=contours[0][:,0,:][:,::-1]    
    cont[:,0]+=t0low
    cont[:,1]+=t1low
    
    new_poly_points=pixel_to_real(metadata[index1]["units_per_pixel"],
                                         metadata[index1]["scan_rotation"],
                                         metadata[index1]["anchor_point"],
                                         cont)

    if len(new_poly_points)<4:
        print("something wrong")
        return None,None,None
    if plotting:
        plt.imshow(im1[:,:,1]>1)
        plt.title("im1_after")
        plt.show()
    
    params["im1shift"]=offset
    return params,shapely.geometry.Polygon(new_poly_points)#,pc2


def close_translation_by_phase_correlation(im1,im2):
    
    dims=im1.shape
    mat=phase_correlation(im1,im2)
    transl,maxval=max_from_2d(mat)
    if transl[0]>dims[0]/2:
        transl[0]-=dims[0]
    if transl[1]>dims[1]/2:
        transl[1]-=dims[1]
    mean=np.mean(mat)
    std=np.std(mat)
    certainty=(maxval-mean)/std
    return transl,certainty

def stitch_all_no_copy(stitchobject,numbreak=np.inf,level_lowe=0.5,overlap_pad=0,rotation_limit=None,relative_scale_limit=None,
                       process_on_loading=None,plotting=False,minimal_number_of_sift_features=15,minimal_correlation_sigma=5,
                       no_sift=False,force=False,maximal_transl=None):

    #if "edgepath" in data:
    edgepath=stitchobject["edgepath"]
    metadata=stitchobject["metadata"]
    index1=edgepath[0,0]

    res_img=stitchobject["res_img"]
    G=stitchobject["graph"]
    counter=stitchobject["counter"]
    infolog=stitchobject["infolog"]
        
        
    while counter < len(edgepath):

        index2=edgepath[counter,1]
        im2=cv2.imread(metadata[edgepath[counter,1]]["filepath"],0)
        if process_on_loading is not None:
            res2=process_on_loading(im2)
        else:
            res2=im2
    
        correct_for_stitchpath=stitchpath_correction(edgepath[counter,0],metadata,infolog)
        print(edgepath[counter])
        
        #plt.imshow(res2[:,:,1])
        #plt.title("res2")
        #plt.show()
        res_params,res_polygon,pc2=stitch_by_overlap_translation(res_img,res2,-1,index2,metadata,level_lowe,
                                    overlap_pad=overlap_pad,
                                    correct_for_stitchpath=correct_for_stitchpath,
                                    infolog=infolog,plotting=plotting,
                                    no_sift=no_sift,
                                    maximal_transl=maximal_transl)

        #res_params,res_polygon=stitch_by_overlap_no_copy(res_img,res2,-1,index2,metadata,level_lowe,translation_only=True,overlap_pad=overlap_pad,
        #                            rotation_limit=rotation_limit,relative_scale_limit=relative_scale_limit,
        #                            correct_for_stitchpath=correct_for_stitchpath,
        #                            infolog=infolog,plotting=plotting,
        #                            minimal_number_of_sift_features=minimal_number_of_sift_features,
        #                            minimal_correlation_sigma=minimal_correlation_sigma)
        
        if res_polygon is None:
            print(str(index2) +" index removed")
            G.remove_edge(edgepath[counter][0],edgepath[counter][1])
            newedgepath=np.array([i for i in nx.bfs_edges(G,source=index1)])
            
            if np.sum(edgepath[:counter]-newedgepath[:counter])==0:
                edgepath=newedgepath
            else:
                print("should never appear ")

            if counter==0:
                pass
            else:
                pass
            """
                stitchobject2=dict()
                stitchobject2["edgepath"]=edgepath
                stitchobject2["res_img"]=res_img
                stitchobject2["graph"]=G
                stitchobject2["counter"]=counter
                stitchobject2["infolog"]=infolog
                stitchobject2["metadata"]=metadata.copy()
                
                return stitchobject2
            """
                
        else:
            #infolog["stitchindices"].append(index2)
            
            metadata[-1]["polygon"]=res_polygon
            infolog[index2]=res_params.copy()
            
            print("step "+str(counter)+" stitched "+str(index2))
            infolog[index2]["im1pos"]=pc2
            infolog["stitchpaths"][index2]=edgepath[counter,0]
            infolog[index2]["step"]=counter
            
            #for testomg
            #infolog[index2]["position"]=metadata[index2]["position"]
            
            counter+=1
            if counter==numbreak:
                break

    stitchobject2=dict()
    stitchobject2["edgepath"]=edgepath
    stitchobject2["res_img"]=res_img
    stitchobject2["graph"]=G
    stitchobject2["counter"]=counter
    stitchobject2["infolog"]=infolog
    stitchobject2["metadata"]=metadata.copy()
    
    return stitchobject2

def stitch_by_overlap_translation(im1,im2,index1,index2,metadata,level_lowe=0.5,
                                  overlap_pad=0,correct_for_stitchpath=None,infolog=None,
                                  plotting=False,maximal_transl=None,no_sift=False):

    translation_only=True
    if maximal_transl is None:
        maximal_transl=np.sqrt(np.sum(np.array(im2.shape)**2))

    if correct_for_stitchpath is None:
        poly2=metadata[index2]["polygon"]
    else:
        #poly3=metadata[index2]["polygon"]
        poly2=shapely.affinity.translate(metadata[index2]["polygon"],correct_for_stitchpath[0],correct_for_stitchpath[1])#1+

    overlap=shapely.intersection(metadata[index1]["polygon"],poly2)
    
    overlap_coord=np.array(overlap.oriented_envelope.exterior.xy).T[:-1]


    if plotting or len(overlap_coord)==0:
        #p3p=-np.array(poly3.exterior.xy)
        p2p=-np.array(poly2.exterior.xy)
        p1p=-np.array(metadata[index1]["polygon"].exterior.xy)
        if len(overlap_coord)==0:
            pass
        else:
            pop=-np.array(overlap.exterior.xy)
            plt.plot(pop[0],pop[1],'r')
        
        #plt.plot(p3p[0],p3p[1],'b')    
        plt.plot(p2p[0],p2p[1],'g')
        plt.plot(p1p[0],p1p[1],'-x',c='k')
        plt.axis("equal")
        plt.show()

    if len(overlap_coord)==0:
        print("empty overlap")
        #a=5/0
        return None,None,None

    #overlapcenter=np.array(overlap.centroid.coords[0])
    #distances=np.empty(len(p2coord))
    #for i in range(len(distances)):
    #    distances[i]=np.sqrt(np.sum((p2coord[i]-overlapcenter)*(p2coord[i]-overlapcenter)))
    #distance=np.max(distances)
    #im1_points=np.empty([4,2])
    #im1_points[0]=overlapcenter+np.array([distance,distance])
    #im1_points[1]=overlapcenter+np.array([distance,-distance])
    #im1_points[2]=overlapcenter+np.array([-distance,-distance])
    #im1_points[3]=overlapcenter+np.array([-distance,distance])
    #print(overlap_coord)
    #print(overlapcenter)
    #print(distance)
    
    

    im1_coord=real_to_pixel(metadata[index1]["units_per_pixel"],
                            metadata[index1]["scan_rotation"],
                            metadata[index1]["anchor_point"],
              np.array(metadata[index1]["polygon"].exterior.xy).T[:-1])
    
    im1_overlap_coord=real_to_pixel(metadata[index1]["units_per_pixel"],
                            metadata[index1]["scan_rotation"],
                            metadata[index1]["anchor_point"],
                            overlap_coord)

        
    lower1=np.min(im1_overlap_coord,axis=0)-overlap_pad
    upper1=np.max(im1_overlap_coord,axis=0)+overlap_pad

    
    #print(correct_for_stitchpath)
    for i in range(len(overlap_coord)):
        overlap_coord[i]-=correct_for_stitchpath
        
    im2_coord=real_to_pixel(metadata[index2]["units_per_pixel"],
                            metadata[index2]["scan_rotation"],
                            metadata[index2]["anchor_point"],
                            overlap_coord)


    lower2=np.min(im2_coord,axis=0)-overlap_pad
    upper2=np.max(im2_coord,axis=0) +overlap_pad
    im1_cropdim=upper1-lower1
    im2_cropdim=upper2-lower2
    
    dimdiff=im2_cropdim-im1_cropdim
    if np.sum(np.abs(dimdiff)>0):
        print("dimdiff>0 some rounding")
    if  dimdiff[0]<0:
        lower1[0]+=1
    elif dimdiff[0]>0:
        lower2[0]+=1
        
    if  dimdiff[1]<0:
        lower1[1]+=1
    elif dimdiff[1]>0:
        lower2[1]+=1

    cropdim=upper2-lower2
    
    im2_cropped=np.zeros([cropdim[0],cropdim[1],2],dtype=np.uint8)
    
    offs2=np.zeros(2,dtype=int)
    l2=np.zeros(2,dtype=int)
    if lower2[0] < 0:
        offs2[0]=-lower2[0]
    else:
        l2[0]=lower2[0]
        
    if lower2[1] < 0:
        offs2[1]=-lower2[1]
    else:
        l2[1]=lower2[1]

    ucpd=upper2-l2+offs2
    if upper2[0]>metadata[index2]["dimensions"][0]:
        ucpd[0]+=(metadata[index2]["dimensions"][0]-upper2[0])

    if upper2[1]>metadata[index2]["dimensions"][1]:
        ucpd[1]+=(metadata[index2]["dimensions"][1]-upper2[1])

    im2_cropped[offs2[0]:ucpd[0],offs2[1]:ucpd[1],:]=im2[l2[0]:upper2[0],l2[1]:upper2[1],:]

    im1_cropped=im1[lower1[0]:upper1[0],lower1[1]:upper1[1],:]
    if im1_cropped.shape[0]!=im2_cropped.shape[0] or im1_cropped.shape[1]!=im2_cropped.shape[1]:
        print("--------------------------------------------------------------------")
        print("pad image1 before (phase_correlation)")
        return None,None,None
    if no_sift:
        sift_pts1=np.zeros(2)
    else:
        try:
            sift_pts1,sift_pts2 = sift_align_matches(im1_cropped[:,:,0],im2_cropped[:,:,0],level_lowe) #clipped
        except:
            sift_pts1=np.zeros(2)

        
    if len(sift_pts1)<4:
        print("no sift matches found")
        sift_pts1,sift_pts2=make_point_translation(np.zeros(2))
    else:
        sift_pts2[:,0]+=lower2[1]
        sift_pts2[:,1]+=lower2[0]
        transl_sift=np.median(sift_pts1-sift_pts2,axis=0)[::-1]
        sift_pts1,sift_pts2=make_point_translation(transl_sift)

        
    #transl=phase_cross_correlation(im1_cropped[:,:,0],im2_cropped[:,:,0],
    #                               reference_mask=im1_cropped[:,:,1]>1, 
    #                               moving_mask=im2_cropped[:,:,1]>1,
    #                               disambiguate=True)[0]
    
    transl,certainty=close_translation_by_phase_correlation(im1_cropped[:,:,0],im2_cropped[:,:,0])
    
    #if trans
    if np.sqrt(np.sum(transl**2))>maximal_transl:
        print(np.sqrt(np.sum(transl**2)))
        print(transl)
        transl=np.zeros(2)
        print("translation bigger than maximal")
    
    pts1,pts2=make_point_translation(transl)
    pts2[:,0]+=lower2[1]
    pts2[:,1]+=lower2[0]
    

    imtransformed,params = align_pair_special(im1_cropped,im2,pts1,pts2,
                                              translation_only=translation_only)
    if no_sift:
        ssim2=-np.inf
    else:
        imtransformed2,params2 = align_pair_special(im1_cropped,im2,sift_pts1,sift_pts2,
                                              translation_only=translation_only)
        offset2=lower1-params2["im1shift"]
        if offset2[0]<0 or offset2[1]<0:#correct_negative0>0 or correct_negative1>0:
            print("offset2--------------------------------------------------------------------")
            print("pad image1 before")
            return None,None,None
        
        imog2=im1[offset2[0]:offset2[0]+imtransformed2.shape[0],offset2[1]:offset2[1]+imtransformed2.shape[1],:]
        new2=img_add_weighted_gray_alpha(imog2,imtransformed2)    

        imgbool2num=1*((imog2[:,:,1].astype(np.uint16)*new2[:,:,1].astype(np.uint16))>0)
        im2check=img_to_uint8_fast(imgbool2num*new2[:,:,0])
        im_orig_check2=img_to_uint8_fast(imgbool2num*imog2[:,:,0])

        ssim2=skimage.metrics.structural_similarity(im_orig_check2, im2check)


    offset=lower1-params["im1shift"]

    if offset[0]<0 or offset[1]<0:#correct_negative0>0 or correct_negative1>0:
        print("offset--------------------------------------------------------------------")
        print("pad image1 before")
        return None,None,None

    imog=im1[offset[0]:offset[0]+imtransformed.shape[0],offset[1]:offset[1]+imtransformed.shape[1],:]
    
    new1=img_add_weighted_gray_alpha(imog,imtransformed)
    

    imgbool1num=1*((imog[:,:,1].astype(np.uint16)*new1[:,:,1].astype(np.uint16))>0)
    im_orig_check1=img_to_uint8_fast(imgbool1num*imog[:,:,0])
    im1check=img_to_uint8_fast(imgbool1num*new1[:,:,0])


    ssim1=skimage.metrics.structural_similarity(im_orig_check1, im1check)
    
    #plt.imshow(im1check)
    #plt.show()
    #plt.imshow(im2check)
    #plt.show()
    
    #print(ssim1)
    #print(ssim2)

    
    if ssim2>ssim1:
        correlation_better=False
    else:
        correlation_better=True
        
    if correlation_better:
        print("phase correlation")
        print(params)
        print(transl)

        
        im1[offset[0]:offset[0]+imtransformed.shape[0],offset[1]:offset[1]+imtransformed.shape[1],:]=new1
        t0low=min(offset[0],lower1[0],np.min(im1_coord[:,0]))
        t1low=min(offset[1],lower1[1],np.min(im1_coord[:,1]))
        t0up=max(offset[0]+imtransformed.shape[0],upper1[0],np.max(im1_coord[:,0]))
        t1up=max(offset[1]+imtransformed.shape[1],upper1[1],np.max(im1_coord[:,1]))

        newpos=np.empty([1,2])
        newpos[0]=offset+np.array(imtransformed.shape)[:2]/2
        new_pos_point=pixel_to_real(metadata[index1]["units_per_pixel"],
                                             metadata[index1]["scan_rotation"],
                                             metadata[index1]["anchor_point"],
                                             newpos)[0]
        
        #thresh,nres=cv2.threshold(imtransformed[:,:,1],1,255,cv2.THRESH_BINARY )

    else:
        print("feature matching")
        print(params2)
        im1[offset2[0]:offset2[0]+imtransformed2.shape[0],offset2[1]:offset2[1]+imtransformed2.shape[1],:]=new2
        t0low=min(offset2[0],lower1[0],np.min(im1_coord[:,0]))
        t1low=min(offset2[1],lower1[1],np.min(im1_coord[:,1]))
        t0up=max(offset2[0]+imtransformed2.shape[0],upper1[0],np.max(im1_coord[:,0]))
        t1up=max(offset2[1]+imtransformed2.shape[1],upper1[1],np.max(im1_coord[:,1]))
        
        newpos=np.empty([1,2])
        newpos[0]=offset2+np.array(imtransformed2.shape)[:2]/2
        new_pos_point=pixel_to_real(metadata[index1]["units_per_pixel"],
                                             metadata[index1]["scan_rotation"],
                                             metadata[index1]["anchor_point"],
                                             newpos)[0]
        
        #thresh,nres=cv2.threshold(imtransformed2[:,:,1],1,255,cv2.THRESH_BINARY )


    #num,newimg=cv2.connectedComponents(
    #    im1[offset[0]:offset[0]+imtransformed.shape[0],offset[1]:offset[1]+imtransformed.shape[1],1])#nres)
    #
    #if num>2:
    #    print("more than two")
    #    print(num)
    #    im1[offset[0]:offset[0]+imtransformed.shape[0],offset[1]:offset[1]+imtransformed.shape[1],:]=imog
    #
    #    return None,None,None





    

    #contours2, hierarchy2 = cv2.findContours(nres, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)#, cv2.CHAIN_APPROX_SIMPLE)
    #cont2=contours2[0][:,0,:][:,::-1]    
    #cont2[:,0]+=offset[0]
    #cont2[:,1]+=offset[1]
    
    #new_poly_points2=pixel_to_real(metadata[index1]["units_per_pixel"],
    #                                     metadata[index1]["scan_rotation"],
    #                                     metadata[index1]["anchor_point"],
    #                                     cont2)
    #polyg2=shapely.geometry.Polygon(new_poly_points2)
    #pc2=polyg2.centroid.coords[0]
    #print(pc2)
    
    #print(transl)
    ##print(new_pos_points[0])
    #print("meta pos2")
    #print(metadata[index2]["position"])

    thresh,nres=cv2.threshold(im1[t0low:t0up,t1low:t1up,1],1,255,cv2.THRESH_BINARY )

    contours, hierarchy = cv2.findContours(nres, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)#, cv2.CHAIN_APPROX_SIMPLE)
    cont=contours[0][:,0,:][:,::-1]    
    cont[:,0]+=t0low
    cont[:,1]+=t1low
    
    new_poly_points=pixel_to_real(metadata[index1]["units_per_pixel"],
                                         metadata[index1]["scan_rotation"],
                                         metadata[index1]["anchor_point"],
                                         cont)

    
    if len(new_poly_points)<4:
        print("something wrong")
        return None,None,None
    if plotting:
        plt.imshow(im1[:,:,1]>1)
        plt.title("im1_after")
        plt.show()
    
    params["im1shift"]=offset
    return params,shapely.geometry.Polygon(new_poly_points),new_pos_point



def stitch_by_overlap_translation2(im1,im2,index1,index2,metadata,level_lowe=0.5,
                                  overlap_pad=0,correct_for_stitchpath=None,infolog=None,
                                  plotting=False,maximal_transl=None,no_sift=False):

    translation_only=True
    if maximal_transl is None:
        maximal_transl=np.sqrt(np.sum(np.array(im2.shape)**2))
    #if len(im2.shape)==2:
    #    im2=img_single_to_double_channel(im2)

    if correct_for_stitchpath is None:
        poly2=metadata[index2]["polygon"]
    else:
        #poly3=metadata[index2]["polygon"]
        poly2=shapely.affinity.translate(metadata[index2]["polygon"],correct_for_stitchpath[0],correct_for_stitchpath[1])#1+

    overlap=shapely.intersection(metadata[index1]["polygon"],poly2)
    
    overlap_coord=np.array(overlap.oriented_envelope.exterior.xy).T[:-1]


    if plotting or len(overlap_coord)==0:
        #p3p=-np.array(poly3.exterior.xy)
        p2p=-np.array(poly2.exterior.xy)
        p1p=-np.array(metadata[index1]["polygon"].exterior.xy)
        if len(overlap_coord)==0:
            pass
        else:
            pop=-np.array(overlap.exterior.xy)
            plt.plot(pop[0],pop[1],'r')
        
        #plt.plot(p3p[0],p3p[1],'b')    
        plt.plot(p2p[0],p2p[1],'g')
        plt.plot(p1p[0],p1p[1],'k')
        plt.axis("equal")
        plt.show()

    if len(overlap_coord)==0:
        print("empty overlap")
        #a=5/0
        return None,None,None

    #overlapcenter=np.array(overlap.centroid.coords[0])
    #distances=np.empty(len(p2coord))
    #for i in range(len(distances)):
    #    distances[i]=np.sqrt(np.sum((p2coord[i]-overlapcenter)*(p2coord[i]-overlapcenter)))
    #distance=np.max(distances)
    #im1_points=np.empty([4,2])
    #im1_points[0]=overlapcenter+np.array([distance,distance])
    #im1_points[1]=overlapcenter+np.array([distance,-distance])
    #im1_points[2]=overlapcenter+np.array([-distance,-distance])
    #im1_points[3]=overlapcenter+np.array([-distance,distance])
    #print(overlap_coord)
    #print(overlapcenter)
    #print(distance)
    
    

    im1_coord=real_to_pixel(metadata[index1]["units_per_pixel"],
                            metadata[index1]["scan_rotation"],
                            metadata[index1]["anchor_point"],
              np.array(metadata[index1]["polygon"].exterior.xy).T[:-1])
    
    im1_overlap_coord=real_to_pixel(metadata[index1]["units_per_pixel"],
                            metadata[index1]["scan_rotation"],
                            metadata[index1]["anchor_point"],
                            overlap_coord)

        
    lower1=np.min(im1_overlap_coord,axis=0)-overlap_pad
    upper1=np.max(im1_overlap_coord,axis=0)+overlap_pad

    
    #print(correct_for_stitchpath)
    for i in range(len(overlap_coord)):
        overlap_coord[i]-=correct_for_stitchpath
        
    im2_coord=real_to_pixel(metadata[index2]["units_per_pixel"],
                            metadata[index2]["scan_rotation"],
                            metadata[index2]["anchor_point"],
                            overlap_coord)


    lower2=np.min(im2_coord,axis=0)-overlap_pad
    upper2=np.max(im2_coord,axis=0) +overlap_pad
    im1_cropdim=upper1-lower1
    im2_cropdim=upper2-lower2
    
    dimdiff=im2_cropdim-im1_cropdim
    if np.sum(np.abs(dimdiff)>0):
        print("dimdiff>0 some rounding")
    if  dimdiff[0]<0:
        lower1[0]+=1
    elif dimdiff[0]>0:
        lower2[0]+=1
        
    if  dimdiff[1]<0:
        lower1[1]+=1
    elif dimdiff[1]>0:
        lower2[1]+=1

    cropdim=upper2-lower2
    
    im2_cropped=np.zeros([cropdim[0],cropdim[1],2],dtype=np.uint8)
    
    offs2=np.zeros(2,dtype=int)
    l2=np.zeros(2,dtype=int)
    if lower2[0] < 0:
        offs2[0]=-lower2[0]
    else:
        l2[0]=lower2[0]
        
    if lower2[1] < 0:
        offs2[1]=-lower2[1]
    else:
        l2[1]=lower2[1]

    ucpd=upper2-l2+offs2
    if upper2[0]>metadata[index2]["dimensions"][0]:
        ucpd[0]+=(metadata[index2]["dimensions"][0]-upper2[0])

    if upper2[1]>metadata[index2]["dimensions"][1]:
        ucpd[1]+=(metadata[index2]["dimensions"][1]-upper2[1])

    im2_cropped[offs2[0]:ucpd[0],offs2[1]:ucpd[1],:]=im2[l2[0]:upper2[0],l2[1]:upper2[1],:]

    im1_cropped=im1[lower1[0]:upper1[0],lower1[1]:upper1[1],:]
    if im1_cropped.shape[0]!=im2_cropped.shape[0] or im1_cropped.shape[1]!=im2_cropped.shape[1]:
        print("--------------------------------------------------------------------")
        print("pad image1 before (phase_correlation)")
        return None,None,None
    if no_sift:
        sift_pts1=np.zeros(2)
    else:
        try:
            sift_pts1,sift_pts2 = sift_align_matches(im1_cropped[:,:,0],im2_cropped[:,:,0],level_lowe) #clipped
        except:
            sift_pts1=np.zeros(2)

        
    if len(sift_pts1)<4:
        print("no sift matches found")
        sift_pts1,sift_pts2=make_point_translation(np.zeros(2))
    else:
        sift_pts2[:,0]+=lower2[1]
        sift_pts2[:,1]+=lower2[0]
        transl_sift=np.median(sift_pts1-sift_pts2,axis=0)[::-1]
        sift_pts1,sift_pts2=make_point_translation(transl_sift)

        
    #transl=phase_cross_correlation(im1_cropped[:,:,0],im2_cropped[:,:,0],
    #                               reference_mask=im1_cropped[:,:,1]>1, 
    #                               moving_mask=im2_cropped[:,:,1]>1,
    #                               disambiguate=True)[0]
    
    transl,certainty=close_translation_by_phase_correlation(im1_cropped[:,:,0],im2_cropped[:,:,0])
    
    #if trans
    if np.sqrt(np.sum(transl**2))>maximal_transl:
        print(np.sqrt(np.sum(transl**2)))
        print(transl)
        transl=np.zeros(2)
        print("translation bigger than maximal")
    
    pts1,pts2=make_point_translation(transl)
    pts2[:,0]+=lower2[1]
    pts2[:,1]+=lower2[0]
    

    imtransformed,params = align_pair_special(im1_cropped,im2,pts1,pts2,
                                              translation_only=translation_only)
    if no_sift:
        ssim2=-np.inf
    else:
        imtransformed2,params2 = align_pair_special(im1_cropped,im2,sift_pts1,sift_pts2,
                                              translation_only=translation_only)
        offset2=lower1-params2["im1shift"]
        if offset2[0]<0 or offset2[1]<0:#correct_negative0>0 or correct_negative1>0:
            print("offset2--------------------------------------------------------------------")
            print("pad image1 before")
            return None,None,None
        
        imog2=im1[offset2[0]:offset2[0]+imtransformed2.shape[0],offset2[1]:offset2[1]+imtransformed2.shape[1],:]
        new2=img_add_weighted_gray_alpha(imog2,imtransformed2)    

        imgbool2num=1*((imog2[:,:,1].astype(np.uint16)*new2[:,:,1].astype(np.uint16))>0)
        im2check=img_to_uint8_fast(imgbool2num*new2[:,:,0])
        im_orig_check2=img_to_uint8_fast(imgbool2num*imog2[:,:,0])

        ssim2=skimage.metrics.structural_similarity(im_orig_check2, im2check)


    offset=lower1-params["im1shift"]

    if offset[0]<0 or offset[1]<0:#correct_negative0>0 or correct_negative1>0:
        print("offset--------------------------------------------------------------------")
        print("pad image1 before")
        return None,None,None

    imog=im1[offset[0]:offset[0]+imtransformed.shape[0],offset[1]:offset[1]+imtransformed.shape[1],:]
    
    new1=img_add_weighted_gray_alpha(imog,imtransformed)
    

    imgbool1num=1*((imog[:,:,1].astype(np.uint16)*new1[:,:,1].astype(np.uint16))>0)
    im_orig_check1=img_to_uint8_fast(imgbool1num*imog[:,:,0])
    im1check=img_to_uint8_fast(imgbool1num*new1[:,:,0])


    ssim1=skimage.metrics.structural_similarity(im_orig_check1, im1check)
    
    #plt.imshow(im1check)
    #plt.show()
    #plt.imshow(im2check)
    #plt.show()
    
    #print(ssim1)
    #print(ssim2)

    
    if ssim2>ssim1:
        correlation_better=False
    else:
        correlation_better=True
        
    if correlation_better:
        print("phase correlation")
        print(params)
        print(transl)

        
        im1[offset[0]:offset[0]+imtransformed.shape[0],offset[1]:offset[1]+imtransformed.shape[1],:]=new1
        t0low=min(offset[0],lower1[0],np.min(im1_coord[:,0]))
        t1low=min(offset[1],lower1[1],np.min(im1_coord[:,1]))
        t0up=max(offset[0]+imtransformed.shape[0],upper1[0],np.max(im1_coord[:,0]))
        t1up=max(offset[1]+imtransformed.shape[1],upper1[1],np.max(im1_coord[:,1]))

        newpos=np.empty([1,2])
        newpos[0]=offset+np.array(imtransformed.shape)[:2]/2
        new_pos_point=pixel_to_real(metadata[index1]["units_per_pixel"],
                                             metadata[index1]["scan_rotation"],
                                             metadata[index1]["anchor_point"],
                                             newpos)[0]
        
        #thresh,nres=cv2.threshold(imtransformed[:,:,1],1,255,cv2.THRESH_BINARY )

    else:
        print("feature matching")
        print(params2)
        im1[offset2[0]:offset2[0]+imtransformed2.shape[0],offset2[1]:offset2[1]+imtransformed2.shape[1],:]=new2
        t0low=min(offset2[0],lower1[0],np.min(im1_coord[:,0]))
        t1low=min(offset2[1],lower1[1],np.min(im1_coord[:,1]))
        t0up=max(offset2[0]+imtransformed2.shape[0],upper1[0],np.max(im1_coord[:,0]))
        t1up=max(offset2[1]+imtransformed2.shape[1],upper1[1],np.max(im1_coord[:,1]))
        
        newpos=np.empty([1,2])
        newpos[0]=offset2+np.array(imtransformed2.shape)[:2]/2
        new_pos_point=pixel_to_real(metadata[index1]["units_per_pixel"],
                                             metadata[index1]["scan_rotation"],
                                             metadata[index1]["anchor_point"],
                                             newpos)[0]
        
        #thresh,nres=cv2.threshold(imtransformed2[:,:,1],1,255,cv2.THRESH_BINARY )


    #num,newimg=cv2.connectedComponents(
    #    im1[offset[0]:offset[0]+imtransformed.shape[0],offset[1]:offset[1]+imtransformed.shape[1],1])#nres)
    #
    #if num>2:
    #    print("more than two")
    #    print(num)
    #    im1[offset[0]:offset[0]+imtransformed.shape[0],offset[1]:offset[1]+imtransformed.shape[1],:]=imog
    #
    #    return None,None,None





    

    #contours2, hierarchy2 = cv2.findContours(nres, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)#, cv2.CHAIN_APPROX_SIMPLE)
    #cont2=contours2[0][:,0,:][:,::-1]    
    #cont2[:,0]+=offset[0]
    #cont2[:,1]+=offset[1]
    
    #new_poly_points2=pixel_to_real(metadata[index1]["units_per_pixel"],
    #                                     metadata[index1]["scan_rotation"],
    #                                     metadata[index1]["anchor_point"],
    #                                     cont2)
    #polyg2=shapely.geometry.Polygon(new_poly_points2)
    #pc2=polyg2.centroid.coords[0]
    #print(pc2)
    
    #print(transl)
    ##print(new_pos_points[0])
    #print("meta pos2")
    #print(metadata[index2]["position"])

    thresh,nres=cv2.threshold(im1[t0low:t0up,t1low:t1up,1],1,255,cv2.THRESH_BINARY )

    contours, hierarchy = cv2.findContours(nres, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)#, cv2.CHAIN_APPROX_SIMPLE)
    cont=contours[0][:,0,:][:,::-1]    
    cont[:,0]+=t0low
    cont[:,1]+=t1low
    
    new_poly_points=pixel_to_real(metadata[index1]["units_per_pixel"],
                                         metadata[index1]["scan_rotation"],
                                         metadata[index1]["anchor_point"],
                                         cont)

    if len(new_poly_points)<4:
        print("something wrong")
        return None,None,None
    if plotting:
        plt.imshow(im1[:,:,1]>1)
        plt.title("im1_after")
        plt.show()
    
    params["im1shift"]=offset
    return params,shapely.geometry.Polygon(new_poly_points),new_pos_point


def make_point_translation(transl):
    pts1=np.zeros([4,2])#,dtype=int)
    pts1[0]=0,0
    pts1[1]=2,0
    pts1[2]=2,2
    pts1[3]=0,2
    pts2=np.zeros([4,2])#,dtype=int)
    for i in range(4):
        pts2=pts1-transl[::-1]
    coffs=np.min(pts2,axis=0)
    for i in range(4):
        pts1[i]-=coffs
        pts2[i]-=coffs
    return pts1,pts2
