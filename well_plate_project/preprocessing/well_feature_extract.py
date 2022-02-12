#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 10:20:09 2020

@author: enzo
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

#%%

def uniform_center_region(full_well):   
    import cv2
    import numpy as np
    # get dimensions
    hh, ww, cc = full_well.shape
    # compute center of image (as integer)
    wc = ww//2
    hc = hh//2
    # create grayscale copy of input as basis of mask
    gray = cv2.cvtColor(full_well,cv2.COLOR_BGR2GRAY)
    # create zeros mask 2 pixels larger in each dimension
    zeros = np.zeros([hh + 2, ww + 2], np.uint8)
    
    lowDiff=10
    upDiff = 15
    area_thresh = 0
    #TODO fix area_thresh as 10% of the total area of the input image
    while(area_thresh == 0):
        # do floodfill at center of image as seed point
        ffimg = cv2.floodFill(gray, zeros, (wc,hc), (255), loDiff =lowDiff, upDiff = upDiff , flags=8)[1] #
        #plt.imshow(full_well), plt.show()
        
        # set rest of ffimg to black
        ffimg[ffimg!=255] = 0
        #plt.imshow(ffimg), plt.show()
        
        
        # get contours, find largest and its bounding box 
        contours = cv2.findContours(ffimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        outer_contour = 0
        for cntr in contours:
            area = cv2.contourArea(cntr)
            print(f'area:{area}')
            if area > area_thresh:
                area_thresh = area
                outer_contour = cntr
                x,y,w,h = cv2.boundingRect(outer_contour)
        #end_for
        if area_thresh == 0:
            lowDiff= max(lowDiff - 1, 0)
            upDiff = min(upDiff + 5, 255)
            print(f'lowDiff:{lowDiff},upDiff:{upDiff}')
    
    
    # draw the filled contour on a black image
    mask = np.full([hh,ww,cc], (0,0,0), np.uint8)
    cv2.drawContours(mask,[outer_contour],0,(255,255,255),thickness=cv2.FILLED)
    
    # mask the input
    masked_img = full_well.copy()
    masked_img[mask == 0] = 0
    #masked_img[mask != 0] = img[mask != 0]
    
    # crop the bounding box region of the masked img
    result = masked_img[y:y+h, x:x+w]
    
    # draw the contour outline on a copy of result
    result_outline = result.copy()
    cv2.drawContours(result_outline,[outer_contour],0,(0,0,255),thickness=1,offset=(-x,-y))
    
    
    # display it
    # plt.imshow(full_well), plt.show()
    # plt.imshow(ffimg), plt.show()
    # plt.imshow(mask), plt.show()
    # plt.imshow(masked_img), plt.show()
    # plt.imshow(result), plt.show()
    # plt.imshow(result_outline), plt.show()
    return result, mask


#%%
def extract_features(well, non_zero = False):    
    #Problem: color fidelity of a photo
    #white balance
    #setting values for base colors 
    img_lab=cv2.cvtColor(well,cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(img_lab)
    blue,green,red = cv2.split(well)
    img_hsv=cv2.cvtColor(well,cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(img_hsv)
    mean, stddev = cv2.meanStdDev(img_lab)
    
    grayImg = cv2.cvtColor(well,cv2.COLOR_BGR2GRAY)

    img_propr = {}
    img_propr["gray"] = image_stats(grayImg, non_zero)
    img_propr["blue"] = image_stats(blue, non_zero)
    img_propr["green"] = image_stats(green, non_zero)
    img_propr["red"] = image_stats(red, non_zero)
    img_propr["L"] = image_stats(l, non_zero)
    img_propr["a"] = image_stats(a, non_zero)
    img_propr["b"] = image_stats(b, non_zero)
    img_propr["H"] = image_stats(H, non_zero)
    img_propr["S"] = image_stats(S, non_zero)
    img_propr["V"] = image_stats(V, non_zero)
    
    return img_propr



#%% Image statistics
def entropy2(labels, base=None):
    from math import log,e
    """ Computes entropy of label distribution. """
    n_labels = len(labels)
    if n_labels <= 1:
        return 0
    value,counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)
    if n_classes <= 1:
        return 0
    ent = 0.
    # Compute entropy
    base = e if base is None else base
    for i in probs:
        ent -= i * log(i, base)
    return ent

def image_stats(single_chan_img, non_zero=False):
    from skimage.measure import shannon_entropy
    from skimage.feature import greycomatrix
    from scipy.stats import skew
    assert len(single_chan_img.shape) == 2
    vect_chan = np.reshape(single_chan_img, -1)
    if non_zero : vect_chan = vect_chan[np.nonzero(vect_chan)]
    stats_dict={}
    stats_dict["mean"] = np.mean(vect_chan)
    stats_dict["stddev"] = np.std(vect_chan)
    #mean, stddev = cv2.meanStdDev(single_chan_img)

    from scipy import stats
    stats_dict["mean_trm30"] = stats.trim_mean(vect_chan, 0.3, axis=0)


    stats_dict["skewness"]= skew(vect_chan)

    #stats_dict["energy"]= 
    # "energy" should be inversely proportional to Shannon entropy
    stats_dict["entropy"] = shannon_entropy(single_chan_img) 
    stats_dict["entropy2"] = entropy2(vect_chan) 
    glcm = np.squeeze(greycomatrix(single_chan_img, distances=[1], angles=[0], symmetric=True, normed=True))
    entropy_glcm = -np.sum(glcm*np.log2(glcm + (glcm==0)))
    stats_dict["entropy_glcm"] = entropy_glcm
    
    from PIL import Image, ImageStat
    im_pil = Image.fromarray(single_chan_img)
    stat = ImageStat.Stat(im_pil)
    stats_dict["mean_PIL"] = stat.mean[0]
    return stats_dict




#%% Well stuff
def circle_well(referenceImage, trainImage, matrix, referenceWells, reduce=True, save = True, **kwargs):
    #output is a pandas well plate with feature dicts
    import pandas as pd
    import json
    
    height, width = referenceImage.shape[0:2]
    well_plate = pd.DataFrame(index = map(chr, range(65, 73)), columns=list(range(1,13)))
    
    if save and kwargs:
        image_name = kwargs.get('image_name', None)
        OUTPUT_FOLDER = kwargs.get('out_folder', None)
        out_image = OUTPUT_FOLDER / image_name
        out_image.mkdir(parents=True, exist_ok=True) 
    
    for key, values in referenceWells.iterrows(): 
        for i, well in enumerate(values):
            
            #print(well) 
            # well = well.astype()
            mask = np.zeros((height,width), np.uint8)
            x=int(well[0]); y= int(well[1]); R=int(well[2])
            single_maks = cv2.circle(mask,(x,y),R,(255,255,255),-1)
            corrected_mask = cv2.warpPerspective(single_maks, matrix, (trainImage.shape[1], trainImage.shape[0]), cv2.WARP_INVERSE_MAP)
            #plt.imshow(corrected_mask), plt.show()
            result = np.where(corrected_mask == np.amax(corrected_mask))
            minx=min(result[0]); maxx=max(result[0])
            miny=min(result[1]); maxy=max(result[1])
            res = cv2.bitwise_and(trainImage,trainImage,mask = corrected_mask)
            cropped = res[minx:maxx, miny:maxy] ####
            feature_dict = {} 
            feature_dict ['full'] = extract_features(cropped)
            if reduce: 
                reduced, mask_redu = uniform_center_region(cropped)
                feature_dict ['reduced'] = extract_features(reduced)        
            if save and kwargs:
                #print(key+str(i+1))
                #plt.imshow(res), plt.show()
                #plt.imshow(cropped), plt.show()
                target_filename =  image_name + '_' + key+str(i+1) + '_crop' + '.png'
                target_path = out_image / target_filename   
                cv2.imwrite(str(target_path), cropped)#####
                if reduce: 
                    target_filename =  image_name + '_' + key+str(i+1) + '_reduc' + '.png'
                    target_path = out_image / target_filename   
                    cv2.imwrite(str(target_path), reduced)
                    target_filename =  image_name + '_' + key+str(i+1) + '_maskreduc' + '.png'
                    target_path = out_image / target_filename   
                    cv2.imwrite(str(target_path), mask_redu)###### 
                file_name=image_name + key+str(i+1) +"_dict.json"
                target_path = out_image / file_name  
                with open(str(target_path),"w+") as file:
                    json.dump(feature_dict, file)
                    #file.write(str(feature_dict))
            well_plate.at[key, i+1] = feature_dict

    return well_plate



def circle_well_mock(reduce=True):
    #output is a pandas well plate with feature dicts
    import pandas as pd
    import numpy as np
    well_plate = pd.DataFrame(index = map(chr, range(65, 73)), columns=list(range(1,13)))
    for key, values in well_plate.iterrows(): 
        #print(key)
        #print(values)
        #print(enumerate(values))
        for i, well in enumerate(values):
            cropped = np.zeros((5, 5, 3), np.uint8)
            feature_dict = {} 
            features = extract_features(cropped)
            feature_dict ['full'] = features
            if reduce: 
                feature_dict ['reduced'] = features               
            well_plate.at[key, i+1] = feature_dict
    return well_plate


#%% Extract good circles
def circle_extract(image, dp = 1.5,  param1=None, param2=None, plot=True):
    
    if plot:
        plt.figure(figsize=(10,10))
        plt.xticks([]), plt.yticks([])
        plt.imshow(image) , plt.show()
    
    height, width = image.shape[:2]
    #maxRadius = int(0.80*(width/14)/2) #12+2
    #minRadius = int(0.75*(width/14)/2) #12+2
    maxRadius = int(0.93*(width/14)/2) #12+2
    minRadius = int(0.79*(width/14)/2) #12+2
    circles = cv2.HoughCircles(image.astype('uint8')
                                , cv2.HOUGH_GRADIENT
                                , dp= dp  #2.21
                                , minDist= 2.1*minRadius #90
                                , param1= param1 #30 50
                                , param2= param2 #90 30
                                , minRadius = minRadius #50
                                , maxRadius = maxRadius #70
                                )
    circles = np.uint16(np.around(circles))
    

    
    if plot:
        image_show = image.copy()
        for i in circles[0,:]:
            cv2.circle(image_show,(i[0],i[1]),i[2],(0,255,255),4)
        plt.figure(figsize=(10,10))
        plt.xticks([]), plt.yticks([])
        plt.imshow(image_show) , plt.show()
        print(circles.shape[1])
    
    assert circles.shape[1] == 96

    
    ind = np.argsort(circles[0,:, 1])   
    ordered_circles=circles[0,ind]
    matrix=ordered_circles.T.reshape((3, 8, 12))
    #TODO improve code
    sidx = matrix[0,:,:].argsort(axis=1)
    out=np.zeros(matrix.shape)
    out[0,:,:] = np.take_along_axis(matrix[0,:,:], sidx, axis=1)
    out[1,:,:] = np.take_along_axis(matrix[1,:,:], sidx, axis=1)
    out[2,:,:] = np.take_along_axis(matrix[2,:,:], sidx, axis=1)
    
    
    out=np.zeros(matrix.shape)
    matrix=ordered_circles.reshape((8, 12, 3))
    sidx = matrix[:,:,0].argsort(axis=1)
    out=np.zeros(matrix.shape)
    out[:,:,0] = np.take_along_axis(matrix[:,:,0], sidx, axis=1)
    out[:,:,1]= np.take_along_axis(matrix[:,:,1], sidx, axis=1)
    out[:,:,2] = np.take_along_axis(matrix[:,:,2], sidx, axis=1)
    
    from scipy import stats
    xm = stats.trim_mean(out[:,:,0], 0.3, axis=0).astype('int')
    ym = stats.trim_mean(out[:,:,1], 0.3, axis=1).astype('int')
    rm = stats.trim_mean(out[:,:,2], 0.3, axis=None).astype('int')
    rm = 0.9*rm
    
    out_round = np.zeros(out.shape)    
    out_round[:,:,0] = np.meshgrid(xm,ym)[0]
    out_round[:,:,1] = np.meshgrid(xm,ym)[1]
    out_round[:,:,2] = rm
    
    
    
    import numpy.lib.recfunctions as nlr
    circles_tuple = nlr.unstructured_to_structured(out_round).astype('O')
    
    import pandas as pd
    reference_wells = pd.DataFrame(circles_tuple, index = map(chr, range(65, 73)), columns=list(range(1,13)))

    
    height, width = image.shape
    mask_query_circ = np.zeros((height,width), np.uint8)
    
    
    circles_round = np.zeros(circles.shape) 
    index = 0
    for i in range(0,8):
        for j in range(0,12):
            circle = np.array((out_round[i,j,0], out_round[i,j,1], out_round[i,j,2]))
            circles_round[0,index,:] = circle
            index +=1
        
    
    circles = np.uint16(np.around(circles_round))
    counter = 0
    for i in circles[0,:]:
        # Draw on mask
        cv2.circle(mask_query_circ,(i[0],i[1]),i[2],(255,255,255),-1)
        #masked_data = ...
        cv2.bitwise_and(image, image, mask=mask_query_circ)    
        counter +=1
        print(counter)


    if plot:
        image_show = image.copy()
        for i in circles[0,:]:
            cv2.circle(image_show,(i[0],i[1]),i[2],(255,255,255),4)
        plt.figure(figsize=(10,10))
        plt.xticks([]), plt.yticks([])
        plt.imshow(image_show) , plt.show()
        print(circles.shape[1])
        
        
    return mask_query_circ, reference_wells






if __name__ == "__main__":
    wp =circle_well_mock()
    
    
    
    
    
    




