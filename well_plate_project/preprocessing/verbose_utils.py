#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 16:32:01 2020

@author: enzo
"""
import cv2
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import json

from well_plate_project.preprocessing.utils import single_channel_equalize, resize_image, derive_image
   
#%% Verbose utilities

def derivative_mask(input_image):   
    input_image = single_channel_equalize(input_image, c_map= 'gray', eq_method = 'clahe')
    derived = derive_image(input_image, derivative = 'sobel')
    ret,thresh1 = cv2.threshold(derived,200,255,cv2.THRESH_BINARY)
    res = resize_image(thresh1, 10)
    return res


#non usefoul mathamatical comparisons
def compare_images(orig_full, to_compare_full):
    #TODO try EDGE surf https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_table_of_contents_feature2d/py_table_of_contents_feature2d.html
    

    print("start_comparing")
    dict_compare = {}
    new_width = min(orig_full.shape[0],to_compare_full.shape[0] )
    new_height = min(orig_full.shape[1],to_compare_full.shape[1] )
    dim = (new_width, new_height)
    # resize image
    orig = cv2.resize(orig_full, dim, interpolation = cv2.INTER_AREA) 
    to_compare = cv2.resize(to_compare_full, dim, interpolation = cv2.INTER_AREA) 

    bw_orig = single_channel_equalize(orig, c_map= 'gray', eq_method = 'clahe')
    bw_to_compare = single_channel_equalize(to_compare, c_map= 'gray', eq_method = 'clahe')
    diff_bw= np.linalg.norm(bw_orig-bw_to_compare)/(new_width*new_height) #np.inf  'fro'
    dict_compare['diff_bw_2'] = diff_bw
    
    lap_orig = derivative_mask(orig)
    lap_to_compare = derivative_mask(to_compare)
    diff_lap= np.linalg.norm(lap_orig-lap_to_compare, ord = np.inf)/(new_width*new_height) #np.inf  'fro'
    dict_compare['duffLAP_inf'] = diff_lap
    print(f"Calculate CORR with {lap_orig.shape} ; {lap_to_compare.shape}")
    corr = signal.correlate2d (lap_orig, lap_to_compare)
    dict_compare['CorrMAX'] = np.round(np.max(corr)).astype('float')
    dict_compare['CorrNORM_inf'] =np.round(np.linalg.norm(corr, ord = np.inf))
    dict_compare['width'] =to_compare_full.shape[0]
    dict_compare['height'] =to_compare_full.shape[1]
    return dict_compare


#Draw Matches
def draw_inliers(mask, img1,kp1,img2,kp2,good):
    """
    draw our inliers (if successfully found the object) or matching keypoints (if failed).
    """
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = mask.ravel().tolist(), # draw only inliers# ravel function returns  - contiguous flattened array 
                   flags = 2)

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

    #plt.imshow(img3, 'gray'),plt.show()
    return img3



def matching_info(referenceImage, keypoints_reference, trainImage_orig, keypoints_train,  matrix, M, matrix_mask, matches, show = False, save = True, **kwargs):
    feature_dict = {}
    
    h,w= referenceImage.shape[0:2]
    pts_image = np.array([[0,0],[0,h-1],[w-1,h-1],[w-1,0]], dtype='float32')
    pts_image = np.array([pts_image])
    dst_image = cv2.perspectiveTransform(pts_image,matrix) # applying perspective algorithm : destination points
    corrected_img = cv2.warpPerspective(trainImage_orig, M, (referenceImage.shape[1], referenceImage.shape[0]), cv2.WARP_INVERSE_MAP)
    
    trainImage = trainImage_orig.copy()
    homography = cv2.polylines(trainImage,[np.int32(dst_image)], True, (0,0,255), 10, cv2.LINE_AA)

    compare_dict = compare_images(trainImage, corrected_img)
    feature_dict.update(compare_dict)

    
    compare_image = draw_inliers(matrix_mask, referenceImage, keypoints_reference, trainImage, keypoints_train, matches)
    

    
    if save and kwargs:
        image_name = kwargs.get('image_name', None)
        OUTPUT_FOLDER = kwargs.get('out_folder', None)
        OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True) 
        print("Holo")
        target_filename =  image_name + '_holo' + '.jpg'
        target_path = OUTPUT_FOLDER / target_filename   
        cv2.imwrite(str(target_path), homography)
        target_filename =  image_name + '_cropped' + '.jpg'
        target_path = OUTPUT_FOLDER / target_filename
        cv2.imwrite(str(target_path), corrected_img)
        target_filename =  image_name + '_cro2cmp' + '.jpg'
        target_path = OUTPUT_FOLDER / target_filename   
        cv2.imwrite(str(target_path), compare_image)
        print("copmuted Compare")
        file_name=image_name +"_dict.json"
        target_path = OUTPUT_FOLDER / file_name  
        print(feature_dict)
        with open(str(target_path),"w+") as file:
            json.dump(feature_dict, file)
    
    if show:
        plt.imshow(trainImage) , plt.show()
        plt.imshow(corrected_img), plt.show()
        plt.imshow(compare_image), plt.show()
        if kwargs:
            image_name = kwargs.get('image_name', None)
            OUTPUT_FOLDER = kwargs.get('out_folder', None)
            target_path = OUTPUT_FOLDER / (image_name + '_trainImage.jpg') 
            cv2.imwrite(str(target_path), trainImage)
            target_path = OUTPUT_FOLDER / (image_name + '_corrected_img.jpg') 
            cv2.imwrite(str(target_path), corrected_img)
            target_path = OUTPUT_FOLDER / (image_name + '_compare_image.jpg') 
            cv2.imwrite(str(target_path), compare_image)

        
    return feature_dict