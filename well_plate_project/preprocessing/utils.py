#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 16:35:40 2020

@author: enzo
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
#%% Utils functions for feature extraction

def horotate(img):
    if img.shape[0]>img.shape[1]:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE )
    return img


def resize_image(img, scale_percent = 70): # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
    return resized


def apply_equalization(img,  eq_method, deblur):
    if deblur:
        img = deblur_img(img)
    if eq_method == 'clahe':
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16,9)) #(4,3)
        equlized_level = clahe.apply(img)
    elif eq_method == 'hist':
        equlized_level = cv2.equalizeHist(img)
    elif eq_method is None:
        equlized_level = img
    return equlized_level

def single_channel_equalize(BRG_input_image, c_map= 'gray', eq_method = 'clahe', deblur= False, channel = [1,1,1]):
    if c_map == 'gray':
        out_image = cv2.cvtColor(BRG_input_image, cv2.COLOR_BGR2GRAY) 
        out_image = apply_equalization(out_image, eq_method, deblur)
    elif c_map == 'HSV':       
        img_hsv=cv2.cvtColor(BRG_input_image,cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(img_hsv)
        H = apply_equalization(H, eq_method, deblur)
        S = apply_equalization(S, eq_method, deblur)
        V = apply_equalization(V, eq_method, deblur)
        out_image = channel[0] * H + channel[1]*S + channel[2]*V
    elif c_map == 'LAB':
        img_lab=cv2.cvtColor(BRG_input_image,cv2.COLOR_BGR2LAB)
        L, a, b = cv2.split(img_lab)
        L = apply_equalization(L, eq_method, deblur)
        a = apply_equalization(a, eq_method, deblur)
        b = apply_equalization(b, eq_method, deblur)
        out_image = channel[0]*L + channel[1]*a + channel[2]*b
    return out_image.astype('uint8')


def extract_img_tags(image_path):
    from PIL import Image, ExifTags
    import pandas as pd
    img = Image.open(str(image_path))
    
    if img._getexif() is not None:  #TODO make more stable controls
        exif = { ExifTags.TAGS[k]: v for k, v in img._getexif().items() if k in ExifTags.TAGS }
        tags = dict(pd.Series(exif)[['ShutterSpeedValue','ApertureValue','BrightnessValue','ExposureBiasValue', 'MaxApertureValue','MeteringMode',
                             'Flash', 'FocalLength', 'ExposureTime', 'Orientation','ISOSpeedRatings', 'FocalLengthIn35mmFilm','FNumber'  ]])
        dict_tags = {'img_tags': tags}
    else:
        dict_tags = {'img_tags': {}}

    return dict_tags



def deblur_img(image):
    from skimage import img_as_float, restoration, img_as_ubyte
    camera = img_as_float(image)
    psf = np.ones((3, 3)) / 9
    deconvolved = restoration.richardson_lucy(camera, psf, 5)
    return img_as_ubyte(deconvolved)


def clustering_cv2_k(image, k=2):
    reshaped_image= np.float32(image.reshape(-1, image.shape[2]))
    stopCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    ret, labels, clusters = cv2.kmeans(reshaped_image, k, None, stopCriteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    clusters = np.uint8(clusters)
    intermediateImage = clusters[labels.flatten()]
    clusteredImage = intermediateImage.reshape((image.shape))

    return clusteredImage

def crop_from_clustered(original_image, clustered_image, area_limit=0.1, tol = 0.075, plot=True):

    grayImage = cv2.cvtColor(clustered_image, cv2.COLOR_BGR2GRAY) 
    _,threshold = cv2.threshold(grayImage, 100, 255,  cv2.THRESH_BINARY) 
    
    contours = cv2.findContours(threshold.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    
    for c in contours:
        # get the bounding rect
        area = cv2.contourArea(c); 
        if  area > area_limit*(original_image.shape[0]*original_image.shape[1]):
            # print(area/(img.shape[0]*img.shape[1]))
            x, y, w, h = cv2.boundingRect(c)
        
            # get the min area rect
            rect = cv2.minAreaRect(c); box = cv2.boxPoints(rect)
            # convert all coordinates floating point values to int
            box = np.int0(box)
            # draw a red 'nghien' rectangle
            # cv2.drawContours(original_image, [box], 0, (0, 255, 0), 5)
            # print(len(contours))
            # cv2.drawContours(original_image, c, -1, (255, 0, 0), 10)
            
    # get width and height of the detected rectangle
    width = int(rect[1][1]); height = int(rect[1][0])
    
    src_pts = box.astype("float32")
    
    # allargo del tol% il rettangolo di crop per eventuali sminchiature
    perc_h = height*tol; perc_w = width*tol
    dst_pts = np.array([[width-perc_w, height-perc_h],
                        [perc_w, height-perc_h],
                        [perc_w, perc_h],
                        [width-perc_w, perc_h]], dtype="float32")
    
    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(original_image, M, (width, height))
    if width<height:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    
    if plot:  
        plt.figure(figsize=(10,10))
        plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), cmap = 'gray', interpolation = 'bicubic') #interpolation = None !!!
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()
    return warped
        
#%% Utils functions for derivatives
def derive_image(single_channel, derivative = 'sobel'):
    if derivative == 'laplacian':
        denoised = cv2.GaussianBlur(single_channel, (7,7), 5);   
        out_image = cv2.Laplacian(denoised,cv2.CV_64F)
    elif derivative == 'sobel':
        denoised = cv2.GaussianBlur(single_channel, (7,7), 5);
        sobelx = cv2.Sobel(single_channel,cv2.CV_64F,1,0,ksize=3)  # x
        sobely = cv2.Sobel(single_channel,cv2.CV_64F,0,1,ksize=3)  # y
        out_image = sobelx+ sobely
    return out_image.astype('uint8')


#%% Utils inutils
def white_balance(img): 
    import cv2
    #Automatic White Balancing with Grayworld assumption
    #https://pippin.gimp.org/image-processing/chapter-automaticadjustments.html
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


