#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 12:52:01 2020

@author: modal
"""
#%% import general pakage 
import numpy as np
import cv2
import matplotlib.pyplot as plt



def import_rotate_gauss(img):
    #%% IMPORT THE IMAGE
    if img.shape[0]>img.shape[1]:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


def gray_contours_perspective(img):

# Convert to Grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_gray = cv2.equalizeHist(img_gray)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_gray = clahe.apply(img_gray)

    # Converting image to a binary image  
    _,threshold = cv2.threshold(img_gray, 100, 255,  cv2.THRESH_BINARY) #140
    plt.imshow(threshold)
    plt.show()
    
    contours = cv2.findContours(threshold.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    for c in contours:
        # get the bounding rect
        area = cv2.contourArea(c); 
        if  area > 0.02*(img.shape[0]*img.shape[1]):
            # print(area/(img.shape[0]*img.shape[1]))
            x, y, w, h = cv2.boundingRect(c)
        
            # get the min area rect
            rect = cv2.minAreaRect(c); box = cv2.boxPoints(rect)
            # convert all coordinates floating point values to int
            box = np.int0(box)
            # draw a red 'nghien' rectangle
            #cv2.drawContours(img, [box], 0, (0, 255, 0), 5)
            #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            #plt.show()

            # print(len(contours))
            # cv2.drawContours(img, c, -1, (255, 255, 0), 5)
            
    # get width and height of the detected rectangle
    width = int(rect[1][1]); height = int(rect[1][0])

    src_pts = box.astype("float32")

    dst_pts = np.array([[width, height],
                        [0, height],
                        [0, 0],
                        [width, 0]], dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(img, M, (width, height))
    if width<height:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped


def crop_rotate(img):
    img = import_rotate_gauss(img)
    #%% CROP MULTIWELL
    warped = gray_contours_perspective(img)
    return warped

    


#%% INIT

def clear_all():
    """Clears all the variables from the workspace of the application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]


#Errori con:  a2_c  c1_b

def load_test_file(): 
    image_file_name = 'c1_b.jpg' #h2_a q1_b
    from well_plate_project.config import data_dir
    raw_data_dir = data_dir / 'raw'
    path = raw_data_dir / 'EXPERIMENTS'
    image_file = path /  image_file_name
    assert image_file.is_file()

    img = cv2.imread(str(image_file))
    plt.imshow(img)
    plt.show()
    return img


#%% MAIN
if __name__ == "__main__":
    clear_all()
    image = load_test_file()
    
    print("Testing all... ")
    #image = watershed_segmentation(image)
    image = crop_rotate(image)
    print("Plotting... ")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

    print("Modules ... ")
    img = import_rotate_gauss(image)
    plt.figure(figsize=(25,25))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

    img = import_rotate_gauss(img)
    warped = gray_contours_perspective(img)

    plt.figure(figsize=(25,25))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

    plt.figure(figsize=(25,25))
    plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), cmap = 'gray', interpolation = 'bicubic') #interpolation = None !!!
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

    print("Done testing ... ")














