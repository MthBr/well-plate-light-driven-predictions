#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 12:52:01 2020

@author: modal
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import pandas as pd

#%% IMPORT THE IMAGE
path = 'IMG/EXPERIMENTS/'
path_test = 'IMG/TEST/'
img = cv2.imread(path+'h2_a.jpg')

if img.shape[0]>img.shape[1]:
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

img = cv2.GaussianBlur(img, (3, 3), 0)

plt.figure(figsize=(25,25))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

#%% CROP MULTIWELL

# Convert to Grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_gray = cv2.equalizeHist(img_gray)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_gray = clahe.apply(img_gray)

# Converting image to a binary image  
_,threshold = cv2.threshold(img_gray, 140, 255,  cv2.THRESH_BINARY) 


# plt.figure(figsize=(25,25))
# plt.imshow(threshold, cmap = 'gray', interpolation = 'bicubic')
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()

# Detecting shapes in image by selecting region  
# with same colors or intensity. 
# contours = cv2.findContours(threshold, cv2.RETR_TREE, 
#                             cv2.CHAIN_APPROX_SIMPLE)
 
contours = cv2.findContours(threshold.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

for c in contours:
    # get the bounding rect
    area = cv2.contourArea(c); 
    if  area > 0.025*(img.shape[0]*img.shape[1]):
        # print(area/(img.shape[0]*img.shape[1]))
        x, y, w, h = cv2.boundingRect(c)
    
        # get the min area rect
        rect = cv2.minAreaRect(c); box = cv2.boxPoints(rect)
        # convert all coordinates floating point values to int
        box = np.int0(box)
        # draw a red 'nghien' rectangle
        # cv2.drawContours(img, [box], 0, (0, 255, 0), 5)
        # print(len(contours))
        # cv2.drawContours(img, c, -1, (255, 255, 0), 5)
        
# get width and height of the detected rectangle
width = int(rect[1][1]); height = int(rect[1][0])

src_pts = box.astype("float32")

# coordinate of the points in box points after the rectangle has been straightened
# dst_pts = np.array([[0, height-1],
#                     [0, 0],
#                     [width-1, 0],
#                     [width-1, height-1]], dtype="float32")
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

# if warped.shape[1]>warped.shape[0]:
#     warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
# img=img[y:y+h,x:x+w]

plt.figure(figsize=(25,25))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

plt.figure(figsize=(25,25))
plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

#%% BRIGHTNESS EQUALIZATION

# # create a CLAHE object (Arguments are optional).
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# b = clahe.apply(warped[:, :, 0])
# g = clahe.apply(warped[:, :, 1])
# r = clahe.apply(warped[:, :, 2])
# equalized = np.dstack((b, g, r))

# # cl1 = clahe.apply(img)

# plt.figure(figsize=(20,20))
# plt.imshow(cv2.cvtColor(equalized, cv2.COLOR_BGR2RGB))
# plt.xticks([]), plt.yticks([])
# plt.show()

#%% EDGE RECOGNITION

# equalized = cv2.cvtColor(equalized, cv2.COLOR_BGR2GRAY); 

# if len(equalized.shape)==2:
#     equalized = np.expand_dims(equalized, 2)

# for idx in range(equalized.shape[2]):
#     img_layer = equalized[:,:,idx].copy()
    
#     # Convolution Filters
#     laplacian = cv2.Laplacian(img_layer,cv2.CV_64F)
#     sobelx = cv2.Sobel(img_layer,cv2.CV_64F,1,0,ksize=3)  # x
#     sobely = cv2.Sobel(img_layer,cv2.CV_64F,0,1,ksize=3)  # y
    
    
#     filtered = laplacian
#     filtered = cv2.convertScaleAbs(filtered)
#     filtered = cv2.equalizeHist(filtered)
    
#     # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     # filtered = clahe.apply(filtered)
    
#     filtered = pd.DataFrame(filtered); filtered[filtered<200]=0; 
#     filtered = np.array(filtered)
#     # edges = cv2.Canny(test,60,100)
    
    
    
#     kernel = np.ones((12,12),np.uint8)
#     denoised = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)
#     kernel = np.ones((3, 3), np.uint8)
    
#     try:
#         if isinstance(eroded,np.ndarray):
#             eroded[:,:,idx] = cv2.erode(denoised, kernel, iterations = 4)
        
#     except:
#         eroded = np.empty((equalized.shape[0],equalized.shape[1],equalized.shape[2])); eroded[:] = np.nan 
#         eroded[:,:,idx] = cv2.erode(denoised, kernel, iterations = 4)
        
    

# pd_eroded = pd.DataFrame(eroded.sum(axis=2)); 
# img_layer = pd.DataFrame(img_layer); img_layer[pd_eroded!=0]=0;

# ax,fig = plt.subplots(figsize=(20,15))
# plt.subplot(131),plt.imshow(filtered)#,cmap = 'gray')
# plt.title('H Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(132),plt.imshow(pd_eroded.values)
# plt.title('K Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(133),plt.imshow(img_layer.values)
# plt.title('J Image'), plt.xticks([]), plt.yticks([])
# plt.show()

# del eroded


# plt.figure(figsize=(20,20))
# plt.imshow(laplacian)
# plt.xticks([]), plt.yticks([])
# plt.show()

#%% LAB IMAGE

# #-----Converting image to LAB Color model----------------------------------- 
# lab = cv2.cvtColor(equalized, cv2.COLOR_BGR2LAB)
# plt.figure(figsize=(20,20))
# plt.imshow(lab)
# plt.xticks([]), plt.yticks([])
# plt.show()

# #-----Splitting the LAB image to different channels-------------------------
# l, a, b = cv2.split(lab)

# #-----Applying CLAHE to L-channel-------------------------------------------
# clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
# cl = clahe.apply(l)
# ca = clahe.apply(a)
# cb = clahe.apply(b)
# # cv2.imshow('CLAHE output', cl)

# #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
# limg = cv2.merge((cl,ca,cb))
# # cv2.imshow('limg', limg)

# plt.figure(figsize=(20,20))
# plt.imshow(b)
# plt.xticks([]), plt.yticks([])
# plt.show()

# # #-----Converting image from LAB Color model to RGB model--------------------
# final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
# plt.figure(figsize=(20,20))
# plt.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
# plt.xticks([]), plt.yticks([])
# plt.show()
# # cv2.imshow('final', final)

#%% CIRCLE RECOGNITION

# Convolution Filters
# laplacian = cv2.Laplacian(equalized,cv2.CV_64F)
# sobelx = cv2.Sobel(equalized,cv2.CV_64F,1,0,ksize=3)  # x
# sobely = cv2.Sobel(equalized,cv2.CV_64F,0,1,ksize=3)  # y

test = warped.copy(); 
# test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY);  
l, a, b = cv2.split(cv2.cvtColor(test, cv2.COLOR_BGR2LAB))
test = b

# kernel = np.array([[-1,-1,-1], 
#                     [-1, 9,-1],
#                     [-1,-1,-1]])
# kernel=np.array([[0,-1,0],[-1,6,-1],[0,-1,0]])
# test = cv2.filter2D(test, -1, kernel) # applying the sharpening kernel to the input image & displaying it.

test = cv2.equalizeHist(test); 
# test = cv2.addWeighted(test, 1.2, test, 0, 0)
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
# test = clahe.apply(test)
# lap = cv2.equalizeHist((laplacian).astype('uint8')); 
# lap = pd.DataFrame(lap); lap[lap<100] = 0; lap[lap>100]=255
# lap = lap.values; lap = cv2.GaussianBlur(lap, (3, 3), 0)#; img = cv.equalizeHist(img)

plt.figure(figsize=(20,20))
plt.imshow(test)
plt.xticks([]), plt.yticks([])
plt.show()

if len(test.shape)<3:
    test=np.expand_dims(test,axis=2)

Z = test.reshape((-1,test.shape[2])) #l.reshape(-1,1)
Z = np.float32(Z)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 9
res,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape(test.shape)

plt.figure(figsize=(10,10))
plt.imshow(res2)
plt.show()

# test = cv2.bitwise_not(test)

# kernel = np.array([[-1,-1,-1], 
#                    [-1, 9,-1],
#                    [-1,-1,-1]])
# kernel=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
# test = cv2.filter2D(test, -1, kernel) # applying the sharpening kernel to the input image & displaying it.
 
# res2 = test   
circles = cv2.HoughCircles(res2.astype('uint8'), cv2.HOUGH_GRADIENT, 2.7, 85, param1=30,param2=90,minRadius=40,maxRadius=45)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(test,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(test,(i[0],i[1]),2,(0,0,255),3)

plt.figure(figsize=(15,15))
plt.imshow(test)
plt.xticks([]), plt.yticks([])
plt.show()













