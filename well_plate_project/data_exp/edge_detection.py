#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 16:39:30 2020

@author: modal
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import pandas as pd


path = 'IMG/EXPERIMENTS/'
path_test = 'IMG/TEST/'
# bg = cv.imread('bg.jpeg',0); bg_eq =  cv.equalizeHist(bg); #bg_eq = cv.GaussianBlur(bg_eq, (1, 1), 7)
img = cv.imread(path_test+'fluo1b.jpeg',0)# cv.IMREAD_COLOR)#; img_eq = cv.equalizeHist(img); #img_eq = cv.GaussianBlur(img_eq, (1, 1), 7)


if img.shape[0]>img.shape[1]:
    img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)

# img = cv.GaussianBlur(img, (3, 3), 0)#; img = cv.equalizeHist(img)
    
ax,fig = plt.subplots(figsize=(50,20))
plt.subplot(131), plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.show()

# convolute with proper kernels
laplacian = cv.Laplacian(img,cv.CV_64F)
sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=3)  # x
sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=3)  # y

ax,fig = plt.subplots(figsize=(30,30))
plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.show()

test=laplacian
test = cv.convertScaleAbs(test)
test = cv.equalizeHist(test)
test = pd.DataFrame(test); test[test<150]=0; test = np.array(test)
edges = cv.Canny(test,60,100)

ax,fig = plt.subplots(figsize=(30,20))
plt.subplot(121),plt.imshow(test)#,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()




# Median
# denoised = cv.medianBlur(test, 7)

# # Morphologic
kernel = np.ones((12,12),np.uint8)
denoised = cv.morphologyEx(test, cv.MORPH_CLOSE, kernel)

# # boxFilter
# denoised = cv.boxFilter(test, ddepth=-1, 7);
   
# #Gaussian Filter
# denoised = cv.GaussianBlur(test, (7,7), 5);
   
  
# str = 'Bilateral Filter';
# denoised = cv.bilateralFilter(test, 21, 21, 5);

# str = 'Non-Local Means Filter';
denoised2 = cv.fastNlMeansDenoising(test, h=41,templateWindowSize=5, searchWindowSize=35);


# ax,fig = plt.subplots(figsize=(30,20))
# plt.subplot(121),plt.imshow(test)#,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(denoised,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()

kernel = np.ones((3, 3), np.uint8)
eroded = cv.erode(denoised, kernel, iterations = 4)
eroded2 = cv.erode(denoised2, kernel, iterations = 4)

# ax,fig = plt.subplots(figsize=(30,20))
# plt.subplot(121),plt.imshow(test)#,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(eroded,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()

#%%

pd_img = pd.DataFrame(img.copy()); pd_img2 = pd.DataFrame(img.copy())
pd_eroded = pd.DataFrame(eroded); pd_eroded2 = pd.DataFrame(eroded2)
pd_img[pd_eroded!=0]=0; pd_img2[pd_eroded2!=0]=0; 
pd_img[pd_img<80] = 0
ax,fig = plt.subplots(figsize=(30,20))
plt.subplot(131),plt.imshow(img)#,cmap = 'gray')
plt.title('H Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(pd_img.values)
plt.title('K Image'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(pd_img2.values)
plt.title('J Image'), plt.xticks([]), plt.yticks([])
plt.show()
#setting alpha=1, beta=1, gamma=0 gives direct overlay of two images


plt.figure(figsize=(25,25))
edo = cv.addWeighted(img, 1, eroded, 0.3, 0)
plt.imshow(edo)

img_eq = cv.equalizeHist(img)
pd_img_eq = pd.DataFrame(img_eq); pd_img_eq[pd_img_eq<110]=0; img_eq = np.array(pd_img_eq); img_eq = np.array(pd_img_eq)

plt.subplot(132), plt.imshow(img_eq, cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])


plt.subplot(133), plt.imshow(img_eq-bg_eq, cmap = 'gray')
plt.title('Difference Image'), plt.xticks([]), plt.yticks([])
plt.show()

img_diff = img_eq-bg_eq
# # from PIL import Image
# # img = np.array(Image.open('fluo1.jpeg'))
# # img.show()

# img[:,:,2]=0
# img[:,:,0]=0

# img_test = pd.DataFrame(img[:,:,1])
# # T1=1; T2=100
# # img_test[(img_test>T1) & (img_test<T2)]=0
# # img[:,:,1] = np.array(img_test)


# img_eq = cv.GaussianBlur(img_eq, (3, 3), 7)
edges = cv.Canny(img_diff,60,200)

ax,fig = plt.subplots(figsize=(30,20))
plt.subplot(121),plt.imshow(img_diff)#,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()


#%%
import numpy as np 
import cv2 
import matplotlib.pyplot as plt

path = 'IMG/EXPERIMENTS/'
path_test = 'IMG/TEST/'
image = 'a2_a.jpg'
# Reading image 
img2 = cv2.imread(path+image, cv2.IMREAD_COLOR) 
   
# Reading same image in another variable and  
# converting to gray scale. 
img = cv2.imread(path+image, cv2.IMREAD_GRAYSCALE) 
img = cv2.equalizeHist(img)
# Converting image to a binary image  
# (black and white only image). 
_,threshold = cv2.threshold(img, 237, 255,  
                            cv2.THRESH_BINARY) 

plt.figure(figsize=(25,25))
plt.imshow(threshold, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
   
   
# Detecting shapes in image by selecting region  
# with same colors or intensity. 
# contours = cv2.findContours(threshold, cv2.RETR_TREE, 
#                             cv2.CHAIN_APPROX_SIMPLE)
 
contours = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
   
# Searching through every region selected to  
# find the required polygon. 
# for cnt in contours : 
#     area = cv2.contourArea(cnt) 
   
#     # Shortlisting the regions based on there area. 
#     if area > 100:  
#         approx = cv2.approxPolyDP(cnt,  
#                                   0.01 * cv2.arcLength(cnt, True), True) 
   
#         # Checking if the no. of sides of the selected region is 7. 
#         if(len(approx) == 4):  
#             cv2.drawContours(img2, [approx], 0, (0, 0, 255), 5) 
   
# # Showing the image along with outlined arrow. 
# plt.figure(figsize=(25,25))
# plt.imshow(img2, cmap = 'gray', interpolation = 'bicubic')
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()

for c in contours:
    # get the bounding rect
    area = cv2.contourArea(c); 
    if  area > 0.025*(img.shape[0]*img.shape[1]):
        print(area/(img.shape[0]*img.shape[1]))
        x, y, w, h = cv2.boundingRect(c)
        
        # draw a green rectangle to visualize the bounding rect
        # cv2.rectangle(img2, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
        # get the min area rect
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        # convert all coordinates floating point values to int
        box = np.int0(box)
        # draw a red 'nghien' rectangle
        cv2.drawContours(img2, [box], 0, (0, 255, 0), 5)

    # finally, get the min enclosing circle
    # (x, y), radius = cv2.minEnclosingCircle(c)
    # convert all values to int
    # center = (int(x), int(y))
    # radius = int(radius)
    # and draw the circle in blue
    # img = cv2.circle(img2, center, radius, (255, 0, 0), 2)

        # print(len(contours))
        cv2.drawContours(img2, c, -1, (255, 255, 0), 5)
   
plt.figure(figsize=(25,25))
plt.imshow(img2, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()












