#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 16:31:49 2020

@author: enzo
"""
#%%
import cv2
import matplotlib.pyplot as plt
import numpy as np
from well_plate_project.config import data_dir

def single_channel_gray(BRG_input_image):
    # # Morphologic
    kernel = np.ones((12,12),np.uint8)
    denoised = cv2.morphologyEx(BRG_input_image, cv2.MORPH_CLOSE, kernel)
    gray_image = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY) 
    #gray_equlized = cv2.equalizeHist(gray_image)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4,3))
    img = clahe.apply(gray_image)
    #img = cv2.equalizeHist(gray_image)
    # Converting image to a binary image  
    # (black and white only image). 
    _,threshold = cv2.threshold(img, 20, 255,  
                                cv2.THRESH_BINARY)
    return img, threshold



def clustering(img):
    # convert to np.float32
    Z = img.reshape((-1,1))
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 12
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return res2

#%%
path_query =  data_dir / 'raw' / 'Match'
image_file = path_query / '1ln.png'
assert image_file.is_file()
queryImage = cv2.imread(str(image_file)) #aswana_cropped_2 aswana_cropped
#plt.imshow(queryImage); plt.show()



#%%
plt.close('all')

#%%
path_train = data_dir / 'raw' / 'exp_v2_crp (t1)' / 'luce_nat'  #foto_tel1  EXPERIMENTS foto_tel1
jpg_path = path_train / '10' / '20201118_090359_cropped.jpg' #20201118_090416   IMG_20201118_090440
good = cv2.imread(str(jpg_path)) #aswana_cropped_2 aswana_cropped

jpg_bad = path_train / '10' / '20201118_090416_cropped.jpg' #20201118_090416   IMG_20201118_090440
bad = cv2.imread(str(jpg_bad)) #aswana_cropped_2 aswana_cropped

#%% 
good_gray, good_gray_binary = single_channel_gray(queryImage) # queryImage good
plt.figure(figsize=(10,10)), plt.imshow(good_gray);plt.show()
plt.figure(figsize=(10,10)), plt.imshow(good_gray_binary);plt.show()

# Detect edges using Canny
canny_output = cv2.Canny(good_gray_binary, 10, 256)
# Find contours
contours, hierarchy = cv2.findContours(good_gray_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

import random as rng
drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
for i in range(len(contours)):
    color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
    cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
# Show in a window
plt.imshow(drawing)

contours_org=contours
hierarchy_org=hierarchy

#%% 

good_gray, good_gray_binary = single_channel_gray(good) # queryImage good
plt.figure(figsize=(10,10)), plt.imshow(good_gray);plt.show()
plt.figure(figsize=(10,10)), plt.imshow(good_gray_binary);plt.show()

clustered = clustering(good_gray)
plt.figure(figsize=(10,10)), plt.imshow(clustered);plt.show()


# Detect edges using Canny
canny_output = cv2.Canny(clustered, 10, 250)
# Find contours
contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

import random as rng
drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
for i in range(len(contours)):
    color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
    cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
# Show in a window
plt.figure(figsize=(10,10)), plt.imshow(drawing);plt.show()



#%% 

good_gray, good_gray_binary = single_channel_gray(bad) # queryImage good
plt.figure(figsize=(10,10)), plt.imshow(good_gray);plt.show()
plt.figure(figsize=(10,10)), plt.imshow(good_gray_binary);plt.show()


#%% 
dst = cv2.Canny(good_gray, 50, 200, None, 3)



# Copy edges to the images that will display the results in BGR
cdst = cv2.cvtColor(good_gray, cv2.COLOR_GRAY2BGR)
cdstP = np.copy(cdst)

# lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)

# if lines is not None:
#     for i in range(0, len(lines)):
#         rho = lines[i][0][0]
#         theta = lines[i][0][1]
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a * rho
#         y0 = b * rho
#         pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
#         pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
#         cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)


linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 0, 0)

if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)


# plt.figure(figsize=(10,10)), plt.imshow(good_gray);plt.show()
# plt.figure(figsize=(10,10)), plt.imshow(cdst);plt.show()
plt.figure(figsize=(10,10)), plt.imshow(cdstP);plt.show()




            
#%%
jpg_bad = path_train / '10' / '20201118_090416_cropped.jpg' #20201118_090416   IMG_20201118_090440
bad = cv2.imread(str(jpg_bad)) #aswana_cropped_2 aswana_cropped























plt.figure(figsize=(10,10))
plt.imshow(lap_orig);plt.show()

lap_good = single_channel_gray(good)
plt.figure(figsize=(10,10))
plt.imshow(lap_good);plt.show()

flipHorizontal = cv2.flip(lap_good, 1)




lap_bad = single_channel_gray(bad)
plt.figure(figsize=(10,10))
plt.imshow(lap_bad);plt.show()






import numpy as np
diff_lap_good = np.linalg.norm(lap_good -lap_orig)/(lap_good.shape[0]*lap_good.shape[1]) #np.inf  'fro'
diff_orig = np.linalg.norm(lap_orig -lap_orig)/(lap_orig.shape[0]*lap_orig.shape[1]) #np.inf  'fro'
diff_lap_bad = np.linalg.norm(lap_bad -lap_orig)/(lap_bad.shape[0]*lap_bad.shape[1]) #np.inf  'fro'


jpg_good_2 = path_train / '10' / '20201118_090420_cropped.jpg'
good_2 = cv2.imread(str(jpg_good_2))
lap_good_2 = single_channel_gray(good_2)

plt.figure(figsize=(10,10))
plt.imshow(cv2.flip(lap_good_2, 1));plt.show()
plt.figure(figsize=(10,10))
plt.imshow(lap_good_2 -cv2.flip(lap_good_2, 1));plt.show()
diff_lap_good_2 = np.linalg.norm(lap_good_2 -lap_orig)/(lap_good_2.shape[0]*lap_good_2.shape[1]) #np.inf  'fro' ord ='fro'


compare_edge_images(queryImage, good_2)


jpg_good_3 = path_train / '10' / '20201118_090422_cropped.jpg'
good_3 = cv2.imread(str(jpg_good_3))
lap_good_3 = single_channel_gray(good_3)
plt.figure(figsize=(10,10))
plt.imshow(lap_good_3);plt.show()
diff_lap_good_3 = np.linalg.norm(lap_good_3 -lap_orig)/(lap_good_2.shape[0]*lap_good_2.shape[1])  #np.inf  'fro'




jpg_bad_2 = path_train / '10' / 'IMG_20201118_090440_cropped.jpg'
bad_2 = cv2.imread(str(jpg_bad_2))
plt.figure(figsize=(10,10))
plt.imshow(bad_2);plt.show()
lap_bad_2 = single_channel_gray(bad_2)
plt.figure(figsize=(10,10))
plt.imshow(lap_bad_2);plt.show()
diff_lap_bad_2 = np.linalg.norm(lap_bad_2 -lap_orig)/(lap_good_2.shape[0]*lap_good_2.shape[1])  #np.inf  'fro'



plt.figure(figsize=(10,10))
plt.imshow(lap_bad_2);plt.show()
plt.figure(figsize=(10,10))
plt.imshow(cv2.flip(lap_bad_2, 1));plt.show()
plt.figure(figsize=(10,10))
plt.imshow(lap_bad_2 -cv2.flip(lap_bad_2, 1));plt.show()