#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 16:31:49 2020

@author: enzo
"""
import cv2
import matplotlib.pyplot as plt

def single_channel_gray(BRG_input_image):
    gray_image = cv2.cvtColor(BRG_input_image, cv2.COLOR_BGR2GRAY) 
    #gray_equlized = cv2.equalizeHist(gray_image)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4,3))
    gray_equlized = clahe.apply(gray_image)
    return gray_equlized

def compute_laplac(input_image):   
    input_image = single_channel_gray(input_image)
    #Gaussian Filter
    #denoised = cv2.GaussianBlur(input_image, (3,3), 3);   
    laplacian = cv2.Laplacian(input_image,cv2.CV_64F).astype('uint8')
    #denoised_laplacian = cv2.GaussianBlur(laplacian, (7,7), 5);   
    sobelx = cv2.Sobel(input_image,cv2.CV_64F,1,0,ksize=3)  # x
    sobely = cv2.Sobel(input_image,cv2.CV_64F,0,1,ksize=3)  # y
    laplacian = sobelx+ sobely
    ret,thresh1 = cv2.threshold(laplacian,200,255,cv2.THRESH_BINARY)
    return laplacian


from well_plate_project.config import data_dir
path_query =  data_dir / 'raw' / 'Match'
image_file = path_query / 'aswana_cropped.jpg'
assert image_file.is_file()
queryImage = cv2.imread(str(image_file)) #aswana_cropped_2 aswana_cropped
#plt.imshow(queryImage); plt.show()



path_train = data_dir / 'raw' / 'EXPERIMENTS_Crp' #foto_tel1  EXPERIMENTS foto_tel1
jpg = path_train / 'a2_c_cropped.jpg' #20201118_090416   IMG_20201118_090440
jpg_bad = path_train / 'b1_a_cropped.jpg' #20201118_090416   IMG_20201118_090440
good = cv2.imread(str(jpg)) #aswana_cropped_2 aswana_cropped
bad = cv2.imread(str(jpg_bad)) #aswana_cropped_2 aswana_cropped


lap_orig = compute_laplac(queryImage)
plt.figure(figsize=(10,10))
plt.imshow(lap_orig);plt.show()

lap_good = compute_laplac(good)
plt.figure(figsize=(10,10))
plt.imshow(lap_good);plt.show()

lap_bad = compute_laplac(bad)
plt.figure(figsize=(10,10))
plt.imshow(lap_bad);plt.show()



import numpy as np
diff_lap_good = np.linalg.norm(lap_orig -lap_good, ord = np.inf) #np.inf  'fro'
diff_lap_bad =np.linalg.norm(lap_orig - lap_bad, ord = np.inf)




jpg_good_2 = path_train / 'd2_a_cropped.jpg'
good_2 = cv2.imread(str(jpg_good_2))
lap_good_2 = compute_laplac(good_2)
diff_lap_good_2 = np.linalg.norm(lap_orig -lap_good_2, ord = np.inf) #np.inf  'fro'





jpg_good_3 = path_train / 'e2_b_cropped.jpg'
good_3 = cv2.imread(str(jpg_good_3))
lap_good_3 = compute_laplac(good_3)
diff_lap_good_3 = np.linalg.norm(lap_orig -lap_good_3, ord = np.inf) #np.inf  'fro'




jpg_bad_2 = path_train / 'd1_a_cropped.jpg'
bad_2 = cv2.imread(str(jpg_bad_2))
lap_bad_2 = compute_laplac(bad_2)
diff_lap_bad_2 = np.linalg.norm(lap_orig -lap_bad_2, ord = np.inf) #np.inf  'fro'






