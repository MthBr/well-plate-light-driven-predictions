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


def resize(img):
    scale_percent = 10 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
    return resized

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
    #denoised_laplacian = cv2.GaussianBlur(thresh1, (3,3), 2);  
    res = resize(thresh1)
    #denoised_laplacian = cv2.GaussianBlur(res, (3,3), 2);  
    return res


from well_plate_project.config import data_dir
path_query =  data_dir / 'raw' / 'Match'
image_file = path_query / '1ln.png'
assert image_file.is_file()
queryImage = cv2.imread(str(image_file)) #aswana_cropped_2 aswana_cropped
#plt.imshow(queryImage); plt.show()



path_train = data_dir / 'raw' / 'exp_v2_crp (t1)' / 'luce_nat'  #foto_tel1  EXPERIMENTS foto_tel1
jpg = path_train / '10' / '20201118_090359_cropped.jpg' #20201118_090416   IMG_20201118_090440
jpg_bad = path_train / '10' / '20201118_090416_cropped.jpg' #20201118_090416   IMG_20201118_090440
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

# from scipy import signal





import numpy as np
diff_lap_good = np.linalg.norm(lap_orig -lap_good, ord = np.inf) #np.inf  'fro'
diff_lap_bad =np.linalg.norm(lap_orig - lap_bad, ord = np.inf)
# cor_good = signal.correlate2d (lap_orig, lap_good)
# cor_bad = signal.correlate2d (lap_orig, lap_bad)

# print('GOOD   -> Corr MAX: ', np.round(np.max(cor_good)), ' Corr NORM: ', 
#       np.round(np.linalg.norm(cor_good, ord = np.inf)))
# print('BAD    -> Corr MAX: ', np.round(np.max(cor_bad)), ' Corr NORM: ', 
#       np.round(np.linalg.norm(cor_bad, ord = np.inf)))




jpg_good_2 = path_train / '10' / '20201118_090420_cropped.jpg'
good_2 = cv2.imread(str(jpg_good_2))
lap_good_2 = compute_laplac(good_2)
plt.figure(figsize=(10,10))
plt.imshow(lap_good_2);plt.show()
diff_lap_good_2 = np.linalg.norm(lap_orig -lap_good_2, ord = np.inf) #np.inf  'fro'





jpg_good_3 = path_train / '10' / '20201118_090422_cropped.jpg'
good_3 = cv2.imread(str(jpg_good_3))
lap_good_3 = compute_laplac(good_3)
plt.figure(figsize=(10,10))
plt.imshow(lap_good_3);plt.show()
diff_lap_good_3 = np.linalg.norm(lap_orig -lap_good_3, ord = np.inf) #np.inf  'fro'




jpg_bad_2 = path_train / '10' / 'IMG_20201118_090440_cropped.jpg'
bad_2 = cv2.imread(str(jpg_bad_2))
plt.figure(figsize=(10,10))
plt.imshow(bad_2);plt.show()
lap_bad_2 = compute_laplac(bad_2)
plt.figure(figsize=(10,10))
plt.imshow(lap_bad_2);plt.show()
diff_lap_bad_2 = np.linalg.norm(lap_orig -lap_bad_2, ord = np.inf) #np.inf  'fro'


# cor_good_2 = signal.correlate2d (lap_orig, lap_good_2)
# cor_good_3 = signal.correlate2d (lap_orig, lap_good_3)
# cor_bad_2 = signal.correlate2d (lap_orig, lap_bad_2)

# print('GOOD_2 -> Corr MAX: ', np.round(np.max(cor_good_2)), ' Corr NORM: ',
#       np.round(np.linalg.norm(cor_good_2, ord = np.inf)))
# print('GOOD_3 -> Corr MAX: ', np.round(np.max(cor_good_3)), ' Corr NORM: ',
#       np.round(np.linalg.norm(cor_good_3, ord = np.inf)))
# print('BAD_2  -> Corr MAX: ', np.round(np.max(cor_bad_2)), ' Corr NORM: ',
#       np.round(np.linalg.norm(cor_bad_2, ord = np.inf)))


