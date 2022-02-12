#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 11:58:48 2020

@author: modal
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import pandas as pd


path = 'IMG/EXPERIMENTS/'
path_test = 'IMG/TEST/'
# bg = cv.imread('bg.jpeg',0); bg_eq =  cv.equalizeHist(bg); #bg_eq = cv.GaussianBlur(bg_eq, (1, 1), 7)
# img = cv2.imread(path+'b2_a.jpg')#[0:25,0:25,:]
# plt.imshow(img)

# for idx in [0,1,2]:
#     print(idx)
#     f = cv2.dft(np.float32(img[:,:,idx]), flags=cv2.DFT_COMPLEX_OUTPUT)
#     f_shift = np.fft.fftshift(f)
#     f_complex = f_shift[:,:,0] + 1j*f_shift[:,:,1]
#     f_abs = np.abs(f_complex) + 1 # lie between 1 and 1e6
#     f_bounded = 20 * np.log(f_abs)
#     f_img = 255 * f_bounded / np.max(f_bounded)
#     f_img = f_img.astype(np.uint8)
    
#     f_img = cv2.idft(f_shift)
#     f_img = cv2.magnitude(f_img[:,:,0],f_img[:,:,1])
    
#     plt.imshow(f_img)
#     plt.show()
    
#%%

img = cv2.imread(path_test+'fluo16_cropped.jpeg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

b = clahe.apply(img[:, :, 0])
g = clahe.apply(img[:, :, 1])
r = clahe.apply(img[:, :, 2])
equalized = np.dstack((b, g, r))

# cl1 = clahe.apply(img)
plt.imshow(cv2.cvtColor(equalized, cv2.COLOR_BGR2RGB))
plt.show()