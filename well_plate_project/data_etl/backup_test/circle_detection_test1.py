#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 12:52:01 2020
Circle Detection inspiration:
https://stackoverflow.com/questions/58109962/how-to-optimize-circle-detection-with-python-opencv

@author: modal
"""
#%% INIT
image_file_name = 'a2_a_cropped.jpg'

from well_plate_project.config import data_dir
path = data_dir / 'raw'
image_file = path /  image_file_name
assert image_file.is_file()


import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import matplotlib.pyplot as plt


# Load in image, convert to gray scale, and Otsu's threshold
image = cv2.imread(str(image_file))
plt.imshow(image)
plt.show()




gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Remove small noise by filtering using contour area
cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

for c in cnts:
    if cv2.contourArea(c) < 1000:
        cv2.drawContours(thresh,[c], 0, (0,0,0), -1)

#cv2.imshow('thresh', thresh)
plt.imshow(cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB)); plt.show()

# Compute Euclidean distance from every binary pixel
# to the nearest zero pixel then find peaks
distance_map = ndimage.distance_transform_edt(thresh)
local_max = peak_local_max(distance_map, indices=False, min_distance=5, labels=thresh)

# Perform connected component analysis then apply Watershed
markers = ndimage.label(local_max, structure=np.ones((3, 3)))[0]
labels = watershed(-distance_map, markers, mask=thresh)

# Iterate through unique labels
for label in np.unique(labels):
    if label == 0:
        continue

    # Create a mask
    mask = np.zeros(gray.shape, dtype="uint8")
    mask[labels == label] = 255

    # Find contours and determine contour area
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    c = max(cnts, key=cv2.contourArea)
    cv2.drawContours(image, [c], -1, (36,255,12), -1)

#cv2.imshow('image', image)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
#cv2.waitKey()