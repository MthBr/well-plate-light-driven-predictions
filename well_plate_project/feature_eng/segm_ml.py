#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 4, 2020

@author: Enx
"""

#%% Clean Memory
for i in list(globals().keys()):
    if(i[0] != '_'):
        exec('del {}'.format(i))


#%%
from well_plate_project.config import data_dir, reportings_dir

# Dataset paths
raw_data_dir = data_dir / 'raw'
interm_data_dir = raw_data = data_dir / 'intermediate'
describe_data_dir = reportings_dir / 'description'


import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.segmentation as seg
#import skimage.filters as filters
#import skimage.draw as draw
import skimage.color as color


image_file_name = 'a2_a_cropped.jpg'

image_file = raw_data_dir / image_file_name
assert image_file.is_file()

original_image = cv2.imread(str(image_file))

img = cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(original_image,cv2.COLOR_BGR2GRAY)

#%%
plt.imshow(img)
plt.show()

#%%
image_felzenszwalb = seg.felzenszwalb(img) 

print(np.unique(image_felzenszwalb).size)

plt.imshow(image_felzenszwalb)
plt.show()


image_felzenszwalb_colored = color.label2rgb(image_felzenszwalb, img, kind='avg')

plt.figure(figsize=(10,10))
plt.imshow(image_felzenszwalb_colored)
plt.show()




