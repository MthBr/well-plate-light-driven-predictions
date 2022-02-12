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



#%%
def felzen (img):
    print("Starting felzenszwalb segmentation... ")
    image_felzenszwalb = seg.felzenszwalb(img) 
    print("Done felzenszwalb segmentation... with")
    print(np.unique(image_felzenszwalb).size)

    print("Starting coloring felzenszwalb segmentation...")
    image_felzenszwalb_colored = color.label2rgb(image_felzenszwalb, img, kind='avg')
    print("Ended coloring felzenszwalb segmentation...")

    return image_felzenszwalb_colored


# %%
def clear_all():
    """Clears all the variables from the workspace of the application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]


def load_test_file(): 
    image_file_name = 'a2_a_cropped.jpg'
    from well_plate_project.config import data_dir
    raw_data_dir = data_dir / 'raw'
    path = raw_data_dir / 'EXPERIMENTS'
    image_file = raw_data_dir /  image_file_name
    assert image_file.is_file()
    img = cv2.imread(str(image_file))
    return img



if __name__ == "__main__":
    clear_all()
    original_image = load_test_file()
    
    print("Testing ... ")
    img = original_image 
    img = cv2.cvtColor(original_image,cv2.COLOR_BGR2LAB)

    plt.imshow(img)
    plt.show()
    
    image_felzenszwalb_colored = felzen(img)

    print("Plotting... ")
    plt.figure(figsize=(10,10))
    plt.imshow(image_felzenszwalb_colored)
    plt.show()

    print("Done")
