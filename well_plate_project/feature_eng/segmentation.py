#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 4, 2020

@author: Enx
"""
from well_plate_project.config import data_dir, reportings_dir

# Dataset paths
raw_data_dir = data_dir / 'raw'
interm_data_dir = raw_data = data_dir / 'intermediate'
describe_data_dir = reportings_dir / 'description'


import numpy as np
import cv2
import matplotlib.pyplot as plt

image_file = '1.jpg'

image_path = str(raw_data_dir / '1.jpg')
assert image_path.is_file()

print(image_path)

original_image = cv2.imread(str(image_path))


# convert our image from RGB Colours Space to HSV to work ahead.
img=cv2.cvtColor(original_image.astype('uint8'),cv2.COLOR_BGR2RGB)









#DBscan 
#https://stackoverflow.com/questions/40142835/image-not-segmenting-properly-using-dbscan
#https://core.ac.uk/download/pdf/79492015.pdf


