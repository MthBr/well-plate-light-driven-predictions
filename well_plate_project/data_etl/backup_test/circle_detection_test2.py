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
raw_data_dir = data_dir / 'raw'
path = raw_data_dir / 'EXPERIMENTS'
image_file = raw_data_dir /  image_file_name
assert image_file.is_file()


import cv2
import numpy as np
import matplotlib.pyplot as plt



# Load in image, convert to gray scale, and Otsu's threshold
image = cv2.imread(str(image_file))
plt.imshow(image)
plt.show()

output = image.copy()
height, width = image.shape[:2]
maxRadius = int(1.05*(width/14)/2) #12+2
minRadius = int(0.79*(width/14)/2) #12+2

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
circles = cv2.HoughCircles(image=gray, 
                           method=cv2.HOUGH_GRADIENT, 
                           dp=1.2, 
                           minDist=2*minRadius, #there is no overlapping, you could say that the distance between two circles is at least the diameter, so minDist could be set to something like 2*minRadius.
                           param1=50,
                           param2=50,
                           minRadius=minRadius,
                           maxRadius=maxRadius                           
                          )

if circles is not None:
    # convert the (x, y) coordinates and radius of the circles to integers
    circlesRound = np.round(circles[0, :]).astype("int")
    # loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circlesRound:
        cv2.circle(output, (x, y), r, (0, 255, 0), 4)

    plt.imshow(output)
else:
    print ('No circles found')



#https://stackoverflow.com/questions/58109962/how-to-optimize-circle-detection-with-python-opencv


