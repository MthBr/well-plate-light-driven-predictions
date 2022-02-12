#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 12:52:01 2020
Circle Detection inspiration:
https://stackoverflow.com/questions/58109962/how-to-optimize-circle-detection-with-python-opencv

@author: modal
"""

import cv2
import matplotlib.pyplot as plt

def cropped_hough(image, blur=True):
    """
    Suppose the well plate (12x8) is cropped and horizontally positioned 
    """
    import numpy as np

    output = image.copy()
    height, width = image.shape[:2]
    maxRadius = int(0.93*(width/14)/2) #12+2
    minRadius = int(0.72*(width/14)/2) #12+2

    assert len(image.shape) < 4
    if len(image.shape) == 3 and image.shape[2] >1: #the second condition is for checking non fictitious 3D-images
        import warnings
        warnings.warn('Attettion: suppossing input is BGR')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if blur:
        image = cv2.GaussianBlur(image, ksize = (7, 7), sigmaX = 1.5)
        
    circles = cv2.HoughCircles(image=image, 
                            method=cv2.HOUGH_GRADIENT, 
                            dp=1.2, 
                            minDist=2*minRadius, #there is no overlapping, you could say that the distance between two circles is at least the diameter, so minDist could be set to something like 2*minRadius.
                            param1=50,
                            param2=30, #50 30 
                            minRadius=minRadius,
                            maxRadius=maxRadius                           
                            )

    print(circles.shape[1])
    print(circles.shape[1]== (12*8))
     
    return circles



def plot_circles(circles, image, cicle_colour = (0, 255, 0), thickness=3 ):  #(0,0,255)
    import numpy as np
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circlesRound = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circlesRound:
            cv2.circle(image, (x, y), r, cicle_colour ,thickness) # draw the outer circle
            cv2.circle(image,(x,y), 1, cicle_colour, thickness) # draw the center of the circle
    else:
        print ('No circles found')
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.show()
    return image


def watershed_segmentation(image):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
    return None


#%% INIT

def clear_all():
    """Clears all the variables from the workspace of the application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]


def load_test_file(): 
    image_file_name = '20a_cropped.jpg'
    from well_plate_project.config import data_dir
    path = data_dir / 'raw' / 'Cropped'
    image_file = path /  image_file_name
    assert image_file.is_file()

    img = cv2.imread(str(image_file))
    plt.imshow(img)
    plt.show()
    return img



if __name__ == "__main__":
    clear_all()
    image = load_test_file()
    
    print("Testing ... ")
    #image = watershed_segmentation(image)
    yuv_img = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv_img)
    single_chan_img = y

    circles = cropped_hough(single_chan_img)

    print("Plotting... ")
    plot_circles(circles, image)




