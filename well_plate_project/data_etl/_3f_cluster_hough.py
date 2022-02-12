#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 12:52:01 2020

@author: modal
"""

import cv2
import matplotlib.pyplot as plt

def cluster_hough(warped):
    import numpy as np
    #%% CIRCLE RECOGNITION

    test = warped.copy(); 
    # test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY);  
    l, a, b = cv2.split(cv2.cvtColor(test, cv2.COLOR_BGR2LAB))
    test = b

    # kernel = np.array([[-1,-1,-1], 
    #                     [-1, 9,-1],
    #                     [-1,-1,-1]])
    # kernel=np.array([[0,-1,0],[-1,6,-1],[0,-1,0]])
    # test = cv2.filter2D(test, -1, kernel) # applying the sharpening kernel to the input image & displaying it.

    test = cv2.equalizeHist(test); 
    # test = cv2.addWeighted(test, 1.2, test, 0, 0)


    #TODO !!!
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    # test = clahe.apply(test)


    # lap = cv2.equalizeHist((laplacian).astype('uint8')); 
    # lap = pd.DataFrame(lap); lap[lap<100] = 0; lap[lap>100]=255
    # lap = lap.values; lap = cv2.GaussianBlur(lap, (3, 3), 0)#; img = cv.equalizeHist(img)

    plt.figure(figsize=(20,20))
    plt.imshow(test)
    plt.xticks([]), plt.yticks([])
    plt.show()

    if len(test.shape)<3:
        test=np.expand_dims(test,axis=2)

    Z = test.reshape((-1,test.shape[2])) #l.reshape(-1,1)
    Z = np.float32(Z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 9
    res,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(test.shape)

    plt.figure(figsize=(10,10))
    plt.imshow(res2)
    plt.show()

    # test = cv2.bitwise_not(test)

    # kernel = np.array([[-1,-1,-1], 
    #                    [-1, 9,-1],
    #                    [-1,-1,-1]])
    # kernel=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    # test = cv2.filter2D(test, -1, kernel) # applying the sharpening kernel to the input image & displaying it.
    
    # res2 = test   
    circles = cv2.HoughCircles(res2.astype('uint8'), cv2.HOUGH_GRADIENT, 2.7, 85, param1=30,param2=90,minRadius=40,maxRadius=45)

    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(test,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(test,(i[0],i[1]),2,(0,0,255),3)


    plt.figure(figsize=(15,15))
    plt.imshow(test)
    plt.xticks([]), plt.yticks([])
    plt.show()
    return test

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
    image_file_name = 'a2_a_cropped.jpg'
    from well_plate_project.config import data_dir
    raw_data_dir = data_dir / 'raw'
    path = raw_data_dir / 'EXPERIMENTS'
    image_file = raw_data_dir /  image_file_name
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

    image = cluster_hough(image)

    print("Plotting... ")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.xticks([]), plt.yticks([])
    plt.show()















