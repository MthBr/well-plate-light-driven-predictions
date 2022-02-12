#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 4, 2020

@author: Enx
"""

#%% Importing and Variable fixing

from well_plate_project.config import data_dir, reportings_dir

# Dataset paths
raw_data_dir = data_dir / 'raw'
interm_data_dir = raw_data = data_dir / 'intermediate'
describe_data_dir = reportings_dir / 'description'

# Use KMeans clustering algorithm from sklearn.cluster to cluster pixels in image
from sklearn.cluster import KMeans

import numpy as np
import cv2
import matplotlib.pyplot as plt


image_name = 'a2_a_cropped.jpg'

image_path = raw_data_dir / image_name
assert image_path.is_file()

#%% Import image


original_image = cv2.imread(str(image_path))


img=original_image.astype('uint8')

plot_img_4hist_rgb(original_image)

plot_img_4hist_rgb(img)


#%% Convert image
# convert our image from RGB Colours Space to HSV to work ahead.
img_rgb=cv2.cvtColor(original_image.astype('uint8'),cv2.COLOR_BGR2RGB)
plot_img_4hist_rgb(img_rgb)


img_hsv=cv2.cvtColor(original_image.astype('uint8'),cv2.COLOR_BGR2HSV)
plot_img_4hist_hsv(img_hsv)
#https://raspberrypi.stackexchange.com/questions/10588/hue-saturation-intensity-histogram-plot

img_lab=cv2.cvtColor(original_image.astype('uint8'),cv2.COLOR_BGR2LAB)
plot_img_4hist_rgb(img_lab)


#%% Reshaping

# For clustering the image using k-means, we first need to convert it into a 2-dimensional array
image_2D = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
reshaped_image = np.float32(original_image.reshape(-1, 3))
reshaped_image_hsv = np.float32(img_hsv.reshape(-1, 3))
reshaped_image_lab = np.float32(img_lab.reshape(-1, img_lab.shape[2]))




#%% Clustering 1
K = 10
stopCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)

ret, labels, clusters = cv2.kmeans(reshaped_image, K, None, stopCriteria, 10, cv2.KMEANS_RANDOM_CENTERS)
clusters = np.uint8(clusters)

intermediateImage = clusters[labels.flatten()]
clusteredImage = intermediateImage.reshape((original_image.shape))

cv2.imwrite("clusteredImage.jpg", clusteredImage)

plot_img_4hist_rgb(clusteredImage)



#%% Clustering2
# tweak the cluster size and see what happens to the Output
kmeans = KMeans(n_clusters=5, random_state=0).fit(image_2D)
clustered = kmeans.cluster_centers_[kmeans.labels_]

# Reshape back the image from 2D to 3D image
clustered_3D = clustered.reshape(img.shape[0], img.shape[1], img.shape[2])
plt.imshow(clustered_3D)
plt.title('Clustered Image')
plt.show()

plot_img_4hist_rgb(clustered_3D)



#%% Remove 1 cluster from image and apply canny edge detection
removedCluster = 1

cannyImage = np.copy(original_image).reshape((-1, 3))
cannyImage[labels.flatten() == removedCluster] = [0, 0, 0]

cannyImage = cv2.Canny(cannyImage,100,200).reshape(original_image.shape)
cv2.imwrite("cannyImage.jpg", cannyImage)
plot_img_4hist_rgb(cannyImage)


#%% Finding contours using opencv

initialContoursImage = np.copy(cannyImage)
imgray = cv2.cvtColor(initialContoursImage, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(imgray, 50, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(initialContoursImage, contours, -1, (0,0,255), cv2.CHAIN_APPROX_SIMPLE)
cv2.imwrite("initialContoursImage.jpg", initialContoursImage)


plot_img_4hist_rgb(initialContoursImage)


#DBscan 
#https://stackoverflow.com/questions/40142835/image-not-segmenting-properly-using-dbscan
#https://core.ac.uk/download/pdf/79492015.pdf


#https://github.com/s4lv4ti0n/clusteringAlgorithms


# %% Define functions

def plot_img_4hist_rgb(image_to_plot):
    # Calulate various hists
    
    plt.figure(figsize=(10,10))

    #plt.subplot(221), plt.imshow(image_to_plot, 'gray')
    plt.subplot(221), plt.imshow(image_to_plot)

    plt.subplot(223), plt.hist(image_to_plot.ravel(),256,[0,256])
    plt.xlim([0,256])
    

    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([image_to_plot],[i],None,[256],[0,256])
        plt.subplot(222), plt.plot(histr,color = col)
        plt.yscale('log')
        plt.xlim([0,256])
        plt.subplot(224), plt.plot(histr,color = col) 
        plt.xlim([0,256])


    plt.show()
    plt.close()
    return


# %%
def plot_img_4hist_hsv(image_to_plot):
    # Calulate various hists
    
    plt.figure(figsize=(10,10))

    #plt.subplot(221), plt.imshow(image_to_plot, 'gray')
    plt.subplot(221), plt.imshow(image_to_plot)
    
    hist = cv2.calcHist( [image_to_plot], [0, 1], None, [180, 256], [0, 180, 0, 256] )
    plt.subplot(223), plt.plot(hist)
    plt.subplot(222), plt.imshow(hist, interpolation = 'nearest')


    plt.show()
    plt.close()
    return
# %%
