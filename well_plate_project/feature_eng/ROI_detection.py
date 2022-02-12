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

image_file_name = 'a2_a_cropped.jpg'

image_file = raw_data_dir / image_file_name
assert image_file.is_file()

original_image = cv2.imread(str(image_file))

img = cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(original_image,cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),7,255,-1)

#plt.figure(figsize=(10,10))
#plt.imshow(img),
#plt.show()


#%%

#

sift = cv2.SIFT()
#kp = sift.detect(img,None)
#kp, des = sift.detectAndCompute(gray,None)


#img=cv2.drawKeypoints(gray,kp)

#plt.figure(figsize=(10,10))
#plt.imshow(img)
#plt.show()




# Load image 
image = cv2.imread(str(image_file), 0) 
image =   img

# Set our filtering parameters 
# Initialize parameter settiing using cv2.SimpleBlobDetector 
params = cv2.SimpleBlobDetector_Params() 


params.minDistBetweenBlobs = 10
params.filterByColor = True
params.maxArea = 10000
params.minThreshold = 10
params.maxThreshold = 200


# Set Area filtering parameters 
params.filterByArea = True
params.minArea = 20  #20 50
  
# Set Circularity filtering parameters 
params.filterByCircularity = True 
params.minCircularity = 0.1 #0.9 0 
  
# Set Convexity filtering parameters 
params.filterByConvexity = True
params.minConvexity = 0.1 # 0.2 0
      
# Set inertia filtering parameters 
params.filterByInertia = True
params.minInertiaRatio = 0.01 #0.01 0.1
  
# Create a detector with the parameters 
detector = cv2.SimpleBlobDetector_create(params) 
      
# Detect blobs 
keypoints = detector.detect(image) 
  
# Draw blobs on our image as red circles 
blank = np.zeros((1, 1))  
blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 0, 255), 
                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
  
number_of_blobs = len(keypoints) 
text = "Number of Circular Blobs: " + str(len(keypoints)) 
cv2.putText(blobs, text, (20, 550), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2) 
  
# Show blobs 
#cv2.imshow("Filtering Circular Blobs Only", blobs) 
#cv2.waitKey(0) 
#cv2.destroyAllWindows() 


plt.figure(figsize=(10,10))
plt.imshow(blobs)
plt.title("Filtering Circular Blobs Only") #get the title property handler
plt.show()






#DBscan 
#https://stackoverflow.com/questions/40142835/image-not-segmenting-properly-using-dbscan
#https://core.ac.uk/download/pdf/79492015.pdf


