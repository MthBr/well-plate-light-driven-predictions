
import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np


#image_file = 'WellPlate_project/feature_eng/2.jpg'
image_file = 'a2_a_cropped.jpg'

print(image_file)

original_image = cv2.imread(image_file)

#img = cv2.cvtColor(original_image.astype('uint8'),cv2.COLOR_BGR2GRAY)
# convert our image from RGB Colours Space to HSV to work ahead.
img=cv2.cvtColor(original_image.astype('uint8'),cv2.COLOR_BGR2LAB)

l, a, b = cv2.split(img)
img = b; img=np.expand_dims(img,axis=2)

#cv2.imshow('image', img) 
plt.figure(figsize=(10,10))
plt.imshow(img)
plt.show()


Z_tot = img.reshape((-1,img.shape[2])) #l.reshape(-1,1)
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
#center[2:K-1][:] = [255,255,255]
res = center[label.flatten()]
res2 = res.reshape((img.shape))

plt.figure(figsize=(10,10))
plt.imshow(res2)
plt.show()




circles = cv2.HoughCircles(res2.astype('uint8'), cv2.HOUGH_GRADIENT, 2.7, 85, param1=30,param2=90,minRadius=40,maxRadius=45)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(res2,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(res2,(i[0],i[1]),2,(0,0,255),3)


plt.figure(figsize=(20,20))
plt.imshow(res2)
plt.xticks([]), plt.yticks([])
plt.show()










