#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 4, 2020


#https://towardsdatascience.com/clustering-with-more-than-two-features-try-this-to-explain-your-findings-b053007d680a

#https://stackoverflow.com/questions/10376974/opencv-clustering-on-more-than-4-channels


@author: Enx
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

#%% Clustering 1
def clustering_cv2_k(image, k=3):
    reshaped_image= np.float32(image.reshape(-1, image.shape[2]))

    stopCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)

    ret, labels, clusters = cv2.kmeans(reshaped_image, k, None, stopCriteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    clusters = np.uint8(clusters)

    intermediateImage = clusters[labels.flatten()]
    clusteredImage = intermediateImage.reshape((image.shape))

    #cv2.imwrite("clusteredImage.jpg", clusteredImage)
    #plot_img_4hist_rgb(clusteredImage)
    return clusteredImage




#%% Clustering2
def clustering_k(img, K=5):
    # Use KMeans clustering algorithm from sklearn.cluster to cluster pixels in image
    from sklearn.cluster import KMeans
    image_2D = np.float32(img.reshape(img.shape[0]*img.shape[1], img.shape[2]))
    # tweak the cluster size and see what happens to the Output
    kmeans = KMeans(n_clusters=K, random_state=0).fit(image_2D)
    clustered = kmeans.cluster_centers_[kmeans.labels_].astype('uint8')

    # Reshape back the image from 2D to 3D image
    clustered_3D = clustered.reshape(img.shape[0], img.shape[1], img.shape[2])

    #plt.imshow(clustered_3D)
    #plt.title('Clustered Image')
    #plt.show()
    #plot_img_4hist_rgb(clustered_3D)
    return clustered_3D





#DBscan 
#https://stackoverflow.com/questions/40142835/image-not-segmenting-properly-using-dbscan
#https://core.ac.uk/download/pdf/79492015.pdf


#https://github.com/s4lv4ti0n/clusteringAlgorithms






# %%
def clear_all():
    """Clears all the variables from the workspace of the application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]





if __name__ == "__main__":
    clear_all()

    from well_plate_project.config import data_dir
    image_file_name = 'IMG_20201118_082858.jpg' # f1_a_cropped a2_a_cropped
    
    path = data_dir / 'raw' / 'exp_v2' / 'luce_ult' /'01'
    image_file = path /  image_file_name
    assert image_file.is_file()
    original_image = cv2.imread(str(image_file))
    plt.imshow(original_image)
    plt.show()
    

    out_c=clustering_cv2_k(original_image, k=2)
    
    plt.imshow(out_c)
    plt.show()
    
    
    #%%
    out=out_c.copy()
    grayImage = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    plt.imshow(grayImage)
    plt.show()
    
    _,threshold = cv2.threshold(grayImage, 100, 255,  cv2.THRESH_BINARY) 
    
    plt.imshow(threshold)
    plt.show()
    
    contours = cv2.findContours(threshold.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    
    for c in contours:
        # get the bounding rect
        area = cv2.contourArea(c); 
        if  area > 0.1*(out.shape[0]*out.shape[1]):
            # print(area/(img.shape[0]*img.shape[1]))
            x, y, w, h = cv2.boundingRect(c)
        
            # get the min area rect
            rect = cv2.minAreaRect(c); box = cv2.boxPoints(rect)
            # convert all coordinates floating point values to int
            box = np.int0(box)
            # draw a red 'nghien' rectangle
            cv2.drawContours(out, [box], 0, (0, 255, 0), 5)
            print(len(contours))
            cv2.drawContours(out, c, -1, (255, 0, 0), 10)
            
    # get width and height of the detected rectangle
    width = int(rect[1][1]); height = int(rect[1][0])
    
    src_pts = box.astype("float32")
    
    # coordinate of the points in box points after the rectangle has been straightened
    # dst_pts = np.array([[0, height-1],
    #                     [0, 0],
    #                     [width-1, 0],
    #                     [width-1, height-1]], dtype="float32")
    perc_h =height*0.075
    perc_w = width*0.075
    dst_pts = np.array([[width-perc_w, height-perc_h],
                        [perc_w, height-perc_h],
                        [perc_w, perc_h],
                        [width-perc_w, perc_h]], dtype="float32")
    
    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(original_image, M, (width, height))
    if width<height:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    
    # if warped.shape[1]>warped.shape[0]:
    #     warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    # img=img[y:y+h,x:x+w]
    
    plt.figure(figsize=(10,10))
    plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    
    
    
    
    plt.figure(figsize=(10,10))
    plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), cmap = 'gray', interpolation = 'bicubic') #interpolation = None !!!
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    
    
    
    
    
    
    
    
    print("Done")

# %%
