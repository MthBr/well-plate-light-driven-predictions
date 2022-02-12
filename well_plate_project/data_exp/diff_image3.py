#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 16:31:49 2020

@author: enzo
"""
#%%
import cv2
import matplotlib.pyplot as plt
import numpy as np
from well_plate_project.config import data_dir

def single_channel_gray(BRG_input_image):
    gray_image = cv2.cvtColor(BRG_input_image, cv2.COLOR_BGR2GRAY) 
    #gray_equlized = cv2.equalizeHist(gray_image)
    #clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4,3))
    #gray_equlized = clahe.apply(gray_image)
    gray_equlized = gray_image
    return gray_equlized

def calculate_keypoints(img, method, single_channel, graphics=False): 
    """
    Gray or single channel input

    https://pysource.com/2018/03/21/feature-detection-sift-surf-obr-opencv-3-4-with-python-3-tutorial-25/

    """
    
    if single_channel=='gray':
        img_single_channel = single_channel_gray(img)
        
        
    print(img_single_channel.shape, type(img_single_channel), img_single_channel.dtype)
    
    #TODO ADD surf https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_table_of_contents_feature2d/py_table_of_contents_feature2d.html
    
    if method=='sift':
    # SIFT
        sift = cv2.SIFT_create(edgeThreshold = 21, sigma = 1.2) #edgeThreshold = 21, sigma = 1.2 #SIFT (Scale-Invariant Feature Transform)
        keypoints_sift, descriptors_sift = sift.detectAndCompute(img_single_channel, None)
        img_sift = cv2.drawKeypoints(img_single_channel, keypoints_sift, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        if graphics == True:
            plt.figure(), plt.imshow(img_sift), plt.title("SIFT"), plt.show()
        return keypoints_sift, descriptors_sift 
    
    elif method=='orb':
    # ORB
        orb = cv2.ORB_create(nfeatures=3000)
        keypoints_orb, descriptors_orb = orb.detectAndCompute(img_single_channel, None)
        img_orb = cv2.drawKeypoints(img_single_channel, keypoints_orb, None, color=(0, 255, 0), flags=0)
        if graphics == True:
            plt.figure(), plt.imshow(img_orb), plt.title("ORB"), plt.show()
        return keypoints_orb, descriptors_orb
    
    elif method=='fast':
    # FAST
        fast = cv2.FastFeatureDetector_create() #FAST algorithm for corner detection 
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create() 
        keypoints_fast = fast.detect(img_single_channel, None)
        keypoints_brief, descriptors_brief = brief.compute(img_single_channel, keypoints_fast)  
        print(len(keypoints_fast), len(keypoints_brief))

        if graphics == True:
            img_fast = cv2.drawKeypoints(img_single_channel, keypoints_fast, None, color=(255, 0, 0))
            img_brief = cv2.drawKeypoints(img_single_channel, keypoints_brief, None, color=(255, 0, 0))   
            plt.figure(), plt.imshow(img_fast), plt.title("Detected FAST keypoints"), plt.show()
            plt.figure(), plt.imshow(img_brief), plt.title("Detected BRIEF keypoints"), plt.show()
        return keypoints_brief, descriptors_brief
        
    elif method=='star':
    # STAR-BRIEF
        star = cv2.xfeatures2d.StarDetector_create() ## only feature
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create() # only descript, NO feature
        keypoints_star = star.detect(img_single_channel, None)
        keypoints_brief, descriptors_brief = brief.compute(img_single_channel, keypoints_star)
        print(len(keypoints_star), len(keypoints_brief))
        if graphics == True:
            img_star = cv2.drawKeypoints(img_single_channel, keypoints_star, None, color=(255, 0, 0))
            img_brief = cv2.drawKeypoints(img_single_channel, keypoints_brief, None, color=(255, 0, 0))    
            plt.figure(), plt.imshow(img_star), plt.title("Detected STAR keypoints"), plt.show()
            plt.figure(), plt.imshow(img_brief), plt.title("Detected BRIEF keypoints"), plt.show()
        return keypoints_brief, descriptors_brief
    return 0


def  feature_matching(descriptions_query, descriptions_train, matching_type, lowe_threshold = 0.75):
    
    if matching_type == 'flann':    
        descriptions_query = descriptions_query.astype('float32')
        descriptions_train = descriptions_train.astype('float32')
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 7) #5
        search_params = dict(checks = 70)   # 50
        flann = cv2.FlannBasedMatcher(index_params, search_params)   
        matches = flann.knnMatch(descriptions_query,descriptions_train,k=2)  
        # store all the good matches as per Lowe's ratio test.
        good_matches = []
        for m,n in matches:
            if m.distance < lowe_threshold*n.distance:
                good_matches.append(m)
    elif matching_type == 'bfhamm':
        descriptions_query = descriptions_query.astype('float32')
        descriptions_train = descriptions_train.astype('float32')
        # matcher takes normType, which is set to cv2.NORM_L2 for SIFT and SURF, cv2.NORM_HAMMING for ORB, FAST and BRIEF
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) #brute force NORM_HAMMING  NORM_L1
        matches = bf.match(descriptions_query, descriptions_train) 
        # store all the good matches as per Lowe's ratio test.
        good_matches = []
        for i, m in enumerate(matches):
            if i < len(matches) - 1 and m.distance < lowe_threshold * matches[i+1].distance:
                good_matches.append(m)
                
    elif matching_type == 'bruteforce':
        # matcher takes normType, which is set to cv2.NORM_L2 for SIFT and SURF, cv2.NORM_HAMMING for ORB, FAST and BRIEF
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True) #brute force NORM_HAMMING  NORM_L1
        matches = bf.match(descriptions_query, descriptions_train) 
        # store all the good matches as per Lowe's ratio test.
        good_matches = []
        for i, m in enumerate(matches):
            if i < len(matches) - 1 and m.distance < lowe_threshold * matches[i+1].distance:
                good_matches.append(m)
    
    return good_matches



#%% Extract Transform
def extract_matrix(good_matches, kp1, kp2, img1, img2): #
    """
    If enough matches are found, we extract the locations of matched keypoints in both the images.
    They are passed to find the perpective transformation. Once we get this 3x3 transformation matrix, 
    we use it to transform the corners of queryImage to corresponding points in trainImage. 
    Then we draw it.
    """

    MIN_MATCHES = 50
    MIN_MATCH_COUNT = 10 #set a condition that atleast 10 matches (defined by MIN_MATCH_COUNT) are to be there to find the object. 
    #Otherwise simply show a message saying not enough matches are present.

    if len(good_matches)>MIN_MATCH_COUNT: #MIN_MATCHES
        # maintaining list of index of descriptors 
        src_points = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
        dst_points = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

        # finding  perspective transformation 
        # between two planes 
        matrix, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC,5.0)
        M, mask_reverse = cv2.findHomography(dst_points, src_points, cv2.RANSAC, 5.0)
        # print(matrix.shape)
        #print(mask == mask_reverse)
        
    else:
        print(f"Not enough matches are found - {len(good_matches)} / {MIN_MATCH_COUNT}")
        matrix = None
        M= None
    
    return matrix, M, mask

#%% Draw Matches
def draw_inliers(mask, img1,kp1,img2,kp2,good):
    """
    draw our inliers (if successfully found the object) or matching keypoints (if failed).
    """
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = mask.ravel().tolist(), # draw only inliers# ravel function returns  - contiguous flattened array 
                   flags = 2)

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

    #plt.imshow(img3, 'gray'),plt.show()
    return img3
#%%
path_query =  data_dir / 'raw' / 'Match'
image_file = path_query / '1ln.png'
assert image_file.is_file()
queryImage = cv2.imread(str(image_file)) #aswana_cropped_2 aswana_cropped
#plt.imshow(queryImage); plt.show()
keypoints_query, descriptions_query = calculate_keypoints(queryImage, 'fast', 'gray', graphics=True)



#%%
keypoints_train, descriptions_train = calculate_keypoints(queryImage, 'fast', 'gray', graphics=True)
matches = feature_matching(descriptions_query, descriptions_train, matching_type='flann', lowe_threshold=0.7)
matrix, M, matrix_mask = extract_matrix(matches, keypoints_query, keypoints_train, queryImage, queryImage)

plt.figure(figsize=(10,10))
plt.imshow(queryImage);plt.show()


h,w= queryImage.shape[0:2]
pts_image = np.array([[0,0],[0,h-1],[w-1,h-1],[w-1,0]], dtype='float32')
pts_image = np.array([pts_image])
dst_image = cv2.perspectiveTransform(pts_image,matrix) # applying perspective algorithm : destination points
corrected_img = cv2.warpPerspective(queryImage, M, (queryImage.shape[1], queryImage.shape[0]), cv2.WARP_INVERSE_MAP)


queryImage_copy = queryImage.copy()
trainImage = queryImage.copy()
homography = cv2.polylines(queryImage_copy,[np.int32(dst_image)], True, (0,0,255), 10, cv2.LINE_AA)
plt.figure(figsize=(10,10)), plt.imshow(homography) , plt.show()
plt.figure(figsize=(10,10)), plt.imshow(corrected_img), plt.show()

#%%
compare_image = draw_inliers(matrix_mask, queryImage_copy, keypoints_query, trainImage, keypoints_train, matches)
plt.figure(figsize=(10,10)),plt.imshow(compare_image), plt.show()


#%%
plt.close('all')

#%%
path_train = data_dir / 'raw' / 'exp_v2_crp (t1)' / 'luce_nat'  #foto_tel1  EXPERIMENTS foto_tel1
jpg_path = path_train / '10' / '20201118_090359_cropped.jpg' #20201118_090416   IMG_20201118_090440
good = cv2.imread(str(jpg_path)) #aswana_cropped_2 aswana_cropped
jpg_good_2 = path_train / '10' / '20201118_090420_cropped.jpg'
good_2 = cv2.imread(str(jpg_good_2))
good = good_2

keypoints_train, descriptions_train = calculate_keypoints(good, 'fast', 'gray', graphics=True)
matches = feature_matching(descriptions_query, descriptions_train, matching_type='flann', lowe_threshold=0.71)
matrix, M, matrix_mask = extract_matrix(matches, keypoints_query, keypoints_train, queryImage, good)


plt.figure(figsize=(10,10))
plt.imshow(good);plt.show()

plt.figure(figsize=(10,10))
plt.imshow(queryImage);plt.show()

h,w= queryImage.shape[0:2]
pts_image = np.array([[0,0],[0,h-1],[w-1,h-1],[w-1,0]], dtype='float32')
pts_image = np.array([pts_image])
dst_image = cv2.perspectiveTransform(pts_image,matrix) # applying perspective algorithm : destination points
corrected_img = cv2.warpPerspective(good, M, (queryImage.shape[1], queryImage.shape[0]), cv2.WARP_INVERSE_MAP)


good_copy = good.copy()
homography = cv2.polylines(good_copy,[np.int32(dst_image)], True, (0,0,255), 10, cv2.LINE_AA)
plt.figure(figsize=(10,10)), plt.imshow(homography) , plt.show()
plt.figure(figsize=(10,10)), plt.imshow(corrected_img), plt.show()

compare_image = draw_inliers(matrix_mask, queryImage, keypoints_query, good_copy, keypoints_train, matches)
plt.figure(figsize=(10,10)), plt.imshow(compare_image), plt.show()
            
            
#%%
jpg_bad = path_train / '10' / '20201118_090416_cropped.jpg' #20201118_090416   IMG_20201118_090440
bad = cv2.imread(str(jpg_bad)) #aswana_cropped_2 aswana_cropped






















plt.figure(figsize=(10,10))
plt.imshow(lap_orig);plt.show()

lap_good = single_channel_gray(good)
plt.figure(figsize=(10,10))
plt.imshow(lap_good);plt.show()

flipHorizontal = cv2.flip(lap_good, 1)




lap_bad = single_channel_gray(bad)
plt.figure(figsize=(10,10))
plt.imshow(lap_bad);plt.show()






import numpy as np
diff_lap_good = np.linalg.norm(lap_good -lap_orig)/(lap_good.shape[0]*lap_good.shape[1]) #np.inf  'fro'
diff_orig = np.linalg.norm(lap_orig -lap_orig)/(lap_orig.shape[0]*lap_orig.shape[1]) #np.inf  'fro'
diff_lap_bad = np.linalg.norm(lap_bad -lap_orig)/(lap_bad.shape[0]*lap_bad.shape[1]) #np.inf  'fro'


jpg_good_2 = path_train / '10' / '20201118_090420_cropped.jpg'
good_2 = cv2.imread(str(jpg_good_2))
lap_good_2 = single_channel_gray(good_2)

plt.figure(figsize=(10,10))
plt.imshow(cv2.flip(lap_good_2, 1));plt.show()
plt.figure(figsize=(10,10))
plt.imshow(lap_good_2 -cv2.flip(lap_good_2, 1));plt.show()
diff_lap_good_2 = np.linalg.norm(lap_good_2 -lap_orig)/(lap_good_2.shape[0]*lap_good_2.shape[1]) #np.inf  'fro' ord ='fro'


compare_edge_images(queryImage, good_2)


jpg_good_3 = path_train / '10' / '20201118_090422_cropped.jpg'
good_3 = cv2.imread(str(jpg_good_3))
lap_good_3 = single_channel_gray(good_3)
plt.figure(figsize=(10,10))
plt.imshow(lap_good_3);plt.show()
diff_lap_good_3 = np.linalg.norm(lap_good_3 -lap_orig)/(lap_good_2.shape[0]*lap_good_2.shape[1])  #np.inf  'fro'




jpg_bad_2 = path_train / '10' / 'IMG_20201118_090440_cropped.jpg'
bad_2 = cv2.imread(str(jpg_bad_2))
plt.figure(figsize=(10,10))
plt.imshow(bad_2);plt.show()
lap_bad_2 = single_channel_gray(bad_2)
plt.figure(figsize=(10,10))
plt.imshow(lap_bad_2);plt.show()
diff_lap_bad_2 = np.linalg.norm(lap_bad_2 -lap_orig)/(lap_good_2.shape[0]*lap_good_2.shape[1])  #np.inf  'fro'



plt.figure(figsize=(10,10))
plt.imshow(lap_bad_2);plt.show()
plt.figure(figsize=(10,10))
plt.imshow(cv2.flip(lap_bad_2, 1));plt.show()
plt.figure(figsize=(10,10))
plt.imshow(lap_bad_2 -cv2.flip(lap_bad_2, 1));plt.show()