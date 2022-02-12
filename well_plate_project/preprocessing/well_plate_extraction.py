# -*- coding: utf-8 -*-
"""


@author: enzo & fabio
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np




#%% Keypoints extraction
def calculate_keypoints(img_single_channel, method='sift', graphics=False, save_name = '' ): 
    """
    Gray or single channel input

    https://pysource.com/2018/03/21/feature-detection-sift-surf-obr-opencv-3-4-with-python-3-tutorial-25/

    """        
    #TODO ADD surf https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_table_of_contents_feature2d/py_table_of_contents_feature2d.html
    
    if method=='sift':
    # SIFT
        sift = cv2.SIFT_create(edgeThreshold = 21, sigma = 1.2) #edgeThreshold = 21, sigma = 1.2 #SIFT (Scale-Invariant Feature Transform)
        keypoints_sift, descriptors_sift = sift.detectAndCompute(img_single_channel, None)
        img_sift = cv2.drawKeypoints(img_single_channel, keypoints_sift, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        if graphics == True:
            plt.figure(), plt.imshow(img_sift), plt.title("SIFT"), plt.show()
            if save_name != '':
                cv2.imwrite(save_name, img_sift)
        keypoints = keypoints_sift; descriptors = descriptors_sift
    
    elif method=='orb':
    # ORB
        orb = cv2.ORB_create(nfeatures=3000)
        keypoints_orb, descriptors_orb = orb.detectAndCompute(img_single_channel, None)
        img_orb = cv2.drawKeypoints(img_single_channel, keypoints_orb, None, color=(0, 255, 0), flags=0)
        if graphics == True:
            plt.figure(), plt.imshow(img_orb), plt.title("ORB"), plt.show()
        keypoints = keypoints_orb; descriptors = descriptors_orb
    
    elif method=='fast':
    # FAST
        fast = cv2.FastFeatureDetector_create() #FAST algorithm for corner detection 
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create() 
        keypoints_fast = fast.detect(img_single_channel, None)
        keypoints_brief, descriptors_brief = brief.compute(img_single_channel, keypoints_fast)  
        if graphics == True:
            print(len(keypoints_fast), len(keypoints_brief))
            img_fast = cv2.drawKeypoints(img_single_channel, keypoints_fast, None, color=(255, 0, 0))
            img_brief = cv2.drawKeypoints(img_single_channel, keypoints_brief, None, color=(255, 0, 0))   
            plt.figure(), plt.imshow(img_fast), plt.title("Detected FAST keypoints"), plt.show()
            plt.figure(), plt.imshow(img_brief), plt.title("Detected BRIEF keypoints"), plt.show()
        keypoints =keypoints_brief; descriptors = descriptors_brief
        
    elif method=='star':
    # STAR-BRIEF
        star = cv2.xfeatures2d.StarDetector_create() ## only feature
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create() # only descript, NO feature
        keypoints_star = star.detect(img_single_channel, None)
        keypoints_brief, descriptors_brief = brief.compute(img_single_channel, keypoints_star)
        if graphics == True:
            print(len(keypoints_star), len(keypoints_brief))
            img_star = cv2.drawKeypoints(img_single_channel, keypoints_star, None, color=(255, 0, 0))
            img_brief = cv2.drawKeypoints(img_single_channel, keypoints_brief, None, color=(255, 0, 0))    
            plt.figure(), plt.imshow(img_star), plt.title("Detected STAR keypoints"), plt.show()
            plt.figure(), plt.imshow(img_brief), plt.title("Detected BRIEF keypoints"), plt.show()
        keypoints = keypoints_brief; descriptors = descriptors_brief
    return keypoints, descriptors, method


#%% Feature Matching
def  feature_matching(descriptions_query, descriptions_train, matching_type = 'flann', lowe_threshold = 0.7, methods = None ):
    #Note: it is not symmetric
    
    #TODO indagare la dipendenza tra metodi e norme (nel caso parte il FRA!)
    try:
       if methods[0] == 'sift' or methods[0] == 'surf':
           norm = cv2.NORM_L2
       else:
           norm = cv2.NORM_HAMMING
    except:
        norm = cv2.NORM_L1
        
    
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
        bf = cv2.BFMatcher(norm, crossCheck=True) #brute force NORM_HAMMING  NORM_L1
        matches = bf.match(descriptions_query, descriptions_train) 
        # store all the good matches as per Lowe's ratio test.
        good_matches = []
        for i, m in enumerate(matches):
            if i < len(matches) - 1 and m.distance < lowe_threshold * matches[i+1].distance:
                good_matches.append(m)
                
    elif matching_type == 'bruteforce':
        # matcher takes normType, which is set to cv2.NORM_L2 for SIFT and SURF, cv2.NORM_HAMMING for ORB, FAST and BRIEF
        bf = cv2.BFMatcher(norm, crossCheck=True) #brute force NORM_HAMMING  NORM_L1
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
    MIN_MATCH_COUNT = 7 #set a condition that atleast 10 matches (defined by MIN_MATCH_COUNT) are to be there to find the object. 
    #Otherwise simply show a message saying not enough matches are present.

    if len(good_matches)>MIN_MATCH_COUNT: #MIN_MATCHES
        # maintaining list of index of descriptors 
        src_points = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
        dst_points = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

        # finding  perspective transformation 
        # between two planes 
        matrix, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC,3.0) #Changed fomr 5  to 3
        M, mask_reverse = cv2.findHomography(dst_points, src_points, cv2.RANSAC, 3.0) #Changed fomr 5  to 3
        #ransacReprojThreshold:
        #dst_points coordinates are measured in pixels with pixel-accurate precision,
        #it makes sense to set this parameter somewhere in the range 1-3
        # print(matrix.shape)
        #print(mask == mask_reverse)
        
    else:
        print(f"Not enough matches are found - {len(good_matches)} / {MIN_MATCH_COUNT}")
        matrix = None
        M= None
    
    return matrix, M, mask






