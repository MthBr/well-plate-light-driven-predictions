# -*- coding: utf-8 -*-
"""
Created on  Nov 20 2020

Ideas

https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html


http://www.ens-lyon.fr/LIP/Arenaire/ERVision/camera_geometry_alignment_final.pdf


https://blog.francium.tech/feature-detection-and-matching-with-opencv-5fd2394a590




@author: modal
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
print(f"CV2 Version: {cv2.__version__}")

#%% Detector1
def show_keypoints(img, gray): #

    sift = cv2.SIFT_create()

    keypoints, descriptors = sift.detectAndCompute(img,None)
    img_points = cv2.drawKeypoints(gray,keypoints,img)
    plt.imshow(img_points), plt.show()

    ## Use This or the one below, One at a time
    #img=cv2.drawKeypoints(img,keypoints,outImage = None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img=cv2.drawKeypoints(img, keypoints, outImage = None, color=(255,0,0))
    plt.imshow(img),
    plt.show()
    return img


#%% Detector1
def compare_features_detect(img, gray): 
    from random import shuffle
    from ssc import ssc
    """
    Gray or single channel input

    https://pysource.com/2018/03/21/feature-detection-sift-surf-obr-opencv-3-4-with-python-3-tutorial-25/

    """
    sift = cv2.SIFT_create() #SIFT (Scale-Invariant Feature Transform)
    #surf = cv2.xfeatures2d.SURF_create() #TODO CMake # SURF (Speeded-Up Robust Features)
    orb = cv2.ORB_create(nfeatures=3000)
   
    
    fast = cv2.FastFeatureDetector_create() #FAST algorithm for corner detection 
    star = cv2.xfeatures2d.StarDetector_create() ## only feature
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create() # only descript, NO feature
    
    
    keypoints_sift, descriptors_sift = sift.detectAndCompute(gray, None)
    #keypoints_surf, descriptors_surf = surf.detectAndCompute(img, None)
    keypoints_orb, descriptors_orb = orb.detectAndCompute(gray, None)
    
    keypoints_fast = fast.detect(gray, None)
    keypoints_star = star.detect(gray, None)
    keypoints_brief, des = brief.compute(gray, keypoints_star)

    # keypoints should be sorted by strength in descending order before feeding to SSC to work correctly
    shuffle(keypoints_fast)  # simulating sorting by score with random shuffle
    selected_keypoints = ssc(keypoints_fast, num_ret_points=1000 , tolerance=0.1 , cols=img.shape[1], rows=img.shape[0])


    img_sift = cv2.drawKeypoints(gray, keypoints_sift, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #img_surf = cv2.drawKeypoints(gray, keypoints_surf, None)
    img_orb = cv2.drawKeypoints(gray, keypoints_orb, None, color=(0, 255, 0), flags=0)
    img_fast = cv2.drawKeypoints(gray, keypoints_fast, None, color=(255, 0, 0))
    img_star = cv2.drawKeypoints(gray, keypoints_star, None, color=(255, 0, 0))
    img_brief = cv2.drawKeypoints(gray, keypoints_brief, None, color=(255, 0, 0))
    img_ssc = cv2.drawKeypoints(gray, selected_keypoints, outImage = None , color=(255, 0, 0))

    plt.figure(), plt.imshow(img_sift), plt.title("SIFT"), plt.show()

    #cv2.imshow("SURF", img_surf)

    plt.figure(), plt.imshow(img_orb), plt.title("ORB"), plt.show()
    
    plt.figure(), plt.imshow(img_star), plt.title("Detected STAR keypoints"), plt.show()
    plt.figure(), plt.imshow(img_brief), plt.title("Detected BRIEF keypoints"), plt.show()

    plt.figure(), plt.imshow(img_fast), plt.title("Detected FAST keypoints"), plt.show()

    plt.figure(), plt.imshow(img_ssc), plt.title("SSC - Selected keypoints"), plt.show()


    del sift
    del orb
    del fast
    return True






#%% Detector1
def detector_sift_flann(queryImage, trainImage): #
    """
    img1 = cv2.imread('box.png',0)          # queryImage
    img2 = cv2.imread('box_in_scene.png',0) # trainImage
    find SIFT features in images and apply the ratio test to find the best matches.
    """

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(queryImage,None)
    kp2, des2 = sift.detectAndCompute(trainImage,None)

    #feature matching
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good_source = []
    good_target = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good_source.append(m)
            good_target.append(n)

    return good_source, kp1, kp2


def detector_sift_bf(queryImage, trainImage): #
    """
    img1 = cv2.imread('box.png',0)          # queryImage
    img2 = cv2.imread('box_in_scene.png',0) # trainImage
    find SIFT features in images and apply the ratio test to find the best matches.
    """

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    keypoints_1, descriptors_1 = sift.detectAndCompute(queryImage,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(trainImage,None)


    #feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True) #brute force NORM_HAMMING
    matches = bf.match(descriptors_1, descriptors_2)

    img3 = cv2.drawMatches(queryImage, keypoints_1, trainImage, keypoints_2, matches[:50], trainImage, flags=2)
    plt.imshow(img3),plt.show()

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m in matches:
        if m.distance < 0.7:
            good.append(m)
        
    return good, keypoints_1, keypoints_2


def detector_orb_bf(queryImage, trainImage): #
    """
    img1 = cv2.imread('box.png',0)          # queryImage
    img2 = cv2.imread('box_in_scene.png',0) # trainImage
    find SIFT features in images and apply the ratio test to find the best matches.
    """

    orb = cv2.ORB_create()

    # find the keypoints and descriptors with SIFT
    keypoints_1, descriptors_1 = orb.detectAndCompute(queryImage,None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(trainImage,None)


    #feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True) #brute force NORM_HAMMING
    matches = bf.match(descriptors_1, descriptors_2)

    img3 = cv2.drawMatches(queryImage, keypoints_1, trainImage, keypoints_2, matches[:50], flags=2)
    plt.imshow(img3),plt.show()

    # store all the good matches as per Lowe's ratio test.
    good = []
    for i, m in enumerate(matches):
        if i < len(matches) - 1 and m.distance < 0.7 * matches[i+1].distance:
            good.append(m)
    
    return good, keypoints_1, keypoints_2

def detector_orb(queryImage, trainImage): #
    # Initiate SIFT detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(queryImage,None)
    kp2, des2 = orb.detectAndCompute(trainImage,None)

    # create BFMatcher object
    # matcher takes normType, which is set to cv2.NORM_L2 for SIFT and SURF, cv2.NORM_HAMMING for ORB, FAST and BRIEF
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) #brute force match NORM_L1
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], flags=2)
    plt.imshow(img3),plt.show()

    return matches, kp1, kp2

def detector_orb_flann(queryImage, trainImage): #

    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(queryImage, None)
    kp2, des2 = orb.detectAndCompute(trainImage, None)

    index_params = dict(algorithm=6,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=2)
    search_params = {}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # As per Lowe's ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return good_matches



#%% Extract Transform
def extract_and_transform(good_source_matches, good_target, kp1, kp2, img1, img2): #
    """
    If enough matches are found, we extract the locations of matched keypoints in both the images.
    They are passed to find the perpective transformation. Once we get this 3x3 transformation matrix, 
    we use it to transform the corners of queryImage to corresponding points in trainImage. 
    Then we draw it.
    """

    MIN_MATCHES = 50
    MIN_MATCH_COUNT = 10 #set a condition that atleast 10 matches (defined by MIN_MATCH_COUNT) are to be there to find the object. 
    #Otherwise simply show a message saying not enough matches are present.

    if len(good_source_matches)>MIN_MATCH_COUNT: #MIN_MATCHES
        # maintaining list of index of descriptors 
        src_points = np.float32([ kp1[m.queryIdx].pt for m in good_source_matches ]).reshape(-1,1,2)
        dst_points = np.float32([ kp2[m.trainIdx].pt for m in good_source_matches ]).reshape(-1,1,2)

        # finding  perspective transformation 
        # between two planes 
        matrix, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC,5.0)
        M, mask_reverse = cv2.findHomography(dst_points, src_points, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist() # ravel function returns  - contiguous flattened array 
        print(matrix)
        
        
        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,matrix) # applying perspective algorithm : destination points

        #M = cv2.getPerspectiveTransform(dst,pts)
        corrected_img = cv2.warpPerspective(img2, M, (img1.shape[1], img1.shape[0]), cv2.WARP_INVERSE_MAP)
    else:
        print(f"Not enough matches are found - {len(good)} / {MIN_MATCH_COUNT}")
        matches_mask = None
        dst=None
        corrected_img= None
    
    return matches_mask, dst, corrected_img


#%% Draw Matches
def draw_inliers(matches_mask, img1,kp1,img2,kp2,good):
    """
    draw our inliers (if successfully found the object) or matching keypoints (if failed).
    """
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matches_mask, # draw only inliers
                   flags = 2)

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

    plt.imshow(img3, 'gray'),plt.show()
    return img3

#%% Pre process
def single_channel_laplac(input_image):    
    #Gaussian Filter
    denoised = cv2.GaussianBlur(input_image, (7,7), 5);   
    laplacian = cv2.Laplacian(denoised,cv2.CV_64F)
    #sobelx = cv2.Sobel(img,cv.CV_64F,1,0,ksize=3)  # x
    #sobely = cv2.Sobel(img,cv.CV_64F,0,1,ksize=3)  # y
    return laplacian

def single_channel_gray(BRG_input_image):
    gray_image = cv2.cvtColor(BRG_input_image, cv2.COLOR_BGR2GRAY) 
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_equlized = clahe.apply(gray_image)
    return gray_equlized


def feture_extraction(img):
    single_chan = single_channel_gray(img)
 
    fast = cv2.FastFeatureDetector_create() #FAST algorithm for corner detection 
    #star = cv2.xfeatures2d.StarDetector_create() ## only feature
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create() # only descript, NO feature
    
    keypoints_fast = fast.detect(single_chan, None)
    #keypoints_star = star.detect(single_chan, None)
    keypoints, descriptors = brief.compute(single_chan, keypoints_fast)
    return keypoints, descriptors
    


# %%
def clear_all():
    """Clears all the variables from the workspace of the application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]


def load_file(image_file_name, data_dir): 
    
    image_file = path /  image_file_name
    assert image_file.is_file()
    img = cv2.imread(str(image_file))
    return img


if __name__ == "__main__":
    clear_all()
    from well_plate_project.config import data_dir

    path = data_dir / 'raw' / 'Cropped'
    queryImage = load_file('bianc_cropped.jpg', path)
    
    
    lap_query = single_channel_laplac(queryImage)
    gray_query = single_channel_gray(queryImage)


    path = data_dir / 'raw' / 'foto_tel1' #foto_tel1
    trainImage = load_file('IMG_20201118_081801.jpg', path)    
    lap_train = single_channel_laplac(queryImage)
    gray_train = single_channel_gray(queryImage)
    

    
    
    #compare_features_detect(queryImage, gray_queryImage)
    #compare_features_detect(trainImage, gray_trainImage)

    #good_source, good_target, kp1, kp2 = detector_sift_flann(gray_queryImage, gray_trainImage) #detector_sift_flann  detector_sift_bf

    #good, kp1, kp2 = detector_orb(gray_queryImage, gray_trainImage)

    #matchesMask, dst, corrected_img = extract_and_transform(good_source, good_target, kp1, kp2, gray_queryImage, gray_trainImage)
    #homography = cv2.polylines(trainImage,[np.int32(dst)], True, (0,0,255), 10, cv2.LINE_AA)
    #plt.imshow(homography) , plt.show()

    #plt.imshow(corrected_img), plt.show()


    #test_image = draw_inliers(matchesMask, queryImage,kp1, trainImage,kp2, good_source)

    print("Done")
