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
def calculate_keypoints(img, method, single_channel, graphics=False): 
    """
    Gray or single channel input

    https://pysource.com/2018/03/21/feature-detection-sift-surf-obr-opencv-3-4-with-python-3-tutorial-25/

    """
    
    if single_channel=='gray':
        img_single_channel = single_channel_gray(img)
    
    elif single_channel=='laplacian':
        img_single_channel = compute_laplac(img)
    elif single_channel=='color':
        img_single_channel = clahe(img)
    elif single_channel=='HSV':
        img_single_channel = HSV(img)
    elif single_channel=='hog':
        img_single_channel = hog(img)
    elif single_channel=='mixed':
        img_single_channel = mixed(img)
        
        
    print(img_single_channel.shape, type(img_single_channel), img_single_channel.dtype)
    
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
def extract_and_transform(good_matches, kp1, kp2, img1, img2): #
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
        
        
        h,w= img1.shape[0:2]
        #pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ])
        pts = np.array([[0,0],[0,h-1],[w-1,h-1],[w-1,0]], dtype='float32')
        pts = np.array([pts])
        dst = cv2.perspectiveTransform(pts,matrix) # applying perspective algorithm : destination points
        # print (pts)
        # print(dst)
        #M = cv2.getPerspectiveTransform(dst,pts)
        corrected_img = cv2.warpPerspective(img2, M, (img1.shape[1], img1.shape[0]), cv2.WARP_INVERSE_MAP)
    else:
        print(f"Not enough matches are found - {len(good_matches)} / {MIN_MATCH_COUNT}")
        mask = None
        dst=None
        corrected_img= None
    
    return mask, dst, corrected_img


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

#%% Pre process
def compute_laplac(input_image):   
    input_image = single_channel_gray(input_image)
    #Gaussian Filter
    denoised = cv2.GaussianBlur(input_image, (7,7), 5);   
    laplacian = cv2.Laplacian(denoised,cv2.CV_64F).astype('uint8')
    denoised_laplacian = cv2.GaussianBlur(laplacian, (7,7), 5);   
    #sobelx = cv2.Sobel(img,cv.CV_64F,1,0,ksize=3)  # x
    #sobely = cv2.Sobel(img,cv.CV_64F,0,1,ksize=3)  # y
    return denoised_laplacian

def single_channel_gray(BRG_input_image):
    gray_image = cv2.cvtColor(BRG_input_image, cv2.COLOR_BGR2GRAY) 
    #gray_equlized = cv2.equalizeHist(gray_image)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4,3))
    gray_equlized = clahe.apply(gray_image)
    return gray_equlized

def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.0,tileGridSize=(3,7))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr

def HSV(img):
    img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(img_hsv)
    clahe = cv2.createCLAHE(clipLimit=1.0,tileGridSize=(7,3))
    V = clahe.apply(V)
    return V

def mixed(img_rgb):
    img_hsv=cv2.cvtColor(img_rgb,cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(img_hsv)
    r, g, b = cv2.split(img_rgb)
    clahe = cv2.createCLAHE(clipLimit=1.0,tileGridSize=(7,7))
    V = cv2.cvtColor(clahe.apply(V), cv2.COLOR_GRAY2BGR)
    g = cv2.cvtColor(clahe.apply(g), cv2.COLOR_GRAY2BGR)
    S = cv2.cvtColor(clahe.apply(S), cv2.COLOR_GRAY2BGR)
    a=1/3
    b=1
    c=1/3
    mixed = a*V+b*g+c*S
    return mixed.astype('uint8')
#plt.imshow(mixed(img)), plt.show()
#queryImage

def hog (img):
    from skimage.feature import hog
    from skimage import exposure
    gray = single_channel_gray(img)
    #multic = clahe(img)
    features, hog_image = hog(gray, orientations=9,
                                  pixels_per_cell=(16, 16),
                                  cells_per_block=(1, 1),
                                  visualize=True) #, multichannel=True) 
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 256))
    overlay = hog_image_rescaled.astype('uint8')
    background = HSV(img)
    added_image = cv2.addWeighted(background,0.7,overlay,0.3,0) 
    return added_image

def feture_extraction(img):
    single_chan = single_channel_gray(img)
 
    fast = cv2.FastFeatureDetector_create() #FAST algorithm for corner detection 
    #star = cv2.xfeatures2d.StarDetector_create() ## only feature
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create() # only descript, NO feature
    
    keypoints_fast = fast.detect(single_chan, None)
    #keypoints_star = star.detect(single_chan, None)
    keypoints, descriptors = brief.compute(single_chan, keypoints_fast)
    return keypoints, descriptors
    

def resize_image(img, scale_percent = 70): # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
    return resized



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




def crop_pipeline(queryImage, keypoints_query, descriptions_query, trainImage):
    # Compare features between input image and its laplacian
    keypoints_train, descriptions_train = calculate_keypoints(trainImage, 'sift', 'gray', graphics=False)
    
    #feature matching
    # print(len(keypoints_query), len(keypoints_train))
    # print(len(descriptions_query), len(descriptions_train))
    matches = feature_matching(descriptions_query, descriptions_train, matching_type='flann', lowe_threshold=0.75)
    
    array_dist=[x.distance for x in matches[:1000]]
    
    
    matches_mask, dst, corrected_img = extract_and_transform(matches, keypoints_query,
                                                              keypoints_train, queryImage, trainImage)
    
    #homography = 
    cv2.polylines(trainImage,[np.int32(dst)], True, (0,0,255), 10, cv2.LINE_AA)
    # plt.imshow(homography) , plt.show()
    #plt.imshow(corrected_img), plt.show()
    
    compare_image = draw_inliers(matches_mask, queryImage, keypoints_query, trainImage, keypoints_train, matches)
    return corrected_img, compare_image, array_dist
        

# %% Afer functions


# Import of the query image
path_query = '../../data/data/raw/Match/'
queryImage = cv2.imread(path_query+'aswana_cropped.jpg') #aswana_cropped_2 aswana_cropped
plt.imshow(queryImage); plt.show()
# queryImage = resize_image(queryImage, 100)

# Compare features between input image and its laplacian
keypoints_query, descriptions_query = calculate_keypoints(queryImage, 'sift', 'gray', graphics=False)


# Import of the train image
path_train = '../../data/data/raw/EXPERIMENTS/'  #foto_tel1  EXPERIMENTS foto_tel1

target_folder= '../../data/data/raw/EXPERIMENTS_Crp/' # EXPERIMENTS_Crp  foto_tel1_crp



# trainImage = cv2.imread(path_train+'IMG_20201118_080910.jpg') #  IMG_20201118_080910 IMG_20201118_082825
# #trainImage = resize_image(trainImage, 50)
# corrected_img, compare_image, array_dist = crop_pipeline(queryImage, keypoints_query, descriptions_query, trainImage)
# plt.imshow(corrected_img), plt.show()
# plt.imshow(compare_image), plt.show()

# print(np.sum(array_dist), np.mean(array_dist), np.std(array_dist))



from well_plate_project.data_etl._3a_circle_detection import cropped_hough, plot_circles
from _0_plot_hists import plot_hist_bgr
from pathlib import Path
path_train = Path(path_train)
target_folder = Path(target_folder)
for jpg in path_train.glob('*.jpg'):
    print(f'Processing {jpg}', end='\n')
    trainImage = cv2.imread(str(jpg))
    corrected_img, compare_image, array_dist = crop_pipeline(queryImage, keypoints_query, descriptions_query, trainImage)
    
    target_filename =  jpg.stem + '_cropped' + jpg.suffix
    # target_filename =  jpg.stem + '_y_yk3_ragYlab' + jpg.suffix   #jpg.name
    #'rag_lab_b7_' + 
    target_path = target_folder / target_filename
    cv2.imwrite(str(target_path), corrected_img)
    
    plt.figure(); plt.imshow(corrected_img); plt.show()
    print(np.sum(array_dist), np.mean(array_dist), np.std(array_dist))
    
    target_filename =  jpg.stem + '_cro2cmp' + jpg.suffix
    target_path = target_folder / target_filename   
    cv2.imwrite(str(target_path), compare_image)
    
    
    target_filename =  jpg.stem + '_circ' + jpg.suffix
    circles = cropped_hough(corrected_img)
    circled_image = plot_circles(circles,queryImage)
    target_path = target_folder / target_filename
    cv2.imwrite(str(target_path), circled_image)
    
    
    # hist = plot_hist_bgr(trainImage)
    # target_filename =  jpg.stem + '_hist' + jpg.suffix
    # target_path = target_folder / target_filename   
    # hist.savefig(str(target_path))
    
    



print("Done")































