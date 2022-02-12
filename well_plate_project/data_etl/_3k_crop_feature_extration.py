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


trainImage = cv2.imread(path_train+'20a.jpg') #  IMG_20201118_080910 IMG_20201118_082825
#trainImage = resize_image(trainImage, 50)
#corrected_img, compare_image, array_dist = crop_pipeline(queryImage, keypoints_query, descriptions_query, trainImage)

#%%
l, a, b = cv2.split(cv2.cvtColor(queryImage, cv2.COLOR_BGR2LAB))
image = b
clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4,3))
image = clahe.apply(image)
circles = cv2.HoughCircles(image.astype('uint8'), cv2.HOUGH_GRADIENT, 2.21, 90, 
                           param1=30,param2=90,minRadius=40,maxRadius=45)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(image,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    #cv2.circle(image,(i[0],i[1]),2,(0,0,255),3)


plt.figure(figsize=(10,10))
plt.imshow(image)
plt.xticks([]), plt.yticks([])
plt.show()


#ind = np.lexsort((circles[0,:,0],circles[0,:,1])) 
# ind = np.argsort(circles[0,:, 1])   
# ordered_circles=circles[0,ind]
# matrix=ordered_circles.reshape((8, 12, 3)) # Row of well 8 (A-H) + 12Column (1-12) + 3 axsis (x,y,R)




# resh_circles = circles.reshape((8, 12, 3))
# ind = np.lexsort((resh_circles[:,:,0],resh_circles[:,:,1])) 
# ord_matr_circ = resh_circles[:,:,ind]


ind = np.argsort(circles[0,:, 1])   
ordered_circles=circles[0,ind]
matrix=ordered_circles.T.reshape((3, 8, 12))
#TODO improve code
sidx = matrix[0,:,:].argsort(axis=1)
out=np.zeros(matrix.shape)
out[0,:,:] = np.take_along_axis(matrix[0,:,:], sidx, axis=1)
out[1,:,:] = np.take_along_axis(matrix[1,:,:], sidx, axis=1)
out[2,:,:] = np.take_along_axis(matrix[2,:,:], sidx, axis=1)


out=np.zeros(matrix.shape)
matrix=ordered_circles.reshape((8, 12, 3))
sidx = matrix[:,:,0].argsort(axis=1)
out=np.zeros(matrix.shape)
out[:,:,0] = np.take_along_axis(matrix[:,:,0], sidx, axis=1)
out[:,:,1]= np.take_along_axis(matrix[:,:,1], sidx, axis=1)
out[:,:,2] = np.take_along_axis(matrix[:,:,2], sidx, axis=1)


import numpy.lib.recfunctions as nlr
circles_tuple = nlr.unstructured_to_structured(out).astype('O')

import pandas as pd
df_out = pd.DataFrame(circles_tuple, index = map(chr, range(65, 73)), columns=list(range(1,13)))


height, width = image.shape
mask = np.zeros((height,width), np.uint8)


circles = np.uint16(np.around(circles))
counter = 0
for i in circles[0,:]:
    # Draw on mask
    cv2.circle(mask,(i[0],i[1]),i[2],(255,255,255),-1)
    masked_data = cv2.bitwise_and(image, image, mask=mask)    
    # Apply Threshold
    # _,thresh = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)

    # # Find Contour
    # cnt = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]

    # #print len(contours)
    # x,y,w,h = cv2.boundingRect(cnt[0])

    # # Crop masked_data
    # crop = masked_data[y:y+h,x:x+w]

    # # Write Files
    # cv2.imwrite("output/crop"+str(counter)+".jpg", crop)

    counter +=1

    print(counter)
    


plt.imshow(masked_data);plt.show()

plt.imshow(mask);plt.show()

# x, y, r = circles[0,:][0]
# rows, cols = image.shape

# for i in range(cols):
#     for j in range(rows):
#         if np.linalg.norm([i-x, j-y]) > r:
#             image[j,i] = 0

# #cv2.imwrite("iris.jpg",image)
# plt.imshow(image, cmap = 'gray', interpolation = 'bicubic')
# plt.xticks([]), plt.yticks([])
# plt.show()

#%%
# Compare features between input image and its laplacian
keypoints_train, descriptions_train = calculate_keypoints(trainImage, 'sift', 'gray', graphics=False)

#feature matching
matches = feature_matching(descriptions_query, descriptions_train, matching_type='flann', lowe_threshold=0.75)
array_dist=[x.distance for x in matches[:1000]]


matrix, M, matrix_mask = extract_matrix(matches, keypoints_query, keypoints_train, queryImage, trainImage)



h,w= queryImage.shape[0:2]
pts_image = np.array([[0,0],[0,h-1],[w-1,h-1],[w-1,0]], dtype='float32')
pts_image = np.array([pts_image])
dst_image = cv2.perspectiveTransform(pts_image,matrix) # applying perspective algorithm : destination points
corrected_img = cv2.warpPerspective(trainImage, M, (queryImage.shape[1], queryImage.shape[0]), cv2.WARP_INVERSE_MAP)




homography = cv2.polylines(trainImage,[np.int32(dst_image)], True, (0,0,255), 10, cv2.LINE_AA)
plt.imshow(homography) , plt.show()
plt.imshow(corrected_img), plt.show()

compare_image = draw_inliers(matrix_mask, queryImage, keypoints_query, trainImage, keypoints_train, matches)


plt.imshow(corrected_img), plt.show()
plt.imshow(compare_image), plt.show()

print(np.sum(array_dist), np.mean(array_dist), np.std(array_dist))





corrected_mask = cv2.warpPerspective(mask, matrix, (trainImage.shape[1], trainImage.shape[0]), cv2.WARP_INVERSE_MAP)
plt.imshow(corrected_mask), plt.show()
#%%



height, width = image.shape
mask = np.zeros((height,width), np.uint8)

circles = np.uint16(np.around(circles))
counter = 0

for key, values in df_out.iterrows(): 
    for i, well in enumerate(values):
        print(key+str(i+1)) 
        print(well) 
        # well = well.astype()
        mask = np.zeros((height,width), np.uint8)
        x=int(well[0]); y= int(well[1]); R=int(well[2])
        single_maks = cv2.circle(mask,(x,y),R,(255,255,255),-1)
        corrected_mask = cv2.warpPerspective(single_maks, matrix, (trainImage.shape[1], trainImage.shape[0]), cv2.WARP_INVERSE_MAP)
        #masked_data = cv2.bitwise_and(image, image, mask=mask) 
        plt.imshow(corrected_mask), plt.show()
        #rect = cv2.boundingRect(corrected_mask)               # function that computes the rectangle of interest
        #squares=np.array([[x-R,y-R], [x+R,y-R], [x-R,y+R],[x+R,y+R]], dtype='float32')
        #squares = np.array([squares])
        #dst_squares = cv2.perspectiveTransform(squares,matrix)[0]
        #cropped_img = trainImage[dst_squares[0,0]:dst_squares[1,0], dst_squares[2]:dst_squares[3]]#[rect[0]:(rect[0]+rect[2]), rect[1]:(rect[1]+rect[3])]#
        
        #l_mask = cv2.cvtColor(corrected_mask,cv2.COLOR_GRAY2BGR)#change mask to a 3 channel image 
        #mask_out=cv2.subtract(l_mask,trainImage)
        #mask_out=cv2.subtract(l_mask,mask_out)        
        #plt.imshow(mask_out), plt.show()
        
        result = np.where(corrected_mask == np.amax(corrected_mask))
        minx=min(result[0]); maxx=max(result[0])
        miny=min(result[1]); maxy=max(result[1])
        res = cv2.bitwise_and(trainImage,trainImage,mask = corrected_mask)
        cropped = res[minx:maxx, miny:maxy]
        plt.imshow(res), plt.show()
        plt.imshow(cropped), plt.show()
        
        
        
    
for well in df_out:
    # Draw on mask
    cv2.circle(mask,(i[0],i[1]),i[2],(255,255,255),-1)
    masked_data = cv2.bitwise_and(image, image, mask=mask)    
    # Apply Threshold
    # _,thresh = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)

    # # Find Contour
    # cnt = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]

    # #print len(contours)
    # x,y,w,h = cv2.boundingRect(cnt[0])

    # # Crop masked_data
    # crop = masked_data[y:y+h,x:x+w]

    # # Write Files
    # cv2.imwrite("output/crop"+str(counter)+".jpg", crop)

    counter +=1

    print(counter)









#%%
from well_plate_project.data_etl._3a_circle_detection import cropped_hough, plot_circles

circles = cropped_hough(queryImage)
circled_image = plot_circles(circles,queryImage)
plt.imshow(circled_image), plt.show()


centers = circles[:,:,0:2]
dst_centers = cv2.perspectiveTransform(centers,matrix)
circlesRound = np.round(dst_centers[0, :]).astype("int")
cicle_colour = (0, 0, 255); thickness=3
for (x, y) in circlesRound:
            cv2.circle(trainImage,(x,y), 10, cicle_colour, thickness) # draw the center of the circle


plt.imshow(trainImage), plt.show()


#%%









#%%

# from _0_plot_hists import plot_hist_bgr
# from pathlib import Path
# path_train = Path(path_train)
# target_folder = Path(target_folder)
# for jpg in path_train.glob('*.jpg'):
#     print(f'Processing {jpg}', end='\n')
#     trainImage = cv2.imread(str(jpg))
#     corrected_img, compare_image, array_dist = crop_pipeline(queryImage, keypoints_query, descriptions_query, trainImage)
    
#     target_filename =  jpg.stem + '_cropped' + jpg.suffix
#     # target_filename =  jpg.stem + '_y_yk3_ragYlab' + jpg.suffix   #jpg.name
#     #'rag_lab_b7_' + 
#     target_path = target_folder / target_filename
#     cv2.imwrite(str(target_path), corrected_img)
    
#     plt.figure(); plt.imshow(corrected_img); plt.show()
#     print(np.sum(array_dist), np.mean(array_dist), np.std(array_dist))
    
#     target_filename =  jpg.stem + '_cro2cmp' + jpg.suffix
#     target_path = target_folder / target_filename   
#     cv2.imwrite(str(target_path), compare_image)
    
    # hist = plot_hist_bgr(trainImage)
    # target_filename =  jpg.stem + '_hist' + jpg.suffix
    # target_path = target_folder / target_filename   
    # hist.savefig(str(target_path))
    
    



print("Done")































