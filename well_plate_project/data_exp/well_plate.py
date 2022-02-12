#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 10:20:09 2020

@author: enzo
"""
import cv2
import numpy as np

def map_names(circles):
    
    
    
    return 0


def read_excel(file_xls):
    import pandas as pd
    df_dict=pd.read_excel(file_xls,None)
    
    return df_dict

#%%
import matplotlib.pyplot as plt
def uniform_center_region2(full_well):   
    import cv2
    import numpy as np
    # get dimensions
    hh, ww, cc = full_well.shape
    # compute center of image (as integer)
    wc = ww//2
    hc = hh//2
    # create grayscale copy of input as basis of mask
    gray = cv2.cvtColor(full_well,cv2.COLOR_BGR2GRAY)
    # create zeros mask 2 pixels larger in each dimension
    zeros = np.zeros([hh + 2, ww + 2], np.uint8)
    
    # do floodfill at center of image as seed point
    ffimg = cv2.floodFill(gray, zeros, (wc,hc), (255), loDiff =3, upDiff = 70 , flags=8)[1] #
    #plt.imshow(full_well), plt.show()
    
    # set rest of ffimg to black
    ffimg[ffimg!=255] = 0
    #plt.imshow(ffimg), plt.show()
    
    
    # get contours, find largest and its bounding box 
    contours = cv2.findContours(ffimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    area_thresh = 0
    outer_contour = 0
    for cntr in contours:
        area = cv2.contourArea(cntr)
        if area > area_thresh:
            area = area_thresh
            outer_contour = cntr
            x,y,w,h = cv2.boundingRect(outer_contour)
        else:
            print('Area')
            print(area)
            print('cntr')
            print(cntr)
            print('cntrs')
            print(contours)
            print("WARNING")
    
    # draw the filled contour on a black image
    mask = np.full([hh,ww,cc], (0,0,0), np.uint8)
    cv2.drawContours(mask,[outer_contour],0,(255,255,255),thickness=cv2.FILLED)
    
    # mask the input
    masked_img = full_well.copy()
    masked_img[mask == 0] = 0
    #masked_img[mask != 0] = img[mask != 0]
    
    # crop the bounding box region of the masked img
    result = masked_img[y:y+h, x:x+w]
    
    # draw the contour outline on a copy of result
    result_outline = result.copy()
    cv2.drawContours(result_outline,[outer_contour],0,(0,0,255),thickness=1,offset=(-x,-y))
    
    
    # display it
    #plt.imshow(full_well), plt.show()
    #plt.imshow(ffimg), plt.show()
    #plt.imshow(mask), plt.show()
    #plt.imshow(masked_img), plt.show()
    #plt.imshow(result), plt.show()
    #plt.imshow(result_outline), plt.show()
    return result, mask



def uniform_center_region(full_well):   
    import cv2
    import numpy as np
    # get dimensions
    hh, ww, cc = full_well.shape
    # compute center of image (as integer)
    wc = ww//2
    hc = hh//2
    # create grayscale copy of input as basis of mask
    gray = cv2.cvtColor(full_well,cv2.COLOR_BGR2GRAY)
    # create zeros mask 2 pixels larger in each dimension
    zeros = np.zeros([hh + 2, ww + 2], np.uint8)
    
    
    
    lowDiff=5
    upDiff = 50
    area_thresh = 0
    #TODO fix area_thresh as 10% of the total area of the input image
    while(area_thresh == 0):
        # do floodfill at center of image as seed point
        ffimg = cv2.floodFill(gray, zeros, (wc,hc), (255), loDiff =lowDiff, upDiff = upDiff , flags=8)[1] #
        #plt.imshow(full_well), plt.show()
        
        # set rest of ffimg to black
        ffimg[ffimg!=255] = 0
        #plt.imshow(ffimg), plt.show()
        
        
        # get contours, find largest and its bounding box 
        contours = cv2.findContours(ffimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        outer_contour = 0
        for cntr in contours:
            area = cv2.contourArea(cntr)
            print(f'area:{area}')
            if area > area_thresh:
                area_thresh = area
                outer_contour = cntr
                x,y,w,h = cv2.boundingRect(outer_contour)
        #end_for
        if area_thresh == 0:
            lowDiff= max(lowDiff - 1, 0)
            upDiff = min(upDiff + 10, 255)
            print(f'lowDiff:{lowDiff},upDiff:{upDiff}')
    
    
    # draw the filled contour on a black image
    mask = np.full([hh,ww,cc], (0,0,0), np.uint8)
    cv2.drawContours(mask,[outer_contour],0,(255,255,255),thickness=cv2.FILLED)
    
    # mask the input
    masked_img = full_well.copy()
    masked_img[mask == 0] = 0
    #masked_img[mask != 0] = img[mask != 0]
    
    # crop the bounding box region of the masked img
    result = masked_img[y:y+h, x:x+w]
    
    # draw the contour outline on a copy of result
    result_outline = result.copy()
    cv2.drawContours(result_outline,[outer_contour],0,(0,0,255),thickness=1,offset=(-x,-y))
    
    
    # display it
    # plt.imshow(full_well), plt.show()
    # plt.imshow(ffimg), plt.show()
    # plt.imshow(mask), plt.show()
    # plt.imshow(masked_img), plt.show()
    # plt.imshow(result), plt.show()
    # plt.imshow(result_outline), plt.show()
    return result, mask










def uniform_center_region3(full_well):   
    import cv2
    import numpy as np
    # get dimensions
    hh, ww, cc = full_well.shape
    # compute center of image (as integer)
    wc = ww//2
    hc = hh//2
    # create grayscale copy of input as basis of mask
    gray = cv2.cvtColor(full_well,cv2.COLOR_BGR2GRAY)
    #b
    # create zeros mask 2 pixels larger in each dimension
    zeros = np.zeros([hh + 2, ww + 2], np.uint8)
    
    # do floodfill at center of image as seed point
    ffimg = cv2.floodFill(gray, zeros, (wc,hc), (255), loDiff =1, upDiff = 50 , flags=8)[1] #
    #ffimg = cv2.floodFill(gray, zeros, (wc,hc), (255), loDiff =3, upDiff = 70 , flags=8)[1] #
    plt.imshow(full_well), plt.show()
    
    # set rest of ffimg to black
    ffimg[ffimg!=255] = 0
    #plt.imshow(ffimg), plt.show()
    
    
    # get contours, find largest and its bounding box 
    contours = cv2.findContours(ffimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    contours = contours[0] if len(contours) == 2 else contours[1]
    area_thresh = 0
    outer_contour = 0
    
    for cntr in contours:
        area = cv2.contourArea(cntr)
        if area > area_thresh:
            area = area_thresh
            outer_contour = cntr
            x,y,w,h = cv2.boundingRect(outer_contour)
        else:
            delta = min (wc//2, hc//2)
            cntr[0][0][0] = max(cntr[0][0][0] -delta, 0)
            cntr[0][0][1] = max(cntr[0][0][1] -delta, 0)
            cntr[1][0][0] = max(cntr[0][0][0] -delta, 0)
            cntr[1][0][1] = min(cntr[0][0][1] +delta,hh, ww) 
            cntr[2][0][0] = min(cntr[1][0][0] +delta,hh, ww)
            cntr[2][0][1] = max(cntr[1][0][1] -delta, 0)
            cntr[3][0][0] = min(cntr[1][0][0] +delta,hh, ww) 
            cntr[3][0][1] = min(cntr[1][0][1] +delta,hh, ww) 
            area = cv2.contourArea(cntr)
            outer_contour = cntr
            x,y,w,h = cv2.boundingRect(outer_contour)
            print('Area')
            print(area)
            print('cntr')
            print(cntr)
            print('cntrs')
            print(contours)
            print("WARNING")
    
    # draw the filled contour on a black image
    mask = np.full([hh,ww,cc], (0,0,0), np.uint8)
    cv2.drawContours(mask,[outer_contour],0,(255,255,255),thickness=cv2.FILLED)
    
    # mask the input
    masked_img = full_well.copy()
    masked_img[mask == 0] = 0
    #masked_img[mask != 0] = img[mask != 0]
    
    # crop the bounding box region of the masked img
    result = masked_img[y:y+h, x:x+w]
    
    # draw the contour outline on a copy of result
    result_outline = result.copy()
    cv2.drawContours(result_outline,[outer_contour],0,(0,0,255),thickness=1,offset=(-x,-y))
    
    
    # display it
    plt.imshow(full_well), plt.show()
    plt.imshow(ffimg), plt.show()
    plt.imshow(mask), plt.show()
    plt.imshow(masked_img), plt.show()
    plt.imshow(result), plt.show()
    plt.imshow(result_outline), plt.show()
    return result, mask
#%%
def white_balance(img): 
    import cv2
    #Automatic White Balancing with Grayworld assumption
    #https://pippin.gimp.org/image-processing/chapter-automaticadjustments.html
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

def extract_features(well, non_zero = False):
    import cv2
    from skimage.measure import shannon_entropy
    from skimage.feature import greycomatrix
    
    #Problem: color fidelity of a photo
    #white balance
    #setting values for base colors 
    img_lab=cv2.cvtColor(well,cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(img_lab)
    b,g,r = cv2.split(well)
    #b = np.reshape(well[:,:,0], -1)
    #g = np.reshape(well[:,:,1], -1)
    #r = np.reshape(well[:,:,2], -1)
    #l = np.reshape(img_lab[:,:,0], -1)
    #a = np.reshape(img_lab[:,:,1], -1)
    #b = np.reshape(img_lab[:,:,2], -1)
    
    
    #img_std_l = np.std(l)
    #img_std_a = np.std(a)
    #img_std_b = np.std(b)
    
    mean, stddev = cv2.meanStdDev(img_lab)
    
    
    #Entropy!   entropy, energy, homogeneity and contrast
    entropy = shannon_entropy(img_lab)

    
    grayImg = cv2.cvtColor(well,cv2.COLOR_BGR2GRAY)
    #glcm = np.squeeze(greycomatrix(grayImg, distances=[1], angles=[0], symmetric=True, normed=True))
    #entropy_glcm = -np.sum(glcm*np.log2(glcm + (glcm==0)))
    
    # computing the mean 
    #b_mean = np.mean(b) 
    #g_mean = np.mean(g) 
    #r_mean = np.mean(r) 
  
    # displaying the most prominent color 
    #if (b_mean > g_mean and b_mean > r_mean): 
    #    print("Blue") 
    #if (g_mean > r_mean and g_mean > b_mean): 
    #    print("Green") 
    #else: 
    #    print("Red")
        
    #img_propr = pandas.Series( [b_mean, g_mean, r_mean],
    #                     index=['B_mean', 'G_mean', 'R_mean'])
    
    img_propr = {}
    img_propr["gray"] = image_stats(grayImg, non_zero)
    img_propr["b"] = image_stats(b, non_zero)
    img_propr["g"] = image_stats(g, non_zero)
    img_propr["r"] = image_stats(r, non_zero)
    img_propr["L"] = image_stats(l, non_zero)
    img_propr["a"] = image_stats(a, non_zero)
    img_propr["b"] = image_stats(b, non_zero)
    
    return img_propr



#%% Image statistics
def entropy2(labels, base=None):
    from math import log,e
    """ Computes entropy of label distribution. """
    n_labels = len(labels)
    if n_labels <= 1:
        return 0
    value,counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)
    if n_classes <= 1:
        return 0
    ent = 0.
    # Compute entropy
    base = e if base is None else base
    for i in probs:
        ent -= i * log(i, base)
    return ent

def image_stats(single_chan_img, non_zero=False):
    from skimage.measure import shannon_entropy
    from skimage.feature import greycomatrix
    from scipy.stats import skew
    assert len(single_chan_img.shape) == 2
    vect_chan = np.reshape(single_chan_img, -1)
    if non_zero : vect_chan = vect_chan[np.nonzero(vect_chan)]
    stats_dict={}
    stats_dict["mean"] = np.mean(vect_chan)
    stats_dict["stddev"] = np.std(vect_chan)
    #mean, stddev = cv2.meanStdDev(single_chan_img)

    stats_dict["skewness"]= skew(vect_chan)

    #stats_dict["energy"]= 
    # "energy" should be inversely proportional to Shannon entropy
    stats_dict["entropy"] = shannon_entropy(single_chan_img) 
    stats_dict["entropy2"] = entropy2(vect_chan) 
    glcm = np.squeeze(greycomatrix(single_chan_img, distances=[1], angles=[0], symmetric=True, normed=True))
    entropy_glcm = -np.sum(glcm*np.log2(glcm + (glcm==0)))
    stats_dict["entropy_glcm"] = entropy_glcm
    
    from PIL import Image, ImageStat
    im_pil = Image.fromarray(single_chan_img)
    stat = ImageStat.Stat(im_pil)
    stats_dict["mean_PIL"] = stat.mean[0]
    return stats_dict



def image_statistics(Z):
    #Input: Z, a 2D array, hopefully containing some sort of peak
    #Output: cx,cy,sx,sy,skx,sky,kx,ky
    #cx and cy are the coordinates of the centroid
    #sx and sy are the stardard deviation in the x and y directions
    #skx and sky are the skewness in the x and y directions
    #kx and ky are the Kurtosis in the x and y directions
    #Note: this is not the excess kurtosis. For a normal distribution
    #you expect the kurtosis will be 3.0. Just subtract 3 to get the
    #excess kurtosis.
    import numpy as np

    h,w = np.shape(Z)

    x = range(w)
    y = range(h)


    #calculate projections along the x and y axes
    yp = np.sum(Z,axis=1)
    xp = np.sum(Z,axis=0)

    #centroid
    cx = np.sum(x*xp)/np.sum(xp)
    cy = np.sum(y*yp)/np.sum(yp)

    #standard deviation
    x2 = (x-cx)**2
    y2 = (y-cy)**2

    sx = np.sqrt( np.sum(x2*xp)/np.sum(xp) )
    sy = np.sqrt( np.sum(y2*yp)/np.sum(yp) )

    #skewness
    x3 = (x-cx)**3
    y3 = (y-cy)**3

    skx = np.sum(xp*x3)/(np.sum(xp) * sx**3)
    sky = np.sum(yp*y3)/(np.sum(yp) * sy**3)

    #Kurtosis
    x4 = (x-cx)**4
    y4 = (y-cy)**4
    kx = np.sum(xp*x4)/(np.sum(xp) * sx**4)
    ky = np.sum(yp*y4)/(np.sum(yp) * sy**4)


    return cx,cy,sx,sy,skx,sky,kx,ky


#%% testings

   
 

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
    image_file_name = 'd1_aC1_crop.jpg'
    from well_plate_project.config import data_dir
    path = data_dir / 'raw' / 'EXPERIMENTS_Crp' / 'd1_a'
    image_file = path /  image_file_name
    assert image_file.is_file()
    return image_file


if __name__ == "__main__":
    clear_all()
    image_file = load_test_file()
    img = cv2.imread(str(image_file))
    print("Plotting... ")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.xticks([]), plt.yticks([])
    plt.show()
    
    
    
    lower_blue = np.array([0, 0, 0])
    upper_blue = np.array([0, 0, 0])
    mask = cv2.inRange(img, lower_blue, upper_blue)
    result = cv2.bitwise_and(img, img, mask=mask)
    print("Plotting2... ")
    plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
    plt.xticks([]), plt.yticks([])
    plt.show()
    
    
    print("Testing ... ")
    image_reduced, mask = uniform_center_region(img)


    print("Plotting... ")
    plt.imshow(cv2.cvtColor(image_reduced, cv2.COLOR_BGR2RGB))
    plt.xticks([]), plt.yticks([])
    plt.show()
    

    
    from PIL import Image, ImageStat
    imgPIL = Image.open(str(image_file))
    stat = ImageStat.Stat(imgPIL)
    print(stat)
    
    
    
    circle_dict = extract_features(img)
    print("Circle")
    print(circle_dict)
    
    circle_dict = extract_features(img, True)
    print("Circle")
    print(circle_dict)
    
    reduced_dict = extract_features(image_reduced, True)
    print("Reduced")
    print(reduced_dict)
    
    reduced_dict = extract_features(image_reduced)
    print("Reduced")
    print(reduced_dict)
    
    
    
    