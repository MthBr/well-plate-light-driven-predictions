#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 10:33:39 2019
Set of functions for data loading and cleaning
@author: modal
"""
from well_plate_project.custom_funcs import benchmark


@benchmark
def load_dataset(file, chunksize=None, column_types=None, parse_dates=False, sep=','):
    df = pd.DataFrame()
    if chunksize!=None:
        for chunk in pd.read_csv(file, chunksize=chunksize, dtype = column_types, parse_dates = parse_dates, sep=sep):
            df = pd.concat([df, chunk], ignore_index=True)
    
        del chunk
    else:
        df = pd.read_csv(file, dtype = column_types, parse_dates = parse_dates, sep=sep)
    
    return df

def save_on_pickle(df,file_name):
    pickling_on = open(file_name,"wb")
    pickle.dump(df, pickling_on)
    pickling_on.close()
    
def load_pickle(file_name):
    pickle_off = open(file_name,"rb")
    df = pickle.load(pickle_off)
    return df


@benchmark
def crop_pipeline(image_file):
    import cv2
    from well_plate_project.data_etl._1f_crop_rotate import crop_rotate

    print(f'Loading image file {str(image_file.name)}...') #.upper
    assert image_file.is_file()

    original_image = cv2.imread(str(image_file))
    #plt.imshow(original_image)
    #plt.show()

    #%% Convert image
    # convert our image from RGB Colours Space to HSV to work ahead.
    img_rgb=cv2.cvtColor(original_image.astype('uint8'),cv2.COLOR_BGR2RGB)
    #plot_img_4hist_rgb(img_rgb)

    img_hsv=cv2.cvtColor(original_image.astype('uint8'),cv2.COLOR_BGR2HSV)
    #plot_img_4hist_hsv(img_hsv)  #https://raspberrypi.stackexchange.com/questions/10588/hue-saturation-intensity-histogram-plot

    img_lab=cv2.cvtColor(original_image.astype('uint8'),cv2.COLOR_BGR2LAB)
    #plot_img_4hist_rgb(img_lab)

    img_bgr = original_image
    #plot_img_4hist_rgb(img_bgr)


    #STEP 0: rotate and crop OR crop and rotate
    cropped_image = crop_rotate(img_bgr) #input and output as BGR
    return cropped_image #circled_image




@benchmark
def image_pipeline(image_file):
    import cv2
    
    from well_plate_project.data_etl._2c_clustering import clustering_cv2_k
    from well_plate_project.data_etl._2b_felzen import felzen
    from well_plate_project.data_etl._2a_rag_merging import segm_rag_merg
    from well_plate_project.data_etl._3a_circle_detection import cropped_hough, plot_circles
    from well_plate_project.data_etl._3f_cluster_hough import cluster_hough

    print(f'Loading image file {str(image_file.name)}...') #.upper
    assert image_file.is_file()

    original_image = cv2.imread(str(image_file))
    #plt.imshow(original_image)
    #plt.show()

    #%% Convert image
    # convert our image from RGB Colours Space to HSV to work ahead.
    img_rgb=cv2.cvtColor(original_image.astype('uint8'),cv2.COLOR_BGR2RGB)
    #plot_img_4hist_rgb(img_rgb)

    img_hsv=cv2.cvtColor(original_image.astype('uint8'),cv2.COLOR_BGR2HSV)
    #plot_img_4hist_hsv(img_hsv)  #https://raspberrypi.stackexchange.com/questions/10588/hue-saturation-intensity-histogram-plot

    img_lab=cv2.cvtColor(original_image.astype('uint8'),cv2.COLOR_BGR2LAB)
    #plot_img_4hist_rgb(img_lab)

    img_bgr = original_image
    #plot_img_4hist_rgb(img_bgr)

    #STEP 0: rotate and crop OR crop and rotate
    #cropped_image = crop_rotate(img_bgr) #input and output as BGR

    #STEP 2: clustering
    image = clustering_cv2_k(cropped_image)
    image = cv2.GaussianBlur(image, ksize = (7, 7), sigmaX = 1.5)

    #STEP 2: rog segmentation
    img = cv2.cvtColor(cropped_image,cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(cropped_image,cv2.COLOR_BGR2GRAY)
    image = segm_rag_merg(img, gray)
    image = cv2.cvtColor(image,cv2.COLOR_LAB2BGR)
    #image = cv2.GaussianBlur(image, ksize = (7, 7), sigmaX = 1.5)


    #STEP 2: rog segmentation
    # img = cv2.cvtColor(cropped_image,cv2.COLOR_BGR2RGB)
    # image = felzen(img)
    # image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    # image = cv2.GaussianBlur(image, ksize = (7, 7), sigmaX = 1.5)


    #STEP 3: cicle on multiple input types, output as list of circles
    cropped_image = cropped_hough(image)

    #cropped_image_rgb =cv2.cvtColor(cropped_image.astype('uint8'),cv2.COLOR_BGR2RGB)
    #circled_image = plot_circles(circles,cropped_image)

    #STEP 4: input as image and list of circles, output mask/cropeed circles

    return cropped_image #circled_image


def circles_image_pipeline(image_file):
    import cv2
    import numpy as np
    from well_plate_project.data_etl._1f_crop_rotate import crop_rotate
    from well_plate_project.data_etl._2c_clustering import clustering_cv2_k
    from well_plate_project.data_etl._2b_felzen import felzen
    from well_plate_project.data_etl._2a_rag_merging import segm_rag_merg
    from well_plate_project.data_etl._3a_circle_detection import cropped_hough, plot_circles
    from well_plate_project.data_etl._3f_cluster_hough import cluster_hough

    print(f'Loading image file {str(image_file.name)}...') #.upper
    assert image_file.is_file()

    original_image = cv2.imread(str(image_file))
    #plt.imshow(original_image)
    #plt.show()

    #%% Convert image
    # convert our image from RGB Colours Space to HSV to work ahead.
    img_rgb=cv2.cvtColor(original_image.astype('uint8'),cv2.COLOR_BGR2RGB)
    #plot_img_4hist_rgb(img_rgb)

    img_hsv=cv2.cvtColor(original_image.astype('uint8'),cv2.COLOR_BGR2HSV)
    #plot_img_4hist_hsv(img_hsv)  #https://raspberrypi.stackexchange.com/questions/10588/hue-saturation-intensity-histogram-plot

    img_lab=cv2.cvtColor(original_image.astype('uint8'),cv2.COLOR_BGR2LAB)
    #plot_img_4hist_rgb(img_lab)

    img_bgr = original_image
    #plot_img_4hist_rgb(img_bgr)

    yuv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)



    y_clust= np.split(yuv_img, yuv_img.shape[-1], -1)[0] #cv2.split(yuv_img)
    y_img = np.squeeze(y_clust, axis=-1)
    circles_y = cropped_hough(y_img)

    import matplotlib.pyplot as plt

    # plt.imshow(y_clust)
    # plt.show()
    # plt.imshow(y_img)
    # plt.show()


    cluster_y = clustering_cv2_k(y_clust)
    circles_clust_y = cropped_hough(cluster_y)
    # plt.imshow(cluster_y)
    # plt.show()


    #cluster_y_5 = clustering_cv2_k(y_clust, k=5)
    #circles_clust_y5 = cropped_hough(cluster_y_5)



    rag_sem = segm_rag_merg(img_lab, y_img)
    rag_sem_bgr = cv2.cvtColor(rag_sem,cv2.COLOR_LAB2BGR)
    circles_rag_sem_lab_y = cropped_hough(rag_sem_bgr) #cropped hough supposes 1 CHANNEL
    # plt.imshow(rag_sem_bgr)
    # plt.show()


    print("Plotting... ")
    circled_image = plot_circles(circles_y, img_bgr, (0, 255, 0),4)
    circled_image = plot_circles(circles_clust_y, circled_image, (255, 0, 0),3)
    circled_image = plot_circles(circles_rag_sem_lab_y, circled_image, (0, 0, 255),2)
    height, width = circled_image.shape[:2]
    h= int(height/15)
    w = int(width/15)
    cv2.putText(circled_image, 'Y:' + str(circles_y.shape[1]),            (w*2,h), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)
    cv2.putText(circled_image, 'clus3Y:' + str(circles_clust_y.shape[1]), (w*4,h), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
    cv2.putText(circled_image, 'ragYlab:' + str(circles_rag_sem_lab_y.shape[1]),   (w*6,h), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)


    return circled_image

def clear_all():
    """Clears all the variables from the workspace of the application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    clear_all()
    print("Testing ... ")
    from well_plate_project.config import data_dir
    source_folder =  data_dir / 'raw' / 'Cropped'
    image_file_name = 'a2_a_cropped.jpg' #h2_a
    image_file = source_folder /  image_file_name
    #image_pipeline(image_file)
    out = circles_image_pipeline(image_file)
    plt.imshow(out)
    plt.show()




