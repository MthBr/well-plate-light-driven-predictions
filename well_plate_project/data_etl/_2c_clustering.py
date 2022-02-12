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





def canny_countour(original_image, lables):
#%% Remove 1 cluster from image and apply canny edge detection
    removedCluster = 1

    cannyImage = np.copy(original_image).reshape((img.shape[0]*img.shape[1], img.shape[2]))
    cannyImage[labels.flatten() == removedCluster] = [0, 0, 0]

    cannyImage = cv2.Canny(cannyImage,100,200).reshape(original_image.shape)
    #cv2.imwrite("cannyImage.jpg", cannyImage)
    #plot_img_4hist_rgb(cannyImage)


    #%% Finding contours using opencv

    initialContoursImage = np.copy(cannyImage)
    imgray = cv2.cvtColor(initialContoursImage, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(imgray, 50, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(initialContoursImage, contours, -1, (0,0,255), cv2.CHAIN_APPROX_SIMPLE)
    #cv2.imwrite("initialContoursImage.jpg", initialContoursImage)
    #plot_img_4hist_rgb(initialContoursImage)

    return initialContoursImage


#DBscan 
#https://stackoverflow.com/questions/40142835/image-not-segmenting-properly-using-dbscan
#https://core.ac.uk/download/pdf/79492015.pdf


#https://github.com/s4lv4ti0n/clusteringAlgorithms


def concatenate_images(list_img):
    assert isinstance(list_img, (list, tuple))
    assert len(list_img)>1

    assert list_img[0].shape == list_img[1].shape
    final_img = np.concatenate((list_img[0], list_img[1]), axis=2) # axes are 0-indexed, i.e. 0, 1, 2
    for img in list_img[2:]:
        assert list_img[0].shape[0:2] == img.shape[0:2]
        #final_img = np.concatenate((final_img, img), axis=2)
        final_img = np.dstack((final_img, img))

    return final_img, len(list_img)


def split_chan_dict(string_3, image): #key, value
    assert len(string_3)==3
    assert image.shape[2] == 3
    dictionary = {
        string_3.charAt(2) :  image[:,:,0],
        string_3.charAt(1) :  image[:,:,1],
        string_3.charAt(0) :  image[:,:,2],
    }
    return dictionary

def analyze_concat_images(imgs_dict, CVK, SK):
    cnt_img, n_images = concatenate_images(list(imgs_dict.values())) # axes are 0-indexed, i.e. 0, 1, 2

    print("Cv2")
    clusterd_cv2 = clustering_cv2_k(cnt_img, CVK)
    clustered_k = clustering_k(cnt_img, SK)


    list_imgs_k = np.array_split(clustered_k, n_images, axis=2)
    list_imgs_k_cv2 = np.dsplit(clusterd_cv2, n_images)
    print("dict")
    dict_k={}
    for key, img in  zip(imgs_dict.keys(), list_imgs_k):
        assert img.shape == img_bgr.shape
        dict_k[key] = img
    
    dict_k_cv={}
    for key, img in  zip(imgs_dict.keys(), list_imgs_k_cv2):
        assert img.shape == img_bgr.shape
        dict_k_cv[key] = img
    
    return dict_k, dict_k_cv


def analyze_concat_channels(img_bgr):
    splitted_img_dict = {}
    for key, value in imgs_dict.items():
        image_3chn = split_chan_dict(key, value)
        splitted_img_dict[key] = image_3chn

    final_channels = np.concatenate((
        splitted_img_dict["RGB"]["B"], 
        splitted_img_dict["LAB"]["L"]), axis=2) # axes are 0-indexed, i.e. 0, 1, 2

    clusterd_cv2 = clustering_cv2_k(final_channels)
    clustered_k = clustering_k(final_channels)

    return 0


# %%
def clear_all():
    """Clears all the variables from the workspace of the application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]


def load_test_file(image_file_name, data_dir): 

    return img


if __name__ == "__main__":
    clear_all()

    from well_plate_project.data_etl._0_plot_hists import plot_hist_rgb, plot_hist_hsv, plot_hist_bgr, plot_hist_lab, plot_any_chnl
    from well_plate_project.config import data_dir
    image_file_name = '20201118_083228.jpg' # f1_a_cropped a2_a_cropped
    
    path = data_dir / 'raw' / 'exp_v2' / 'luce_ult_es' /'01'
    image_file = path /  image_file_name
    assert image_file.is_file()
    original_image = cv2.imread(str(image_file))
    plt.imshow(original_image)
    plt.show()
    
    print("Testing ... ")

    img_rgb=cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
    img_hsv=cv2.cvtColor(original_image,cv2.COLOR_BGR2HSV)
    img_lab=cv2.cvtColor(original_image,cv2.COLOR_BGR2LAB)
    img_yuv=cv2.cvtColor(original_image,cv2.COLOR_BGR2YUV)
    img_bgr = original_image

    imgs_dict =	{
    #"RGB": img_rgb,
    #"HSV": img_hsv,
    "LAB": img_lab,
    "YUV": img_yuv,
    }
    
    #plot_hist_hsv(img_hsv)
    #plot_hist_lab(img_lab)
    #plot_hist_rgb(img_rgb)
    #plot_hist_bgr(img_bgr)
    #plot_hist_yuv(img_yuv)

    CVK=9
    SK=9
    dict_k, dict_k_cv = analyze_concat_images(imgs_dict, CVK, SK)

    conated_images= ''.join(str(p) for p in imgs_dict.keys())
    target_folder = data_dir / 'intermediate' / 'TEST_CLUST'

    for name, image in  imgs_dict.items():
        assert dict_k[name].shape == original_image.shape
        assert dict_k_cv[name].shape == original_image.shape

        print("Plotting... ")
        plt.imshow(image)
        plt.title(name+'_Orig')
        plt.show()
        target_filename= name+'_Orig_'+ conated_images + '.jpg'
        target_path = target_folder / target_filename
        print(target_path)
        writeStatus=cv2.imwrite(str(target_path), image)
        if writeStatus is True:
            print("image written")
        else:
            print("problem")


        fig=plot_any_chnl(image)
        plt.show()
        target_filename= name+'_Orig_chn_'+ conated_images + '.jpg'
        target_path = target_folder / target_filename
        fig.savefig(str(target_path))

        print("Plotting 2 ")
        plt.imshow(dict_k[name])
        plt.title(name+'_SK'+ str(SK))
        plt.show()
        target_filename= name+'_SK' + str(SK) + conated_images + '.jpg'
        target_path = target_folder / target_filename
        cv2.imwrite(str(target_path), dict_k[name])

        fig=plot_any_chnl(dict_k[name])
        plt.show()
        target_filename= name+ '_chl'+'_SK'+ str(SK)+ conated_images + '.jpg'
        target_path = target_folder / target_filename
        fig.savefig(str(target_path))


        print("Plotting 3 ")
        plt.imshow(dict_k_cv[name])
        plt.title(name+'_CVK'+ str(SK))
        plt.show()
        target_filename= name+'_CVK'+ str(CVK)+ conated_images + '.jpg'
        target_path = target_folder / target_filename
        cv2.imwrite(str(target_path), dict_k_cv[name])

        fig=plot_any_chnl(dict_k_cv[name])
        plt.show()
        target_filename= name+ '_chl'+'_CVK'+ str(CVK) + conated_images + '.jpg'
        target_path = target_folder / target_filename
        fig.savefig(str(target_path))

    print("Done")

# %%
