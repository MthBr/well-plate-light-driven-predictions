#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 4, 2020

Tools for Investigating the importance of colorspaces for image classification
https://arxiv.org/pdf/1902.00267.pdf

Look at:
https://towardsdatascience.com/understand-and-visualize-color-spaces-to-improve-your-machine-learning-and-deep-learning-models-4ece80108526

Steganalysis based on an ensemblecolorspace approach
https://arxiv.org/pdf/2002.02413.pdf


@author: Enx
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

#for 3d stuff
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

import matplotlib
#matplotlib.use('Agg')
#plt.ioff()

# %% Define functions

def plot_hist_rgb(img_rgb):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=[2*x for x in plt.rcParams["figure.figsize"]]) #

    # Calulate various hists
    axs[0, 0].imshow(img_rgb)

    axs[1, 0].hist(img_rgb.ravel(),256,[0,256])
    axs[1, 0].axis(xmin=0,xmax=256)
    

    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([img_rgb],[i],None,[256],[0,256])
        axs[0, 1].plot(histr,color = col)
        axs[0, 1].set_yscale('log')
        axs[0, 1].axis(xmin=0,xmax=256)

        axs[1, 1].plot(histr,color = col) 
        axs[1, 1].axis(xmin=0,xmax=256)

    fig.suptitle('RGB')
    plt.show()
    plt.close(fig)
    return fig


def plot_3d_rgb():
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    r,g,b = cv2.split(img_rgb)
    pixel_colors = img_rgb.reshape((np.shape(img_rgb)[0]*np.shape(img_rgb)[1], 3))
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Red")
    axis.set_ylabel("Green")
    axis.set_zlabel("Blue")
    plt.show()
    plt.close(fig)
    return fig



def plot_hist_bgr(image_to_plot):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=[2*x for x in plt.rcParams["figure.figsize"]]) #

    # Calulate various hists
    axs[0, 0].imshow(image_to_plot)

    axs[1, 0].hist(image_to_plot.ravel(),256,[0,256])
    axs[1, 0].axis(xmin=0,xmax=256)
    

    color = ('r','g','b')
    for i,col in enumerate(color):
        histr = cv2.calcHist([image_to_plot],[i],None,[256],[0,256])
        axs[0, 1].plot(histr,color = col)
        axs[0, 1].set_yscale('log')
        axs[0, 1].axis(xmin=0,xmax=256)

        axs[1, 1].plot(histr,color = col) 
        axs[1, 1].axis(xmin=0,xmax=256)

    #fig.suptitle('BGR')
    #plt.show()
    #plt.close(fig)
    return fig

def plot_hist_yuv(img_yuv):
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=[2*x for x in plt.rcParams["figure.figsize"]]) #

    colormap_u = np.array([[[i,255-i,0] for i in range(256)]],dtype=np.uint8)
    colormap_v = np.array([[[0,255-i,i] for i in range(256)]],dtype=np.uint8)
    y, u, v = cv2.split(img_yuv)

    # Convert back to BGR so we can apply the LUT
    y = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
    u = cv2.cvtColor(u, cv2.COLOR_GRAY2BGR)
    v = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)

    u_mapped = cv2.LUT(u, colormap_u)
    v_mapped = cv2.LUT(v, colormap_v)

    # Calulate various hists
    axs[0, 0].imshow(img_yuv); axs[0, 0].axis("off")


    axs[0, 1].imshow(y); axs[0, 1].axis("off"); axs[0, 1].set_title("Y'")  

    axs[1, 0].remove()
    
    axs[1, 1].hist(y.ravel(),256,[0,256]); axs[1, 1].axis(xmin=0,xmax=256)

    axs[0, 2].imshow(u_mapped); axs[0, 2].axis("off"); axs[0, 2].set_title("U")  
    axs[1, 2].hist(u_mapped.ravel(),256,[0,256])
    axs[1, 2].axis(xmin=0,xmax=256)
    axs[1, 2].set_yscale('log')


    axs[0, 3].imshow(v_mapped)
    axs[0, 3].axis("off"); axs[0, 3].set_title("V") 
    axs[1, 3].hist(v_mapped.ravel(),256,[0,256])
    axs[1, 3].axis(xmin=0,xmax=256); axs[1, 3].set_yscale('log')

    fig.suptitle('YUV')
    plt.show(fig)
    # plt.close(fig)
    return fig
# %%
def plot_hist_hsv(img_hsv):
    
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=[2*x for x in plt.rcParams["figure.figsize"]]) 

    # Calulate various hists
    axs[0, 0].imshow(img_hsv)
    axs[0, 0].axis("off")

    hist = cv2.calcHist( [img_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256] )
    axs[0, 1].imshow(hist)

    axs[0, 2].imshow(hist, interpolation = 'nearest')
    
    axs[0,3].set_visible(False)
    axs[1,0].remove()

    lu1=img_hsv[...,0].flatten()
    axs[1, 1].hist(lu1*360,bins=360,range=(0.0,360.0),histtype='stepfilled', color='r', label='Hue')
    axs[1, 1].set_title("Hue")
    #axs[1, 1].set_xlabel("Value")
    #axs[1, 1].set_ylabel("Frequency")
    axs[1, 1].legend()

    lu2=img_hsv[...,1].flatten()
    axs[1, 2].hist(lu2,bins=100,range=(0.0,1.0),histtype='stepfilled', color='g', label='Saturation')
    axs[1, 2].set_title("Saturation")   
    #axs[1, 2].set_xlabel("Value")    
    #axs[1, 2].set_ylabel("Frequency")
    axs[1, 2].legend()

    lu3=img_hsv[...,2].flatten()
    axs[1, 3].hist(lu3*255,bins=256,range=(0.0,255.0),histtype='stepfilled', color='b', label='Intesity')
    axs[1, 3].set_title("Intensity")   
    #axs[1, 3].set_xlabel("Value")    
    #axs[1, 3].set_ylabel("Frequency")
    axs[1, 3].legend()


    fig.suptitle('HSV')
    fig.show()
    plt.show()
    # plt.close(fig)
    return fig




def extract_single_dim_from_LAB_convert_to_RGB(image,idim):
    '''
    image is a single lab image of shape (None,None,3)
    '''
    from skimage.color import lab2rgb 
    z = np.zeros(image.shape)
    if idim != 0 :
        z[:,:,0]=80 ## I need brightness to plot the image along 1st or 2nd axis
    z[:,:,idim] = image[:,:,idim]
    z = lab2rgb(z)
    return(z)

def plot_hist_lab(img_LAB):
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=[2*x for x in plt.rcParams["figure.figsize"]]) # figsize=(9,9)
    axs[0, 0].imshow(img_LAB)
    axs[0, 0].axis("off")

    l,a,b = cv2.split(img_LAB)
    axs[1, 0].imshow(l,cmap='gray')
    axs[1, 0].axis("off")


    lab_rgb_gray = extract_single_dim_from_LAB_convert_to_RGB(img_LAB,0) 
    axs[0, 1].imshow(lab_rgb_gray); 
    axs[0, 1].axis("off")
    axs[0, 1].set_title("L: lightness")
    axs[1, 1].hist(lab_rgb_gray.ravel())


    lab_rgb_gray = extract_single_dim_from_LAB_convert_to_RGB(img_LAB,1) 
    axs[0, 2].imshow(lab_rgb_gray); 
    axs[0, 2].axis("off")
    axs[0, 2].set_title("A: color spectrums \n green to red")
    axs[1, 2].hist(lab_rgb_gray.ravel())


    lab_rgb_gray = extract_single_dim_from_LAB_convert_to_RGB(img_LAB,2) 
    axs[0, 3].imshow(lab_rgb_gray)
    axs[0, 3].axis("off")
    axs[0, 3].set_title("B: color spectrums \n blue to yellow")
    axs[1, 3].hist(lab_rgb_gray.ravel())

    fig.suptitle('LAB')
    plt.show()
    # plt.close(fig)
    return fig

def plot_any_chnl(image_to_plot):

    n_chnl = image_to_plot.shape[2] #number of channels
    
    fig, axs = plt.subplots(nrows=2, ncols=n_chnl+1, figsize=[2*x for x in plt.rcParams["figure.figsize"]]) 

    assert(n_chnl ==  3)

    colormap_u = np.array([[[i,255-i,0] for i in range(256)]],dtype=np.uint8)
    colormap_v = np.array([[[0,255-i,i] for i in range(256)]],dtype=np.uint8)
    y, u, v = cv2.split(image_to_plot)

    # Convert back to BGR so we can apply the LUT
    y = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
    u = cv2.cvtColor(u, cv2.COLOR_GRAY2BGR)
    v = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)

    u_mapped = cv2.LUT(u, colormap_u)
    v_mapped = cv2.LUT(v, colormap_v)

    # Calulate various hists
    axs[0, 0].imshow(image_to_plot)
    axs[0, 0].axis("off")

    color = ('c','m','k')
    for i,col in enumerate(color):
        histr = cv2.calcHist([image_to_plot],[i],None,[256],[0,256])
        axs[1, 0].plot(histr,color = col)
        axs[1, 0].set_yscale('log')
        axs[1, 0].axis(xmin=0,xmax=256)


    axs[0, 1].imshow(y)
    axs[0, 1].axis("off")
    axs[0, 1].set_title("ch1")  
    
    axs[1, 1].hist(y.ravel(),256,[0,256])
    axs[1, 1].axis(xmin=0,xmax=256)

    axs[0, 2].imshow(u_mapped)
    axs[0, 2].axis("off")
    axs[0, 2].set_title("ch1")  
    axs[1, 2].hist(u_mapped.ravel(),256,[0,256])
    axs[1, 2].axis(xmin=0,xmax=256)
    axs[1, 2].set_yscale('log')


    axs[0, 3].imshow(v_mapped)
    axs[0, 3].axis("off")
    axs[0, 3].set_title("cg2") 
    axs[1, 3].hist(v_mapped.ravel(),256,[0,256])
    axs[1, 3].axis(xmin=0,xmax=256)
    axs[1, 3].set_yscale('log')


    fig.suptitle('3channels')
    plt.show()
    # plt.close(fig)
    return fig




# %%
def clear_all():
    """Clears all the variables from the workspace of the application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]


def load_test_file(): 
    image_file_name = '20201118_083241.jpg' #20201118_083216
    from well_plate_project.config import data_dir
    path = data_dir / 'raw' /'exp_v2' / 'luce_ult_easy' /'01' #/ 'EXPERIMENTS' / 
    image_file = path /  image_file_name
    assert image_file.is_file()
    img = cv2.imread(str(image_file))
    return img



if __name__ == "__main__":
    clear_all()
    original_image = load_test_file()
    #plt.imshow(original_image)
    #plt.show()

    print("Testing impage plotting ")
    
    img=original_image.astype('uint8')
    
    hist = plot_hist_bgr(img)
     
     
    plot_hist_rgb(original_image)
    plot_hist_bgr(img)
    plot_any_chnl(img)
    #device, color_header, color_data, analysis_images = pcv.analyze_color(img, imagename, mask, 256, device, debug="print")

    #%% Convert image
    # convert our image from RGB Colours Space to HSV to work ahead.
    img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plot_hist_rgb(img_rgb)
    plot_any_chnl(img_rgb)

    img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    plot_hist_hsv(img_hsv)
    plot_any_chnl(img_hsv)
    #https://raspberrypi.stackexchange.com/questions/10588/hue-saturation-intensity-histogram-plot

    img_lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    plot_hist_lab(img_lab)
    plot_any_chnl(img_lab)

    img_yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
    plot_hist_yuv(img_yuv)





















# %%
