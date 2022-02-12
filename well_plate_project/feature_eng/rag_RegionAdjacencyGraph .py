#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 4, 2020

@author: Enx
"""
#%% Clean Memory
for i in list(globals().keys()):
    if(i[0] != '_'):
        exec('del {}'.format(i))

#%%
from well_plate_project.config import data_dir, reportings_dir

# Dataset paths
raw_data_dir = data_dir / 'raw'
interm_data_dir = raw_data = data_dir / 'intermediate'
describe_data_dir = reportings_dir / 'description'


import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.segmentation as seg



image_file_name = 'a2_a_cropped.jpg'

image_file = raw_data_dir / image_file_name
assert image_file.is_file()

original_image = cv2.imread(str(image_file))

img = cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(original_image,cv2.COLOR_BGR2GRAY)

#%%
plt.imshow(img)
plt.show()

#%%
from skimage import segmentation, filters, color
from skimage.future import graph


#Segment image
labels = segmentation.slic(img, compactness=30, n_segments=800)
#n_segments=400, start_label=1)


#Create RAG
g = graph.rag_mean_color(img, labels)
#Draw RAG
gplt = graph.show_rag(labels, g, img)

cbar = plt.colorbar(gplt)
plt.show()
plt.close()


# %%
edges = filters.sobel(gray)
edges_rgb = color.gray2rgb(edges)

g = graph.rag_boundary(labels, edges)
lc = graph.show_rag(labels, g, edges_rgb, img_cmap=None, edge_cmap='viridis',
                    edge_width=1.2)
plt.colorbar(lc, fraction=0.03)
plt.show()

from skimage import io
#io.show()

# %%
labels = segmentation.slic(img, compactness=30, n_segments=400)
g = graph.rag_mean_color(img, labels)

out = graph.show_rag(labels, g, img)
plt.figure()
plt.title("RAG with all edges shown in green.")
cbar = plt.colorbar(out)
plt.show()
plt.close()

# The color palette used was taken from
# http://www.colorcombos.com/color-schemes/2/ColorCombo2.html
from matplotlib import colors
cmap = colors.ListedColormap(['#6599FF', '#ff9900'])
out = graph.show_rag(labels, g, img, node_color="#ffde00", colormap=cmap,
                     thresh=30, desaturate=True)
plt.figure()
plt.title("RAG with edge weights less than 30, color "
          "mapped between blue and orange.")
plt.imshow(out)

plt.figure()
plt.title("All edges drawn with viridis colormap")
out = graph.show_rag(labels, g, img, colormap=viridis,
                     desaturate=True)

plt.imshow(out)
plt.show()


# %%
