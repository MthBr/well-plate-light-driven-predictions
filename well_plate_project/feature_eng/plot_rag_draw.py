"""
======================================
Drawing Region Adjacency Graphs (RAGs)
======================================

This example constructs a Region Adjacency Graph (RAG) and draws it with
the `rag_draw` method.
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
from skimage import data, segmentation
from skimage.future import graph
from matplotlib import pyplot as plt


#img = data.coffee()
labels = segmentation.slic(img, compactness=30, n_segments=400, start_label=1)
g = graph.rag_mean_color(img, labels)

fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(6, 8))

ax[0].set_title('RAG drawn with default settings')
lc = graph.show_rag(labels, g, img, ax=ax[0])
# specify the fraction of the plot area that will be used to draw the colorbar
fig.colorbar(lc, fraction=0.03, ax=ax[0])

ax[1].set_title('RAG drawn with grayscale image and viridis colormap')
lc = graph.show_rag(labels, g, img,
                    img_cmap='gray', edge_cmap='viridis', ax=ax[1])
fig.colorbar(lc, fraction=0.03, ax=ax[1])

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()
