"""
==========================
Region Boundary based RAGs
==========================

Construct a region boundary RAG with the ``rag_boundary`` function. The
function  :py:func:`skimage.future.graph.rag_boundary` takes an
``edge_map`` argument, which gives the significance of a feature (such as
edges) being present at each pixel. In a region boundary RAG, the edge weight
between two regions is the average value of the corresponding pixels in
``edge_map`` along their shared boundary.

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
from skimage.future import graph
from skimage import data, segmentation, color, filters, io
from matplotlib import pyplot as plt


#img = data.coffee()
gimg = color.rgb2gray(img)

labels = segmentation.slic(img, compactness=30, n_segments=400, start_label=1)
edges = filters.sobel(gimg)
edges_rgb = color.gray2rgb(edges)

g = graph.rag_boundary(labels, edges)
lc = graph.show_rag(labels, g, edges_rgb, img_cmap=None, edge_cmap='viridis',
                    edge_width=1.2)

plt.colorbar(lc, fraction=0.03)
io.show()
