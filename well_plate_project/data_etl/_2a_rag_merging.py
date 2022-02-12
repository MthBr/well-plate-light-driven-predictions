"""
============================================
Hierarchical Merging of Region Boundary RAGs
============================================

This example demonstrates how to perform hierarchical merging on region
boundary Region Adjacency Graphs (RAGs). Region boundary RAGs can be
constructed with the :py:func:`skimage.future.graph.rag_boundary` function.
The regions with the lowest edge weights are successively merged until there
is no edge with weight less than ``thresh``. The hierarchical merging is done
through the :py:func:`skimage.future.graph.merge_hierarchical` function.
For an example of how to construct region boundary based RAGs, see
:any:`plot_rag_boundary`.

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

from skimage import data, segmentation, filters, color
from skimage.future import graph
#from matplotlib import pyplot as plt


def weight_boundary(graph, src, dst, n):
    """
    Handle merging of nodes of a region boundary region adjacency graph.

    This function computes the `"weight"` and the count `"count"`
    attributes of the edge between `n` and the node formed after
    merging `src` and `dst`.


    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the "weight" and "count" attributes to be
        assigned for the merged node.

    """
    default = {'weight': 0.0, 'count': 0}

    count_src = graph[src].get(n, default)['count']
    count_dst = graph[dst].get(n, default)['count']

    weight_src = graph[src].get(n, default)['weight']
    weight_dst = graph[dst].get(n, default)['weight']

    count = count_src + count_dst
    return {
        'count': count,
        'weight': (count_src * weight_src + count_dst * weight_dst)/count
    }


def merge_boundary(graph, src, dst):
    """Call back called before merging 2 nodes.

    In this case we don't need to do any computation here.
    """
    pass


def segm_rag_merg(img, gray):
    edges = filters.sobel(gray)
    labels = segmentation.slic(img, compactness=30, n_segments=700, start_label=1)
    g = graph.rag_boundary(labels, edges)

    #graph.show_rag(labels, g, img)
    #plt.title('Initial RAG')

    labels2 = graph.merge_hierarchical(labels, g, thresh=0.01, rag_copy=False,
                                    in_place_merge=True,
                                    merge_func=merge_boundary,
                                    weight_func=weight_boundary)

    #graph.show_rag(labels, g, img)
    #plt.title('RAG after hierarchical merging')
    #plt.figure()

    out = color.label2rgb(labels2, img, kind='avg', bg_label=0)

    return out


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
    image_file_name = 'a2_a_cropped.jpg'
    from well_plate_project.config import data_dir
    path = data_dir / 'raw' / 'Cropped'
    image_file = path /  image_file_name
    assert image_file.is_file()

    img = cv2.imread(str(image_file))
    plt.imshow(img)
    plt.show()
    return img



if __name__ == "__main__":
    clear_all()
    original_image = load_test_file()
    
    print("Testing ... ")


    img = cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(original_image,cv2.COLOR_BGR2GRAY)
    #gray=color.rgb2gray(img)
    out = segm_rag_merg(img, gray)
    
    print("Plotting... ")
    plt.imshow(out)
    plt.title('Final segmentation')
    plt.show()
