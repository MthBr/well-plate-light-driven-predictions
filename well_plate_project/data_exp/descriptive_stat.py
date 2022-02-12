#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 15:04:03 2021

@author: enzo
"""

#%% 
def plot_save_GMM(bins, count_r, count_g, count_b, type_name, target_filename):
    fig = plt.figure()
    plt.bar(bins[:-1], count_r/(count_r.sum()), color='r', alpha=0.7)
    plt.bar(bins[:-1], count_g/(count_g.sum()), color='g', alpha=0.7)
    plt.bar(bins[:-1], count_b/(count_b.sum()), color='b', alpha=0.7)
    plt.xlabel('Intensity Value')
    
    from sklearn.mixture import GaussianMixture
    import scipy.stats as stats
    # Fit GMM
    gmm = GaussianMixture(n_components = 7)
    
    
    f_r= (count_r/(count_r.sum())).reshape(-1,1)
    gmm = gmm.fit(f_r)
    weights = gmm.weights_
    means = gmm.means_
    covars = gmm.covariances_ 
    f_axis = f_r.copy().ravel()
    f_axis.sort()
    #plt.plot(f_axis,weights[0]*stats.norm.pdf(f_axis,means[0],np.sqrt(covars[0])).ravel(), c='red')
    #plt.plot(f_axis,weights[1]*stats.norm.pdf(f_axis,means[1],np.sqrt(covars[1])).ravel(), c='red')

    plt.title(type_name)
    #target_filename =  'rgb_'+ type_name + '_all' +'.jpg'
    target_path = out_fold / target_filename   
    plt.savefig(str(target_path))
    
    return fig


#%% 
def plot_save(bins, count_r, count_g, count_b, type_name, target_filename):
    fig = plt.figure()
    plt.bar(bins[:-1], count_r/(count_r.sum()), color='r', alpha=0.5, bottom=0.001)
    plt.bar(bins[:-1], count_g/(count_g.sum()), color='g', alpha=0.5, bottom=0.001)
    plt.bar(bins[:-1], count_b/(count_b.sum()), color='b', alpha=0.5, bottom=0.001)
    plt.xlabel('Intensity Value')
    plt.ylabel('Relative Frequency')
    plt.yscale('log')
    plt.ylim((0,0.3))  #043   043   07

    #plt.title(type_name)
    target_path = out_fold / target_filename   
    plt.savefig(str(target_path), dpi=600)
    
    return fig

#%% 





from well_plate_project.config import data_dir, reportings_dir
import cv2
import numpy as np

image_folder = data_dir / 'raw' / 'exp_v2' / 'luce_ult_easy' # luce_nat_easy  luce_ult_easy

out_fold = reportings_dir / 'stats' # luce_nat luce_nat_es_hw luce_nat_es_hw_hist  UV


nb_bins = 256
count_r_hw = np.zeros(nb_bins)
count_r = np.zeros(nb_bins)
count_g_hw = np.zeros(nb_bins)
count_g = np.zeros(nb_bins)
count_b_hw = np.zeros(nb_bins)
count_b = np.zeros(nb_bins)

for folder in image_folder.iterdir():
    print(f'Processing {folder.name}', end='\n')
    ind = 0
    for jpg in folder.rglob('*.jpg'): #TODO add jpeg
        print(f'Processing {jpg}', end='\n')
        image_name = jpg.stem
        img1 = cv2.imread(str(jpg))
        b,g,r = cv2.split(img1)
        hist_r = np.histogram(r, bins=nb_bins, range=[0, 255])
        hist_g = np.histogram(g, bins=nb_bins, range=[0, 255])
        hist_b = np.histogram(b, bins=nb_bins, range=[0, 255])
        if image_name.startswith('2020'):      
            count_r += hist_r[0]
            count_g += hist_g[0]
            count_b += hist_b[0]
        else:
            count_r_hw += hist_r[0]
            count_g_hw += hist_g[0]
            count_b_hw += hist_b[0]
            
        
        
#%%     
import matplotlib.pyplot as plt
 
type_name= 'luce_ult_easy3'

bins = hist_r[1]


target_filename =  'rgb_'+ type_name + '_sams' '.jpg'
plot_save(bins, count_r, count_g, count_b, type_name, target_filename)


target_filename =  'rgb_'+ type_name + '_hw' '.jpg'
plot_save(bins, count_r_hw, count_g_hw, count_b_hw, type_name, target_filename)



count_r_all = count_r + count_r_hw
count_g_all = count_g + count_g_hw
count_b_all = count_b + count_b_hw
target_filename =  'rgb_'+ type_name + '_all' +'.jpg'
plot_save(bins, count_r_all, count_g_all, count_b_all, type_name, target_filename)







#%%
fig = plt.figure()
plt.bar(bins[:-1], count_r/(1), color='r', alpha=0.7)
plt.bar(bins[:-1], count_g/(1), color='g', alpha=0.7)
plt.bar(bins[:-1], count_b/(1), color='b', alpha=0.7)
plt.xlabel('Intensity Value')

from sklearn.mixture import GaussianMixture
import scipy.stats as stats
# Fit GMM
gmm = GaussianMixture(n_components = 7, covariance_type='full')

f_r= (count_r/(1)).reshape(-1,1)


gmm = gmm.fit(f_r)

gmm_x=bins[:-1]
#%% TEST

f_r = np.exp(-np.power(gmm_x - 150, 2.) / (2 * np.power(10, 2.))).reshape(-1, 1)
gmm = GaussianMixture(n_components = 1, covariance_type='full')
gmm = gmm.fit(f_r)
gmm_y = np.exp(gmm.score_samples(gmm_x.reshape(-1, 1)))

plt.figure()
plt.plot(gmm_x, f_r, color="crimson", lw=4, label="GMM")
plt.show()

plt.figure()
plt.plot(gmm_x, gmm_y, color="crimson", lw=4, label="GMM")
plt.show()

#%%



gmm_y = gmm.score_samples(gmm_x.reshape(-1, 1))

plt.plot(gmm_x, gmm_y, color="crimson", lw=4, label="GMM")


weights = gmm.weights_
means = gmm.means_
covars = gmm.covariances_ 
f_axis = f_r.copy().ravel()
f_axis.sort()
#plt.plot(f_axis,weights[0]*stats.norm.pdf(f_axis,means[0],np.sqrt(covars[0])).ravel(), c='red')
#plt.plot(f_axis,weights[1]*stats.norm.pdf(f_axis,means[1],np.sqrt(covars[1])).ravel(), c='red')

plt.title(type_name)
target_filename =  'rgb_'+ type_name + '_all' +'.jpg'
target_path = out_fold / target_filename   
plt.savefig(str(target_path))

























#%% TEST
#https://stackoverflow.com/questions/35990467/fit-two-gaussians-to-a-histogram-from-one-set-of-data-python

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Define simple gaussian
def gauss_function(x, amp, x0, sigma):
    return amp * np.exp(-(x - x0) ** 2. / (2. * sigma ** 2.))

# Generate sample from three gaussian distributions
samples = np.random.normal(-0.5, 0.2, 2000)
samples = np.append(samples, np.random.normal(-0.1, 0.07, 5000))
samples = np.append(samples, np.random.normal(0.2, 0.13, 10000))
gmm_x=(bins[:-1]/bins[:-1].max()) +10
samples = np.exp(-np.power(gmm_x - 0.5, 2.) / (2 * np.power(0.3, 2.)))
# samples= (count_r/(count_r.max()))


# Fit GMM
gmm = GaussianMixture(n_components=1, covariance_type="full", tol=0.001)
gmm = gmm.fit(X=np.expand_dims(samples, 1))

# Evaluate GMM
#gmm_x = np.linspace(-2, 1.5, 5000)
#gmm_x=bins[:-1]
gmm_y = np.exp(gmm.score_samples(gmm_x.reshape(-1, 1)))

# Construct function manually as sum of gaussians
gmm_y_sum = np.full_like(gmm_x, fill_value=0, dtype=np.float32)
for m, c, w in zip(gmm.means_.ravel(), gmm.covariances_.ravel(), gmm.weights_.ravel()):
    gauss = gauss_function(x=gmm_x, amp=1, x0=m, sigma=np.sqrt(c))
    gmm_y_sum += gauss / np.trapz(gauss, gmm_x) * w

# Make regular histogram
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[8, 5])
#ax.hist(samples, bins=gmm_x, density=None, alpha=0.5, color="#0070FF")
ax.plot(gmm_x, samples, color="crimson", lw=4, label="GMM")
ax.plot(gmm_x, gmm_y, color="blue", lw=4, label="GMM")
#ax.plot(gmm_x, gmm_y_sum, color="black", lw=4, label="Gauss_sum", linestyle="dashed")

# Annotate diagram
ax.set_ylabel("Probability density")
ax.set_xlabel("Arbitrary units")

# Make legend
plt.legend()

plt.show()

#%%

