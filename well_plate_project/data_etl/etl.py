#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Version 1, Extraction Transformation Loading
Cropping the images in a folder and identifying the wells
@author:
"""
#%% Clean Memory and import pakages
for i in list(globals().keys()):
    if(i[0] != '_'):
        exec('del {}'.format(i))



from well_plate_project.data_etl.etl_utils import crop_pipeline, image_pipeline, circles_image_pipeline

from well_plate_project.config import data_dir, reportings_dir

#%%  Dataset paths
raw_data_dir = data_dir / 'raw'
interm_data_dir = raw_data = data_dir / 'intermediate'
describe_data_dir = reportings_dir / 'description'



#%%  all images construction
import cv2
import matplotlib.pyplot as plt


#%%  Search the source folder for scanned JPG files, auto crop and save each to the target folder


match_folder = data_dir / 'raw' / 'Match'
source_folder =  data_dir / 'raw' / 'EXPERIMENTS'
source_folder =  data_dir / 'raw' / 'foto_tel1'
target_folder = data_dir / 'raw' / 'EXPERIMENTS_Crp'

for jpg in source_folder.glob('*.jpg'):
    print(f'Processing {jpg}', end='\n')
    circled_image = crop_pipeline(jpg)
    target_filename =  jpg.stem + '_cropped' + jpg.suffix
   # target_filename =  jpg.stem + '_y_yk3_ragYlab' + jpg.suffix   #jpg.name
    #'rag_lab_b7_' + 
    target_path = target_folder / target_filename
    cv2.imwrite(str(target_path), circled_image)


print('Done')

