#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 15:40:41 2020

@author: enzo
"""
#%% IMPORT LIBRARIES
from well_plate_project.preprocessing import single_channel_equalize, circle_extract, matching_info, circle_well, horotate
from well_plate_project.preprocessing import calculate_keypoints, feature_matching, extract_matrix, circle_well_mock
from well_plate_project.preprocessing import match_target_feat, extract_img_tags, extract_featuresAB

import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import pickle
import numpy as np
from well_plate_project.config import data_dir
#%% IMPORT struc
path_query =  data_dir / 'raw' / 'Match'
image_file_structure = path_query / 'UV_crp.jpg' #1ln_b 2uv_b
assert image_file_structure.is_file()
imgR_structure = cv2.imread(str(image_file_structure)) 
plt.imshow(imgR_structure)

#%% IMPORT FILES
image_file = path_query / 'UV_big.jpg'  #1ln  2uv
assert image_file.is_file()
imgR = cv2.imread(str(image_file)) 
plt.imshow(imgR)


#%% Image train feature creation
plt.close('all')
mask_query_circ, reference_wells = circle_extract(single_channel_equalize(imgR, c_map = 'LAB', channel = [1,0,1]),
                                                  dp = 2.9, param1=None, param2=None, plot=True) #2.93
imgR_single_channel = single_channel_equalize(imgR_structure, eq_method = 'clahe', c_map = 'LAB',  channel = [0,0,1])
keypoints_reference, descriptions_reference, model = calculate_keypoints(imgR_single_channel)

target_df = extract_featuresAB('Matrici multiwell.xlsx')

#%% Import of the train image

data_dict_df = {}
path_train = data_dir / 'raw' / 'exp_v2' / 'UV'
output_folder = data_dir / 'raw' / 'exp_v2_crp' / 'UV' # EXPERIMENTS_Crp  foto_tel1_crp



for folder in path_train.iterdir():
    print(f'Processing {folder.name}', end='\n')
    ind = 0
    for jpg in folder.rglob('*.jpg'):
        print(f'Processing {jpg}', end='\n')
        out_fold = Path(str(jpg).replace(str(path_train), str(output_folder))).parent
        print(f'Out Folder {out_fold}', end='\n')
        image_name = jpg.stem
        img1 = cv2.imread(str(jpg))
        img1 = horotate(img1)
        #Feature saving
        img_tags = extract_img_tags(str(jpg))
        data_dict_df[folder.name + '_'+ str(ind)] = img_tags
        
        # Homografy calculation
        img1_single_channel = single_channel_equalize(img1, eq_method = 'clahe', c_map = 'LAB',  channel = [0,0,1])
        keypoints1_query, descriptions1_query, model= calculate_keypoints(img1_single_channel)
        
        err = 10
        lowe_tre = 0.680
        I = np.identity(3)
        err_tresh = 0.75 
        lowe_delta = 0.001 # 001 0025
        while (err>err_tresh and lowe_tre<0.750):
            matches = feature_matching(descriptions_reference, descriptions1_query, matching_type='flann', lowe_threshold=lowe_tre)
            if len(matches) < 70: 
                lowe_tre += lowe_delta
                continue
            matrix, M, matrix_mask = extract_matrix(matches, keypoints_reference, keypoints1_query, imgR, img1)
            err = np.linalg.norm(I - M.dot(matrix) , ord = np.inf)
            pixels = cv2.countNonZero(matrix_mask)
            power = np.linalg.norm(matrix)
            powerM = np.linalg.norm(M)
            print(f'err:{err}, lowe:{lowe_tre}, pixels:{pixels}, power:{power}, len:{len(matches)}')
        
            corrected_mask = cv2.warpPerspective(mask_query_circ, matrix, (img1.shape[1], img1.shape[0]), cv2.WARP_INVERSE_MAP)
            #Verbose debug
            full_dict={}
            #imgR_single_channel  imgR
            info_dict = matching_info(imgR_single_channel, keypoints_reference, img1, keypoints1_query,  matrix, M, matrix_mask, matches, show = False, save = True, 
                                      image_name = image_name + '_'+ str(lowe_tre), out_folder = out_fold)
            full_dict[str(lowe_tre)] = info_dict
            full_dict[str(lowe_tre)].update({'I-n*m':err})
            full_dict[str(lowe_tre)].update({'pixels':pixels})
            full_dict[str(lowe_tre)].update({'power':power})
            full_dict[str(lowe_tre)].update({'powerM':powerM})
            full_dict[str(lowe_tre)].update({'len':len(matches)})
            
            file_name=image_name + '_'+ str(lowe_tre) +"_dict.json"
            target_path = out_fold / file_name
            import json
            with open(str(target_path),"w+") as file:
                json.dump(full_dict, file)
            target_filename =  image_name + '_'+ str(lowe_tre)+'_corr_mask' + '.jpg'
            target_path = out_fold / target_filename   
            cv2.imwrite(str(target_path), corrected_mask)
            
            lowe_tre += lowe_delta
            
        
        reduced = False
        if (err < err_tresh):
            print(f'Error at {image_name}  with err: {err}', end='\n')
            # Feature extract
            well_plate_dict = circle_well(imgR, img1, matrix, reference_wells, reduce=reduced, save = True, image_name = image_name, out_folder = out_fold)
          
            target_filename =  image_name + '_df' + '.pkl'
            target_path = out_fold / target_filename
            with open(str(target_path),"wb+") as file:
                pickle.dump(well_plate_dict, file)
    
            data_dict_df[folder.name + '_'+ str(ind)].update({'well_features' : well_plate_dict})
            data_dict_df[folder.name + '_'+ str(ind)]['img_tags']['mock'] = False
        else:
             well_plate_dict = circle_well_mock(reduce=reduced)
             data_dict_df[folder.name + '_'+ str(ind)].update({'well_features' : well_plate_dict})
             data_dict_df[folder.name + '_'+ str(ind)]['img_tags']['mock'] = True
        
        ind += 1
    
    

#%% Image train feature creation

df_ml = match_target_feat(data_dict_df, target_df)

    
output_folder = data_dir / 'processed' / 'UV'
target_filename =  'all_dict' + '.pkl'
target_path = output_folder / target_filename
with open(str(target_path),"wb+") as file:
    pickle.dump(data_dict_df, file)


output_folder = data_dir / 'processed' / 'UV'
target_filename =  'all_ml_df' + '.pkl'
target_path = output_folder / target_filename
with open(str(target_path),"wb+") as file:
    pickle.dump(df_ml, file)