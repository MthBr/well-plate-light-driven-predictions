#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 11:53:45 2020

@author: enzo
"""

# from well_plate_project.model import majority_voting


#%% Importing
import pandas as pd
# from well_plate_project.model import train_reg_model_extended, train_model, train_model_extended, train_model_reg

from well_plate_project.model import combine_class, train_classification_model, train_class_regression_model, train_regression_model


ml_df_name = 'all_ml_df.pkl'

from well_plate_project.config import data_dir
path = data_dir / 'processed' / 'luce_nat_es_hw'  #  UV  luce_ult_es2  luce_nat_es_hw_hist
#luce_nat luce_nat_es_hw UV  luce_nat_es_sam  luce_nat_es_hw_hist
df_file = path /  ml_df_name
assert df_file.is_file()
ml_df_FULL = pd.read_pickle(str(df_file))

#cleaning
ml_df = ml_df_FULL[ml_df_FULL['mock']==False]
index_names = ml_df[(ml_df['well_plate_name']=='08')].index 
ml_df.drop(index_names, inplace = True) 


#%% Importing
#ml_df = ml_df.replace('0 nM', '25 nM')
#ml_df = ml_df.replace('25 nM', '50 nM')
# ml_df[ml_df['class_target']=='0 nM']['class_target'] = '25 nM'

non_features=['well_plate_name', 'well_name', 'class_target','value_target', 'wp_image_prop','wp_image_version', 'dict_values']
#'class_target'
tags = ['ShutterSpeedValue','ApertureValue','BrightnessValue','ExposureBiasValue', 'MaxApertureValue','MeteringMode',
                         'Flash',  'FocalLength', 'ExposureTime', 'Orientation', 'FocalLengthIn35mmFilm','FNumber'  ]
#'ISOSpeedRatings',

channels=["gray", 'blue','green','red','L','a','b','H','S','V']

# non_features.extend(['full_'+s+'_'+"mean" for s in channels])
# non_features.extend(['full_'+s+'_'+"mean_PIL" for s in channels])
# non_features.extend(['full_'+s+'_'+"mean_trm30" for s in channels])
# non_features.extend(['full_'+s+'_'+"stddev" for s in channels])
# non_features.extend(['full_'+s+'_'+"skewness" for s in channels])
# non_features.extend(['full_'+s+'_'+"entropy" for s in channels])
# non_features.extend(['full_'+s+'_'+"entropy2" for s in channels])
# non_features.extend(['full_'+s+'_'+"entropy_glcm" for s in channels])


# channels=["gray", 'blue','green','red','a','S']
# non_features.extend(['full_'+s+'_'+"skewness" for s in channels])
# non_features.extend(['full_'+s+'_'+"entropy" for s in channels])
# non_features.extend(['full_'+s+'_'+"entropy2" for s in channels])
# non_features.extend(['full_'+s+'_'+"entropy_glcm" for s in channels])

#non_features.extend(tags)

features_ml = list(ml_df.columns.difference(non_features))

#%% Extract feature





# a = train_model_reg(ml_df, non_features, target_class='class_target', target_reg = 'value_target', verbose=3)

# train_reg_model_extended(ml_df, non_features, verbose=3)
# a= train_model(ml_df, non_features, verbose=2)
# a=train_model_extended(ml_df, non_features, verbose=3)

# combined_classifier, combined_regression, scaler_X, scaler_y = train_class_regression_model(ml_df, non_features, mode = 'proba', target_class='class_target', 
#                              target_reg = 'value_target', verbose=2)



combined_model, scaler_X = train_classification_model(ml_df, non_features, verbose=4)

# combined_regression, scaler_X, scaler_y = 
# train_regression_model(ml_df, non_features, verbose=4)




#%% Extract feature

ml_df_name = 'all_ml_df.pkl'

path = data_dir / 'processed' / 'luce_nat_inc'  # luce_nat_inc  luce_uv_inc
df_file = path /  ml_df_name
assert df_file.is_file()
ml_df_FULL = pd.read_pickle(str(df_file))

#cleaning
ml_df_inc = ml_df_FULL[ml_df_FULL['mock']==False]

#%%
#ml_df = ml_df.replace('0 nM', '25 nM')
#ml_df = ml_df.replace('25 nM', '50 nM')
# ml_df[ml_df['class_target']=='0 nM']['class_target'] = '25 nM'


features = list(ml_df_inc.columns.difference(non_features))
X = ml_df_inc[features].copy()

import numpy as np
# Standardizing the features  
X.loc[:,X.dtypes ==np.float] = scaler_X.fit_transform(X.loc[:,X.dtypes ==np.float].values)
X = pd.get_dummies(X)
    

        
X = combine_class(X, 'proba',  combined_model, None) 
X = pd.get_dummies(X)

X_classes = X[X.columns[-6:]]

wells = pd.DataFrame({'well_plate_name': ml_df_inc.well_plate_name,
                       'well_name': ml_df_inc.well_name,
                       'wp_version': ml_df_inc.wp_image_version,
                       'class': X_classes[X_classes.idxmax(axis=1)].columns,
                        }, index= X.index)

y_pred = wells.merge(X_classes, left_index=True, right_index=True)



# %%
