#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 10:20:09 2020

@author: enzo
"""
#%% Utils
import pandas as pd


def read_excel(file_xls):
    import pandas as pd
    df_dict=pd.read_excel(file_xls,None)
    
    return df_dict


def map_df_worksheet(basic_structure_df, version, df_weel_plates, well_plate_names):
    import unittest
    case = unittest.TestCase()
    case.assertListEqual(df_weel_plates.columns.tolist(), ['well_plate_name', 'well_name', 'class_target', 'value_target'])
    
    row=8
    col=12
    row_name=list(map(chr, range(ord('A'), ord('H')+1)))
    col_name=[str(each) for each in range(1,13)]
    well_names = [f'{a}{b}' for a in row_name for b in col_name]

    count = 0
    for x,y in zip(well_plate_names[0::2], well_plate_names[1::2]):
        print(x, '+', y)
        
        start_row =  1 + count*4 + count*row
        end_row =  start_row + row
        
        temp_dataframe = basic_structure_df.iloc[start_row:end_row, 1:1+col]
        temp_dataframe.columns = list(basic_structure_df.iloc[start_row-1, 1:1+col])
        temp_dataframe.index = row_name
        stacked = temp_dataframe.stack().reset_index()
        stacked.insert(loc=0, column='well_name', value=well_names, allow_duplicates = False)
        stacked.insert(0, 'well_plate_name', x+version)
        stacked = stacked.drop(['level_0'], axis=1)
        stacked = stacked.rename(columns={"level_1": "class_target", 0: "value_target"})
        #stacked.set_index(['well_plate_name','well_name'])
        df_weel_plates=pd.concat([df_weel_plates, stacked], ignore_index=True) #, sort=True)
        
        
        temp_dataframe = basic_structure_df.iloc[start_row:end_row, (1+col+2):(1+col+2)+col]
        temp_dataframe.columns = list(basic_structure_df.iloc[start_row-1, (1+col+2):(1+col+2)+col])
        temp_dataframe.index = row_name
        stacked = temp_dataframe.stack().reset_index()
        stacked.insert(loc=0, column='well_name', value=well_names[:len(stacked)], allow_duplicates = False)
        stacked.insert(0, 'well_plate_name', y+version)
        stacked = stacked.drop(['level_0'], axis=1)
        stacked = stacked.rename(columns={"level_1": "class_target", 0: "value_target"})
        #stacked.set_index(['well_plate_name','well_name'])
        df_weel_plates=pd.concat([df_weel_plates, stacked], ignore_index=True) #, sort=True)

        count += 1
    
    return df_weel_plates



#%% matchfeature
def  reform(elmnts):
    result = {'_'.join([l0_key, l1_key, l2_key]): values for l0_key, l0_dict in elmnts.items() for l1_key, l1_dict in l0_dict.items() for l2_key, values in l1_dict.items() }
    return result

def match_target_feat(data_dict_df, target_df):
    columns = ['well_plate_name', 'wp_image_version', 'wp_image_prop', 'well_name', 'dict_values']
    df_weel_plates = pd.DataFrame(columns=columns)
    
    #Test
    #key_name='01_2'
    #dict_list = data_dict_df[key_name]
    
    
    for key_name, dict_list in data_dict_df.items():
        tags_dict = dict_list['img_tags']
        #ser_tag = pd.Series(tags_dict)
        dataframe=dict_list['well_features']
        stacked = dataframe.stack().reset_index()
        stacked.insert(loc=0, column='well_name', value=(stacked["level_0"] + stacked["level_1"].astype(str)), allow_duplicates = False)
        stacked = stacked.drop(['level_0'], axis=1)
        stacked = stacked.drop(['level_1'], axis=1)
        name_vect = key_name.split("_")
        stacked.insert(0, 'wp_image_version', name_vect[1])
        stacked.insert(0, 'well_plate_name', name_vect[0].upper())
        stacked = stacked.rename(columns={0: "dict_values"})
        for ind, value in tags_dict.items():
            stacked.insert(0,ind, value)
        df_weel_plates = df_weel_plates.append(stacked)


    reformed = df_weel_plates['dict_values'].apply(lambda d: pd.Series(reform(d)))
    result = pd.concat([df_weel_plates, reformed], axis=1, sort=False)

    ml_df = pd.merge(target_df,result,on=['well_plate_name', 'well_name' ])
    print("Done")
    return ml_df

#%% INIT

def clear_all():
    """Clears all the variables from the workspace of the application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]


def load_test_file(file_name = 'Matrici multiwell.xlsx'): 
    from well_plate_project.config import data_dir
    path = data_dir / 'raw' / 'matrix'
    file = path /  file_name
    assert file.is_file()
    return file

    

def extract_featuresAB(xls_file_name):
    xls_file = load_test_file(xls_file_name)
    dict_df = read_excel(str(xls_file))
    
    keys = list(dict_df.keys())
    
    columns = ['well_plate_name', 'well_name', 'class_target', 'value_target']
    
    df_well_plates = pd.DataFrame(columns=columns)
    
    
    last_letter = 'V'
    well_plate_names = list(map(chr, range(ord('A'), ord('I')+1)))  #Italian order! -J,K
    well_plate_names.extend(list(map(chr, range(ord('L'), ord(last_letter)+1))))
    
    worksheet = keys[0]
    version = '1' if not worksheet[-1].isnumeric() else worksheet[-1]
    basic_structure_df = dict_df[worksheet]
    df_well_plates = map_df_worksheet(basic_structure_df, version, df_well_plates, well_plate_names)
    
    
    worksheet = keys[1]
    version = '1' if not worksheet[-1].isnumeric() else worksheet[-1]
    basic_structure_df = dict_df[worksheet]
    df_well_plates = map_df_worksheet(basic_structure_df, version, df_well_plates, well_plate_names)
    
    from well_plate_project.config import data_dir
    import pickle
    print("Saving...")  
    target_filename =  'matrici_multiwell' + '_df' + '.pkl'
    target_path = data_dir / 'raw' / 'matrix' / target_filename
    with open(str(target_path),"wb+") as file:
        pickle.dump(df_well_plates, file)
    print("Done")
    return df_well_plates
    

def extract_features_xls(xls_file_name):
    xls_file = load_test_file(xls_file_name)
    dict_df = read_excel(str(xls_file))
    
    keys = list(dict_df.keys())
    
    columns = ['well_plate_name', 'well_name', 'class_target', 'value_target']
    
    df_well_plates = pd.DataFrame(columns=columns)
    
    worksheet = keys[1] #SECOND worksheet
    version = ''
    basic_structure_df = dict_df[worksheet]
    well_plate_names = list(map(lambda s: f'{s:02d}', range(1, 10+1)))
    df_well_plates = map_df_worksheet(basic_structure_df, version, df_well_plates, well_plate_names)
    
    from well_plate_project.config import data_dir
    import pickle
    print("Saving...")  
    target_filename =  'piastra_novembre' + '_df' + '.pkl'
    target_path = data_dir / 'raw' / 'matrix' / target_filename
    with open(str(target_path),"wb+") as file:
        pickle.dump(df_well_plates, file)
    print("Done")
    return df_well_plates
    

if __name__ == "__main__":
    clear_all()
    #xls_file_name = 'Matrici multiwell.xlsx'
    #extract_featuresAB(xls_file_name)
    
    #xls_file_name = 'PIASTRA novembre.xlsx'
    #extract_features(xls_file_name)





    



    

