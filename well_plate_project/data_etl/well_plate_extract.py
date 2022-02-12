#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 10:20:09 2020

@author: enzo
"""
import pandas as pd

def map_dict_worksheet(basic_structure_df, version, dict_weel_plates):
    
    row=8
    col=12
    row_name=list(map(chr, range(ord('A'), ord('H')+1)))
    col_name=[str(each) for each in range(1,13)]


    last_letter = 'V'
    well_plate_names = list(map(chr, range(ord('A'), ord('I')+1)))  #Italian order! -J,K
    well_plate_names.extend(list(map(chr, range(ord('L'), ord(last_letter)+1))))
    
    count = 0
    for x,y in zip(well_plate_names[0::2], well_plate_names[1::2]):
        print(x, '+', y)
        
        start_row =  1 + count*3 + count*row
        end_row =  start_row + row
        
        dict_weel_plates[x+version] = basic_structure_df.iloc[start_row:end_row, 1:1+col]
        dict_weel_plates[x+version].columns = col_name
        dict_weel_plates[x+version].index = row_name
        
        dict_weel_plates[y+version] = basic_structure_df.iloc[start_row:end_row, (1+col+2):(1+col+2)+col]
        dict_weel_plates[y+version].columns = col_name
        dict_weel_plates[y+version].index = row_name
        count += 1
    
    return dict_weel_plates


def read_excel(file_xls):
    import pandas as pd
    df_dict=pd.read_excel(file_xls,None)
    
    return df_dict


def map_df_worksheet(basic_structure_df, version, df_weel_plates):
    import unittest
    case = unittest.TestCase()
    case.assertListEqual(df_weel_plates.columns.tolist(), ['well_plate_name', 'well_name', 'class_target', 'value_target'])
    
    row=8
    col=12
    row_name=list(map(chr, range(ord('A'), ord('H')+1)))
    col_name=[str(each) for each in range(1,13)]
    well_names = [f'{a}{b}' for a in row_name for b in col_name]

    last_letter = 'V'
    well_plate_names = list(map(chr, range(ord('A'), ord('I')+1)))  #Italian order! -J,K
    well_plate_names.extend(list(map(chr, range(ord('L'), ord(last_letter)+1))))
    
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



#%% testings



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




def test_dict():
    clear_all()
    file_name = 'Matrici multiwell.xlsx'
    xls_file = load_test_file(file_name)
    dict_df = read_excel(str(xls_file))
    
    keys = list(dict_df.keys())
    KEY_WORD = "Matrici "
    dict_weel_plates = {}
    
    worksheet = keys[0]
    version = '1' if not worksheet[-1].isnumeric() else worksheet[-1]
    basic_structure_df = dict_df[worksheet]
    dict_weel_plates = map_dict_worksheet(basic_structure_df, version, dict_weel_plates)
    
    
    worksheet = keys[1]
    version = '1' if not worksheet[-1].isnumeric() else worksheet[-1]
    basic_structure_df = dict_df[worksheet]
    dict_weel_plates = map_dict_worksheet(basic_structure_df, version, dict_weel_plates)
    
    from well_plate_project.config import data_dir
    import pickle
    print("Saving...")  
    target_filename =  'matrici_multiwell' + '_dict_df' + '.pkl'
    target_path = data_dir / 'raw' / 'matrix' / target_filename
    with open(str(target_path),"wb+") as file:
        pickle.dump(dict_weel_plates, file)
    print("Done")
    return 0
    

if __name__ == "__main__":
    clear_all()
    file_name = 'Matrici multiwell.xlsx'
    xls_file = load_test_file(file_name)
    dict_df = read_excel(str(xls_file))
    
    keys = list(dict_df.keys())
    KEY_WORD = "Matrici "
    
    columns = ['well_plate_name', 'well_name', 'class_target', 'value_target']
    
    df_weel_plates = pd.DataFrame(columns=columns)
    
    worksheet = keys[0]
    version = '1' if not worksheet[-1].isnumeric() else worksheet[-1]
    basic_structure_df = dict_df[worksheet]
    df_weel_plates = map_df_worksheet(basic_structure_df, version, df_weel_plates)
    
    
    worksheet = keys[1]
    version = '1' if not worksheet[-1].isnumeric() else worksheet[-1]
    basic_structure_df = dict_df[worksheet]
    df_weel_plates = map_df_worksheet(basic_structure_df, version, df_weel_plates)
    
    from well_plate_project.config import data_dir
    import pickle
    print("Saving...")  
    target_filename =  'matrici_multiwell' + '_df' + '.pkl'
    target_path = data_dir / 'raw' / 'matrix' / target_filename
    with open(str(target_path),"wb+") as file:
        pickle.dump(df_weel_plates, file)
    print("Done")


