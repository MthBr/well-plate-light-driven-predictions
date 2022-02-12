
#%%
import pandas as pd
import pickle
from well_plate_project.preprocessing import extract_features_xls, match_target_feat
target_df = extract_features_xls('PIASTRA novembre.xlsx')


  
from well_plate_project.config import data_dir
ml_df_name = 'all_ml_df.pkl'
dict_filename =  'all_dict.pkl'

path_process = data_dir / 'processed'



folder = path_process / 'luce_uv_inc'

print(f'Processing {folder.name}', end='\n')
df_file = folder /  dict_filename
print(f'Full {df_file}', end='\n')
if df_file.is_file():
    data_dict_df = pd.read_pickle(str(df_file))
    print("mathcing")
    df_ml = match_target_feat(data_dict_df, target_df)
    target_path = folder / ml_df_name
    with open(str(target_path),"wb") as file:
        pickle.dump(df_ml, file)



#%%
for folder in path_process.iterdir():
    print(f'Processing {folder.name}', end='\n')
    df_file = folder /  dict_filename
    print(f'Full {df_file}', end='\n')
    if df_file.is_file():
        data_dict_df = pd.read_pickle(str(df_file))
        print("mathcing")
        df_ml = match_target_feat(data_dict_df, target_df)
        target_path = folder / ml_df_name
        with open(str(target_path),"wb") as file:
            pickle.dump(df_ml, file)
