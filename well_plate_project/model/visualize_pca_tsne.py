#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 12:03:55 2020

@author: enzo
"""
#%%
import pandas as pd
    

ml_df_name = 'all_ml_df.pkl'

from well_plate_project.config import data_dir
path = data_dir / 'processed' / 'luce_nat'  #UV luce_nat
df_file = path /  ml_df_name
assert df_file.is_file()
ml_df_FULL = pd.read_pickle(str(df_file))



ml_df = ml_df_FULL[ml_df_FULL['mock']==False]


#%% Machine learning


from sklearn.preprocessing import StandardScaler

non_features=['well_plate_name', 'well_name', 'class_target', 'value_target', 'wp_image_prop','wp_image_version', 'dict_values']
features = ml_df.columns.difference(non_features)


#clan features
#features_red = [col for col in features if not col.endswith('mean')]
#features_red = [col for col in features_red if not col.endswith('skewness')]
#features = features_red


# Separating out the features
x = ml_df.loc[:, features].values
# Separating out the target
y_real = ml_df.loc[:,['value_target']].values
y_class= ml_df.loc[:,['class_target']].values
y_real_class = ml_df.loc[:,['value_target','class_target']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)


targets = ml_df['class_target'].unique()



#%% TODO
#TODO pca su meno componenti (solo le 3 che siaccavallano)
#TODO t-sne


#%% Visualization & Statistics!


#%% Visualize 2D Projection
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, ml_df[['class_target']]], axis = 1)

import matplotlib.pyplot as plt
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ml_df['class_target'].unique()
#colors = ['r', 'g', 'b']
for target in targets:
    indicesToKeep = finalDf['class_target'] == target
    plt.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               #, c = color
               , cmap = "nipy_spectral"
               , s = 50)
ax.legend(targets)
ax.grid()




#%% Visualize 2D Projection

import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.datasets import load_boston

n_components = 4

pca = PCA(n_components=n_components)
components = pca.fit_transform(x)

total_var = pca.explained_variance_ratio_.sum() * 100
targets = ml_df['class_target'].unique()
labels = {str(i): f"PC {i+1}" for i in range(n_components)}
labels['color'] = 'Median Price'

fig = px.scatter_matrix(
    components,
    #color=targets,
    dimensions=range(n_components),
    labels=labels,
    title=f'Total Explained Variance: {total_var:.2f}%',
)
fig.update_traces(diagonal_visible=False)
fig.show()





import plotly.express as px

df = px.data.iris()
features = ["sepal_width", "sepal_length", "petal_width", "petal_length"]

fig = px.scatter_matrix(
    df,
    dimensions=features,
    color="species"
)
fig.update_traces(diagonal_visible=False)
fig.show()







#%% Visualize 3D Projection

from sklearn.preprocessing import RobustScaler
x = ml_df.loc[:, features].values
x_normalized = StandardScaler().fit_transform(x)




from sklearn.decomposition import PCA
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['pc 1', 'pc 2', 'pc 3' ])
finalDf = pd.concat([principalDf, ml_df[['class_target']]], axis = 1)

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111 , projection='3d') 
ax.set_title('2 component PCA', fontsize = 20)
ax.view_init(-90, 0)
targets = ml_df['class_target'].unique()
#colors = ['r', 'g', 'b']
for target in targets:
    indicesToKeep = finalDf['class_target'] == target
    plt.scatter(finalDf.loc[indicesToKeep, 'pc 1']
               , finalDf.loc[indicesToKeep, 'pc 2']
               , finalDf.loc[indicesToKeep, 'pc 3']
               #, c = color
               , cmap = "nipy_spectral")
ax.legend(targets)
ax.grid()




ax.view_init(30, 30)
plt.draw()



for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)











pca = PCA(n_components=3)
components = pca.fit_transform(x_normalized)
components.show()
# import plotly.express as px
# var = pca.explained_variance_ratio_.sum()
# fig = px.scatter_3d(components, x=0, y=1, z=2, 
#                     color=ml_df['class_target'],title=f'Total Explained Variance: {var}',
#                     labels={'0':'PC1', '1':'PC2', '2':'PC3'})
# fig.show()



pca = PCA(scale=Flase, projection=3, n_components=3)

pca.show()









#%% Statistics

ml_df.dtypes
ml_df = ml_df.astype({'value_target': 'float'})
ml_df.dtypes

prop = ml_df.describe(include='all')



import matplotlib.pyplot as plt
from well_plate_project.config import reportings_dir
out_folder =  reportings_dir / 'figures' 






params = {'axes.titlesize':'3',
          'xtick.labelsize':'2',
          'ytick.labelsize':'2'}
plt.rcParams.update(params)



fig1 = plt.figure()  
fig_name = 'boxplot.svg'
target_path = out_folder / fig_name      
boxplot = ml_df.boxplot()
fig1.savefig(str(target_path), format="svg")
plt.close()










fig1 = plt.figure(dpi=1200)
fig_name = 'boxplot.png'
target_path = out_folder / fig_name      
boxplot = ml_df.boxplot(figsize=(10,9))
plt.xticks(rotation = 90)
fig1.savefig(str(target_path), format="png",  dpi=1200)
plt.close()




fig_name = 'hist.png'; target_path = out_folder / fig_name
plt.figure(dpi=1200)
fig = ml_df.hist(figsize=(10,9), ec="k", xlabelsize = 3, ylabelsize = 3)
#[x.title.set_size(2) for x in fig.ravel()]
#fig = ax_hist.get_figure()
#fig.savefig(str(target_path), format="svg")
plt.tight_layout()
plt.savefig(str(target_path), format="png",  dpi=1200) #,dpi=9000
plt.close()







fig_name = 'hist_grp.png'; target_path = out_folder / fig_name
plt.figure(dpi=1200)
ax_hist = ml_df.groupby('class_target').hist(figsize=(10,9), ec="k")
plt.tight_layout()
plt.savefig(str(target_path), format="png",  dpi=1200) #,dpi=9000
plt.close()





# fig1 = plt.figure(); fig_name = 'scatter.svg';   target_path = out_folder / fig_name
# from pandas.plotting import scatter_matrix
# fig_scatt = scatter_matrix(ml_df, alpha=0.2, figsize=(70, 70), diagonal='kde')
# fig1.savefig(str(target_path), format="svg")








