#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 11:53:45 2020

@author: enzo
"""

# from well_plate_project.model import majority_voting


#%% Importing
import pandas as pd
import numpy as np 

ml_df_name = 'all_ml_df.pkl'

from well_plate_project.config import data_dir
path = data_dir / 'processed' / 'luce_nat'  # UV
df_file = path /  ml_df_name
assert df_file.is_file()
ml_df_FULL = pd.read_pickle(str(df_file))

#cleaning
ml_df = ml_df_FULL[ml_df_FULL['mock']==False]

#%% Feature preparation

from sklearn.preprocessing import StandardScaler

wells = ml_df[['well_plate_name', 'wp_image_version']].copy()
#wells['unique'] = wells['well_plate_name'] + '_'+ wells['wp_image_version']


wells = wells.drop_duplicates()
from sklearn.model_selection import train_test_split
w_train, w_test = train_test_split(wells, test_size=0.20)

X_train = pd.merge(ml_df, w_train, on=['well_plate_name', 'wp_image_version'],right_index=True)

X_test = pd.merge(ml_df, w_test, on=['well_plate_name', 'wp_image_version'],right_index=True)


y_train = ml_df.loc[X_train.index, 'class_target' ]
y_test = ml_df.loc[X_test.index, 'class_target' ]


X_train = X_train.drop(columns=['well_plate_name', 'wp_image_version'])

X_test = X_test.drop(columns=['well_plate_name', 'wp_image_version'])



# Standardizing the features


non_features=['well_plate_name', 'well_name', 'class_target', 'value_target', 'wp_image_prop','wp_image_version', 'dict_values']
features = list(ml_df.columns.difference(non_features))

# Separating out the features
X_train = X_train[features]
X_test = X_test[features]


scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train.values), 
                 index=X_train.index, columns=X_train.columns)

X_test = pd.DataFrame(scaler.transform(X_test.values), 
                 index=X_test.index, columns=X_test.columns)


# Separating out the target


targets = ml_df['class_target'].unique()
labels = np.sort(ml_df['class_target'].unique())



#%% Predict mlp

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(256,128,64,32),activation="relu",random_state=1).fit(X_train, y_train)
y_mlp_pred=mlp.predict(X_test)
y_mlp_prob=mlp.predict_proba(X_test)

# print("MLPClassifier")
# print(classification_report(y_test, y_pred, target_names=targets))
# from sklearn.metrics import plot_confusion_matrix
# import matplotlib.pyplot as plt
# fig=plot_confusion_matrix(clf, X_test, y_test,display_labels=targets)
# fig.figure_.suptitle("Confusion Matrix ")
# plt.show()



#%% Predict Random Forest

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=50)
rf.fit(X_train, y_train)
y_rf_pred=rf.predict(X_test)
y_rf_prob=rf.predict_proba(X_test)


# from sklearn.metrics import classification_report
# print("RandomForestClassifier")
# print(classification_report(y_test, y_pred, target_names=targets))

# feat = pd.Series(clf.feature_importances_, index = features)

# plt.figure(figsize=(10,30))
# feat.sort_values().plot(kind='barh')

#%% Predict XGB

import xgboost as XGB
xgb = XGB.XGBClassifier()
xgb.fit(X_train, y_train)
y_xgb_pred = xgb.predict(X_test)
y_xgb_prob = xgb.predict_proba(X_test)

# print("xgboost")
# print(classification_report(y_test, y_pred, target_names=targets))

# feat = pd.Series(xgb.feature_importances_, index = features)

# plt.figure(figsize=(10,30))
# feat.sort_values().plot(kind='barh')
# plt.show()

#%% Combiner

y_tot_prob = pd.DataFrame((y_xgb_prob+y_rf_prob+y_mlp_prob)/3, index = X_test.index, columns=labels)
y_pred = pd.DataFrame({'label': y_tot_prob.idxmax(axis=1),
                       'confidence': y_tot_prob.max(axis=1),
                       'actual_label': ml_df.class_target,
                       'wellplate': ml_df.well_plate_name,
                       'well': ml_df.well_name}, index=X_test.index)#.dropna()

# y_test = pd.DataFrame(y_test); y_test['banana'] = y_test.index

import numpy as np

from sklearn.metrics import classification_report
print("Mean")

# y_pred = xgb.classes_[np.argmax(y_tot_prob.values, axis=1)]
print(classification_report(y_test, y_pred.label, target_names=targets))








