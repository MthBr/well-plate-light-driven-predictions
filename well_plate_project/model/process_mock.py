#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 09:49:54 2020

@author: enzo
"""
#%% Machine learning
import pandas as pd
    

ml_df_name = 'all_ml_df.pkl'

from well_plate_project.config import data_dir
path = data_dir / 'processed' / 'luce_nat'  #UV luce_nat
df_file = path /  ml_df_name
assert df_file.is_file()
ml_df_FULL = pd.read_pickle(str(df_file))

a=ml_df_FULL.groupby('mock')['full_L_entropy'].count()



#%% Machine learning
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
tips = sns.load_dataset("tips")
sns.boxplot(y='full_L_entropy', data=ml_df_FULL, hue = 'mock'  )
plt.show()

#%% Machine learning


ml_df = ml_df_FULL


from sklearn.preprocessing import StandardScaler

non_features=['well_plate_name', 'well_name', 'class_target', 'value_target', 'wp_image_prop','wp_image_version', 'dict_values', 'mock']
features = ml_df.columns.difference(non_features)


#clan features
#features_red = [col for col in features if not col.endswith('mean')]
#features_red = [col for col in features_red if not col.endswith('skewness')]
#features = features_red


# Separating out the features
x = ml_df.loc[:, features].values
# Separating out the target
ml_df['mock'] = ml_df['mock'].astype('int')
y_class= ml_df['mock'].values
# Standardizing the features
x = StandardScaler().fit_transform(x)


targets = ml_df['mock'].unique().astype('str')

#%% Predict


#TODO linear, tanto per  (non ha senso su molte features)
#SVM
#MLP

#TODO random forest regressor + MAPE, RMSE


#%% SVM

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( x, y_class.ravel(), test_size=0.20, random_state=42)



#Import svm model
from sklearn import svm


#We’ll create two objects from SVM, to create two different classifiers; one with Polynomial kernel, and another one with RBF kernel:
#Train the model using the training sets
rbf = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(X_train, y_train)
poly = svm.SVC(kernel='poly', degree=3, C=1).fit(X_train, y_train)


#To calculate the efficiency of the two models, we’ll test the two classifiers using the test data set:
poly_pred = poly.predict(X_test)
rbf_pred = rbf.predict(X_test)

from sklearn.metrics import accuracy_score, f1_score
#Finally, we’ll calculate the accuracy and f1 scores for SVM with Polynomial kernel:
poly_accuracy = accuracy_score(y_test, poly_pred)
poly_f1 = f1_score(y_test, poly_pred, average='weighted')
print('Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy*100))
print('F1 (Polynomial Kernel): ', "%.2f" % (poly_f1*100))

#In the same way, the accuracy and f1 scores for SVM with RBF kernel:
rbf_accuracy = accuracy_score(y_test, rbf_pred)
rbf_f1 = f1_score(y_test, rbf_pred, average='weighted')
print('Accuracy (RBF Kernel): ', "%.2f" % (rbf_accuracy*100))
print('F1 (RBF Kernel): ', "%.2f" % (rbf_f1*100))


from sklearn.metrics import classification_report
print("SVC-PolynomialKernel")
print(classification_report(y_test, poly_pred, target_names=targets))
print("SVC-RBF")
print(classification_report(y_test, rbf_pred, target_names=targets))


#%% Predict mlp


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( x, y_class.ravel(), test_size=0.20, random_state=42)

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(256,128,64,32),activation="relu",random_state=1).fit(X_train, y_train)
y_pred=clf.predict(X_test)

print("MLPClassifier")
print(classification_report(y_test, y_pred, target_names=targets))
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
fig=plot_confusion_matrix(clf, X_test, y_test,display_labels=targets)
fig.figure_.suptitle("Confusion Matrix ")
plt.show()







#%% Predict

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( x, y_class.ravel(), test_size=0.25, random_state=42)


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
clf = RandomForestClassifier(n_estimators=50)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)


from sklearn.metrics import classification_report
print("RandomForestClassifier")
print(classification_report(y_test, y_pred, target_names=targets))


feat = pd.Series(clf.feature_importances_, index = features)

plt.figure(figsize=(10,30))
feat.sort_values().plot(kind='barh')

import xgboost as XGB
xgb = XGB.XGBClassifier()
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

print("xgboost")
print(classification_report(y_test, y_pred, target_names=targets))

feat = pd.Series(xgb.feature_importances_, index = features)

plt.figure(figsize=(10,30))
feat.sort_values().plot(kind='barh')
plt.show()

#print(classification_report(y_true, y_pred, target_names=target_names))


#%% INIT

def clear_all():
    """Clears all the variables from the workspace of the application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]



if __name__ == "__main__":
    #clear_all()
    
    ml_df_name = 'all_ml_df.pkl'
    
    from well_plate_project.config import data_dir
    path = data_dir / 'processed' / 'luce_nat_easy'
    df_file = path /  ml_df_name
    assert df_file.is_file()
    ml_df = pd.read_pickle(str(df_file))
    
    
    




















