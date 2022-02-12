#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 09:49:54 2020

@author: enzo
"""

from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


from sklearn.base import ClassifierMixin

class CombClass(ClassifierMixin):
    def __init__(self):
        return
    
    def _buildmodels(self):
        from sklearn.neural_network import MLPClassifier
        mlp = MLPClassifier(hidden_layer_sizes=(256,128,64,32),activation="relu",random_state=1)
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=50,max_features='log2')
        import xgboost as XGB
        xgb = XGB.XGBClassifier()
        self.mlp = mlp
        self.rf = rf
        self.xgb = xgb
    
    def fit(self, X, y):
        self._buildmodels()
        
        self.mlp = self.mlp.fit(X,y)
        self.rf = self.rf.fit(X,y)
        self.xgb = self.xgb.fit(X,y)
        return self
     
    def predict(self, X):
        y_xgb_prob=self.xgb.predict_proba(X)
        y_rf_prob = self.rf.predict_proba(X)
        y_mlp_prob= self.mlp.predict_proba(X)
        # self.proba = pd.DataFrame((y_xgb_prob+y_rf_prob+y_mlp_prob)/3, index = X.index, columns=self.xgb.classes_)
        self.proba = (y_xgb_prob+y_rf_prob+y_mlp_prob)/3
        self.classes_ = self.xgb.classes_
        self.prediction = self.classes_[self.proba.argmax(axis=1)]
        return self.prediction
    
    def predict_proba(self, X):
        y_xgb_prob=self.xgb.predict_proba(X)
        y_rf_prob = self.rf.predict_proba(X)
        y_mlp_prob= self.mlp.predict_proba(X)
        # self.proba = pd.DataFrame((y_xgb_prob+y_rf_prob+y_mlp_prob)/3, index = X.index, columns=self.xgb.classes_)
        self.proba = (y_xgb_prob+y_rf_prob+y_mlp_prob)/3
        return self.proba



def train_model(ml_df, non_feature_vect, target='class_target', verbose = 3, save_model = True):
    #non_features=['well_plate_name', 'well_name', 'class_target', 'value_target', 'wp_image_prop','wp_image_version', 'dict_values']
    features = list(ml_df.columns.difference(non_feature_vect))
    
    # Separating out the features
    X = ml_df[features]
    
    # Separating out the target
    y_class= ml_df[target]
    
    # Standardizing the features
    X = pd.DataFrame(StandardScaler().fit_transform(X.values), index=X.index, columns=X.columns)
    
    targets = ml_df[target].unique()
    # labels = np.sort(targets)
    
    # split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.20)
    
    combined_model = CombClass().fit(X_train, y_train)
    # print(combined_model.predict(X_test))
    
    y_pred = pd.DataFrame({'label': combined_model.predict(X_test),
                       'confidence': combined_model.predict_proba(X_test).max(axis=1),
                       'actual_label': ml_df.class_target,
                       'wellplate': ml_df.well_plate_name,
                       'well': ml_df.well_name}, index=X_test.index)
    
    
    if verbose > 0:
        from sklearn.metrics import classification_report, plot_confusion_matrix    
        print("Combined Results")
        print(classification_report(y_test, y_pred.label, target_names=targets))
    if verbose > 1:
        import matplotlib.pyplot as plt
        fig=plot_confusion_matrix(combined_model, X_test, y_test,display_labels=targets)
        fig.figure_.suptitle("Combined Model - Confusion Matrix")
        plt.show()
        
    combined_model = CombClass().fit(X, y_class)
    
    if save_model:
        print('banana')
        
    return combined_model


      
def train_model_extended(ml_df, non_feature_vect, target='class_target', verbose = 3, save_model = True):
    #non_features=['well_plate_name', 'well_name', 'class_target', 'value_target', 'wp_image_prop','wp_image_version', 'dict_values']
    features = list(ml_df.columns.difference(non_feature_vect))
    
    # Separating out the features
    x = ml_df[features]
    
    # Separating out the target
    y_class= ml_df[target]
    
    # Standardizing the features
    x = pd.DataFrame(StandardScaler().fit_transform(x.values), index=x.index, columns=x.columns)
    
    targets = ml_df[target].unique()
    labels = np.sort(targets)
    
    # split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x, y_class, test_size=0.20)
    
    # Predict mlp
    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(256,128,64,32),activation="relu",random_state=1).fit(X_train, y_train)
    y_mlp_pred=mlp.predict(X_test)
    y_mlp_prob=mlp.predict_proba(X_test)
    
    
    #Predict Random Forest
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=50,max_features='log2')
    rf.fit(X_train, y_train)
    y_rf_pred=rf.predict(X_test)
    y_rf_prob=rf.predict_proba(X_test)
    
    # Predict XGB
    import xgboost as XGB
    xgb = XGB.XGBClassifier()
    xgb.fit(X_train, y_train)
    y_xgb_pred = xgb.predict(X_test)
    y_xgb_prob = xgb.predict_proba(X_test)
    print(y_rf_prob)


    # Combiner
    y_tot_prob = pd.DataFrame((y_xgb_prob+y_rf_prob+y_mlp_prob)/3, index = X_test.index, columns=labels)
    y_pred = pd.DataFrame({'label': y_tot_prob.idxmax(axis=1),
                           'confidence': y_tot_prob.max(axis=1),
                           'actual_label': ml_df.class_target,
                           'wellplate': ml_df.well_plate_name,
                           'well': ml_df.well_name}, index=X_test.index)#.dropna()
    
    
    if verbose > 0:
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import classification_report, plot_confusion_matrix, ConfusionMatrixDisplay
        import matplotlib.pyplot as plt
        
        print("Combined Results")
        print(classification_report(y_test, y_pred.label, target_names=targets))
        cm=confusion_matrix(y_test, y_pred.label) #, labels=targets
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=targets)
        disp.plot()
        disp.figure_.suptitle("Combined Confusion Matrix")
        plt.show()
    if verbose >1:
        print("XGB Results")
        print(classification_report(y_test, y_xgb_pred, target_names=targets))
        print("R4nD0m F0r357 Results")
        print(classification_report(y_test, y_rf_pred, target_names=targets))
        print("MLP Results")
        print(classification_report(y_test, y_mlp_pred, target_names=targets))
    if verbose > 2:
        fig=plot_confusion_matrix(xgb, X_test, y_test,display_labels=targets)
        fig.figure_.suptitle("XGB Confusion Matrix")
        plt.show()
        fig=plot_confusion_matrix(rf, X_test, y_test,display_labels=targets)
        fig.figure_.suptitle("R4nD0m F0r357 Confusion Matrix")
        plt.show()
        fig=plot_confusion_matrix(mlp, X_test, y_test,display_labels=targets)
        fig.figure_.suptitle("MLP Confusion Matrix")
        plt.show()

    return 0



    # from sklearn.metrics import classification_report
    # print("RandomForestClassifier")
    # print(classification_report(y_test, y_pred, target_names=targets))
    
    # feat = pd.Series(clf.feature_importances_, index = features)
    
    # plt.figure(figsize=(10,30))
    # feat.sort_values().plot(kind='barh')








    
    # print("xgboost")
    # print(classification_report(y_test, y_pred, target_names=targets))
    
    # feat = pd.Series(xgb.feature_importances_, index = features)
    
    # plt.figure(figsize=(10,30))
    # feat.sort_values().plot(kind='barh')
    # plt.show()














# #%% Machine learning
# import pandas as pd
    

# ml_df_name = 'all_ml_df.pkl'

# from well_plate_project.config import data_dir
# path = data_dir / 'processed' / 'luce_nat'  #UV luce_nat
# df_file = path /  ml_df_name
# assert df_file.is_file()
# ml_df_FULL = pd.read_pickle(str(df_file))




# #%% Machine learning


# ml_df = ml_df_FULL[ml_df_FULL['mock']==False]


# from sklearn.preprocessing import StandardScaler

# non_features=['well_plate_name', 'well_name', 'class_target', 'value_target', 'wp_image_prop','wp_image_version', 'dict_values']
# features = ml_df.columns.difference(non_features)


# #clan features
# #features_red = [col for col in features if not col.endswith('mean')]
# #features_red = [col for col in features_red if not col.endswith('skewness')]
# #features = features_red


# # Separating out the features
# x = ml_df.loc[:, features].values
# # Separating out the target
# y_real = ml_df.loc[:,['value_target']].values
# y_class= ml_df.loc[:,['class_target']].values
# y_real_class = ml_df.loc[:,['value_target','class_target']].values
# # Standardizing the features
# x = StandardScaler().fit_transform(x)


# targets = ml_df['class_target'].unique()

# #%% Predict


# #TODO linear, tanto per  (non ha senso su molte features)
# #SVM
# #MLP

# #TODO random forest regressor + MAPE, RMSE

# #%% Linear


# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split( x, y_real_class, test_size=0.25, random_state=42)

# import numpy as np
# from sklearn.linear_model import LinearRegression
# model = LinearRegression().fit(X_train, y_train[:,0])

# y_pred = model.predict(X_test)


# from sklearn import metrics
# print("LinearRegression:")
# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test[:,0], y_pred))
# print('Mean Squared Error:', metrics.mean_squared_error(y_test[:,0], y_pred))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test[:,0], y_pred)))

# df_linear_regressor = pd.DataFrame({'Actual': y_test[:,0],'Actual_class': y_test[:,1], 'Predicted': y_pred,'DiffPerc': np.abs(y_test[:,0]-y_pred)/y_pred, 'DiffAbs': np.abs(y_test[:,0]-y_pred) })
# df_linear_regressor

# #%% SVM

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split( x, y_class.ravel(), test_size=0.20, random_state=42)



# #Import svm model
# from sklearn import svm


# #We’ll create two objects from SVM, to create two different classifiers; one with Polynomial kernel, and another one with RBF kernel:
# #Train the model using the training sets
# rbf = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(X_train, y_train)
# poly = svm.SVC(kernel='poly', degree=3, C=1).fit(X_train, y_train)


# #To calculate the efficiency of the two models, we’ll test the two classifiers using the test data set:
# poly_pred = poly.predict(X_test)
# rbf_pred = rbf.predict(X_test)

# from sklearn.metrics import accuracy_score, f1_score
# #Finally, we’ll calculate the accuracy and f1 scores for SVM with Polynomial kernel:
# poly_accuracy = accuracy_score(y_test, poly_pred)
# poly_f1 = f1_score(y_test, poly_pred, average='weighted')
# print('Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy*100))
# print('F1 (Polynomial Kernel): ', "%.2f" % (poly_f1*100))

# #In the same way, the accuracy and f1 scores for SVM with RBF kernel:
# rbf_accuracy = accuracy_score(y_test, rbf_pred)
# rbf_f1 = f1_score(y_test, rbf_pred, average='weighted')
# print('Accuracy (RBF Kernel): ', "%.2f" % (rbf_accuracy*100))
# print('F1 (RBF Kernel): ', "%.2f" % (rbf_f1*100))


# from sklearn.metrics import classification_report
# print("SVC-PolynomialKernel")
# print(classification_report(y_test, poly_pred, target_names=targets))
# print("SVC-RBF")
# print(classification_report(y_test, rbf_pred, target_names=targets))


# #%% Predict mlp


# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split( x, y_class.ravel(), test_size=0.20, random_state=42)

# from sklearn.neural_network import MLPClassifier
# clf = MLPClassifier(hidden_layer_sizes=(256,128,64,32),activation="relu",random_state=1).fit(X_train, y_train)
# y_pred=clf.predict(X_test)

# print("MLPClassifier")
# print(classification_report(y_test, y_pred, target_names=targets))
# from sklearn.metrics import plot_confusion_matrix
# import matplotlib.pyplot as plt
# fig=plot_confusion_matrix(clf, X_test, y_test,display_labels=targets)
# fig.figure_.suptitle("Confusion Matrix ")
# plt.show()







# #%% Predict

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split( x, y_class.ravel(), test_size=0.25, random_state=42)


# from sklearn.ensemble import RandomForestClassifier
# from sklearn.datasets import make_classification
# clf = RandomForestClassifier(n_estimators=50)
# clf.fit(X_train, y_train)


# y_pred = clf.predict(X_test)


# from sklearn.metrics import classification_report
# print("RandomForestClassifier")
# print(classification_report(y_test, y_pred, target_names=targets))


# feat = pd.Series(clf.feature_importances_, index = features)

# plt.figure(figsize=(10,30))
# feat.sort_values().plot(kind='barh')

# import xgboost as XGB
# xgb = XGB.XGBClassifier()
# xgb.fit(X_train, y_train)
# y_pred = xgb.predict(X_test)

# print("xgboost")
# print(classification_report(y_test, y_pred, target_names=targets))

# feat = pd.Series(xgb.feature_importances_, index = features)

# plt.figure(figsize=(10,30))
# feat.sort_values().plot(kind='barh')
# plt.show()

# #print(classification_report(y_true, y_pred, target_names=target_names))


# #%% INIT

# def clear_all():
#     """Clears all the variables from the workspace of the application."""
#     gl = globals().copy()
#     for var in gl:
#         if var[0] == '_': continue
#         if 'func' in str(globals()[var]): continue
#         if 'module' in str(globals()[var]): continue

#         del globals()[var]



# if __name__ == "__main__":
#     #clear_all()
    
#     ml_df_name = 'all_ml_df.pkl'
    
#     from well_plate_project.config import data_dir
#     path = data_dir / 'processed' / 'luce_nat_easy'
#     df_file = path /  ml_df_name
#     assert df_file.is_file()
#     ml_df = pd.read_pickle(str(df_file))
    
    
    




















