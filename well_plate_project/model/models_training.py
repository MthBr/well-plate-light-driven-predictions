#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 09:49:54 2020

@author: enzo
"""
# import warnings
# warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np 

from sklearn.base import ClassifierMixin, RegressorMixin

class CombClass(ClassifierMixin):
    def __init__(self):
        return
    
    def _buildmodels(self):
        from sklearn.neural_network import MLPClassifier
        mlp = MLPClassifier(hidden_layer_sizes=(256,128,64,32),activation="relu",random_state=1, max_iter=1000)
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=50,max_features='log2')
        import xgboost as XGB
        xgb = XGB.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        self.mlp = mlp
        self.rf = rf
        self.xgb = xgb
    
    def fit(self, X, y):
        self._buildmodels()
        
        self.mlp = self.mlp.fit(X,y)
        self.rf = self.rf.fit(X,y)
        self.xgb = self.xgb.fit(X,type(y)([list(self.rf.classes_).index(element)for element in y]))
        self.classes_ = self.rf.classes_
        return self
     
    def predict(self, X, model='combined'):
        if model == 'combined':
            y_xgb_prob=self.xgb.predict_proba(X)
            y_rf_prob = self.rf.predict_proba(X)
            y_mlp_prob= self.mlp.predict_proba(X)
            # self.proba = pd.DataFrame((y_xgb_prob+y_rf_prob+y_mlp_prob)/3, index = X.index, columns=self.xgb.classes_)
            self.proba = (y_xgb_prob+y_rf_prob+y_mlp_prob)/3
            self.prediction = self.classes_[self.proba.argmax(axis=1)]
            
        elif model == 'xgb': 
            self.prediction = self.rf.classes_[self.xgb.predict(X)]
        elif model == 'rf':
            self.prediction = self.rf.predict(X)
        elif model == 'mlp':
            self.prediction = self.mlp.predict(X)
        
        return self.prediction
    
    def predict_proba(self, X):
        y_xgb_prob=self.xgb.predict_proba(X)
        y_rf_prob = self.rf.predict_proba(X)
        y_mlp_prob= self.mlp.predict_proba(X)
        # self.proba = pd.DataFrame((y_xgb_prob+y_rf_prob+y_mlp_prob)/3, index = X.index, columns=self.xgb.classes_)
        self.proba = (y_xgb_prob+y_rf_prob+y_mlp_prob)/3
        return self.proba
    
    
class CombRegress(RegressorMixin):
    def __init__(self):
        return
    
    
    def _buildmodels(self):
        from sklearn.neural_network import MLPRegressor
        mlp = MLPRegressor(hidden_layer_sizes=(256,128,64,32),
                           max_iter=1000,
                           activation="relu")
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=50,max_features='log2', criterion='mae')
        import xgboost as XGB
        xgb = XGB.XGBRegressor()
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
        y_xgb_pred=self.xgb.predict(X)
        y_rf_pred = self.rf.predict(X)
        y_mlp_pred= self.mlp.predict(X)
        self.prediction = (y_xgb_pred+y_rf_pred+y_mlp_pred)/3
        return self.prediction
 
    
    
def train_classification_model(ml_df, non_feature_vect, target='class_target', verbose = 3, save_model = True):
    #non_features=['well_plate_name', 'well_name', 'class_target', 'value_target', 'wp_image_prop','wp_image_version', 'dict_values']
    features = list(ml_df.columns.difference(non_feature_vect))
    
    # Separating out the features
    X = ml_df[features].copy()
    
    # Separating out the target
    y_class= ml_df[target]
    
    # Standardizing the features
    scaler_X = MinMaxScaler()
    X.loc[:,X.dtypes ==np.float] = scaler_X.fit_transform(X.loc[:,X.dtypes ==np.float].values)
    X = pd.get_dummies(X)
    
    
    targets = ml_df[target].unique()
    # labels = np.sort(targets)

    if verbose > 0:
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
            
        from sklearn.metrics import classification_report, plot_confusion_matrix    
        print("Combined Results")
        print(classification_report(y_test, y_pred.label, target_names=targets))
    if verbose > 1:
        import matplotlib.pyplot as plt
        fig=plot_confusion_matrix(combined_model, X_test, y_test,display_labels=targets)
        fig.figure_.suptitle("Combined Model - Confusion Matrix")
        plt.show()
    if verbose > 2:
        print("XGB Results")
        y_xgb_pred= combined_model.predict(X_test, model='xgb')
        y_xgb_pred = pd.Series(y_xgb_pred, index=X_test.index, name='xgb_pred')
        print(classification_report(y_test, y_xgb_pred, target_names=targets))
        print("R4nD0m F0r357 Results")
        y_rf_pred= combined_model.predict(X_test, model='rf')
        y_rf_pred = pd.Series(y_rf_pred, index=X_test.index, name='rf_pred')
        print(classification_report(y_test, y_rf_pred, target_names=targets))
        print("MLP Results")
        y_mlp_pred= combined_model.predict(X_test, model='mlp')
        y_mlp_pred = pd.Series(y_mlp_pred, index=X_test.index, name='mlp_pred')
        print(classification_report(y_test, y_mlp_pred, target_names=targets))
    if verbose > 3:
        fig=plot_confusion_matrix(combined_model.xgb, X_test,
                                  type(y_test)([list(targets).index(element)for element in y_test]),
                                  display_labels=targets)
        fig.figure_.suptitle("XGB Confusion Matrix")
        plt.show()
        fig=plot_confusion_matrix(combined_model.rf, X_test, y_test,display_labels=targets)
        fig.figure_.suptitle("R4nD0m F0r357 Confusion Matrix")
        plt.show()
        fig=plot_confusion_matrix(combined_model.mlp, X_test, y_test,display_labels=targets)
        fig.figure_.suptitle("MLP Confusion Matrix")
        plt.show()
        
    combined_model = CombClass().fit(X, y_class)
    
    if save_model:
        print('banana')
        
    return combined_model, scaler_X




def combine_class(X, mode, classification_model, y_classes):
    
    if mode== 'proba':
        # Concateno le probabilità
        X_prob = pd.DataFrame(classification_model.predict_proba(X),index=X.index,  columns=classification_model.classes_)
        X = pd.concat([X,X_prob], axis=1)
    elif mode=='pred_class':
        #Concateno le classi predette
        X = pd.concat([X,pd.Series(classification_model.predict(X),index = X.index, name='class_target')], axis=1)
    elif mode=='real_class':  # cattivone
        #Concateno le classi vere
        X = pd.concat([X,pd.Series(y_classes[X.index],index = X.index, name='class_target')], axis=1)  
    return X


def train_class_regression_model(ml_df, non_feature_vect, mode = 'proba', target_class=None,
                           target_reg = 'value_target',verbose = 3, save_model = True):
    
    #non_features=['well_plate_name', 'well_name', 'class_target', 'value_target', 'wp_image_prop','wp_image_version', 'dict_values']
    features = list(ml_df.columns.difference(non_feature_vect))
    
    # Separating out the features
    X = ml_df[features].copy()
    
    # Separating out the target
    y_class= ml_df[target_class]
    y_reg = ml_df[target_reg]
    
    # Standardizing the features  
    scaler_X = MinMaxScaler()
    X.loc[:,X.dtypes ==np.float] = scaler_X.fit_transform(X.loc[:,X.dtypes ==np.float].values)
    X = pd.get_dummies(X)
    
    targets = ml_df[target_class].unique()
    # labels = np.sort(targets)
    
    if verbose >0: 
        # split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.20)
        
        combined_classifier= CombClass().fit(X_train, y_train_class)  
        
        X_train = combine_class(X_train, mode,  combined_classifier, y_class)
        X_test = combine_class(X_test, mode, combined_classifier, y_class)
            
        X_train = pd.get_dummies(X_train); X_test = pd.get_dummies(X_test)
        
        y_train_reg = y_reg[y_train_class.index]; y_test_reg = y_reg[y_test_class.index]
        scaler_y = MinMaxScaler()
        y_train_reg = pd.Series(scaler_y.fit_transform(y_train_reg.values.reshape(-1, 1)).ravel(),
                                index=y_train_reg.index, name=target_reg)
        combined_regression = CombRegress().fit(X_train, y_train_reg)
        from sklearn import metrics
        print("Combined Results")
        y_pred = scaler_y.inverse_transform(combined_regression.predict(X_test).reshape(-1, 1)).ravel()
        y_pred = pd.Series(y_pred, index=X_test.index, name='comb_pred')
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_reg.values, y_pred.values))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_reg.values, y_pred.values)))
    if verbose > 1:
        print("R4nD0m F0r357 Results")
        y_rf_pred = scaler_y.inverse_transform(combined_regression.rf.predict(X_test).reshape(-1, 1)).ravel()
        y_rf_pred = pd.Series(y_rf_pred, index=X_test.index, name='rf_pred')
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_reg.values, y_rf_pred.values))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_reg.values, y_rf_pred.values)))
        
        print("XGB Results")
        y_xgb_pred = scaler_y.inverse_transform(combined_regression.xgb.predict(X_test).reshape(-1, 1)).ravel()
        y_xgb_pred = pd.Series(y_xgb_pred, index=X_test.index, name='rf_pred')
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_reg.values, y_xgb_pred.values))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_reg.values, y_xgb_pred.values)))
        
        print("MLP Results")
        y_mlp_pred = scaler_y.inverse_transform(combined_regression.mlp.predict(X_test).reshape(-1, 1)).ravel()
        y_mlp_pred = pd.Series(y_mlp_pred, index=X_test.index, name='rf_pred')
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_reg.values, y_mlp_pred.values))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_reg.values, y_mlp_pred.values)))
        
    
    
    
    combined_classifier= CombClass().fit(X, y_class)  
    X = combine_class(X, mode, combined_classifier, y_class) 
    X = pd.get_dummies(X)
    scaler_y = MinMaxScaler()
    y_reg = pd.Series(scaler_y.fit_transform(y_reg.values.reshape(-1, 1)).ravel(),
                                index=y_reg.index, name=target_reg)  
    combined_regression = CombRegress().fit(X, y_reg)
    
    
    return combined_classifier, combined_regression, scaler_X, scaler_y



def train_regression_model(ml_df, non_feature_vect, target_reg = 'value_target',verbose = 3, save_model = True):

    #non_features=['well_plate_name', 'well_name', 'class_target', 'value_target', 'wp_image_prop','wp_image_version', 'dict_values']
    features = list(ml_df.columns.difference(non_feature_vect))
    
    # Separating out the features
    X = ml_df[features].copy()
    
    # Separating out the target
    y_reg= ml_df[target_reg]
    
    # Standardizing the features
    scaler_X = MinMaxScaler()
    X.loc[:,X.dtypes ==np.float] = scaler_X.fit_transform(X.loc[:,X.dtypes ==np.float].values)  
    X = pd.get_dummies(X)
    
    if verbose >0:
        from sklearn import metrics
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.20)
        
        scaler_y = MinMaxScaler()
        y_train_reg = pd.Series(scaler_y.fit_transform(y_train_reg.values.reshape(-1, 1)).ravel(),
                                index=y_train_reg.index, name=target_reg)    
        combined_regression = CombRegress().fit(X_train, y_train_reg)
        
        print("Combined Results")
        y_pred = scaler_y.inverse_transform(combined_regression.predict(X_test).reshape(-1, 1)).ravel()
        y_pred = pd.Series(y_pred, index=X_test.index, name='comb_pred')
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_reg.values, y_pred.values))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_reg.values, y_pred.values)))
        
     
        
    
    
    scaler_y = MinMaxScaler()
    y_reg = pd.Series(scaler_y.fit_transform(y_reg.values.reshape(-1, 1)).ravel(),
                            index=y_reg.index, name=target_reg) 
    combined_regression = CombRegress().fit(X, y_reg)


    return  combined_regression, scaler_X, scaler_y













