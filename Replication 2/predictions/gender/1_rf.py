#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import sklearn as sk
import pyreadr
import pickle
import time
from sklearn.inspection import plot_partial_dependence
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


# In[ ]:


result = pyreadr.read_r('Data/df_gender.RData') 
df_miss = result["df_miss"] 
df_test = result["df_test"] 
df_train= result["df_train"]

y_train = df_train['sex_ref_dem']
X_train = df_train.drop(['sex_ref_dem', 'data_id'], axis=1)
y_test = df_test['sex_ref_dem']
X_test = df_test.drop(['sex_ref_dem', 'data_id'], axis=1)
X_test.dtypes


# In[ ]:


#Build the model:
param_grid = {
    'n_estimators': [100, 200, 300, 500, 800],    #ntrees number of trees in the foreset
    'max_depth': [5, 10, 25, 50, 75],          #max_depth max number of levels in each decision tree
    'max_samples': [0.632, 1.0],        #sample_rate
    'min_samples_leaf': [1, 3, 5],      #min_rows min number of data points allowed in a leaf node
    'min_samples_split': [2,5, 10],     # min number of data points placed in a node before the node is split
    'max_features': ['sqrt', 0.3]    #mtries
}

rf = RandomForestClassifier(bootstrap=True, oob_score=True)
grid_rf = RandomizedSearchCV(rf, param_grid, cv = 10, verbose = 2, n_iter = 250)
grid_rf.fit(X_train, y_train)


# In[ ]:


#Evaluate the model:
print("accuracy on training set: %f" % grid_rf.best_estimator_.score(X_train, y_train))
print("accuracy on test set: %f" % grid_rf.best_estimator_.score(X_test, y_test))

# Compute the accuracy scores
accuracy = accuracy_score(y_test, grid_rf.best_estimator_.predict(X_test))
balanced_accuracy = balanced_accuracy_score(y_test, grid_rf.best_estimator_.predict(X_test))
precision = precision_score(y_test, grid_rf.best_estimator_.predict(X_test), average='macro')
recall = recall_score(y_test, grid_rf.best_estimator_.predict(X_test), average='macro')
f1 = f1_score(y_test, grid_rf.best_estimator_.predict(X_test), average='macro')

# Print the accuracy scores
print("Accuracy:", accuracy)
print("Balanced accuracy:", balanced_accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print(classification_report(y_test, grid_rf.best_estimator_.predict(X_test)))


# In[ ]:


d = X_test.copy()
d['trueY']= y_test
d['data_id'] = df_test['data_id']
d.reset_index(inplace=True)
d[['predY0','predY1']] = grid_rf.best_estimator_.predict_proba(X_test)
d['predY'] = grid_rf.best_estimator_.predict(X_test)


# In[ ]:


d = X_test.copy()
d['trueY']= y_test
d['data_id'] = df_test['data_id']
d.reset_index(inplace=True)
d[['predY0','predY1']] = grid_rf.best_estimator_.predict_proba(X_test)
d['predY'] = grid_rf.best_estimator_.predict(X_test)
d['grp'] = 'test'

e = X_train.copy()
e['trueY']= y_train
e['data_id'] = df_train['data_id']
e.reset_index(inplace=True)
e[['predY0','predY1']] = grid_rf.best_estimator_.predict_proba(X_train)
e['predY'] = grid_rf.best_estimator_.predict(X_train)
e['grp'] = 'train'

X = df_miss.drop(['sex_ref_dem', 'data_id'], axis=1)
f = X.copy()
f['data_id'] = df_miss['data_id']
f.reset_index(inplace=True)
f[['predY0','predY1']] = grid_rf.best_estimator_.predict_proba(X)
f['predY'] = grid_rf.best_estimator_.predict(X)
f['grp'] = ''

final = d.append(e, ignore_index=True)
final = final.append(f, ignore_index=True)
final.to_csv('Output/rf_predictions.csv', index = False)


# In[ ]:


pickle.dump(grid_rf.best_estimator_, open('Output/rf_model.pkl', 'wb'))


# In[ ]:


from sklearn.inspection import permutation_importance
start_time = time.time()
result = permutation_importance(
    grid_rf.best_estimator_, X_test, y_test, n_repeats=1000, random_state=42, n_jobs=10
)
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

feature_names = [f"{X_test.columns[i]}" for i in range(X_test.shape[1])]
forest_importances = pd.Series(result.importances_mean, index=feature_names)

mean = pd.Series(result.importances_mean, index=feature_names, name = 'mean').to_frame()
std = pd.Series(result.importances_std, index=feature_names, name = 'std').to_frame()
df=pd.concat([mean,std],axis=1)
df.to_csv('Output/rf_varimp.csv', index = True)


# In[ ]:




