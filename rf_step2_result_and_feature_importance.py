# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn import metrics
import matplotlib.pyplot as plt

data = pd.read_excel('.\data.xlsx')#data   
df_result = pd.DataFrame()  
model_exp = RandomForestClassifier(n_estimators= 69, max_depth=18, random_state=20, min_samples_leaf=40, 
                                   min_samples_split=3, max_features = 0.5)# parameters from step1
x = data.iloc[:,1:]
x = x.astype(float)
y = data.iloc[:, 0]
y = y.values
feature_list = x.columns.values.tolist() 
x = x.values

model = model_exp
df_result = pd.DataFrame()
split_n = 10
kf = KFold(n_splits = split_n,shuffle = True,random_state = None) # ten-fold cross-validation
aa = 0
bb = 0
cc = 0
j=0 
df_importance = pd.DataFrame(feature_list)
plt.clf()
for train_index, test_index in kf.split(x): 
    j=j+1               
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index] 
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    precision_  = metrics.precision_score(y_test, y_pred) #precision
    recall_  = metrics.recall_score(y_test, y_pred) #recall
    f1_ = metrics.f1_score(y_test, y_pred) #f1 value
    print('precision = ', precision_)
    print('recall = ', recall_)
    print('f1 = ', f1_)
    aa = aa + precision_
    bb = bb + recall_ 
    cc = cc + f1_
    list_combine = [[precision_,recall_,f1_]]
    df1 = pd.DataFrame(list_combine)
    df_result = df_result.append(df1)
    importance = model.feature_importances_ #feature importance
    df_imp =pd.DataFrame(importance)
    df_importance = pd.concat([df_importance, df_imp], axis= 1)

precision_m = aa/split_n
recall_m  = bb/split_n
f1_m = cc/split_n
df_result.to_csv('rf_result.csv') 
df_importance.to_csv('feature_importances.csv')  



