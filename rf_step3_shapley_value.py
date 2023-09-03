# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import shap

#data
data = pd.read_excel('.\data.xlsx')#data 
y = data.loc[:,['trust']]
y = y.values
x = data.iloc[:,1:]
X = x.values

#model fit
model = RandomForestClassifier(n_estimators= 69, max_depth=18, random_state=20, min_samples_leaf=40, 
                                   min_samples_split=3, max_features = 0.5)
model.fit(x,y)

explainer = shap.TreeExplainer(model,x) # tree explainer 
shap_values = explainer.shap_values(x)  # shapley values
shap.summary_plot(shap_values[0], x, show = False) # figure
plt.savefig('./shap.png')




