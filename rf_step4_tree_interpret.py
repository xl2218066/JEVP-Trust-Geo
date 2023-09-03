# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 15:20:51 2023

@author: Xu Liang
"""

from __future__ import division
from IPython.display import Image, display
#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor,\
                         export_graphviz
from treeinterpreter import treeinterpreter as ti
from sklearn.ensemble import RandomForestClassifier
import sys
sys.path.append(r'E:\1 Geo Psychol 2 - Trust\R1\tree_interpret') 
from tree_interp_functions import *

# Set default matplotlib settings
plt.rcParams['figure.figsize'] = (10, 7)
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['figure.titlesize'] = 26
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 16

# Set seaborn colours
sns.set_style('darkgrid')
sns.set_palette('colorblind')
blue, green, red, purple, yellow, cyan, xl1, xl2, xl3, xl4 = sns.color_palette('colorblind')

#shap.initjs()  
data = pd.read_excel(r'.\data.xlsx', header = 0)#data
y = data.loc[:,['trust']]
x = data.iloc[:,1:]
X = x.values
rf_bin_clf = RandomForestClassifier(n_estimators= 69, max_depth=18, random_state=20, min_samples_leaf=40, 
                                   min_samples_split=3, max_features = 0.5)
rf_bin_clf.fit(x,y)

# contribution
rf_bin_clf_pred, rf_bin_clf_bias, rf_bin_clf_contrib = ti.predict(rf_bin_clf, x)

#figure-all
df = plot_obs_feature_contrib(rf_bin_clf,
                              rf_bin_clf_contrib,
                              x,
                              y,
                              3,
                              order_by='contribution',
                              violin=True
                             )
plt.tight_layout()
plt.savefig('contribution_plot_violin_rf.png')

#non-liner analysis for each input
##Elevation Mean
plt.close()
plot_single_feat_contrib('Elevation Mean', rf_bin_clf_contrib, x,
                         class_index=1, add_smooth=True, frac=0.1)
plt.savefig(r'.\Elevation Mean_contribution_rf.png')

##Elevation STD
plt.close()
plot_single_feat_contrib('Elevation STD', rf_bin_clf_contrib, x,
                         class_index=1, add_smooth=True, frac=0.1)
plt.savefig('.\Elevation STD_contribution_rf.png')

##Elevation CV
plt.close()
plot_single_feat_contrib('Elevation CV', rf_bin_clf_contrib, x,
                         class_index=1, add_smooth=True, frac=0.1)
plt.savefig(r'.\Elevation CV_contribution_rf.png')




