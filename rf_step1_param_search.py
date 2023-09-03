# -*- coding: utf-8 -*-

import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn import neighbors, svm, tree, ensemble, linear_model, neural_network, naive_bayes
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import multiprocessing
from multiprocessing import Process,Queue,Pool


def find_params(para_dict, estimator, x_train, y_train):
    gsearch = GridSearchCV(estimator, param_grid=para_dict, scoring = None,
                           n_jobs=-1,  cv=5)
    gsearch.fit(x_train, y_train)
    return gsearch.best_params_, gsearch.best_score_


def run_n_estimators(x_train, y_train):
    clf = ensemble.RandomForestClassifier(
            n_estimators=2,             
            #criterion='mse',             
            max_depth=None,              
            min_samples_split=2,         
            min_samples_leaf=1,         
            max_features='sqrt',         
            max_leaf_nodes=None,        
            bootstrap=True,              
            min_weight_fraction_leaf=0,
            #class_weight="balanced",
            random_state=20,
            n_jobs=-1)                   
    
    # 1 iterations
    i=1
    param_test1 = {
        'n_estimators': [i for i in range(1, 201, 5)]
    }
    best_params, best_score = find_params(param_test1, clf, x_train, y_train)
    print('model_rf', i, ':')
    print(best_params, ':best_score:', best_score)
    clf.set_params(n_estimators=best_params['n_estimators'])
    return best_params, best_score

def run_search(x_train, y_train):
    clf = ensemble.RandomForestClassifier(
            n_estimators=2,             
            #criterion='mse',             
            max_depth=None,              
            min_samples_split=2,         
            min_samples_leaf=1,          
            max_features='auto',        
            max_leaf_nodes=None,         
            bootstrap=True,              
            min_weight_fraction_leaf=0,
            #class_weight="balanced",
            random_state=20,
            n_jobs=5)                   
    
    # 1 terations
    i=1
    param_test1 = {
        'n_estimators': [i for i in range(1, 201, 2)]
    }
    best_params, best_score = find_params(param_test1, clf, x_train, y_train)
    print('model_rf', i, ':')
    print(best_params, ':best_score:', best_score)
    clf.set_params(n_estimators=best_params['n_estimators'])
    n_estimators_best = best_params
    
    # 2.1 max_depth, min_samples_split, min_samples_leaf
    param_test2_1 = {
        'max_depth': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        'min_samples_split' : [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        'min_samples_leaf' : [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    }
    best_params, best_score = find_params(param_test2_1, clf, x_train, y_train)

    # 2.2 max_depth, min_samples_split, min_samples_leaf
    max_d = best_params['max_depth']
    min_ss = best_params['min_samples_split']
    min_sl = best_params['min_samples_leaf']
    param_test2_2 = {
        'max_depth': [max_d-2, max_d, max_d+2],
        'min_samples_split': [min_ss-2, min_ss, min_ss+2],
        'min_samples_leaf' : [min_sl-2, min_sl, min_sl+2]
    }
    best_params, best_score = find_params(param_test2_2, clf, x_train, y_train)
    clf.set_params(max_depth=best_params['max_depth'], min_samples_split=best_params['min_samples_split'],
                   min_samples_leaf=best_params['min_samples_leaf'])
    print('model_rf', i, ':')
    print(best_params, ':best_score:', best_score)
    max_depth_best = best_params

    # 3.1 max_features：
    param_test3_1 = {
        'max_features': [0.3, 0.5, 0.7, 0.9]
    }
    best_params, best_score = find_params(param_test3_1, clf, x_train, y_train)

    #3.2 max_features
    max_f = best_params['max_features']
    param_test3_2 = {
        'max_features': [max_f-0.1, max_f, max_f+0.1]
    }
    best_params, best_score = find_params(param_test3_2, clf, x_train, y_train)
    clf.set_params(max_features=best_params['max_features'])
    print('model_rf', i, ':')
    print(best_params, ':best_score:', best_score)
    max_features_best = best_params
     
    return n_estimators_best, max_depth_best,max_features_best,best_score

if __name__ == '__main__':
    data = pd.read_excel('data.xlsx')#data   
    df_result = pd.DataFrame()   
    x = data.iloc[:,1:]
    y = data.iloc[:,0]          
    n_estimators_best, max_depth_best,max_features_best,best_score = run_search(x,y)
    abc = dict(**n_estimators_best,**max_depth_best, **max_features_best)
    aaa = pd.DataFrame.from_dict(abc, orient='index')
    bbb = aaa.T
    bbb['input'] = 'trust'
    df_result = df_result.append(bbb)
    df_result.to_csv('param_result.csv')
