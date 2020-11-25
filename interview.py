# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 22:42:27 2020

@author: tjwag
"""

import os
from numpy import percentile
import matplotlib.pyplot as plt
import pandas as pd
import math
import seaborn as sns

os.chdir('C:/Users/tjwag/Desktop/MS Data Analytics/')
df = pd.read_csv('interview_dataset.csv')

#visualize data
quartiles = percentile(df['production'], [0, 25, 50, 75, 100])
plt.hist(x=df['production'], bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Production (Mboe)')
plt.ylabel('Frequency')
plt.title('Histogram of Production')
plt.show()

df_corr = df.corr()

#visiualize corr
corr = df_corr
ax = sns.heatmap(corr,vmin=-1, vmax=1, center=0,cmap=sns.diverging_palette(20, 220, n=200),    square=True)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45,horizontalalignment='right')
plt.show(sns)

#remove values correlation coefficients >-.8 and <-.8    
df_fc = df_corr[((df_corr>-.80)&(df_corr<.80))|(df_corr==1)]
df_filt = df_fc.dropna(1) 
df_cols_ccfilt = df_filt.columns  

#filter out colinear variables from main dataset
df_ccfilt = df[df_cols_ccfilt]

#normalize values in each column using min/max 
df_scaled = df_ccfilt
df_scaled.columns = df_cols_ccfilt

#impute missing values with KNN to enable regression alogrithm
import numpy as np
from sklearn.impute import KNNImputer
import math
from matplotlib import pyplot
k = math.sqrt(len(df_scaled['production']))
imputer = KNNImputer(n_neighbors=31, weights="uniform")
df_scaled_imputed = imputer.fit_transform(df_scaled)
df_scaled_imputed = pd.DataFrame(df_scaled_imputed)
df_scaled_imputed.columns = df_cols_ccfilt 

#randomly sample for training (85%)and test sets (15%)
import random
train_set_index = random.sample(list(df_scaled_imputed.index),int(round(len(df_scaled_imputed)*.85,0)))
test_set_index = set(df_scaled_imputed.index) - set(train_set_index)

#create test and training DFs
df_test = df_scaled_imputed.loc[test_set_index,:]
df_test.columns = df_cols_ccfilt
df_train = df_scaled_imputed.loc[train_set_index,:]
df_train.columns = df_cols_ccfilt

#training dataset
train_ind = df_train.iloc[:,:-1]
train_dep = df_train.iloc[:,-1]

#test dataset 
test_ind = df_test.iloc[:,:-1]
ind_cols = test_ind.columns
test_dep = df_test.iloc[:,-1]

#DecisionTreeRegress algortihm  
from sklearn.tree import DecisionTreeRegressor
regr = DecisionTreeRegressor(min_samples_leaf=10)
model = regr.fit(train_ind,train_dep)
pred_regr = regr.predict(test_ind)

#assess model
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error 
correlation, p_value = pearsonr(pred_regr,test_dep)
mean_error = mean_absolute_error(pred_regr,test_dep)
print('\n Using DecisionTreeRegression ML algo + KNN (n = 39) data imputation:')
print('Correlation: ' + str(round(correlation,3)), ', p-val: ' + str(round(p_value,3)))
print('Mean absolute error: ' + str(round(mean_error,3)))


#decision tree, KNN imputed data, CART feature importance 
importance = model.feature_importances_
# summarize feature importance
pyplot.barh([x for x in range(len(importance))], importance, tick_label = list(ind_cols))
plt.ylabel('Features')
plt.xlabel('Importance')
plt.title('Feature importance')
plt.xlim([0,.18])
pyplot.show()

pyplot.scatter(test_dep,pred_regr)
m, b = np.polyfit(test_dep,pred_regr, 1)
plt.plot(test_dep, m*test_dep + b)
plt.ylabel('Predicted Production (MBOE)')
plt.xlabel('Actual Production (MBOE)')
plt.show()



######Alternative: drop high % NaN (Water sat & treatment press) and impute those with lower missing NaN % 



df_scaled = pd.DataFrame(df_scaled)
df_scaled2 = df_scaled.drop('breakdown pressure', axis = 1)
df_scaled3 = df_scaled2.drop('water saturation', axis = 1)
alt_cols = df_scaled3.columns
k = math.sqrt(len(df_scaled['production']))
#print(k)
imputer = KNNImputer(n_neighbors=31, weights="uniform")
df_scaled_imputed2 = imputer.fit_transform(df_scaled3)
df_scaled_imputed2 = pd.DataFrame(df_scaled_imputed2)
df_scaled_imputed2.columns = alt_cols

#randomly sample for training (80%)and test sets (20%)
import random

#create test and training DFs
df_test2 = df_scaled_imputed2.loc[test_set_index,:]
df_test2.columns = alt_cols
df_train2 = df_scaled_imputed2.loc[train_set_index,:]
df_train2.columns = alt_cols

#training dataset
train_ind2 = df_train2.iloc[:,:-1]
train_dep2 = df_train2.iloc[:,-1]

#test dataset 
test_ind2 = df_test2.iloc[:,:-1]
ind_cols2 = test_ind2.columns
test_dep2 = df_test2.iloc[:,-1]

#DecisionTreeRegress algortihm  
from sklearn.tree import DecisionTreeRegressor
regr2 = DecisionTreeRegressor(min_samples_leaf = 10)
model2 = regr2.fit(train_ind2,train_dep2)
pred_regr2 = regr2.predict(test_ind2)

#assess model
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error 
correlation2, p_value2 = pearsonr(pred_regr2,test_dep2)
mean_error2 = mean_absolute_error(pred_regr2,test_dep2)
print('\n Using DecisionTreeRegression ML algo + KNN (n = 39) data imputation, minus water sat + ISIPs:')
print('Correlation: ' + str(round(correlation2,3)), ', p-val: ' + str(round(p_value2,3)))
print('Mean absolute error: ' + str(round(mean_error2,3)))


#decision tree, KNN imputed data (less water sat + breakdown press), CART feature importance 
importance2 = model2.feature_importances_
pyplot.barh([x for x in range(len(importance2))], importance2, tick_label = list(alt_cols[:-1]))
plt.ylabel('Features')
plt.xlabel('Importance')
plt.title('Feature importance')
plt.xlim([0,.18])
pyplot.show()
pyplot.scatter(test_dep2,pred_regr2)
m, b = np.polyfit(test_dep2,pred_regr2, 1)
plt.plot(test_dep2, m*test_dep2 + b)
plt.ylabel('Predicted Production (MBOE)')
plt.xlabel('Actual Production (MBOE)')
pyplot.show()



######### Refine Model using Top 5 most influential Features ##############

#normalize values in each column using min/max 
df_scaled3 = df_scaled[['proppant volume','isip','youngs modulus','tvd (ft)','production']]
cols_lim = df_scaled3.columns

#impute missing values with KNN to enable regression alogrithm
imputer = KNNImputer(n_neighbors=31, weights="uniform")
df_scaled_imputed3 = imputer.fit_transform(df_scaled3)
df_scaled_imputed3 = pd.DataFrame(df_scaled_imputed3)
df_scaled_imputed3.columns = cols_lim

#create test and training DFs
df_test3 = df_scaled_imputed3.loc[test_set_index,:]
df_test3.columns = cols_lim
df_train3 = df_scaled_imputed3.loc[train_set_index,:]
df_train3.columns = cols_lim

#training dataset
train_ind3 = df_train3.iloc[:,:-1]
train_dep3 = df_train3.iloc[:,-1]

#test dataset 
test_ind3 = df_test3.iloc[:,:-1]
ind_cols3 = test_ind3.columns
test_dep3 = df_test3.iloc[:,-1]

#DecisionTreeRegress algortihm  
from sklearn.tree import DecisionTreeRegressor
regr3 = DecisionTreeRegressor(min_samples_leaf = 15, max_depth = None)
model3 = regr3.fit(train_ind3,train_dep3)
pred_regr3 = regr3.predict(test_ind3)

#assess model
correlation3, p_value3 = pearsonr(pred_regr3,test_dep3)
mean_error3 = mean_absolute_error(pred_regr3,test_dep3)
print('\n Using DecisionTreeRegression ML algo + KNN (n = 39) data imputation, Top 5 features:')
print('Correlation: ' + str(round(correlation3,3)), ', p-val: ' + str(round(p_value3,3)))
print('Mean absolute error: ' + str(round(mean_error3,3)))

#decision tree, KNN imputed data (less water sat + breakdown press), CART feature importance 
importance3 = model3.feature_importances_
pyplot.barh([x for x in range(len(importance3))], importance3, tick_label = list(cols_lim[:-1]))
plt.ylabel('Features')
plt.xlabel('Importance')
plt.title('Feature importance')
plt.xlim([0,.5])
pyplot.show()

#predicted v actual visual
pyplot.scatter(test_dep3,pred_regr3)
m, b = np.polyfit(test_dep3,pred_regr3, 1)
plt.plot(test_dep3, m*test_dep3 + b)
plt.title('Top features predicting production')
plt.ylabel('Predicted Production (MBOE)')
plt.xlabel('Actual Production (MBOE)')


