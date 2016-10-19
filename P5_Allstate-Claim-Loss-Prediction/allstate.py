# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 22:52:41 2016

@author: jianzhang
"""
import pandas as pd
import numpy as np
from time import time
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error as mae
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from scipy.stats import pearsonr

#Read Allstate dataset
allstate = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
allstate.head()
test.head()

#Factorize categorical variables
feature_col = list(allstate.columns[1:-1])

cats = [name for name in feature_col if 'cat' in name]
for name in cats:
    allstate[name] = pd.factorize(allstate[name], sort=True)[0]
    test[name] = pd.factorize(test[name], sort=True)[0]
 
#Data Exploration
len(cats)   #Number of categorical variables
len(feature_col)- len(cats)  #Number of numerical variables
allstate.isnull().sum().sum()  #Check for missing value
allstate['loss'].describe()  #Range of target variable is huge

#EDA on loss
sns.distplot(allstate["loss"])  #We should scaling targer variable
sns.distplot(np.log(allstate["loss"]))  #Ideal normal distribution
plt.xlabel("Log(loss)")

#EDA on predict variables
numerical_col = [name for name in feature_col if name not in cats]
sns.boxplot(allstate[numerical_col])  #Boxplot for 14 numerical variables

colMatrix = allstate[numerical_col].corr() #Covariance matrix
sns.heatmap(colMatrix, square=True)        #Heatmap of numerical variables
   
#Extract Feature and target columns
X = allstate[feature_col]
y = np.log(allstate['loss'])
predict_X = test[feature_col]

#Try Linear Regression as benchmark algorithm
reg_A = linear_model.LinearRegression()
start = time()
reg_A.fit(X,y)
end = time()
print "Linear Regression Finish in {:.4f}".format(end - start)
predict_A = reg_A.predict(predict_X)
#mae(np.exp(y), np.exp(predict_A))
#Submission
submission = pd.read_csv("sample_submission.csv")
submission.iloc[:, 1] = np.exp(predict_A)
submission.to_csv('linear.csv', index=None)

#Ridge Regression
ridge = linear_model.RidgeCV(alphas = [0.1, 1, 10, 50])
ridge.fit(X, y)
alpha_ridge = ridge.alpha_
reg_B = linear_model.Ridge(alpha = alpha_ridge)
start = time()
reg_B.fit(X,y)
end = time()
print "Ridge Regression Finish in {:.4f}".format(end - start)

predict_B = reg_B.predict(predict_X)
#mae(y, predict_B)
#Submission
submission = pd.read_csv("sample_submission.csv")
submission.iloc[:, 1] = np.exp(predict_B)
submission.to_csv('ridge.csv', index=None)

#LASSO Regression
lasso = linear_model.LassoCV(alphas = [0.1, 1, 10, 50])
lasso.fit(X, y)
alpha_lasso = lasso.alpha_
reg_C = linear_model.Lasso(alpha = alpha_lasso)
start = time()
reg_C.fit(X,y)
end = time()
print "LASSO Regression Finish in {:.4f}".format(end - start)
#Submission
predict_C = reg_C.predict(predict_X)
#mae(y, predict_C)
submission = pd.read_csv("sample_submission.csv")
submission.iloc[:, 1] = np.exp(predict_C)
submission.to_csv('lasso.csv', index=None)

#Random Forest
reg_D = RandomForestRegressor(n_estimators = 50)
reg_D.fit(X,y)
print "Random Forest Finish"
predict_D = reg_D.predict(predict_X)
submission = pd.read_csv("sample_submission.csv")
submission.iloc[:, 1] = np.exp(predict_D)
submission.to_csv('rf.csv', index=None)

#Feature Importance
importances = reg_D.feature_importances_
std = np.std([tree.feature_importances_ for tree in reg_D.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12,6))
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()

#XGBoost
xgdmat = xgb.DMatrix(X, y)
testmat = xgb.DMatrix(predict_X)
params = {'eta': 0.01, 'subsample': 0.8, 'colsample_bytree': 0.8, 
          'objective': 'reg:linear','min_child_weight':3} 
        
reg_E = xgb.train(params, xgdmat, 1000)
predict_E = reg_E.predict(testmat)
submission = pd.read_csv("sample_submission.csv")
submission.iloc[:, 1] = np.exp(predict_E)
submission.to_csv('xgboost.csv', index=None)

#Ensemble models together
ridge_sub = pd.read_csv("ridge.csv")
rf_sub = pd.read_csv("rf1000.csv")
xg_sub = pd.read_csv("xgboost.csv")
ridge_pre = ridge_sub.iloc[:, 1]
rf_pre = rf_sub.iloc[:, 1]
xg_pre = xg_sub.iloc[:, 1]

#Calculate Pearson Correlation
pearsonr(ridge_pre, xg_pre)
pearsonr(rf_pre, xg_pre)

#Ensemble Ridge Regression with Xgboost
ensemble1 = 0.9*xg_pre + 0.1*ridge_pre
ensemble2 = 0.8*xg_pre + 0.2*ridge_pre
ensemble3 = 0.5*xg_pre + 0.3*rf_pre + 0.2*ridge_pre
ensemble4 = 0.5*xg_pre + 0.4*rf_pre + 0.2*ridge_pre
ensemble5 = 0.6*xg_pre + 0.3*rf_pre + 0.1*ridge_pre

submission.iloc[:, 1] = ensemble5
submission.to_csv('ensemble5.csv', index=None)
