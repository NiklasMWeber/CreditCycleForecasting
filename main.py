# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 11:22:08 2021

@author: Niklas Weber
"""

import tools
import numpy as np
from esig import tosig as ts
import sklearn.preprocessing as pre
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# =============================================================================
# Test computing signature and logsig:
# 
# stream = np.array([[1.0,1.0], [3.0,4.0],[5.0,2.0],[8.0,6.0]])
#                    
# depth = 3
# 
# sig = ts.stream2sig(stream,depth)
# logsig = ts.stream2logsig(stream,depth)
# print(ts.sigdim(2,3))
# print(sig) 
# print("\n",logsig)
# =============================================================================

# Import Data
x_1,x_2,x_3,x_4_1,x_4_2,y = tools.importData()
X,Y,year = tools.prepareData(x_1,x_2,x_3,x_4_1,x_4_2,y)

# Standardize Data
scaler = pre.StandardScaler()
X_scaled = scaler.fit_transform(X) #-mean() --> /std
max_abs_scaler = pre.MaxAbsScaler()
Y_scaled = max_abs_scaler.fit_transform(Y-np.mean(Y))# -mean --> range [-1,1]



'''
# Split in train and test
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled,Y_scaled, test_size = 0.3)

# Train and fit ridge regression
ridge_reg = Ridge(alpha = 0, fit_intercept=True)
ridge_reg.fit(X_train,Y_train)

predict_test = ridge_reg.predict(X_test)
predict_train = ridge_reg.predict(X_train)

# Plot
fig = plt.figure()
fig1 = fig.add_subplot(2,1,1)
xAxes = range(1,np.shape(predict_test)[0]+1)
fig1.plot(xAxes, Y_test, label = 'true')
fig1.plot(xAxes, predict_test, label = 'predict')
fig1.legend()
fig1.set_title('Testing Set')
print('Test R = ', ridge_reg.score(X_test, Y_test))

fig2 = fig.add_subplot(2,1,2)
xAxes = range(1,np.shape(predict_train)[0]+1)
fig2.plot(xAxes, Y_train, label = 'true')
fig2.plot(xAxes, predict_train, label = 'predict')
fig2.legend()
fig2.set_title('Training Set')
print('Train R = ', ridge_reg.score(X_train, Y_train))
'''